# Verification: Full WikiText-2 PPL table — OPT-13B

Single-session re-run that re-computes one clean PPL row per configuration of interest and prints a final table. All 8 runs in one Colab kernel so within-table deltas are noise-free.

**Runs (8 total):**

| # | Label              | Smoothing                       | Quantizer (W / A)             |
|---|--------------------|---------------------------------|-------------------------------|
| 1 | FP16               | —                               | none (fp16)                   |
| 2 | Naive W8A8         | none                            | per_tensor / per_tensor       |
| 3 | O1 max α=0.5       | max,   α=0.5                    | per_tensor / per_token        |
| 4 | O2 max α=0.5       | max,   α=0.5                    | per_tensor / per_tensor       |
| 5 | C  pct (Task02 best) | percentile p=0.995, α=0.5      | per_channel / per_token       |
| 6 | C  max α=0.5       | max,   α=0.5                    | per_channel / per_token       |
| 7 | O1 per-layer α     | max,   α(l) (Task 05 schedule)  | per_tensor / per_token        |
| 8 | C  per-layer α     | max,   α(l) (Task 05 schedule)  | per_channel / per_token       |

> ⚠ **Row 5 (p, α) placeholder.** The Task 02 percentile sweep on 13B has not been run yet (PROGRESS.md status: "cells ready"). Defaults `p=0.995, α=0.5` follow the 2.7B winner and the 6.7B verification placeholder. **Replace `PCT_P` / `PCT_A` in Cell 4 with the actual Task 02 13B winner once that sweep produces one.** With severity spread 46.3× on 13B (the sharpest of the ladder), it would not be surprising if the winner drifts back toward higher p (more conservative percentile) — confirm before locking the row.

**Prereqs on Drive:**
- `act_scales/opt-13b.pt` — for max-based smoothing (rows 3,4,6,7,8) and α(l) severity build (rows 7,8). Already generated (Task 05 severity diagnostic used it).
- `act_percentiles/opt-13b/p0.995.pt` — for row 5. Needs Task 02 calibration to be run on 13B first; calibration buffer is ~21 GB and lives in CPU RAM, so use Colab Pro+ High-RAM for that step (verification eval itself only needs A100-40GB).

**Hardware:**
- **A100-40GB sufficient** for the verification eval. OPT-13B in bf16 ≈ 26 GB; forward pass at seq_len=2048 needs ~2-4 GB peak activations; total ~28-30 GB with comfortable headroom on a 40 GB card. T4 / L4 / A100-MIG / 24 GB cards are **not** options at seq_len=2048.
- Per-run wallclock: ~7-10 min on A100-40GB. Full 8-run sweep ≈ 60-80 min end-to-end.
- **Don't reduce seq_len to fit memory** — that would invalidate cross-model PPL comparability with the rest of the ladder (CLAUDE.md Challenge #2).

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm

from google.colab import drive
drive.mount('/content/drive')

import os, shutil
SAVE_DIR = "/content/drive/MyDrive/thesis_results/verification/opt-13b"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mirror max-scales into smoothquant_repo path (same convention as Task 01/05)
DRIVE_SCALES = "/content/drive/MyDrive/thesis_results/act_scales/opt-13b.pt"
REPO_SCALES  = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-13b.pt"
assert os.path.exists(DRIVE_SCALES), f"missing: {DRIVE_SCALES}"
os.makedirs(os.path.dirname(REPO_SCALES), exist_ok=True)
shutil.copy2(DRIVE_SCALES, REPO_SCALES)

# Verify percentile file for Task 02 winner row (CHANGE p value if Task 02 13B winner is different)
PCT_PATH = "/content/drive/MyDrive/thesis_results/act_percentiles/opt-13b/p0.995.pt"
if not os.path.exists(PCT_PATH):
    print(f"⚠  WARNING: missing {PCT_PATH}")
    print(f"   Either run task02 generate_act_percentiles_cells.md for 13B first,")
    print(f"   or row 5 will be auto-skipped from the RUNS list in Cell 4.")
else:
    print("pct scales :", PCT_PATH)

print("max scales :", REPO_SCALES)
!nvidia-smi
!free -g  # CPU RAM check — High-RAM Colab gives ~83 GB; standard ~12 GB
```

---

## Cell 2: Percentile-smoothing helper (for row 5)

Same helper as in the 1.3B / 2.7B / 6.7B verification files.

```python
%%writefile /content/percentile_smooth.py
"""Percentile-based smoothing for SmoothQuant (OPT only)."""
import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer


@torch.no_grad()
def _per_channel_weight_stat(fcs, p_w, dtype):
    stacked = torch.cat([fc.weight.abs() for fc in fcs], dim=0)
    if p_w >= 1.0:
        scales = stacked.max(dim=0).values
    else:
        scales = stacked.float().quantile(p_w, dim=0).to(stacked.dtype)
    return scales.to(dtype).clamp_(min=1e-5)


@torch.no_grad()
def smooth_ln_fcs_pct(ln, fcs, act_scales, alpha=0.5, p_w=1.0):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype).clamp_(min=1e-5)
    weight_scales = _per_channel_weight_stat(fcs, p_w, dtype).to(device)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm_pct(model, scales, alpha=0.5, p_w=1.0):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_pct(attn_ln, qkv, qkv_input_scales, alpha, p_w)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs_pct(ffn_ln, fc1, fc1_input_scales, alpha, p_w)
```

---

## Cell 3: Per-layer α schedule (for rows 7, 8)

OPT-13B has **40 decoder layers** (hidden=5120). Severity spread on 13B is **46.3×** (Task 05 diagnostic) — the sharpest of the ladder, so the α(l) range will be the most pronounced (essentially the full [0.5, 0.9] envelope).

```python
import re, torch

SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-13b.pt"
ALPHA_MIN, ALPHA_MAX = 0.5, 0.9

raw = torch.load(SCALES_PATH, map_location="cpu")
LAYER_RE = re.compile(r"model\.decoder\.layers\.(\d+)\.(.+)")

sev_by_layer = {}
for name, vec in raw.items():
    m = LAYER_RE.match(name)
    if not m:
        continue
    suffix = m.group(2)
    if not (suffix.startswith("self_attn.q_proj") or suffix == "fc1" or suffix.startswith("fc1")):
        continue
    layer = int(m.group(1))
    v = vec.float().abs()
    s = (v.max() / v.median().clamp(min=1e-12)).item()
    sev_by_layer.setdefault(layer, []).append(s)

layers = sorted(sev_by_layer.keys())
sev = torch.tensor([sum(sev_by_layer[l]) / len(sev_by_layer[l]) for l in layers])
sev_norm = sev / sev.max()
alpha_per_layer = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sev_norm).tolist()

assert len(alpha_per_layer) == 40, f"expected 40 layers, got {len(alpha_per_layer)}"
print(f"per-layer α range: [{min(alpha_per_layer):.3f}, {max(alpha_per_layer):.3f}]  (40 layers)")
print(f"non-default count: {sum(1 for a in alpha_per_layer if abs(a-0.5)>1e-3)}/40")
print(f"severity spread (max/min): {sev.max().item() / sev.min().item():.2f}×")
```

```python
%%writefile /content/smooth_per_layer.py
"""Per-layer α extension of smoothquant.smooth.smooth_lm (OPT only)."""
import re
import torch
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from smoothquant.smooth import smooth_ln_fcs

LAYER_RE = re.compile(r"model\.decoder\.layers\.(\d+)$")

@torch.no_grad()
def smooth_lm_per_layer(model, scales, alpha_schedule):
    if not isinstance(alpha_schedule, (list, tuple)):
        raise TypeError("alpha_schedule must be a list/tuple of floats")
    for name, module in model.named_modules():
        if not isinstance(module, OPTDecoderLayer):
            continue
        m = LAYER_RE.match(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        alpha = float(alpha_schedule[layer_idx])

        attn_ln = module.self_attn_layer_norm
        qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
        qkv_input_scales = scales[name + ".self_attn.q_proj"]
        smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

        ffn_ln = module.final_layer_norm
        fc1 = module.fc1
        fc1_input_scales = scales[name + ".fc1"]
        smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
```

---

## Cell 4: Run all 8 configurations

Same WikiText-2-raw eval protocol as Task 01/02/05: 40 samples × 2048-token windows.

```python
import sys
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")

import torch, torch.nn as nn, json, time, tqdm, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from percentile_smooth import smooth_lm_pct
from smooth_per_layer import smooth_lm_per_layer

MODEL = "facebook/opt-13b"
SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-13b.pt"
SAVE_DIR    = "/content/drive/MyDrive/thesis_results/verification/opt-13b"

# Task 02 winner placeholder for OPT-13B — REPLACE once the 13B sweep finishes.
PCT_P, PCT_A = 0.995, 0.5
PCT_PATH    = f"/content/drive/MyDrive/thesis_results/act_percentiles/opt-13b/p{PCT_P:g}.pt"

INCLUDE_PCT_ROW = os.path.exists(PCT_PATH)
if not INCLUDE_PCT_ROW:
    print(f"⚠  Skipping row 5 — {PCT_PATH} not found.")


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
        self.n_samples = n_samples
    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n = self.n_samples
        for i in tqdm.tqdm(range(n), desc="PPL"):
            batch = self.dataset[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * 2048)
        return torch.exp(torch.stack(nlls).sum() / (n * 2048))


print("Loading tokenizer + dataset + scales...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL)
dataset    = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator  = Evaluator(dataset, tokenizer, "cuda")
act_scales = torch.load(SCALES_PATH)
pct_scales = torch.load(PCT_PATH) if INCLUDE_PCT_ROW else None

# Quantizer presets
O1_QPARAMS    = dict(weight_quant="per_tensor",  act_quant="per_token",  quantize_bmm_input=True)
O2_QPARAMS    = dict(weight_quant="per_tensor",  act_quant="per_tensor", quantize_bmm_input=True)
C_QPARAMS     = dict(weight_quant="per_channel", act_quant="per_token",  quantize_bmm_input=True)
NAIVE_QPARAMS = dict(weight_quant="per_tensor",  act_quant="per_tensor", quantize_bmm_input=True)

RUNS = [
    {"label": "1_FP16",            "smooth": "none",                                "qparams": None},
    {"label": "2_Naive_W8A8",      "smooth": "none",                                "qparams": NAIVE_QPARAMS},
    {"label": "3_O1_max_a0.5",     "smooth": ("max", 0.5),                          "qparams": O1_QPARAMS},
    {"label": "4_O2_max_a0.5",     "smooth": ("max", 0.5),                          "qparams": O2_QPARAMS},
]
if INCLUDE_PCT_ROW:
    RUNS.append({"label": f"5_C_pct{PCT_P}_a{PCT_A}", "smooth": ("pct", PCT_A, PCT_P), "qparams": C_QPARAMS})
RUNS += [
    {"label": "6_C_max_a0.5",      "smooth": ("max", 0.5),                          "qparams": C_QPARAMS},
    {"label": "7_O1_perlayer",     "smooth": ("perlayer", alpha_per_layer),         "qparams": O1_QPARAMS},
    {"label": "8_C_perlayer",      "smooth": ("perlayer", alpha_per_layer),         "qparams": C_QPARAMS},
]

results = []
for i, run in enumerate(RUNS, 1):
    print(f"\n{'='*60}\n  Run {i}/{len(RUNS)}: {run['label']}\n{'='*60}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    sm = run["smooth"]
    if sm == "none":
        pass
    elif sm[0] == "max":
        smooth_lm(model, act_scales, sm[1])
    elif sm[0] == "pct":
        _, alpha, p = sm
        smooth_lm_pct(model, pct_scales, alpha=alpha, p_w=p)
    elif sm[0] == "perlayer":
        smooth_lm_per_layer(model, act_scales, sm[1])
    else:
        raise ValueError(f"unknown smooth spec: {sm}")

    if run["qparams"] is not None:
        model = quantize_model(model, **run["qparams"])

    ppl = evaluator.evaluate(model).item()
    elapsed = time.time() - t0
    print(f">>> {run['label']}: PPL = {ppl:.4f}  ({elapsed:.0f}s)")

    rec = {
        "model": MODEL,
        "label": run["label"],
        "smooth": (sm if isinstance(sm, str) else (sm[0] if sm[0] != "perlayer" else "perlayer")),
        "qparams": run["qparams"],
        "ppl": round(ppl, 4),
        "seconds": round(elapsed, 1),
    }
    results.append(rec)
    with open(f"{SAVE_DIR}/opt-13b_{run['label']}.json", "w") as f:
        json.dump(rec, f, indent=2)

    del model
    torch.cuda.empty_cache()

with open(f"{SAVE_DIR}/opt-13b_summary.json", "w") as f:
    json.dump({"results": results, "alpha_per_layer": alpha_per_layer}, f, indent=2)
print(f"\nsaved -> {SAVE_DIR}/opt-13b_summary.json")
```

---

## Cell 5: Final results table

```python
fp16_ppl = next(r for r in results if r["label"] == "1_FP16")["ppl"]

print(f"\n{'='*70}")
print(f"  OPT-13B — WikiText-2 PPL verification (single session)")
print(f"{'='*70}")
print(f"\n{'#':<3} {'config':<26} {'PPL':>8} {'Δ vs FP16':>12}")
print("-" * 60)
for r in results:
    delta = r["ppl"] - fp16_ppl
    print(f"{r['label'][:2]:<3} {r['label'][2:]:<26} {r['ppl']:>8.4f} {delta:>+12.4f}")

def ppl(label):
    rows = [r for r in results if r["label"] == label]
    return rows[0]["ppl"] if rows else None

print(f"\nHeadline deltas:")
print(f"  Naive W8A8 − FP16          = {ppl('2_Naive_W8A8')   - fp16_ppl:+.4f}")
print(f"  O1/max α=0.5 − FP16        = {ppl('3_O1_max_a0.5')  - fp16_ppl:+.4f}")
print(f"  O2/max α=0.5 − FP16        = {ppl('4_O2_max_a0.5')  - fp16_ppl:+.4f}")
print(f"  C/max α=0.5 − FP16         = {ppl('6_C_max_a0.5')   - fp16_ppl:+.4f}")
pct_label = f"5_C_pct{PCT_P}_a{PCT_A}"
if ppl(pct_label) is not None:
    print(f"  C/pct p={PCT_P} α={PCT_A} − FP16 = {ppl(pct_label) - fp16_ppl:+.4f}")
print(f"  O1 per-layer − O1 α=0.5    = {ppl('7_O1_perlayer')  - ppl('3_O1_max_a0.5'):+.4f}  (negative → per-layer wins)")
print(f"  C  per-layer − C  α=0.5    = {ppl('8_C_perlayer')   - ppl('6_C_max_a0.5'):+.4f}  (negative → per-layer wins)")
```

---

## Notes

- **Naive W8A8** = `weight_quant=per_tensor, act_quant=per_tensor`, no smoothing — same convention across the verification ladder for cross-model comparability. Expect a *very* large delta here: the 6.7B run already showed +15.24 PPL above FP16, and severity spread grows from 22.9× (6.7B) to 46.3× (13B), so the naive collapse should be at least as severe — possibly worse.
- **Row 5 (Task 02 winner)** is a placeholder at `(0.995, 0.5)`. **Replace this** with the actual Task 02 13B winner when that sweep runs (`PCT_P` and `PCT_A` constants in Cell 4). Cell 4 auto-skips row 5 if the percentile scales file is missing.
- **Per-layer α expectation on 13B**: based on the refined rule (per-layer α wins when global α=0.5 is suboptimal for some layers), 13B should show a per-layer C win at least as large as 6.7B's −0.043. With severity spread 46×, α(l) will span essentially the full [0.5, 0.9] envelope, and the worst-affected layers are well separated from the rest. O1 per-layer is expected to lose again (per-tensor W penalty unchanged).
- **Convergence check to watch:** if C/perlayer ≈ C/pct on 13B as it did on 1.3B and 6.7B, the "two routes to the same floor" claim is locked in across four models — strong cross-scale evidence for the thesis chapter.
- All 8 results land at `/content/drive/MyDrive/thesis_results/verification/opt-13b/`.
