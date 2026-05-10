# Verification: WikiText-2 PPL table — OPT-30B (max-scales only, no percentile row)

Single-session re-run mirroring the 1.3B / 2.7B / 6.7B / 13B verification files, with **row 5 (percentile + C) removed**: percentile calibration on 30B requires a ~35 GB top-K buffer (CPU-resident) and we already know from 1.3B / 6.7B / 13B that C+per-layer-α reaches the same PPL floor as C+pct. Saves a ~30-min calibration step plus Pro+ High-RAM requirement, costs nothing the per-layer row doesn't already deliver.

**Runs (7 total):**

| # | Label              | Smoothing                       | Quantizer (W / A)             |
|---|--------------------|---------------------------------|-------------------------------|
| 1 | FP16               | —                               | none (fp16)                   |
| 2 | Naive W8A8         | none                            | per_tensor / per_tensor       |
| 3 | O1 max α=0.5       | max,   α=0.5                    | per_tensor / per_token        |
| 4 | O2 max α=0.5       | max,   α=0.5                    | per_tensor / per_tensor       |
| 6 | C  max α=0.5       | max,   α=0.5                    | per_channel / per_token       |
| 7 | O1 per-layer α     | max,   α(l) (Task 05 schedule)  | per_tensor / per_token        |
| 8 | C  per-layer α     | max,   α(l) (Task 05 schedule)  | per_channel / per_token       |

(Row numbering kept consistent with the rest of the ladder; row 5 just absent.)

**Prereqs on Drive:**
- `act_scales/opt-30b.pt` — for max-based smoothing (rows 3,4,6,7,8) and α(l) severity build (rows 7,8). Generate once via the SmoothQuant repo's `examples/generate_act_scales.py` on Pile-val 512×512; 80 GB GPU is enough (~62 GB total budget for the calibration pass — see PROGRESS.md hardware note).

**Hardware:**
- **A100-80GB or H100-80GB required.** OPT-30B fp16 ≈ 60 GB; PPL eval at seq_len=2048 needs ~3-5 GB activations on top → ~65 GB total. 40 GB cards are not an option.
- Per-run wallclock: ~15-20 min on A100-80GB. Full 7-run sweep ≈ 2-3 hours end-to-end.
- **Use safetensors** (HF's `facebook/opt-30b` ships sharded safetensors by default); `device_map="auto"` then streams weights shard-by-shard to GPU without materialising the full 60 GB on CPU first.
- **Don't reduce seq_len to fit memory** — invalidates cross-model PPL comparability with the rest of the ladder (CLAUDE.md Challenge #2).

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm safetensors

from google.colab import drive
drive.mount('/content/drive')

import os, shutil
SAVE_DIR = "/content/drive/MyDrive/thesis_results/verification/opt-30b"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mirror max-scales into smoothquant_repo path (same convention as Task 01/05)
DRIVE_SCALES = "/content/drive/MyDrive/thesis_results/act_scales/opt-30b.pt"
REPO_SCALES  = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-30b.pt"
assert os.path.exists(DRIVE_SCALES), f"missing: {DRIVE_SCALES} — run generate_act_scales.py first"
os.makedirs(os.path.dirname(REPO_SCALES), exist_ok=True)
shutil.copy2(DRIVE_SCALES, REPO_SCALES)

print("max scales :", REPO_SCALES)
!nvidia-smi
!free -g  # CPU RAM check
```

---

## Cell 2: Per-layer α schedule (for rows 7, 8)

OPT-30B has **48 decoder layers** (hidden=7168). Severity spread for 30B is not yet measured — Cell 2 prints it.

```python
import re, torch

SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-30b.pt"
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

assert len(alpha_per_layer) == 48, f"expected 48 layers, got {len(alpha_per_layer)}"
print(f"per-layer α range: [{min(alpha_per_layer):.3f}, {max(alpha_per_layer):.3f}]  (48 layers)")
print(f"non-default count: {sum(1 for a in alpha_per_layer if abs(a-0.5)>1e-3)}/48")
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

## Cell 3: Run all 7 configurations

Same WikiText-2-raw eval protocol as the rest of the ladder: 40 samples × 2048-token windows.

```python
import sys
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")

import torch, torch.nn as nn, json, time, tqdm, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from smooth_per_layer import smooth_lm_per_layer

MODEL = "facebook/opt-30b"
SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-30b.pt"
SAVE_DIR    = "/content/drive/MyDrive/thesis_results/verification/opt-30b"


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

# Quantizer presets
O1_QPARAMS    = dict(weight_quant="per_tensor",  act_quant="per_token",  quantize_bmm_input=True)
O2_QPARAMS    = dict(weight_quant="per_tensor",  act_quant="per_tensor", quantize_bmm_input=True)
C_QPARAMS     = dict(weight_quant="per_channel", act_quant="per_token",  quantize_bmm_input=True)
NAIVE_QPARAMS = dict(weight_quant="per_tensor",  act_quant="per_tensor", quantize_bmm_input=True)

# Note: row 5 (percentile + C) intentionally omitted — see header.
RUNS = [
    {"label": "1_FP16",            "smooth": "none",                                "qparams": None},
    {"label": "2_Naive_W8A8",      "smooth": "none",                                "qparams": NAIVE_QPARAMS},
    {"label": "3_O1_max_a0.5",     "smooth": ("max", 0.5),                          "qparams": O1_QPARAMS},
    {"label": "4_O2_max_a0.5",     "smooth": ("max", 0.5),                          "qparams": O2_QPARAMS},
    {"label": "6_C_max_a0.5",      "smooth": ("max", 0.5),                          "qparams": C_QPARAMS},
    {"label": "7_O1_perlayer",     "smooth": ("perlayer", alpha_per_layer),         "qparams": O1_QPARAMS},
    {"label": "8_C_perlayer",      "smooth": ("perlayer", alpha_per_layer),         "qparams": C_QPARAMS},
]

results = []
for i, run in enumerate(RUNS, 1):
    print(f"\n{'='*60}\n  Run {i}/{len(RUNS)}: {run['label']}\n{'='*60}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    sm = run["smooth"]
    if sm == "none":
        pass
    elif sm[0] == "max":
        smooth_lm(model, act_scales, sm[1])
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
    with open(f"{SAVE_DIR}/opt-30b_{run['label']}.json", "w") as f:
        json.dump(rec, f, indent=2)

    del model
    torch.cuda.empty_cache()

with open(f"{SAVE_DIR}/opt-30b_summary.json", "w") as f:
    json.dump({"results": results, "alpha_per_layer": alpha_per_layer}, f, indent=2)
print(f"\nsaved -> {SAVE_DIR}/opt-30b_summary.json")
```

---

## Cell 4: Final results table

```python
fp16_ppl = next(r for r in results if r["label"] == "1_FP16")["ppl"]

print(f"\n{'='*70}")
print(f"  OPT-30B — WikiText-2 PPL verification (single session, no percentile row)")
print(f"{'='*70}")
print(f"\n{'#':<3} {'config':<26} {'PPL':>10} {'Δ vs FP16':>14}")
print("-" * 64)
for r in results:
    delta = r["ppl"] - fp16_ppl
    print(f"{r['label'][:2]:<3} {r['label'][2:]:<26} {r['ppl']:>10.4f} {delta:>+14.4f}")

def ppl(label):
    return next(r for r in results if r["label"] == label)["ppl"]

print(f"\nHeadline deltas:")
print(f"  Naive W8A8 − FP16        = {ppl('2_Naive_W8A8')   - fp16_ppl:+.4f}")
print(f"  O1/max α=0.5 − FP16      = {ppl('3_O1_max_a0.5')  - fp16_ppl:+.4f}")
print(f"  O2/max α=0.5 − FP16      = {ppl('4_O2_max_a0.5')  - fp16_ppl:+.4f}")
print(f"  C/max α=0.5 − FP16       = {ppl('6_C_max_a0.5')   - fp16_ppl:+.4f}")
print(f"  O1 per-layer − O1 α=0.5  = {ppl('7_O1_perlayer')  - ppl('3_O1_max_a0.5'):+.4f}  (negative → per-layer wins)")
print(f"  C  per-layer − C  α=0.5  = {ppl('8_C_perlayer')   - ppl('6_C_max_a0.5'):+.4f}  (negative → per-layer wins)")
```

---

## Notes

- **Row 5 omitted by design.** Cross-ladder evidence (1.3B / 6.7B / 13B) shows C+per-layer-α reaches at least the C+pct floor; running percentile calibration on 30B would cost time and Pro+ High-RAM for confirmatory rather than discriminative information.
- **Naive W8A8 expectation**: 13B was +4315.73 PPL above FP16. 30B is likely worse — the chapter's "naive collapse" curve gets one more dramatic data point.
- **Per-layer α expectation**: based on 13B's −0.187 / −0.180 wins on C / O1, 30B should show a similar or larger per-layer win on both schemes given even sharper outliers.
- All 7 results land at `/content/drive/MyDrive/thesis_results/verification/opt-30b/`.
