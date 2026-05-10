# Task 05: Per-Layer α — OPT-1.3B (W8A8, O1 + C schemes)

## What this notebook does

We test whether a **per-layer α** tied to the layer's outlier severity (`max(|X|)/median(|X|)`) beats a **global α** under both SmoothQuant schemes:
- **O1**: per-tensor weight + per-token activation (paper's headline scheme).
- **C**: per-channel weight + per-token activation (Task 01's winner).

Why max-based only: the diagnostic was computed on max-based per-channel act-scales, so we evaluate on max-based smoothing — no percentile mixing. Headroom on 1.3B for O1/max is FP16 14.46 vs O1/global-α=0.5 ≈ 14.68, so a smarter α policy has somewhere to go.

**Formula (zero free parameters):**

```
sev(l)   = mean over {q_proj, fc1} of  max(|X|) / median(|X|)   for layer l
sev_norm = sev(l) / max_l sev(l)         ∈ [0, 1]
α(l)     = α_min + (α_max − α_min) · sev_norm(l)

α_min = 0.5,  α_max = 0.9
```

Peak-severity layers (1–3) → α ≈ 0.9; deep layers → α near 0.5; layer 0 → α near 0.5. Reads straight off the diagnostic, no tuning.

**Runs (4 total, ~15 min each on T4):**
1. O1 global α=0.5 — paper-baseline control, re-run for clean in-notebook comparison.
2. O1 per-layer α(l) — main test vs (1).
3. C global α=0.5 — clean control for the C side, re-run in same env.
4. C per-layer α(l) — does the per-layer effect stack with the per-channel-weight scheme?

All 4 runs share the same environment (transformers/torch/cuDNN versions, scales file, Evaluator instance), so the within-notebook deltas are noise-free even if absolute PPL drifts vs older Task 01 sessions.

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets

from google.colab import drive
drive.mount('/content/drive')

import os, shutil
os.makedirs("/content/drive/MyDrive/thesis_results/task05_perlayer", exist_ok=True)

# Use the EXACT same scales file Task 01 used: copy the Drive canonical file
# into smoothquant_repo/act_scales/ (Task 01's read path) and load from there.
# This eliminates any path/version ambiguity between tasks.
DRIVE_SCALES = "/content/drive/MyDrive/thesis_results/act_scales/opt-1.3b.pt"
REPO_SCALES  = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-1.3b.pt"
assert os.path.exists(DRIVE_SCALES), f"missing: {DRIVE_SCALES}"
os.makedirs(os.path.dirname(REPO_SCALES), exist_ok=True)
shutil.copy2(DRIVE_SCALES, REPO_SCALES)
print(f"copied  {DRIVE_SCALES}  ->  {REPO_SCALES}")
print(f"size: {os.path.getsize(REPO_SCALES)/1024**2:.2f} MB")
!nvidia-smi
```

---

## Cell 2: Build the per-layer α schedule from severity

```python
import re, torch
import matplotlib.pyplot as plt

SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-1.3b.pt"
ALPHA_MIN, ALPHA_MAX = 0.5, 0.9

raw = torch.load(SCALES_PATH, map_location="cpu")
LAYER_RE = re.compile(r"model\.decoder\.layers\.(\d+)\.(.+)")

# Per-layer severity = mean of (max/median) across q_proj and fc1
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

print("Per-layer α schedule (OPT-1.3B):")
for l, s, a in zip(layers, sev.tolist(), alpha_per_layer):
    print(f"  layer {l:>2}:  severity={s:>6.2f}  α={a:.3f}")

# Quick visualisation — sanity-check the curve shape matches what we saw in the diagnostic
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(layers, sev.tolist(), 'o-', color="#d62728", label="severity (max/median)")
ax1.set_xlabel("decoder layer index"); ax1.set_ylabel("severity", color="#d62728")
ax2 = ax1.twinx()
ax2.plot(layers, alpha_per_layer, 's--', color="#1f77b4", label="α(l)")
ax2.set_ylabel("α(l)", color="#1f77b4")
ax2.set_ylim(ALPHA_MIN - 0.05, ALPHA_MAX + 0.05)
plt.title("OPT-1.3B: severity-driven per-layer α")
plt.tight_layout(); plt.show()

# alpha_per_layer is a list of length 24 (OPT-1.3B has 24 decoder layers)
assert len(alpha_per_layer) == 24, f"expected 24 layers, got {len(alpha_per_layer)}"
```

---

## Cell 3: Per-layer-aware smoothing helper

`smoothquant.smooth.smooth_lm` takes a single scalar α. We wrap it: walk the OPT decoder layers ourselves, look up α from a list indexed by layer, and call the lower-level `smooth_ln_fcs` with the right per-layer α.

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
    """alpha_schedule: list[float] of length num_decoder_layers, indexed by layer idx.

    Functionally identical to smooth_lm except α is read per-layer instead of
    being a single scalar."""
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

        # Attention: smooth attn_ln + qkv with q_proj's input scale
        attn_ln = module.self_attn_layer_norm
        qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
        qkv_input_scales = scales[name + ".self_attn.q_proj"]
        smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

        # FFN: smooth ffn_ln + fc1 with fc1's input scale
        ffn_ln = module.final_layer_norm
        fc1 = module.fc1
        fc1_input_scales = scales[name + ".fc1"]
        smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

print("smooth_lm_per_layer ready")
```

```python
import sys
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")
from smooth_per_layer import smooth_lm_per_layer
print("import OK")
```

---

## Cell 4: PPL eval loop (3 global controls + 1 per-layer run)

Same eval protocol as Task 01: WikiText-2 raw, 40 samples × 2048-token windows.

```python
import torch, torch.nn as nn, json, time, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model

MODEL = "facebook/opt-1.3b"
SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-1.3b.pt"  # same path Task 01 used
SAVE_DIR = "/content/drive/MyDrive/thesis_results/task05_perlayer"

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
tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda")
act_scales = torch.load(SCALES_PATH)

O1_QPARAMS = dict(weight_quant="per_tensor",  act_quant="per_token", quantize_bmm_input=True)
C_QPARAMS  = dict(weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)

RUNS = [
    {"label": "O1_global_a0.5", "scheme": "O1", "kind": "global",   "alpha": 0.5,             "qparams": O1_QPARAMS},
    {"label": "O1_perlayer",    "scheme": "O1", "kind": "perlayer", "alpha": alpha_per_layer, "qparams": O1_QPARAMS},
    {"label": "C_global_a0.5",  "scheme": "C",  "kind": "global",   "alpha": 0.5,             "qparams": C_QPARAMS},
    {"label": "C_perlayer",     "scheme": "C",  "kind": "perlayer", "alpha": alpha_per_layer, "qparams": C_QPARAMS},
]

results = []
for i, run in enumerate(RUNS, 1):
    print(f"\n{'='*60}\n  Run {i}/{len(RUNS)}: {run['label']}\n{'='*60}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    if run["kind"] == "global":
        smooth_lm(model, act_scales, run["alpha"])
    else:
        smooth_lm_per_layer(model, act_scales, run["alpha"])
    model = quantize_model(model, **run["qparams"])
    ppl = evaluator.evaluate(model).item()
    elapsed = time.time() - t0
    print(f">>> {run['label']}: PPL = {ppl:.4f}  ({elapsed:.0f}s)")
    rec = {
        "model": MODEL, "label": run["label"], "scheme": run["scheme"], "kind": run["kind"],
        "alpha": run["alpha"], "ppl": round(ppl, 4), "seconds": round(elapsed, 1),
    }
    results.append(rec)
    with open(f"{SAVE_DIR}/opt-1.3b_{run['label']}.json", "w") as f:
        json.dump(rec, f, indent=2)
    del model; torch.cuda.empty_cache()

print("\n\nSummary:")
print(f"{'label':<22} {'scheme':<6} {'PPL':>8}")
for r in results:
    print(f"{r['label']:<22} {r['scheme']:<6} {r['ppl']:>8.4f}")

o1_global   = next(r for r in results if r["label"] == "O1_global_a0.5")
o1_perlayer = next(r for r in results if r["label"] == "O1_perlayer")
c_global    = next(r for r in results if r["label"] == "C_global_a0.5")
c_perlayer  = next(r for r in results if r["label"] == "C_perlayer")
print(f"\nO1: per-layer − global α=0.5 = {o1_global['ppl'] - o1_perlayer['ppl']:+.4f}  (positive → per-layer wins)")
print(f"C : per-layer − global α=0.5 = {c_global['ppl']  - c_perlayer['ppl']:+.4f}  (positive → per-layer wins)")

with open(f"{SAVE_DIR}/opt-1.3b_summary.json", "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nsaved -> {SAVE_DIR}/opt-1.3b_summary.json")
```

---

## What the outcome means

- **O1 per-layer beats O1 global α=0.5 by ≥ 0.05 PPL** → per-layer α is real on 1.3B/O1; repeat on 2.7B.
- **O1 per-layer ≈ O1 global (within ~0.02 PPL)** → diagnostic is real but the optimum is flat in α at this scale; try 2.7B (severity spread 23×).
- **C per-layer beats C global α=0.5 (from task01)** → effect stacks with per-channel-weight scheme; strongest result.
- **C per-layer loses to C global** → C already absorbs the per-layer variation through per-channel weight scales; per-layer α is redundant once weights are per-channel. That itself is a clean thesis observation.

Even a tie is publishable framing: "global α is provably suboptimal in principle (severity variation 13× on 1.3B, 46× on 13B), yet max-smoothing absorbs most of it" — strengthens the percentile-smoothing story (Task 02) by eliminating per-layer α as a competing explanation.
