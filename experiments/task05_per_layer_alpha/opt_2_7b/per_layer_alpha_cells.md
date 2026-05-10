# Task 05: Per-Layer α — OPT-2.7B (W8A8, O1 + C schemes)

## What this notebook does

Same experiment as `opt_1_3b/per_layer_alpha_cells.md`, on OPT-2.7B. The 1.3B run produced **+0.10 PPL improvement** for C + per-layer α over C + global α=0.5; we expect the gain to be at least as large here because the per-layer severity spread is **22.9×** on 2.7B vs 13.1× on 1.3B (more headroom for a layer-aware α).

Same diagnostic-driven α(l) formula, max-based scales only, 32 decoder layers.

**Formula (zero free parameters):**

```
sev(l)   = mean over {q_proj, fc1} of  max(|X|) / median(|X|)   for layer l
sev_norm = sev(l) / max_l sev(l)         ∈ [0, 1]
α(l)     = α_min + (α_max − α_min) · sev_norm(l)

α_min = 0.5,  α_max = 0.9
```

**Runs (4 total):**
1. O1 global α=0.5
2. O1 per-layer α(l)
3. C global α=0.5
4. C per-layer α(l)

All 4 in one session — within-notebook deltas are noise-free.

**Hardware:** OPT-2.7B fp16 ≈ 5.3 GB; bf16 same. Fits comfortably on T4 (14.5 GB) or L4. Each PPL run ~10–15 min on T4, faster on A100/L4.

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

# Copy Drive scales into smoothquant_repo path so this notebook reads from the
# exact same on-disk path Task 01's notebook uses for OPT-2.7B.
DRIVE_SCALES = "/content/drive/MyDrive/thesis_results/act_scales/opt-2.7b.pt"
REPO_SCALES  = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-2.7b.pt"
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

SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-2.7b.pt"
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

print("Per-layer α schedule (OPT-2.7B):")
for l, s, a in zip(layers, sev.tolist(), alpha_per_layer):
    print(f"  layer {l:>2}:  severity={s:>6.2f}  α={a:.3f}")

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(layers, sev.tolist(), 'o-', color="#d62728", label="severity (max/median)")
ax1.set_xlabel("decoder layer index"); ax1.set_ylabel("severity", color="#d62728")
ax2 = ax1.twinx()
ax2.plot(layers, alpha_per_layer, 's--', color="#1f77b4", label="α(l)")
ax2.set_ylabel("α(l)", color="#1f77b4")
ax2.set_ylim(ALPHA_MIN - 0.05, ALPHA_MAX + 0.05)
plt.title("OPT-2.7B: severity-driven per-layer α")
plt.tight_layout(); plt.show()

# OPT-2.7B has 32 decoder layers
assert len(alpha_per_layer) == 32, f"expected 32 layers, got {len(alpha_per_layer)}"
```

---

## Cell 3: Per-layer-aware smoothing helper

Identical helper as the 1.3B notebook — OPT layout is the same, only depth differs.

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

## Cell 4: PPL eval loop (4 runs)

Same eval protocol: WikiText-2 raw, 40 samples × 2048-token windows.

```python
import torch, torch.nn as nn, json, time, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model

MODEL = "facebook/opt-2.7b"
SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-2.7b.pt"
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
    with open(f"{SAVE_DIR}/opt-2.7b_{run['label']}.json", "w") as f:
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

with open(f"{SAVE_DIR}/opt-2.7b_summary.json", "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nsaved -> {SAVE_DIR}/opt-2.7b_summary.json")
```

---

## Expectation

On 1.3B: C per-layer beat C global by **+0.0961 PPL** (severity spread 13.1×). On 2.7B the spread is 22.9× — if the diagnostic→α coupling is real, the C-side gain should be similar or larger. O1 may again show only a noise-floor difference (per-tensor weights bottleneck the activation-side improvement).
