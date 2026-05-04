# Task 02: Generate Activation Percentiles — OPT-2.7B (A100)

One calibration pass produces per-channel exact `quantile(|X|, p)` for `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}`. Implementation uses a per-channel top-K buffer sized for `p_min = 0.90` (`K ≈ 0.10·N` where `N = 512·512 = 262 144`).

Output: one `.pt` per `p` at `/content/drive/MyDrive/thesis_results/act_percentiles/opt-2.7b/p<value>.pt`. Each file is a `dict[name -> tensor[in_features]]` matching the shape of `act_scales/opt-2.7b.pt`. We also compute `p=1.0` from the same buffer in-memory and diff it against the existing `act_scales/opt-2.7b.pt` as a pipeline-correctness check, but **do not save** that row — fp16 storage in the buffer perturbs the per-channel max by up to ~0.5% relative (one fp16 ULP), which would muddy the sweep's `p=1.0` row. The real max-smoothing baseline is the Task 01 result.

Hardware: A100 (Colab Pro or JarvisLabs). GPU memory peak ≈ 17 GB (model fp16 ≈ 5.3 GB, top-K buffers fp16 ≈ 8 GB across 64 OPT-2.7B smoothing sites — 32 layers × {q_proj, fc1}, in_features = 2560 — activations + interpolation scratch).

**Prereq**: the upstream max scales `act_scales/opt-2.7b.pt` must already be on Drive (we diff against them in Cell 3). If not, generate them first via `experiments/task01_scheme_exploration/opt_2_7b/generate_act_scales_cells.md` (or the equivalent task01 procedure for this model).

Cells follow Task 01's self-contained pattern: helper Python is `%%writefile`-emitted inside the cell, so the notebook does not depend on the latest `experiments/task02_percentile_smoothing/percentile_calibration.py` being pushed to GitHub yet.

---

## Cell 1: Setup

```python
# Clone repos + install
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Pile validation set (calibration data)
!mkdir -p smoothquant_repo/dataset
!wget -q -O smoothquant_repo/dataset/val.jsonl.zst \
    https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst

# Drive output dir for the per-p percentile files
!mkdir -p /content/drive/MyDrive/thesis_results/act_percentiles/opt-2.7b

# Verify
!nvidia-smi
!ls -la /content/drive/MyDrive/thesis_results/act_scales/opt-2.7b.pt
!ls -la smoothquant_repo/dataset/val.jsonl.zst
```

---

## Cell 2: Write the top-K calibration helper

```python
%%writefile /content/percentile_calibration.py
"""Per-channel exact-percentile calibration via top-K buffers (Task 02)."""
import functools
import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def _smoothing_site_names(model):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    sites = []
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            sites.append(f"{name}.self_attn.q_proj")
            sites.append(f"{name}.fc1")
    return sites


@torch.no_grad()
def get_act_percentiles(
    model,
    tokenizer,
    dataset_path,
    percentiles=(1.0, 0.999, 0.995, 0.99, 0.95, 0.90),
    num_samples=512,
    seq_len=512,
    buffer_device="cuda",
    buffer_dtype=torch.float16,
    safety_margin=16,
):
    p_list = sorted(set(float(p) for p in percentiles))
    if not p_list:
        raise ValueError("`percentiles` must be non-empty")
    for p in p_list:
        if not (0.0 < p <= 1.0):
            raise ValueError(f"percentile {p} must be in (0, 1]")

    p_min = p_list[0]
    n_total = num_samples * seq_len
    k = max(2, min(n_total, math.ceil((1.0 - p_min) * n_total) + safety_margin))

    model.eval()
    device = next(model.parameters()).device

    site_names = _smoothing_site_names(model)
    if not site_names:
        raise RuntimeError("No OPT decoder layers found — is this an OPT model?")
    sites = {name: None for name in site_names}
    for name, module in model.named_modules():
        if name in sites and isinstance(module, nn.Linear):
            sites[name] = module
    missing = [n for n, m in sites.items() if m is None]
    if missing:
        raise RuntimeError(f"Unresolved smoothing sites: {missing[:3]}...")

    buffers = {}
    for name, lin in sites.items():
        buffers[name] = torch.full(
            (k, lin.in_features),
            fill_value=float("-inf"),
            dtype=buffer_dtype,
            device=buffer_device,
        )

    def update_buffer(name, x):
        in_features = x.shape[-1]
        new = x.reshape(-1, in_features).abs().to(buffer_dtype)
        if buffer_device == "cpu":
            new = new.cpu()
        elif new.device.type != "cuda":
            new = new.to(buffer_device)
        combined = torch.cat([buffers[name], new], dim=0)
        topk = torch.topk(combined, k=k, dim=0, largest=True, sorted=False).values
        buffers[name].copy_(topk)

    def stat_input_hook(_m, inputs, _output, name):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(x, tuple):
            x = x[0]
        update_buffer(name, x)

    hooks = []
    for name, lin in sites.items():
        hooks.append(lin.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.shuffle(seed=42)
        for i in tqdm(range(num_samples), desc="Top-K calibration"):
            input_ids = tokenizer(
                dataset[i]["text"],
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
            ).input_ids.to(device)
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    out = {f"{p:g}": {} for p in p_list}
    for name, buf in buffers.items():
        sorted_buf = torch.sort(buf, dim=0).values
        k_eff = sorted_buf.shape[0]
        for p in p_list:
            pos = p * (n_total - 1)
            idx_f = float(k_eff) - float(n_total) + pos
            if idx_f < 0:
                raise ValueError(
                    f"p={p} below buffer floor (k={k_eff}, N={n_total})"
                )
            lo = max(0, min(k_eff - 1, math.floor(idx_f)))
            hi = max(0, min(k_eff - 1, math.ceil(idx_f)))
            if hi == lo:
                val = sorted_buf[lo].clone()
            else:
                w = idx_f - lo
                val = sorted_buf[lo].float() * (1.0 - w) + sorted_buf[hi].float() * w
                val = val.to(sorted_buf.dtype)
            out[f"{p:g}"][name] = val.detach().cpu()
    return out
```

---

## Cell 3: Run calibration, validate against upstream max, save only p<1.0 files

We pass `p=1.0` into `get_act_percentiles` to get the per-channel max from the top-K buffer for an in-memory diff against the existing `act_scales/opt-2.7b.pt`. That gives us the pipeline-correctness check, but the `p=1.0` row itself is **not saved** — fp16 storage in the buffer perturbs it by up to ~0.5% relative (one fp16 ULP), and that noise is enough to confound the sweep's `p=1.0` row. The real reference baseline lives in Task 01's max-smoothing results.

```python
import sys, os, time
sys.path.insert(0, "/content")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from percentile_calibration import get_act_percentiles

MODEL = "facebook/opt-2.7b"
ORIG_MAX_PATH = "/content/drive/MyDrive/thesis_results/act_scales/opt-2.7b.pt"
DATASET_PATH = "/content/llm-quantization-thesis/smoothquant_repo/dataset/val.jsonl.zst"
OUT_DIR = "/content/drive/MyDrive/thesis_results/act_percentiles/opt-2.7b"

# Calibrate p=1.0 in-memory for the validation diff, then drop it before saving.
PERCENTILES_TO_SAVE = [0.999, 0.995, 0.99, 0.95, 0.90]
PERCENTILES_FOR_CALIBRATION = [1.0] + PERCENTILES_TO_SAVE

print("Loading tokenizer + model (fp16, GPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

print(f"Calibrating exact percentiles {PERCENTILES_FOR_CALIBRATION} via top-K buffer ...")
t0 = time.time()
act_pct = get_act_percentiles(
    model=model,
    tokenizer=tokenizer,
    dataset_path=DATASET_PATH,
    percentiles=PERCENTILES_FOR_CALIBRATION,
    num_samples=512,
    seq_len=512,
    buffer_device="cuda",        # OPT-2.7B buffers ~8 GB on A100, comfortable
    buffer_dtype=torch.float16,
)
print(f"Calibration done in {time.time() - t0:.1f}s")

# --- Pipeline-correctness check: in-memory diff of p=1.0 vs upstream max ---
orig = torch.load(ORIG_MAX_PATH)
ours_max = act_pct["1"]
common = [k for k in ours_max.keys() if k in orig]
print(f"\nValidating top-K pipeline against upstream max ({len(common)} sites)...")
max_abs_diff = 0.0
max_rel_diff = 0.0
worst_name = None
for name in common:
    a = orig[name].float().cpu()
    b = ours_max[name].float().cpu()
    abs_diff = (a - b).abs()
    rel_diff = abs_diff / a.clamp(min=1e-8)
    if abs_diff.max().item() > max_abs_diff:
        max_abs_diff = abs_diff.max().item()
        worst_name = name
    max_rel_diff = max(max_rel_diff, rel_diff.max().item())
print(f"  worst abs diff: {max_abs_diff:.6f}  (channel in '{worst_name}')")
print(f"  worst rel diff: {max_rel_diff:.6f}")
print(f"  expected: rel diff ~= one fp16 ULP (~5e-3 worst case). >1e-2 means a real bug.")
assert max_rel_diff < 1e-2, "Top-K p=1.0 disagrees with upstream max — investigate before trusting percentile rows."

# --- Save only p<1.0 files. p=1.0 is fp16-noisy; baseline comes from Task 01. ---
os.makedirs(OUT_DIR, exist_ok=True)
print(f"\nSaving {len(PERCENTILES_TO_SAVE)} per-p files (p=1.0 deliberately skipped):")
for p in PERCENTILES_TO_SAVE:
    p_key = f"{p:g}"
    out_path = os.path.join(OUT_DIR, f"p{p_key}.pt")
    torch.save(act_pct[p_key], out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  saved p={p_key:>6} -> {out_path}  ({size_mb:.2f} MB)")

print("\nKeys per file (sample):", list(act_pct[f"{PERCENTILES_TO_SAVE[0]:g}"].keys())[:3])
```

---

## Cell 4: Verify all per-p files are well-formed

```python
import torch, os

OUT_DIR = "/content/drive/MyDrive/thesis_results/act_percentiles/opt-2.7b"
for fname in sorted(os.listdir(OUT_DIR)):
    if not fname.endswith(".pt"):
        continue
    path = os.path.join(OUT_DIR, fname)
    d = torch.load(path)
    sample_name = next(iter(d.keys()))
    v = d[sample_name]
    print(f"{fname:>14} | {len(d):>3} sites | sample {sample_name} -> shape {tuple(v.shape)}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
```

Expected pattern: as `p` decreases (0.999 → 0.90), the per-channel max in each file should also decrease (we're reading lower in the sorted distribution).

---

## Notes

- **Buffer size**: `K = ⌈0.10 · 262144⌉ + 16 = 26 230` rows × `in_features=2560` per site, fp16. Total across 64 OPT-2.7B sites ≈ 8 GB GPU memory. Combined with the fp16 model (~5.3 GB) and activations, A100-40 has room to spare.
- **Hooked sites only**: 64 linears (32 layers × {q_proj, fc1}). out_proj and fc2 inputs aren't smoothed, so we skip them.
- **Reproducibility**: `dataset.shuffle(seed=42)` matches the upstream `get_act_scales`. Calibration sample order is identical to Task 01's max scales.
- **Canonical source**: same code lives in `experiments/task02_percentile_smoothing/percentile_calibration.py`. Cell 2 is a runtime copy so the notebook is self-contained on Colab.
