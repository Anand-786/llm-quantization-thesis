# Verification: Generate Activation Scales — LLaMA-7B

Generates `llama-7b.pt` via `smoothquant_repo/examples/generate_act_scales.py` with the cached Pile-val calibration set on Drive, then saves to Drive for reuse by `full_table_cells.md`.

`huggyllama/llama-7b` is open access (no HF token needed). The smoothquant repo's calibration code (`smoothquant.calibration.get_act_scales`) is model-agnostic and treats Llama via `AutoModelForCausalLM`, so no code changes are required to produce per-channel input-max scales keyed by Llama's standard module names (`model.layers.{i}.self_attn.{q,k,v,o}_proj`, `model.layers.{i}.mlp.{gate,up,down}_proj`).

**Hardware:** Llama-7B in fp16 ≈ 13.5 GB. **A100-40GB recommended.** Expect ~15-25 min for 512 samples × 512 seq_len.

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm sentencepiece

from google.colab import drive
drive.mount('/content/drive')

import os, shutil

# Reuse the cached Pile-val calibration file already on Drive (built for Task 04).
DRIVE_DATA = "/content/drive/MyDrive/thesis_results/datasets/val.jsonl.zst"
REPO_DATA  = "/content/llm-quantization-thesis/smoothquant_repo/dataset/val.jsonl.zst"
assert os.path.exists(DRIVE_DATA), f"missing: {DRIVE_DATA} — build it once via Task 04 cell 1."
os.makedirs(os.path.dirname(REPO_DATA), exist_ok=True)
shutil.copy2(DRIVE_DATA, REPO_DATA)

os.makedirs("/content/llm-quantization-thesis/smoothquant_repo/act_scales", exist_ok=True)

!nvidia-smi
!ls -la /content/llm-quantization-thesis/smoothquant_repo/dataset/val.jsonl.zst
!python -c "from smoothquant.calibration import get_act_scales; print('smoothquant OK')"
```

---

## Cell 2: Generate activation scales for LLaMA-7B

```python
%cd /content/llm-quantization-thesis/smoothquant_repo

!python examples/generate_act_scales.py \
    --model-name huggyllama/llama-7b \
    --output-path act_scales/llama-7b.pt \
    --dataset-path dataset/val.jsonl.zst \
    --num-samples 512 \
    --seq-len 512
```

---

## Cell 3: Save scales to Drive + sanity check

```python
import os, torch
SRC = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/llama-7b.pt"
DST_DIR = "/content/drive/MyDrive/thesis_results/act_scales"
os.makedirs(DST_DIR, exist_ok=True)

!cp {SRC} {DST_DIR}/

scales = torch.load(SRC, map_location="cpu")
print(f"#entries: {len(scales)}")

# Spot-check expected Llama keys
sample_keys = [k for k in scales.keys() if "layers.0." in k]
print("layer-0 keys:")
for k in sample_keys:
    print(f"  {k:55s} shape={tuple(scales[k].shape)}  max={scales[k].max().item():.3f}")

# Quick severity preview for layer 0
import re
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")
sevs = []
for name, vec in scales.items():
    m = LAYER_RE.match(name)
    if not m: continue
    if m.group(2) not in ("self_attn.q_proj", "mlp.gate_proj"): continue
    v = vec.float().abs()
    sevs.append((int(m.group(1)), m.group(2), (v.max()/v.median().clamp(min=1e-12)).item()))
sevs.sort()
print(f"\nseverity (max/median) — first 4 layers:")
for s in sevs[:8]:
    print(f"  layer {s[0]:2d}  {s[1]:18s}  {s[2]:7.2f}×")
```

---

## Expected output

`llama-7b.pt` present at both `smoothquant_repo/act_scales/llama-7b.pt` and `/content/drive/MyDrive/thesis_results/act_scales/llama-7b.pt`. Dict with 224 entries (32 layers × 7 linears).

After this cell, run [`full_table_cells.md`](full_table_cells.md) — it consumes only the Drive copy.
