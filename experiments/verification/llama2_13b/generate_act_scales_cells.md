# Verification: Generate Activation Scales — Llama-2-13B

Generates `llama-2-13b.pt` via `smoothquant_repo/examples/generate_act_scales.py` with the cached Pile-val calibration set on Drive, then saves to Drive for reuse by `full_table_cells.md`.

Uses `NousResearch/Llama-2-13b-hf` — ungated mirror of `meta-llama/Llama-2-13b-hf` (same SHAs, no HF token / license-acceptance needed). The smoothquant repo's `get_act_scales` is model-agnostic (driven by `AutoModelForCausalLM` + per-Linear pre-hooks).

**Hardware:** Llama-2-13B in fp16 ≈ 26 GB. **A100-40GB required** (fits with ~10 GB headroom for calibration activations). Expect ~25-40 min for 512 samples × 512 seq_len.

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

# Reuse the cached Pile-val calibration file already on Drive.
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

## Cell 2: Generate activation scales for Llama-2-13B

```python
%cd /content/llm-quantization-thesis/smoothquant_repo

!python examples/generate_act_scales.py \
    --model-name NousResearch/Llama-2-13b-hf \
    --output-path act_scales/llama-2-13b.pt \
    --dataset-path dataset/val.jsonl.zst \
    --num-samples 512 \
    --seq-len 512
```

---

## Cell 3: Save scales to Drive + sanity check

```python
import os, torch
SRC = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/llama-2-13b.pt"
DST_DIR = "/content/drive/MyDrive/thesis_results/act_scales"
os.makedirs(DST_DIR, exist_ok=True)

!cp {SRC} {DST_DIR}/

scales = torch.load(SRC, map_location="cpu")
print(f"#entries: {len(scales)}")

sample_keys = [k for k in scales.keys() if "layers.0." in k]
print("layer-0 keys:")
for k in sample_keys:
    print(f"  {k:55s} shape={tuple(scales[k].shape)}  max={scales[k].max().item():.3f}")

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

`llama-2-13b.pt` present at both `smoothquant_repo/act_scales/llama-2-13b.pt` and `/content/drive/MyDrive/thesis_results/act_scales/llama-2-13b.pt`. Dict with 280 entries (40 layers × 7 linears).

After this cell, run [`full_table_cells.md`](full_table_cells.md) — it consumes only the Drive copy.
