# Verification: Generate Activation Scales — Falcon-7B

Generates `falcon-7b.pt` via `smoothquant_repo/examples/generate_act_scales.py` with the cached Pile-val calibration set on Drive.

Uses `tiiuae/falcon-7b` — open access, no HF token. The smoothquant repo's calibration code is model-agnostic; it hooks pre-Linear forwards via `AutoModelForCausalLM`, so Falcon's fused `query_key_value` and `dense_h_to_4h` linears get recorded under their natural HF names without any model-specific patching.

**Hardware:** Falcon-7B in fp16 ≈ 13.5 GB. **A100-40GB recommended.** Expect ~15-25 min for 512 samples × 512 seq_len.

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm einops

from google.colab import drive
drive.mount('/content/drive')

import os, shutil

# Reuse cached Pile-val calibration file on Drive.
DRIVE_DATA = "/content/drive/MyDrive/thesis_results/datasets/val.jsonl.zst"
REPO_DATA  = "/content/llm-quantization-thesis/smoothquant_repo/dataset/val.jsonl.zst"
assert os.path.exists(DRIVE_DATA), f"missing: {DRIVE_DATA} — build it once via Task 04 cell 1."
os.makedirs(os.path.dirname(REPO_DATA), exist_ok=True)
shutil.copy2(DRIVE_DATA, REPO_DATA)

os.makedirs("/content/llm-quantization-thesis/smoothquant_repo/act_scales", exist_ok=True)

!nvidia-smi
!python -c "from smoothquant.calibration import get_act_scales; print('smoothquant OK')"
```

---

## Cell 2: Generate activation scales for Falcon-7B

```python
%cd /content/llm-quantization-thesis/smoothquant_repo

!python examples/generate_act_scales.py \
    --model-name tiiuae/falcon-7b \
    --output-path act_scales/falcon-7b.pt \
    --dataset-path dataset/val.jsonl.zst \
    --num-samples 512 \
    --seq-len 512
```

---

## Cell 3: Save scales to Drive + sanity check

```python
import os, torch
SRC = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/falcon-7b.pt"
DST_DIR = "/content/drive/MyDrive/thesis_results/act_scales"
os.makedirs(DST_DIR, exist_ok=True)

!cp {SRC} {DST_DIR}/

scales = torch.load(SRC, map_location="cpu")
print(f"#entries: {len(scales)}")

# Falcon naming: transformer.h.<i>.self_attention.query_key_value
#                transformer.h.<i>.mlp.dense_h_to_4h
sample_keys = [k for k in scales.keys() if "h.0." in k]
print("layer-0 keys:")
for k in sample_keys:
    print(f"  {k:60s} shape={tuple(scales[k].shape)}  max={scales[k].max().item():.3f}")

import re
LAYER_RE = re.compile(r"transformer\.h\.(\d+)\.(.+)")
sevs = []
for name, vec in scales.items():
    m = LAYER_RE.match(name)
    if not m: continue
    if m.group(2) not in ("self_attention.query_key_value", "mlp.dense_h_to_4h"): continue
    v = vec.float().abs()
    sevs.append((int(m.group(1)), m.group(2), (v.max()/v.median().clamp(min=1e-12)).item()))
sevs.sort()
print(f"\nseverity (max/median) — first 4 layers:")
for s in sevs[:8]:
    print(f"  layer {s[0]:2d}  {s[1]:32s}  {s[2]:7.2f}×")
```

---

## Expected output

`falcon-7b.pt` present at both `smoothquant_repo/act_scales/falcon-7b.pt` and `/content/drive/MyDrive/thesis_results/act_scales/falcon-7b.pt`. Falcon-7B uses `parallel_attn=True` with a single `input_layernorm` absorbing into both QKV and FFN, so the two recorded sites (`query_key_value` and `dense_h_to_4h`) share the same input distribution — their per-channel max vectors should look very similar.

After this cell, run [`full_table_cells.md`](full_table_cells.md) — it consumes only the Drive copy.
