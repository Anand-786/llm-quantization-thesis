# Task 01: Generate Activation Scales — OPT-6.7B

Generates `opt-6.7b.pt` via `smoothquant_repo/examples/generate_act_scales.py` with the Pile validation set, then saves to Drive for reuse by the scheme comparison and alpha sweep cells.

One-time setup. Expect ~15-25 min on T4 (512 samples × 512 seq_len). Model is ~13 GB in FP16; the script uses `device_map="sequential"` which layer-loads across available memory.

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

# Download Pile validation set (calibration data)
!mkdir -p smoothquant_repo/dataset
!wget -q -O smoothquant_repo/dataset/val.jsonl.zst \
    https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst

# Create act_scales output dir
!mkdir -p smoothquant_repo/act_scales

# Verify
!nvidia-smi
!ls -la smoothquant_repo/dataset/val.jsonl.zst
!python -c "from smoothquant.calibration import get_act_scales; print('smoothquant OK')"
```

---

## Cell 2: Generate activation scales for OPT-6.7B

```python
%cd /content/llm-quantization-thesis/smoothquant_repo

!python examples/generate_act_scales.py \
    --model-name facebook/opt-6.7b \
    --output-path act_scales/opt-6.7b.pt \
    --dataset-path dataset/val.jsonl.zst \
    --num-samples 512 \
    --seq-len 512
```

---

## Cell 3: Save scales to Drive

```python
!mkdir -p /content/drive/MyDrive/thesis_results/act_scales
!cp /content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-6.7b.pt \
    /content/drive/MyDrive/thesis_results/act_scales/

# Verify
!ls -la /content/drive/MyDrive/thesis_results/act_scales/
```

---

## Expected output

`opt-6.7b.pt` present at both `smoothquant_repo/act_scales/` and `/content/drive/MyDrive/thesis_results/act_scales/`. File size should be similar in magnitude to `opt-1.3b.pt` (a dict of per-layer input activation max-abs tensors, not the model weights).
