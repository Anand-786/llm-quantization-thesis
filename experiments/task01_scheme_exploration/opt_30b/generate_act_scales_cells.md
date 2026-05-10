# Task 01: Generate Activation Scales — OPT-30B

Generates `opt-30b.pt` via `smoothquant_repo/examples/generate_act_scales.py` with the Pile validation set, then saves to Drive for reuse by the scheme comparison and alpha sweep cells.

**Hardware**: requires Colab Pro **A100 (80 GB)**. OPT-30B is ~60 GB in FP16 — does not fit on a 40 GB A100 at calibration-time activation footprint, and not on any smaller GPU. Same constraint applies as for OPT-6.7B / OPT-13B (see Challenges in CLAUDE.md): never reduce seq_len to fit memory; that breaks PPL comparability.

One-time setup. Expect ~60-90 min on A100-80GB (512 samples × 512 seq_len). The script uses `device_map="sequential"` which layer-loads across available memory.

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

# Verify A100-80GB is attached
!nvidia-smi
!ls -la smoothquant_repo/dataset/val.jsonl.zst
!python -c "from smoothquant.calibration import get_act_scales; print('smoothquant OK')"
```

---

## Cell 2: Generate activation scales for OPT-30B

```python
%cd /content/llm-quantization-thesis/smoothquant_repo

!python examples/generate_act_scales.py \
    --model-name facebook/opt-30b \
    --output-path act_scales/opt-30b.pt \
    --dataset-path dataset/val.jsonl.zst \
    --num-samples 512 \
    --seq-len 512
```

---

## Cell 3: Save scales to Drive

```python
!mkdir -p /content/drive/MyDrive/thesis_results/act_scales
!cp /content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-30b.pt \
    /content/drive/MyDrive/thesis_results/act_scales/

# Verify
!ls -la /content/drive/MyDrive/thesis_results/act_scales/
```

---

## Expected output

`opt-30b.pt` present at both `smoothquant_repo/act_scales/` and `/content/drive/MyDrive/thesis_results/act_scales/`. File size should be similar in magnitude to `opt-13b.pt` (a dict of per-layer input activation max-abs tensors, not the model weights).
