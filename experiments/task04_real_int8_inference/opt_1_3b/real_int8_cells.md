# Task 04: Real INT8 Inference — OPT-1.3B — Colab Cells

Demonstrates the actual 50% memory reduction (and latency win) of INT8 inference using torch-int's CUTLASS kernels via `smoothquant.opt.Int8OPTForCausalLM`. See [../experiment_plan.md](../experiment_plan.md) for the kernel limitation caveat (per-tensor only) and overall design.

**Configs run here:**

1. FP16 (anchor)
2. INT8 paper recipe — `mit-han-lab/opt-1.3b-smoothquant` from HF (O3 + max smoothing, α=0.5)
3. INT8 ours — local export with `smooth_lm_pct` (p=0.999, α=0.9 — Task 02 winner) + static calibration on Pile

**Hardware:** T4 (free Colab) is sufficient for 1.3B. INT8 GEMM via CUTLASS works on sm_75+.

**Prerequisites on Drive:**
- `/content/drive/MyDrive/thesis_results/act_percentiles/opt-1.3b/p0.999.pt` (Task 02)
- `/content/drive/MyDrive/thesis_results/datasets/val.jsonl.zst` (Pile validation set — see Cell 1 if missing)

---

## Cell 1: Setup

`torch-int` is the painful part — it has to be built from source against the CUDA toolchain PyTorch was built with. Colab supplies a matching nvcc, so this generally works, but compile time is ~5–10 min on T4.

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm

# torch-int: CUTLASS INT8 GEMM kernels
%cd /content
!git clone --recursive https://github.com/Guangxuan-Xiao/torch-int.git
%cd /content/torch-int
# the build script needs the conda env's CUDA; on Colab we set TORCH_CUDA_ARCH_LIST.
# T4 = 7.5, A100 = 8.0. We list both so this notebook runs on either.
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0"
!bash environment.sh || true   # creates pip deps; some lines may fail on Colab — ok
!bash build_cutlass.sh
!python setup.py install

%cd /content/llm-quantization-thesis

import sys
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")
sys.path.insert(0, "/content/llm-quantization-thesis")  # for experiments.task02_*

from google.colab import drive
drive.mount('/content/drive')

# Pull our percentile scales from Drive (Task 02 winner = p=0.999)
!mkdir -p act_percentiles/opt-1.3b
!cp /content/drive/MyDrive/thesis_results/act_percentiles/opt-1.3b/p0.999.pt act_percentiles/opt-1.3b/

# Pile val set for static calibration (Path B)
!mkdir -p dataset
!cp /content/drive/MyDrive/thesis_results/datasets/val.jsonl.zst dataset/ 2>/dev/null || \
   (echo "Pile val.jsonl.zst not on Drive. Downloading..." && \
    wget -q https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst -O dataset/val.jsonl.zst && \
    cp dataset/val.jsonl.zst /content/drive/MyDrive/thesis_results/datasets/)

!nvidia-smi
!python -c "import torch_int; print('torch-int OK')"
!python -c "from smoothquant.opt import Int8OPTForCausalLM; print('Int8OPTForCausalLM OK')"
```

> **If `torch-int` build fails:** the most common cause is a CUDA-toolkit/PyTorch mismatch on Colab. Check `nvcc --version` vs `torch.version.cuda`. If they differ, you may need to `apt install` the matching CUDA toolkit before `build_cutlass.sh`. As a fallback, this task can be skipped on this runtime — the fake-quant numbers from Tasks 01–03 still drive the thesis story; Task 04 is purely the deployment-side demo.

---

## Cell 2: Config

```python
MODEL = "facebook/opt-1.3b"
RUNNER = "experiments/task04_real_int8_inference/run_real_int8_eval.py"
EXPORTER = "experiments/task04_real_int8_inference/export_our_int8.py"

# Task 02 OPT-1.3B winner
P_PCT     = 0.999
ALPHA_PCT = 0.9
PCT_SCALES = f"act_percentiles/opt-1.3b/p{P_PCT}.pt"

# HF prequantized paper model
HF_INT8 = "mit-han-lab/opt-1.3b-smoothquant"

# Where our local INT8 export will be saved
LOCAL_INT8 = "int8_models/opt-1.3b-ours"

OUT_DIR = "results/task04"
!mkdir -p {OUT_DIR}

print(f"Ours: smooth_lm_pct  p={P_PCT}  alpha={ALPHA_PCT}  scales={PCT_SCALES}")
```

---

## Cell 3: FP16 anchor

```python
!python {RUNNER} \
    --mode fp16 \
    --model_path {MODEL} \
    --tokenizer_path {MODEL} \
    --config_label FP16 \
    --save_json {OUT_DIR}/opt-1.3b_realint8_FP16.json
```

---

## Cell 4: INT8 paper recipe (HF prequantized)

```python
!python {RUNNER} \
    --mode int8_hf \
    --model_path {HF_INT8} \
    --tokenizer_path {MODEL} \
    --config_label INT8-paper \
    --save_json {OUT_DIR}/opt-1.3b_realint8_INT8-paper.json
```

---

## Cell 5: Export our INT8 model (Path B)

This applies `smooth_lm_pct` then runs static per-tensor calibration on the Pile val set (512 samples × 512 tokens, ~5–10 min on T4) and saves the resulting INT8 model.

```python
!python {EXPORTER} \
    --model_name {MODEL} \
    --pct_scales {PCT_SCALES} \
    --alpha {ALPHA_PCT} \
    --p_w {P_PCT} \
    --dataset_path dataset/val.jsonl.zst \
    --num_samples 512 \
    --seq_len 512 \
    --output_path {LOCAL_INT8}

!ls -lh {LOCAL_INT8}
```

---

## Cell 6: Evaluate our INT8 model

```python
!python {RUNNER} \
    --mode int8_local \
    --model_path {LOCAL_INT8} \
    --tokenizer_path {MODEL} \
    --config_label INT8-ours \
    --save_json {OUT_DIR}/opt-1.3b_realint8_INT8-ours.json
```

---

## Cell 7: Backup + comparison table

```python
!mkdir -p /content/drive/MyDrive/thesis_results/task04
!cp {OUT_DIR}/opt-1.3b_realint8_*.json /content/drive/MyDrive/thesis_results/task04/

# Optionally back up the exported INT8 weights too — large file, ~1.3GB
# !cp -r {LOCAL_INT8} /content/drive/MyDrive/thesis_results/task04/

import json, glob

ORDER = ["FP16", "INT8-paper", "INT8-ours"]
COLS  = ["size_mb", "wikitext2_ppl", "lambada_last_token_acc", "lambada_latency_ms_per_sample"]

rows_by_label = {}
for f in sorted(glob.glob(f"{OUT_DIR}/opt-1.3b_realint8_*.json")):
    r = json.load(open(f))
    rows_by_label[r["config_label"]] = r

header = ["config"] + COLS
print("  ".join(f"{h:>30}" for h in header))
print("-" * (32 * len(header)))
for label in ORDER:
    r = rows_by_label.get(label)
    if r is None:
        continue
    cells = [f"{label:>30}"]
    for c in COLS:
        v = r.get(c)
        cells.append(f"{v:>30.4f}" if isinstance(v, (int, float)) else f"{'-':>30}")
    print("  ".join(cells))

# Headline numbers for the thesis
fp16 = rows_by_label.get("FP16", {})
ours = rows_by_label.get("INT8-ours", {})
if fp16 and ours:
    ratio = ours["size_mb"] / fp16["size_mb"]
    speedup = fp16.get("lambada_latency_ms_per_sample", 0) / max(ours.get("lambada_latency_ms_per_sample", 1), 1e-9)
    print(f"\nMemory: INT8-ours / FP16 = {ratio*100:.1f}%  (lower is better; ~50% expected)")
    print(f"Latency speedup: {speedup:.2f}x")
```

---

## What to look for

- **Size**: INT8-paper and INT8-ours should both be ~50% of FP16. This is the headline plot for the thesis.
- **WikiText-2 PPL**: INT8-ours should be ≤ INT8-paper (our smoothing was tuned to win on PPL). Both should be close to FP16.
- **LAMBADA latency**: INT8 should be faster than FP16 on T4/A100 thanks to INT8 tensor cores. Don't over-read absolute ms — the proxy is the speedup ratio.
- **LAMBADA accuracy**: sanity check only; ±1% is noise on 1000 samples.
