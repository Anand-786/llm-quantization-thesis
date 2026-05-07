# Task 03: Zero-Shot Eval — OPT-1.3B — Colab Cells

Compares the paper's recipe (O1/O2 + max-smoothing) against ours (C + percentile-smoothing) on the SmoothQuant paper's 7 zero-shot tasks. See [../experiment_plan.md](../experiment_plan.md) for rationale and config table.

**Configs run here (in order):**
1. FP16 (anchor)
2. W8A8-naive (floor)
3. SQ-O1 + max, α=0.5
4. SQ-O2 + max, α=0.5
5. SQ-C + max, α=0.9 (internal reference — isolates the "smoothing-statistic" axis)
6. SQ-C + percentile, α=0.9, p=0.95 (**ours**; α/p are placeholders pending Task 02 winner)

**Runtime estimate:** T4 ≈ 30–45 min/config × 6 ≈ 3–4 h. A100 ≈ 1 h. HellaSwag is the slowest task; if T4 OOMs lower `BATCH` from 8 → 4 → 2.

**Prerequisites:**
- `act_scales/opt-1.3b.pt` on Drive (from Task 01) — used by configs 3, 4, 5.
- `act_percentiles/opt-1.3b/p0.95.pt` on Drive (from Task 02) — used by config 6. **If Task 02 picked a different winner, update `P_PCT` and `ALPHA_PCT` in Cell 2 below.**

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm
!pip install -q lm-eval==0.4.4

import sys
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")
sys.path.insert(0, "/content/llm-quantization-thesis")  # for experiments.task02_*

from google.colab import drive
drive.mount('/content/drive')

# Copy Task 01 max scales + Task 02 percentile scales locally
!mkdir -p smoothquant_repo/act_scales
!cp /content/drive/MyDrive/thesis_results/act_scales/opt-1.3b.pt smoothquant_repo/act_scales/

!mkdir -p act_percentiles/opt-1.3b
!cp /content/drive/MyDrive/thesis_results/act_percentiles/opt-1.3b/*.pt act_percentiles/opt-1.3b/

!nvidia-smi
!ls -la smoothquant_repo/act_scales/ act_percentiles/opt-1.3b/
!python -c "from smoothquant.smooth import smooth_lm; print('smoothquant OK')"
!python -c "from experiments.task02_percentile_smoothing.percentile_smooth import smooth_lm_pct; print('percentile smooth OK')"
!python -c "import importlib.metadata; print('lm_eval', importlib.metadata.version('lm_eval'))"
```

---

## Cell 2: Config — edit these after Task 02 picks its winner

```python
# Paths
MODEL = "facebook/opt-1.3b"
SCRIPT = "experiments/task03_zero_shot_eval/opt_1_3b/run_zero_shot_t3.py"
MAX_SCALES = "smoothquant_repo/act_scales/opt-1.3b.pt"

# Alphas (locked from Task 01 PPL sweep on 1.3B)
ALPHA_O1 = 0.5
ALPHA_O2 = 0.5
ALPHA_C_MAX = 0.9

# Percentile-smoothing knobs — REPLACE with Task 02's PPL winner before running
P_PCT     = 0.95
ALPHA_PCT = 0.9
PCT_SCALES = f"act_percentiles/opt-1.3b/p{P_PCT}.pt"

BATCH = 8  # drop to 4 or 2 if HellaSwag OOMs

OUT_DIR = "results/task03"
!mkdir -p {OUT_DIR}
print(f"Percentile config -> p={P_PCT}, alpha={ALPHA_PCT}, file={PCT_SCALES}")
```

---

## Cell 3: FP16 anchor

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --config_label FP16 \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-1.3b_zeroshot_FP16.json
```

---

## Cell 4: W8A8-naive (no smoothing, per-tensor W + per-tensor A)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label W8A8-naive \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-1.3b_zeroshot_W8A8-naive.json
```

---

## Cell 5: SQ-O1 + max smoothing (paper recipe)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha {ALPHA_O1} \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_tensor --act_quant per_token \
    --config_label SQ-O1-max \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-1.3b_zeroshot_SQ-O1-max.json
```

---

## Cell 6: SQ-O2 + max smoothing (paper recipe)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha {ALPHA_O2} \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label SQ-O2-max \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-1.3b_zeroshot_SQ-O2-max.json
```

---

## Cell 7: SQ-C + max smoothing (internal reference — Task 01 best)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha {ALPHA_C_MAX} \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-C-max \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-1.3b_zeroshot_SQ-C-max.json
```

---

## Cell 8: SQ-C + percentile smoothing (ours)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method percentile --alpha {ALPHA_PCT} --p_w {P_PCT} \
    --act_scales_path {PCT_SCALES} \
    --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-C-pct \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-1.3b_zeroshot_SQ-C-pct.json
```

---

## Cell 9: Backup + comparison table

```python
!mkdir -p /content/drive/MyDrive/thesis_results/task03
!cp {OUT_DIR}/opt-1.3b_zeroshot_*.json /content/drive/MyDrive/thesis_results/task03/

import json, glob

TASKS = ["lambada_openai", "hellaswag", "piqa", "winogrande", "openbookqa", "rte", "copa"]
PRIMARY = {
    "lambada_openai": "acc,none",
    "hellaswag":      "acc_norm,none",
    "piqa":           "acc_norm,none",
    "winogrande":     "acc,none",
    "openbookqa":     "acc_norm,none",
    "rte":            "acc,none",
    "copa":           "acc,none",
}

# Order matches reading flow: anchors first, then paper, then ours
ORDER = ["FP16", "W8A8-naive", "SQ-O1-max", "SQ-O2-max", "SQ-C-max", "SQ-C-pct"]

rows_by_label = {}
for f in sorted(glob.glob(f"{OUT_DIR}/opt-1.3b_zeroshot_*.json")):
    r = json.load(open(f))
    label = r["config_label"]
    row = {"config": label}
    for t in TASKS:
        m = r["results"].get(t, {})
        v = m.get(PRIMARY[t])
        if v is None:
            for k, val in m.items():
                if isinstance(val, (int, float)):
                    v = val
                    break
        row[t] = v
    nums = [row[t] for t in TASKS if isinstance(row[t], (int, float))]
    row["avg"] = sum(nums) / len(nums) if nums else None
    rows_by_label[label] = row

rows = [rows_by_label[l] for l in ORDER if l in rows_by_label]

header = ["config"] + TASKS + ["avg"]
print("  ".join(f"{h:>14}" for h in header))
print("-" * (16 * len(header)))
for row in rows:
    cells = [f"{row['config']:>14}"]
    for t in TASKS + ["avg"]:
        v = row.get(t)
        cells.append(f"{v:>14.4f}" if isinstance(v, (int, float)) else f"{'-':>14}")
    print("  ".join(cells))
```

---

## What to look for

- **SQ-C-pct vs SQ-C-max**: same scheme, only smoothing statistic differs → isolates Task 02's contribution.
- **SQ-C-pct vs SQ-O1-max / SQ-O2-max**: ours vs paper recipe → headline comparison.
- **SQ-C-max vs SQ-O1-max**: re-confirms Task 01's scheme finding at task level (PPL → accuracy).
- LAMBADA + HellaSwag are the most reliable signals. RTE/COPA are noisy (small val sets) — don't over-read.
