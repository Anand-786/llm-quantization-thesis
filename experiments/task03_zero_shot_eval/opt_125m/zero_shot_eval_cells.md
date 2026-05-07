# Task 03: Zero-Shot Eval — OPT-125M — Colab Cells

Same 7-task suite and 6-config layout as the OPT-1.3B run; see [../experiment_plan.md](../experiment_plan.md) for rationale. Only the alpha values and the model name differ — α tuned to OPT-125M's Task 01 PPL optima.

**Configs run here (in order):**
1. FP16 (anchor)
2. W8A8-naive (floor)
3. SQ-O1 + max, α=0.5
4. SQ-O2 + max, α=0.5
5. SQ-C + max, α=0.5 (internal reference — Task 01's C best on 125M was α=0.5, PPL 27.600)
6. SQ-C + percentile, α=0.5, p=0.95 (**ours**; α/p are placeholders pending Task 02 winner on 125M)

**Runtime estimate:** T4 ≈ 5–10 min/config × 6 ≈ 30–60 min total. 125M is small; HellaSwag dataset loading dominates.

**Prerequisites:**
- `act_scales/opt-125m.pt` on Drive (from Task 01) — used by configs 3, 4, 5.
- `act_percentiles/opt-125m/p0.95.pt` on Drive (from Task 02) — used by config 6. **If Task 02 picks a different winner on 125M, update `P_PCT` and `ALPHA_PCT` in Cell 2.**

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
sys.path.insert(0, "/content/llm-quantization-thesis")

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p smoothquant_repo/act_scales
!cp /content/drive/MyDrive/thesis_results/act_scales/opt-125m.pt smoothquant_repo/act_scales/

!mkdir -p act_percentiles/opt-125m
!cp /content/drive/MyDrive/thesis_results/act_percentiles/opt-125m/*.pt act_percentiles/opt-125m/

!nvidia-smi
!ls -la smoothquant_repo/act_scales/ act_percentiles/opt-125m/
!python -c "from smoothquant.smooth import smooth_lm; print('smoothquant OK')"
!python -c "from experiments.task02_percentile_smoothing.percentile_smooth import smooth_lm_pct; print('percentile smooth OK')"
!python -c "import importlib.metadata; print('lm_eval', importlib.metadata.version('lm_eval'))"
```

---

## Cell 2: Config — edit these after Task 02 picks its winner on 125M

```python
MODEL = "facebook/opt-125m"
SCRIPT = "experiments/task03_zero_shot_eval/opt_125m/run_zero_shot_t3.py"
MAX_SCALES = "smoothquant_repo/act_scales/opt-125m.pt"

# Alphas (locked from Task 01 PPL sweep on 125M — step=0.1, 9 levels)
ALPHA_O1 = 0.5    # paper default; near-best for O1 on 125M (28.298)
ALPHA_O2 = 0.5    # per-tensor W: 0.5 is the safe choice
ALPHA_C_MAX = 0.5 # C's PPL min on 125M = 27.600 at alpha=0.5

# Percentile-smoothing knobs — REPLACE with Task 02's PPL winner on 125M
P_PCT     = 0.95
ALPHA_PCT = 0.5
PCT_SCALES = f"act_percentiles/opt-125m/p{P_PCT}.pt"

BATCH = 8

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
    --save_json {OUT_DIR}/opt-125m_zeroshot_FP16.json
```

---

## Cell 4: W8A8-naive

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label W8A8-naive \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-125m_zeroshot_W8A8-naive.json
```

---

## Cell 5: SQ-O1 + max smoothing

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha {ALPHA_O1} \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_tensor --act_quant per_token \
    --config_label SQ-O1-max \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-125m_zeroshot_SQ-O1-max.json
```

---

## Cell 6: SQ-O2 + max smoothing

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha {ALPHA_O2} \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label SQ-O2-max \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-125m_zeroshot_SQ-O2-max.json
```

---

## Cell 7: SQ-C + max (internal reference)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha {ALPHA_C_MAX} \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-C-max \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-125m_zeroshot_SQ-C-max.json
```

---

## Cell 8: SQ-C + percentile (ours)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method percentile --alpha {ALPHA_PCT} --p_w {P_PCT} \
    --act_scales_path {PCT_SCALES} \
    --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-C-pct \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-125m_zeroshot_SQ-C-pct.json
```

---

## Cell 9: Backup + comparison table

```python
!mkdir -p /content/drive/MyDrive/thesis_results/task03
!cp {OUT_DIR}/opt-125m_zeroshot_*.json /content/drive/MyDrive/thesis_results/task03/

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
ORDER = ["FP16", "W8A8-naive", "SQ-O1-max", "SQ-O2-max", "SQ-C-max", "SQ-C-pct"]

rows_by_label = {}
for f in sorted(glob.glob(f"{OUT_DIR}/opt-125m_zeroshot_*.json")):
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

## Reading the results

- 125M is the **most permissive** scale: outliers are weakest, every scheme has the most headroom. Task 01 already showed C wins 9/9 alphas on 125M PPL — expect the zero-shot accuracy gap between schemes to be **smaller** here than at 1.3B / 6.7B. The headline finding "percentile + C beats max + O1" needs to hold at 1.3B and ideally 6.7B for the thesis claim; 125M is a sanity / spectrum-completion data point.
- LAMBADA + HellaSwag still the most reliable signals. RTE/COPA noisy.
