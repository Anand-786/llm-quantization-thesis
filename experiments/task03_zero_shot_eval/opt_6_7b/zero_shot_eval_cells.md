# Task 03: Zero-Shot Eval — OPT-6.7B — Colab Cells

Compares the paper's recipe (O1/O2 + max-smoothing) against ours (C + percentile-smoothing) on the SmoothQuant paper's 7 zero-shot tasks. See [../experiment_plan.md](../experiment_plan.md) for rationale and config table.

**Configs run here (in order):**
1. FP16 (anchor)
2. W8A8-naive (floor)
3. SQ-O1 + max, α=0.5
4. SQ-O2 + max, α=0.5
5. SQ-C + percentile, α=?, p=? (**ours** — fill in from Task 02 OPT-6.7B winner)

After the comparison table, an **optional** SQ-C + max cell is provided for ad-hoc analysis only — not part of the saved comparison.

## Runtime / hardware

**A100 required.** OPT-6.7B fp16 weights are ~13.3 GB; T4 (14.6 GB) cannot hold the model plus activations for batched zero-shot eval. CLAUDE.md Challenge #2 documents the same constraint for PPL eval at seq_len=2048; zero-shot uses shorter sequences but the model itself still doesn't fit comfortably on T4. Run on Colab Pro A100 (40 GB).

Estimated time on A100: 5 configs × ~15–25 min ≈ 1.5–2 h. HellaSwag dominates.

## Prerequisites

- `act_scales/opt-6.7b.pt` on Drive (from Task 01) — used by configs 3, 4. **(Already generated per PROGRESS.md.)**
- `act_percentiles/opt-6.7b/p<chosen>.pt` on Drive (from Task 02) — used by config 5. **Pending Task 02 OPT-6.7B sweep.** Update `P_PCT` and `ALPHA_PCT` in Cell 2 once that sweep picks a winner.

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate "datasets<3.0.0" zstandard tqdm
!pip install -q lm-eval==0.4.4

import sys
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")
sys.path.insert(0, "/content/llm-quantization-thesis")  # for experiments.task02_*

from google.colab import drive
drive.mount('/content/drive')

# Copy Task 01 max scales + Task 02 percentile scales locally
!mkdir -p smoothquant_repo/act_scales
!cp /content/drive/MyDrive/thesis_results/act_scales/opt-6.7b.pt smoothquant_repo/act_scales/

!mkdir -p act_percentiles/opt-6.7b
!cp /content/drive/MyDrive/thesis_results/act_percentiles/opt-6.7b/*.pt act_percentiles/opt-6.7b/ 2>/dev/null || echo "(no percentile files yet — run Task 02 6.7B first)"

!nvidia-smi
!ls -la smoothquant_repo/act_scales/ act_percentiles/opt-6.7b/ 2>/dev/null
!python -c "from smoothquant.smooth import smooth_lm; print('smoothquant OK')"
!python -c "from experiments.task02_percentile_smoothing.percentile_smooth import smooth_lm_pct; print('percentile smooth OK')"
!python -c "import importlib.metadata; print('lm_eval', importlib.metadata.version('lm_eval'))"
```

**Note:** `datasets<3.0.0` pin is required — newer versions removed loading-script support, which breaks PIQA/COPA in lm-eval v0.4.4. See CLAUDE.md Challenges #3.

---

## Cell 2: Config — fill in P_PCT/ALPHA_PCT after Task 02 6.7B winner is known

```python
# Paths
MODEL = "facebook/opt-6.7b"
SCRIPT = "experiments/task03_zero_shot_eval/opt_6_7b/run_zero_shot_t3.py"
MAX_SCALES = "smoothquant_repo/act_scales/opt-6.7b.pt"

# Alphas (paper default 0.5 for O1/O2)
ALPHA_O1 = 0.5
ALPHA_O2 = 0.5

# Percentile-smoothing knobs — REPLACE with Task 02 OPT-6.7B winner before running Cell 7
# Trend across the ladder (Task 02 memory): 1.3B → p=0.999/α=0.9; 2.7B → p=0.995/α=0.5.
# Optimum drifts to lower α and lower p as scale grows; expect 6.7B to continue that trend.
# Reasonable placeholder until the actual sweep finishes:
P_PCT     = 0.99   # placeholder
ALPHA_PCT = 0.5    # placeholder
PCT_SCALES = f"act_percentiles/opt-6.7b/p{P_PCT}.pt"

BATCH = 8  # A100 has headroom; if rare OOM on HellaSwag drop to 4

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
    --save_json {OUT_DIR}/opt-6.7b_zeroshot_FP16.json
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
    --save_json {OUT_DIR}/opt-6.7b_zeroshot_W8A8-naive.json
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
    --save_json {OUT_DIR}/opt-6.7b_zeroshot_SQ-O1-max.json
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
    --save_json {OUT_DIR}/opt-6.7b_zeroshot_SQ-O2-max.json
```

---

## Cell 7: SQ-C + percentile smoothing (ours)

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method percentile --alpha {ALPHA_PCT} --p_w {P_PCT} \
    --act_scales_path {PCT_SCALES} \
    --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-C-pct \
    --batch_size {BATCH} \
    --save_json {OUT_DIR}/opt-6.7b_zeroshot_SQ-C-pct.json
```

---

## Cell 8: Backup + comparison table

```python
!mkdir -p /content/drive/MyDrive/thesis_results/task03
!cp {OUT_DIR}/opt-6.7b_zeroshot_*.json /content/drive/MyDrive/thesis_results/task03/

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

ORDER = ["FP16", "W8A8-naive", "SQ-O1-max", "SQ-O2-max", "SQ-C-pct"]

rows_by_label = {}
for f in sorted(glob.glob(f"{OUT_DIR}/opt-6.7b_zeroshot_*.json")):
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

- **6.7B is the discriminating data point.** SmoothQuant's paper Table 1 shows per-tensor activation quantization drops to 39.9% accuracy at this scale — outliers are severe, so smoothing recipes have meaningful room to differentiate. Unlike 1.3B (where the W8A8 floor is −1.1pp from FP16) or 2.7B (where the floor is only −0.5pp), 6.7B should show a real W8A8-naive collapse and let the recipes spread out on accuracy.
- **SQ-C-pct vs SQ-O1-max / SQ-O2-max**: ours vs paper recipe → headline comparison. Watch whether C-pct's PPL advantage from Task 02 finally translates to a clear zero-shot gap here.
- **SQ-C-pct vs FP16**: how much of the FP16 ceiling does our compound recipe recover at the scale where it matters?
- LAMBADA + HellaSwag are the most reliable signals. RTE/COPA are noisy (small val sets) — don't over-read.

---

## Cell 9 (optional): SQ-C + max smoothing — ad-hoc analysis only

Run only if you want to look at the C scheme's behaviour under max-smoothing as a separate side-quest. Output is **not** included in the saved comparison table or backed up to Drive.

```python
!python {SCRIPT} \
    --model_path {MODEL} \
    --smooth --smooth_method max --alpha 0.5 \
    --act_scales_path {MAX_SCALES} \
    --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-C-max \
    --batch_size {BATCH} \
    --save_json /tmp/opt-6.7b_zeroshot_SQ-C-max.json
```
