# Task 01: Zero-Shot Benchmark Eval — OPT-1.3B — Colab Cells

Mirrors the SmoothQuant paper's 7-task zero-shot suite (LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA) for OPT-1.3B. Paper only reports these for OPT-175B; we run them on 1.3B to add task-level signal alongside our PPL results.

**Alpha policy — best-of-each (same protocol the paper uses for OPT-175B):**
Each scheme is evaluated at its own best alpha rather than at a single fixed alpha. This is exactly what the SmoothQuant paper does: Table 4 reports each variant at the alpha selected by grid search. Comparing each at its tuned optimum is the fair comparison; using a single alpha would penalize whichever scheme's optimum is furthest from that point.

- O1 (per-tensor W, per-token A) → alpha = 0.5 (paper's default; matches O1's optimum on 1.3B PPL sweep — 14.686 at 0.5)
- O2 (per-tensor W, per-tensor A) → alpha = 0.5 (paper's default; per-tensor W collapses at high alpha so 0.5 is the safe optimum)
- C / SQ-PCW-PT (per-channel W, per-token A) → alpha = 0.9 (best from your 1.3B alpha sweep — 14.617 at 0.9)
- D / SQ-PCW-TEN (per-channel W, per-tensor A) → alpha = 0.9 (extrapolated, no full sweep yet — see note in Cell 8)
- FP16 and W8A8-naive included as anchors (no smoothing, no alpha)

**Runtime estimate (T4):** ~25–45 min per config × 6 configs ≈ 3–4.5 hours. HellaSwag is the slowest task (~10k val items). If you hit OOM lower `--batch_size` from 8 to 4 or 2.

Copy-paste each cell into Colab. Run in order.

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

# lm-evaluation-harness (the standard zero-shot eval tool, used by SmoothQuant authors)
!pip install -q lm-eval==0.4.4

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy activation scales from Drive
!mkdir -p smoothquant_repo/act_scales
!cp /content/drive/MyDrive/thesis_results/act_scales/*.pt smoothquant_repo/act_scales/

# Verify
!nvidia-smi
!ls -la smoothquant_repo/act_scales/
!python -c "from smoothquant.smooth import smooth_lm; print('smoothquant OK')"
!python -c "import lm_eval; print('lm_eval', lm_eval.__version__)"
```

---

## Cell 2: Write the parameterized zero-shot eval script

```python
%%writefile /content/llm-quantization-thesis/run_zero_shot.py
"""
Zero-shot benchmark eval for SmoothQuant configs.
Runs the 7 tasks the paper uses for OPT-175B:
  LAMBADA (lambada_openai), HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA.

Loads an OPT model, optionally applies smooth_lm + quantize_model, then evaluates
via lm-evaluation-harness in zero-shot (num_fewshot=0).
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model

import lm_eval
from lm_eval.models.huggingface import HFLM


TASKS = [
    "lambada_openai",
    "hellaswag",
    "piqa",
    "winogrande",
    "openbookqa",
    "rte",
    "copa",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--act_scales_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--weight_quant", type=str, default="per_channel",
                        choices=["per_channel", "per_tensor"])
    parser.add_argument("--act_quant", type=str, default="per_token",
                        choices=["per_token", "per_tensor"])
    parser.add_argument("--quantize_bmm", action="store_true", default=True)
    parser.add_argument("--config_label", type=str, default="unknown")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    start = time.time()

    print("=" * 60)
    print(f"  Config: {args.config_label}")
    print(f"  Model:  {args.model_path}")
    print(f"  Smooth: {args.smooth} (alpha={args.alpha})")
    print(f"  Quant:  {args.quantize}")
    if args.quantize:
        print(f"  Weight: {args.weight_quant}")
        print(f"  Act:    {args.act_quant}")
        print(f"  BMM:    {args.quantize_bmm}")
    print(f"  Tasks:  {TASKS}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )

    if args.smooth:
        act_scales = torch.load(args.act_scales_path)
        smooth_lm(model, act_scales, args.alpha)
        print("Smoothing applied.")

    if args.quantize:
        model = quantize_model(
            model,
            weight_quant=args.weight_quant,
            act_quant=args.act_quant,
            quantize_bmm_input=args.quantize_bmm,
        )
        print("Quantization applied.")

    # Wrap for lm-eval-harness
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
    )

    elapsed = time.time() - start

    # Pull primary metric per task
    summary = {}
    for task, metrics in results["results"].items():
        # Each task has a primary metric like "acc,none" or "acc_norm,none"
        # For clarity, keep both acc and acc_norm where present.
        clean = {k: v for k, v in metrics.items()
                 if any(m in k for m in ["acc", "perplexity", "ppl"]) and "stderr" not in k}
        summary[task] = clean

    print("\n" + "=" * 60)
    print(f"  Config {args.config_label} — Zero-shot results")
    print("=" * 60)
    for task, metrics in summary.items():
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {task:<20} {metric_str}")
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    out = {
        "config_label": args.config_label,
        "model": args.model_path,
        "smooth": args.smooth,
        "alpha": args.alpha if args.smooth else None,
        "quantize": args.quantize,
        "weight_quant": args.weight_quant if args.quantize else None,
        "act_quant": args.act_quant if args.quantize else None,
        "quantize_bmm": args.quantize_bmm if args.quantize else None,
        "tasks": TASKS,
        "results": summary,
        "raw_results": results["results"],
        "duration_seconds": round(elapsed, 1),
    }

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved to {args.save_json}")


if __name__ == "__main__":
    main()
```

---

## Cell 3: FP16 baseline

```python
%cd /content/llm-quantization-thesis

!python run_zero_shot.py \
    --model_path facebook/opt-1.3b \
    --act_scales_path smoothquant_repo/act_scales/opt-1.3b.pt \
    --config_label FP16 \
    --batch_size 8 \
    --save_json results/task01/opt-1.3b_zeroshot_FP16.json
```

---

## Cell 4: W8A8 naive (no smoothing)

```python
!python run_zero_shot.py \
    --model_path facebook/opt-1.3b \
    --act_scales_path smoothquant_repo/act_scales/opt-1.3b.pt \
    --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label W8A8-naive \
    --batch_size 8 \
    --save_json results/task01/opt-1.3b_zeroshot_W8A8-naive.json
```

---

## Cell 5: SQ-O1 (per-tensor W, per-token A) — alpha = 0.5

```python
!python run_zero_shot.py \
    --model_path facebook/opt-1.3b \
    --act_scales_path smoothquant_repo/act_scales/opt-1.3b.pt \
    --smooth --alpha 0.5 --quantize \
    --weight_quant per_tensor --act_quant per_token \
    --config_label SQ-O1 \
    --batch_size 8 \
    --save_json results/task01/opt-1.3b_zeroshot_SQ-O1.json
```

---

## Cell 6: SQ-O2 (per-tensor W, per-tensor A) — alpha = 0.5

```python
!python run_zero_shot.py \
    --model_path facebook/opt-1.3b \
    --act_scales_path smoothquant_repo/act_scales/opt-1.3b.pt \
    --smooth --alpha 0.5 --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label SQ-O2 \
    --batch_size 8 \
    --save_json results/task01/opt-1.3b_zeroshot_SQ-O2.json
```

---

## Cell 7: SQ-PCW-PT (Config C — per-channel W, per-token A) — alpha = 0.9

```python
!python run_zero_shot.py \
    --model_path facebook/opt-1.3b \
    --act_scales_path smoothquant_repo/act_scales/opt-1.3b.pt \
    --smooth --alpha 0.9 --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-PCW-PT \
    --batch_size 8 \
    --save_json results/task01/opt-1.3b_zeroshot_SQ-PCW-PT.json
```

---

## Cell 8: SQ-PCW-TEN (Config D — per-channel W, per-tensor A) — alpha = 0.9

**Note on alpha choice:** D wasn't included in the 1.3B alpha sweep (only O1 vs C were). α=0.9 is extrapolated from C's behaviour: D shares the per-channel W axis with C, and per-channel W absorbs migrated difficulty well at high alpha. The per-tensor A axis (D's constraint, shared with O2) actually benefits from high alpha because less difficulty stays on activations. So D at 0.9 should beat D at 0.5. If a reviewer questions this, a follow-up D-arm alpha sweep is cheap to add. Document the choice in PROGRESS.md after results land.

```python
!python run_zero_shot.py \
    --model_path facebook/opt-1.3b \
    --act_scales_path smoothquant_repo/act_scales/opt-1.3b.pt \
    --smooth --alpha 0.9 --quantize \
    --weight_quant per_channel --act_quant per_tensor \
    --config_label SQ-PCW-TEN \
    --batch_size 8 \
    --save_json results/task01/opt-1.3b_zeroshot_SQ-PCW-TEN.json
```

---

## Cell 9: Backup + summary

```python
!mkdir -p /content/drive/MyDrive/thesis_results/task01
!cp results/task01/opt-1.3b_zeroshot_*.json /content/drive/MyDrive/thesis_results/task01/

import json, glob

TASKS = ["lambada_openai", "hellaswag", "piqa", "winogrande", "openbookqa", "rte", "copa"]

def primary(metrics):
    # Prefer acc_norm (normalized accuracy) when present, else acc, else first numeric
    for key in ["acc_norm,none", "acc,none"]:
        if key in metrics:
            return metrics[key]
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return v
    return None

rows = []
for f in sorted(glob.glob("results/task01/opt-1.3b_zeroshot_*.json")):
    r = json.load(open(f))
    row = {"config": r["config_label"]}
    for t in TASKS:
        row[t] = primary(r["results"].get(t, {}))
    row["avg"] = sum(v for v in row.values() if isinstance(v, (int, float))) / len(TASKS)
    rows.append(row)

# Print as a table
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

## What to expect

- FP16 sets the ceiling. W8A8-naive sets the floor (typically a noticeable drop on LAMBADA/HellaSwag).
- SQ-O1 should recover most of the W8A8-naive drop. SQ-O2 less so (per-tensor activations are harder).
- SQ-PCW-PT at alpha=0.9 should match or slightly exceed FP16 if your PPL story carries over to task accuracy. If C wins on PPL but loses on a task, that's a finding worth noting in PROGRESS.md — task accuracy can be more sensitive to specific outlier behaviour than NLL.
- COPA is small (100 items) and noisy — don't read too much into ±2% deltas there. LAMBADA and HellaSwag are the most reliable signals.
