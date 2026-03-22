"""
Colab Runner for LLM Quantization Thesis
==========================================

HOW TO USE IN COLAB:
====================
Open a new Colab notebook. You only need TWO cells.

CELL 1 — Setup (run once per session):
---------------------------------------
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!pip install -q torch transformers accelerate datasets lm-eval smoothquant tqdm numpy
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi

CELL 2 — Run experiment (edit and rerun as needed):
----------------------------------------------------
%cd /content/llm-quantization-thesis
!python colab/colab_runner.py --model opt-1.3b --scheme O3 --alpha 0.5

That's it. Change the arguments and rerun Cell 2 for each experiment.


ARGUMENTS:
    --model     : opt-125m, opt-1.3b, opt-2.7b, opt-6.7b (short names work)
    --scheme    : fp16, O1, O2, O3, O1_pcw, O2_pcw, O3_pcw, naive_w8a8
    --alpha     : smoothing strength, default 0.5
    --no_smooth : flag, skip smoothing (for naive baselines)
    --skip_zeroshot : flag, skip zero-shot eval (faster, perplexity only)
    --skip_ppl  : flag, skip perplexity eval

EXAMPLES:
    # FP16 baseline
    !python colab/colab_runner.py --model opt-1.3b --scheme fp16

    # Naive W8A8 without smoothing
    !python colab/colab_runner.py --model opt-1.3b --scheme naive_w8a8 --no_smooth

    # SmoothQuant-O3
    !python colab/colab_runner.py --model opt-1.3b --scheme O3 --alpha 0.5

    # Quick test (perplexity only, smallest model)
    !python colab/colab_runner.py --model opt-125m --scheme O3 --skip_zeroshot

    # Sanity check (verifies everything works, ~2 min)
    !python colab/colab_runner.py --sanity
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime

# ─── Fix paths ───────────────────────────────────────────────────────────────
# This script lives in colab/ so project root is one level up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Where SmoothQuant activation scales live (cloned in Cell 1)
SCALES_DIR = os.path.join(PROJECT_ROOT, "smoothquant_repo", "act_scales")

# Where to save results on Google Drive
DRIVE_RESULTS = "/content/drive/MyDrive/thesis_results"

# Also save a copy inside the repo's results folder
LOCAL_RESULTS = os.path.join(PROJECT_ROOT, "experiments", "task01_scheme_exploration", "results")

# ─── Imports from our shared code ────────────────────────────────────────────
from shared.model_utils import (
    load_model_and_tokenizer, get_gpu_info, resolve_model_name
)
from shared.save_utils import save_result, print_result_summary
from shared.eval_utils import run_full_evaluation
from shared.quant_utils import get_act_scales, smooth_layer, get_scheme_info

# ─── Import the Task 01 functions ────────────────────────────────────────────
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "task01_scheme_exploration"))
from run_scheme_eval import (
    apply_smoothing,
    quantize_model_weights_and_activations,
)


# ─── Sanity Check ────────────────────────────────────────────────────────────

def run_sanity_check():
    """Quick check that everything is set up correctly."""
    import torch
    from shared.quant_utils import quantize_tensor_absmax, quantize_tensor_per_token

    checks = []

    # 1. GPU
    gpu = get_gpu_info()
    ok = gpu["available"]
    checks.append(("GPU available", ok, f"{gpu['name']} ({gpu['memory_gb']}GB)" if ok else "NO GPU"))

    # 2. Activation scales exist
    ok = os.path.isdir(SCALES_DIR) and len(os.listdir(SCALES_DIR)) > 0
    n_files = len(os.listdir(SCALES_DIR)) if os.path.isdir(SCALES_DIR) else 0
    checks.append(("Activation scales found", ok, f"{n_files} files in {SCALES_DIR}"))

    # 3. Drive mounted
    ok = os.path.isdir("/content/drive/MyDrive")
    checks.append(("Google Drive mounted", ok, ""))

    # 4. Load tiny model
    try:
        model, tok = load_model_and_tokenizer("opt-125m")
        inputs = tok("Hello world", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        checks.append(("Model load + forward pass", True, "OPT-125M OK"))
        del model, tok
        torch.cuda.empty_cache()
    except Exception as e:
        checks.append(("Model load + forward pass", False, str(e)[:80]))

    # 5. Quantization functions
    try:
        x = torch.randn(4, 768)
        _ = quantize_tensor_absmax(x)
        _ = quantize_tensor_per_token(x)
        checks.append(("Quantization functions", True, ""))
    except Exception as e:
        checks.append(("Quantization functions", False, str(e)[:80]))

    # Print
    print("\n" + "=" * 60)
    print("  SANITY CHECK")
    print("=" * 60)
    all_ok = True
    for name, ok, detail in checks:
        icon = "✅" if ok else "❌"
        line = f"  {icon} {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        if not ok:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("  All checks passed! You're ready to run experiments.")
    else:
        print("  Some checks failed. Fix the issues above before proceeding.")
    print("=" * 60 + "\n")
    return all_ok


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Colab runner for quantization experiments")
    p.add_argument("--sanity", action="store_true", help="Run sanity check only")
    p.add_argument("--model", type=str, default="opt-1.3b")
    p.add_argument("--scheme", type=str, default="O3",
                   help="fp16 | O1 | O2 | O3 | O1_pcw | O2_pcw | O3_pcw | naive_w8a8")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--no_smooth", action="store_true")
    p.add_argument("--skip_zeroshot", action="store_true")
    p.add_argument("--skip_ppl", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.sanity:
        run_sanity_check()
        return

    start = time.time()
    hf_name = resolve_model_name(args.model)
    gpu = get_gpu_info()
    do_smooth = (args.scheme != "fp16") and (not args.no_smooth)

    print("\n" + "=" * 60)
    print("  TASK 01: Quantization Scheme Exploration")
    print(f"  Model:    {hf_name}")
    print(f"  Scheme:   {args.scheme}")
    print(f"  Alpha:    {args.alpha}")
    print(f"  Smooth:   {do_smooth}")
    print(f"  GPU:      {gpu['name']} ({gpu['memory_gb']}GB)")
    print("=" * 60 + "\n")

    # 1. Load model
    model, tokenizer = load_model_and_tokenizer(hf_name)

    # 2. Apply smoothing
    if do_smooth:
        apply_smoothing(model, hf_name, args.alpha, scales_dir=SCALES_DIR)

    # 3. Apply quantization
    if args.scheme != "fp16":
        quantize_model_weights_and_activations(model, args.scheme)

    # 4. Evaluate
    eval_results = run_full_evaluation(
        model, tokenizer,
        skip_zeroshot=args.skip_zeroshot,
        skip_ppl=args.skip_ppl,
    )

    elapsed = time.time() - start

    # 5. Build result dict
    result = {
        "task": "task01_scheme_exploration",
        "model": hf_name,
        "scheme": args.scheme,
        "alpha": args.alpha if do_smooth else None,
        "smoothed": do_smooth,
        "metrics": eval_results.get("zeroshot", {}),
        "avg_zeroshot_acc": eval_results.get("avg_zeroshot_acc"),
        "wikitext2_ppl": eval_results.get("wikitext2_ppl"),
        "gpu": gpu["name"],
        "gpu_memory_gb": gpu["memory_gb"],
        "duration_seconds": round(elapsed, 1),
    }

    print_result_summary(result)

    # 6. Save — both locally and to Drive
    save_result(result, LOCAL_RESULTS)

    drive_task_dir = os.path.join(DRIVE_RESULTS, "task01")
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive_task_dir, exist_ok=True)
        save_result(result, drive_task_dir)
        print(f"  Also saved to Drive → {drive_task_dir}")
    else:
        print("  ⚠️  Drive not mounted — result saved locally only.")

    print(f"\nDone in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()