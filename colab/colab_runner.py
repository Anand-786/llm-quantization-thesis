"""
Colab Runner — thin wrapper that calls run_scheme_eval.py
==========================================================

COLAB SETUP (Cell 1 — run once per session):

    !git clone https://github.com/Anand-786/llm-quantization-thesis.git
    %cd /content/llm-quantization-thesis
    !pip install -q torch transformers accelerate datasets lm-eval smoothquant tqdm numpy
    !git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
    from google.colab import drive
    drive.mount('/content/drive')
    !nvidia-smi

RUN EXPERIMENTS (Cell 2 — edit and rerun):

    # Sanity check:
    %cd /content/llm-quantization-thesis
    !python colab/colab_runner.py --sanity

    # FP16 baseline:
    !python colab/colab_runner.py --model opt-125m --scheme fp16 --skip_zeroshot

    # SmoothQuant-O3:
    !python colab/colab_runner.py --model opt-1.3b --scheme O3 --alpha 0.5
"""

import subprocess
import sys
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "task01_scheme_exploration", "run_scheme_eval.py")


def run_sanity():
    """Quick checks before running real experiments."""
    print("=" * 60)
    print("  SANITY CHECK")
    print("=" * 60)

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_properties(0).name
        mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"  ✅ GPU: {name} ({mem}GB)")
    else:
        print(f"  ❌ No GPU! Change Colab runtime type.")
        return

    # Activation scales
    scales = os.path.join(PROJECT_ROOT, "smoothquant_repo", "act_scales")
    if os.path.isdir(scales) and len(os.listdir(scales)) > 0:
        n = len([f for f in os.listdir(scales) if f.endswith(".pt")])
        print(f"  ✅ Activation scales: {n} files found")
    else:
        print(f"  ❌ No activation scales at {scales}")
        print(f"     Run: !git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo")
        return

    # Drive
    if os.path.isdir("/content/drive/MyDrive"):
        print(f"  ✅ Google Drive mounted")
    else:
        print(f"  ⚠️  Google Drive not mounted (results won't save to Drive)")

    # Eval script exists
    if os.path.isfile(EVAL_SCRIPT):
        print(f"  ✅ Eval script found")
    else:
        print(f"  ❌ Eval script missing at {EVAL_SCRIPT}")
        return

    # SmoothQuant import
    try:
        from smoothquant.smooth import smooth_lm
        print(f"  ✅ SmoothQuant library importable")
    except ImportError as e:
        print(f"  ❌ Cannot import smoothquant: {e}")
        print(f"     Run: !pip install smoothquant")
        return

    # Quick model load test
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from shared.model_utils import load_model_and_tokenizer
        model, tok = load_model_and_tokenizer("opt-125m")
        inp = tok("Hello", return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            model(**inp)
        del model, tok
        torch.cuda.empty_cache()
        print(f"  ✅ Model load + forward pass (OPT-125M)")
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return

    print("=" * 60)
    print("  ALL CHECKS PASSED — ready to run experiments!")
    print("=" * 60)


def main():
    args = sys.argv[1:]

    if "--sanity" in args:
        run_sanity()
        return

    # Forward all arguments to run_scheme_eval.py
    # Add default --drive_path if not specified
    if "--drive_path" not in " ".join(args):
        args.extend(["--drive_path", "thesis_results/task01"])

    cmd = [sys.executable, EVAL_SCRIPT] + args
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
