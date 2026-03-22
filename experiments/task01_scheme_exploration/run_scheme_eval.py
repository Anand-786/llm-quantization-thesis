"""
Task 01: Quantization Scheme Exploration
=========================================
Uses SmoothQuant's own smooth_lm() for smoothing (proven, tested code)
and our own simulated quantization wrappers for different schemes.

Usage:
    python run_scheme_eval.py --model opt-1.3b --scheme O3 --alpha 0.5
    python run_scheme_eval.py --model opt-1.3b --scheme fp16
    python run_scheme_eval.py --model opt-1.3b --scheme naive_w8a8 --no_smooth
"""

import argparse
import sys
import os
import time
import torch
import torch.nn as nn

# Fix paths — works whether called from project root or from this directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from shared.model_utils import load_model_and_tokenizer, get_gpu_info, resolve_model_name
from shared.save_utils import save_result, print_result_summary
from shared.eval_utils import run_full_evaluation
from shared.quant_utils import (
    load_act_scales,
    quantize_tensor_absmax,
    quantize_tensor_per_token,
    quantize_tensor_per_channel,
)


# ─── Smoothing (uses SmoothQuant's own code) ─────────────────────────────────

def apply_smoothing(model, model_name, alpha, scales_dir):
    """
    Apply SmoothQuant smoothing using their own smooth_lm function.
    This is proven, tested code — no reimplementation needed.
    """
    from smoothquant.smooth import smooth_lm
    
    act_scales = load_act_scales(model_name, scales_dir)
    print(f"Applying smoothing with alpha={alpha} ...")
    smooth_lm(model, act_scales, alpha)
    print("  Smoothing applied successfully.")


# ─── Simulated Quantization ──────────────────────────────────────────────────

class FakeQuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that simulates quantization.
    Weights are quantized once at init. Activations are quantized each forward.
    """
    def __init__(self, original: nn.Linear, w_quant_fn, a_quant_fn):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bias = original.bias
        # Quantize weights once (simulated: quant then dequant back to float)
        self.weight = nn.Parameter(w_quant_fn(original.weight.data), requires_grad=False)
        self.a_quant_fn = a_quant_fn

    def forward(self, x):
        return nn.functional.linear(self.a_quant_fn(x), self.weight, self.bias)


# Scheme definitions: (weight_quantizer, activation_quantizer)
SCHEME_MAP = {
    "O1":     (quantize_tensor_absmax,      quantize_tensor_per_token),
    "O2":     (quantize_tensor_absmax,      quantize_tensor_absmax),
    "O3":     (quantize_tensor_absmax,      quantize_tensor_absmax),
    "O1_pcw": (quantize_tensor_per_channel, quantize_tensor_per_token),
    "O2_pcw": (quantize_tensor_per_channel, quantize_tensor_absmax),
    "O3_pcw": (quantize_tensor_per_channel, quantize_tensor_absmax),
    "naive_w8a8": (quantize_tensor_absmax,  quantize_tensor_absmax),
}


def apply_fake_quantization(model, scheme):
    """Replace all nn.Linear modules with FakeQuantLinear wrappers."""
    if scheme not in SCHEME_MAP:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from: {list(SCHEME_MAP.keys())}")
    
    w_fn, a_fn = SCHEME_MAP[scheme]
    print(f"Applying fake quantization: scheme={scheme}")
    
    replaced = 0
    for name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, FakeQuantLinear(child, w_fn, a_fn))
                replaced += 1
    
    print(f"  Replaced {replaced} linear layers.")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 01: Quantization Scheme Exploration")
    p.add_argument("--model", type=str, default="opt-1.3b",
                   help="Short name (opt-125m, opt-1.3b, ...) or full HF ID")
    p.add_argument("--scheme", type=str, default="O3",
                   help="fp16 | O1 | O2 | O3 | O1_pcw | O2_pcw | O3_pcw | naive_w8a8")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Smoothing migration strength (default: 0.5)")
    p.add_argument("--no_smooth", action="store_true",
                   help="Skip smoothing (for naive baselines)")
    p.add_argument("--scales_dir", type=str, default=None,
                   help="Dir with activation scale .pt files (auto-detected if not set)")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Where to save result JSON")
    p.add_argument("--drive_path", type=str, default=None,
                   help="Google Drive subpath (e.g. thesis_results/task01)")
    p.add_argument("--skip_zeroshot", action="store_true")
    p.add_argument("--skip_ppl", action="store_true")
    return p.parse_args()


def find_scales_dir():
    """Auto-detect where activation scales are, checking common locations."""
    candidates = [
        os.path.join(PROJECT_ROOT, "smoothquant_repo", "act_scales"),  # Colab clone
        os.path.join(PROJECT_ROOT, "smoothquant", "act_scales"),
        os.path.join("/content", "smoothquant_repo", "act_scales"),    # Colab absolute
        os.path.join("/content", "smoothquant", "act_scales"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Cannot find activation scales directory.\n"
        f"Checked: {candidates}\n"
        f"Please clone SmoothQuant repo or pass --scales_dir explicitly."
    )


def main():
    args = parse_args()
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
        scales_dir = args.scales_dir or find_scales_dir()
        apply_smoothing(model, hf_name, args.alpha, scales_dir)

    # 3. Apply quantization
    if args.scheme != "fp16":
        apply_fake_quantization(model, args.scheme)

    # 4. Evaluate
    eval_results = run_full_evaluation(
        model, tokenizer,
        skip_zeroshot=args.skip_zeroshot,
        skip_ppl=args.skip_ppl,
    )

    elapsed = time.time() - start

    # 5. Build result
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

    # 6. Save
    save_dir = args.save_dir or os.path.join(SCRIPT_DIR, "results")
    save_result(result, save_dir)

    if args.drive_path:
        drive_full = os.path.join("/content/drive/MyDrive", args.drive_path)
        if os.path.isdir("/content/drive/MyDrive"):
            save_result(result, drive_full)
        else:
            print("  Drive not mounted, skipping Drive save.")

    print(f"\nDone in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()