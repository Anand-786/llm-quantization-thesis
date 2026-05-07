"""Task 03 zero-shot eval driver for OPT-1.3B (and other OPT sizes).

Single binary that supports both smoothing variants:
  --smooth_method max         -> upstream smoothquant.smooth.smooth_lm
  --smooth_method percentile  -> experiments.task02_percentile_smoothing.percentile_smooth.smooth_lm_pct

Quantization always goes through smoothquant.fake_quant.quantize_model so the
per-tensor / per-channel / per-token axes are flagged the same way as Task 01.

Eval is run via lm-evaluation-harness in zero-shot. Primary metric per task is
recorded as reported by the harness; full raw results are also saved.

Expected to be invoked from a Colab cell; see zero_shot_eval_cells.md.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Upstream SmoothQuant
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model

# Task 02 percentile smoothing — repo-local import
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from experiments.task02_percentile_smoothing.percentile_smooth import smooth_lm_pct

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--act_scales_path", default=None,
                   help="Path to .pt of {layer_name -> tensor[in_features]}. "
                        "Required if --smooth is set.")
    p.add_argument("--smooth", action="store_true")
    p.add_argument("--smooth_method", choices=["max", "percentile"], default="max")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--p_w", type=float, default=1.0,
                   help="Weight-side percentile for percentile smoothing. "
                        "Ignored unless --smooth_method=percentile.")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--weight_quant", choices=["per_channel", "per_tensor"],
                   default="per_channel")
    p.add_argument("--act_quant", choices=["per_token", "per_tensor"],
                   default="per_token")
    p.add_argument("--quantize_bmm", action="store_true", default=True)
    p.add_argument("--config_label", default="unknown")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    start = time.time()

    print("=" * 64)
    print(f"  Config:        {args.config_label}")
    print(f"  Model:         {args.model_path}")
    print(f"  Smooth:        {args.smooth} ({args.smooth_method}, alpha={args.alpha}, p_w={args.p_w})")
    print(f"  Quant:         {args.quantize} (W={args.weight_quant}, A={args.act_quant}, bmm={args.quantize_bmm})")
    print(f"  Tasks:         {TASKS}")
    print(f"  Batch size:    {args.batch_size}")
    print("=" * 64)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )

    if args.smooth:
        assert args.act_scales_path, "--act_scales_path required when --smooth is set"
        scales = torch.load(args.act_scales_path)
        if args.smooth_method == "max":
            smooth_lm(model, scales, args.alpha)
        else:
            smooth_lm_pct(model, scales, alpha=args.alpha, p_w=args.p_w)
        print(f"Smoothing applied ({args.smooth_method}).")

    if args.quantize:
        model = quantize_model(
            model,
            weight_quant=args.weight_quant,
            act_quant=args.act_quant,
            quantize_bmm_input=args.quantize_bmm,
        )
        print("Quantization applied.")

        # Sanity check: read realized state off a W8A8Linear to prove
        # weight/act quantization axes match what was requested. Hard-fail if not.
        from smoothquant.fake_quant import W8A8Linear
        sample = next((m for m in model.modules() if isinstance(m, W8A8Linear)), None)
        if sample is None:
            raise RuntimeError("No W8A8Linear found after quantize_model — "
                               "quantization did not apply.")
        n_w8a8 = sum(1 for m in model.modules() if isinstance(m, W8A8Linear))
        realized_w = getattr(sample, "weight_quant_name", None)
        realized_a = getattr(sample, "act_quant_name", None)
        print(f"  Verified: {n_w8a8} W8A8Linear modules; "
              f"weight_quant={realized_w}, act_quant={realized_a}")
        assert realized_w == args.weight_quant, (
            f"Weight quant mismatch: requested {args.weight_quant}, got {realized_w}")
        assert realized_a == args.act_quant, (
            f"Act quant mismatch: requested {args.act_quant}, got {realized_a}")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
    )

    elapsed = time.time() - start

    # Pull primary acc metrics for printing; keep raw for archival.
    summary = {}
    for task, metrics in results["results"].items():
        clean = {
            k: v for k, v in metrics.items()
            if any(m in k for m in ["acc", "perplexity", "ppl"]) and "stderr" not in k
        }
        summary[task] = clean

    print("\n" + "=" * 64)
    print(f"  {args.config_label} — zero-shot results")
    print("=" * 64)
    for task, metrics in summary.items():
        line = "  ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                         for k, v in metrics.items())
        print(f"  {task:<20} {line}")
    print(f"\n  Total time: {elapsed:.0f}s")

    out = {
        "config_label": args.config_label,
        "model": args.model_path,
        "smooth": args.smooth,
        "smooth_method": args.smooth_method if args.smooth else None,
        "alpha": args.alpha if args.smooth else None,
        "p_w": args.p_w if (args.smooth and args.smooth_method == "percentile") else None,
        "act_scales_path": args.act_scales_path if args.smooth else None,
        "quantize": args.quantize,
        "weight_quant": args.weight_quant if args.quantize else None,
        "act_quant": args.act_quant if args.quantize else None,
        "quantize_bmm": args.quantize_bmm if args.quantize else None,
        "tasks": TASKS,
        "results": summary,
        "raw_results": results["results"],
        "duration_seconds": round(elapsed, 1),
        "seed": args.seed,
        "lm_eval_version": __import__("importlib.metadata", fromlist=["version"]).version("lm_eval"),
    }

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Saved -> {args.save_json}")


if __name__ == "__main__":
    main()
