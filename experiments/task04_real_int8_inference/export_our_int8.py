"""Export an Int8OPTForCausalLM checkpoint smoothed with our percentile method.

Pipeline (mirrors smoothquant_repo/examples/export_int8_model.py but swaps
max-smoothing for percentile-smoothing):

  1. Load fp16 OPT model.
  2. Apply experiments.task02_percentile_smoothing.percentile_smooth.smooth_lm_pct
     using the per-model winning (p, alpha) from Task 02.
  3. Run smoothquant.calibration.get_static_decoder_layer_scales on the Pile
     validation set to get per-tensor static activation scales.
  4. Convert via Int8OPTForCausalLM.from_float and save_pretrained.

Note: the resulting INT8 model is per-tensor W + per-tensor static A (i.e.
paper's O3 recipe at the kernel level), with our percentile-smoothing applied
upstream. This is the only granularity supported by torch-int's CUTLASS
kernels — see experiment_plan.md for context.
"""
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.calibration import get_static_decoder_layer_scales
from experiments.task02_percentile_smoothing.percentile_smooth import smooth_lm_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="e.g. facebook/opt-1.3b")
    ap.add_argument("--pct_scales", required=True,
                    help="path to act_percentiles/opt-<size>/p<p>.pt")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--p_w", type=float, required=True,
                    help="passed to smooth_lm_pct as p_w (must match the file)")
    ap.add_argument("--dataset_path", required=True,
                    help="Pile val.jsonl.zst for static calibration")
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--output_path", required=True,
                    help="dir to save the INT8 model")
    args = ap.parse_args()

    print(f"loading {args.model_name} (fp16)")
    model = OPTForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"loading percentile scales from {args.pct_scales}")
    pct_scales = torch.load(args.pct_scales)

    print(f"applying percentile smoothing  alpha={args.alpha}  p_w={args.p_w}")
    smooth_lm_pct(model, pct_scales, alpha=args.alpha, p_w=args.p_w)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"dataset not found: {args.dataset_path}\n"
            "Download from https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst"
        )

    print(f"static calibration on {args.dataset_path} "
          f"({args.num_samples} samples × {args.seq_len} tokens)")
    decoder_layer_scales, _raw = get_static_decoder_layer_scales(
        model, tokenizer, args.dataset_path,
        num_samples=args.num_samples, seq_len=args.seq_len,
    )

    print("converting to Int8OPTForCausalLM")
    int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    int8_model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved INT8 model to {out}")


if __name__ == "__main__":
    main()
