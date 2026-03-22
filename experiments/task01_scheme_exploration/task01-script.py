"""
Task 01: Quantization Scheme Exploration
=========================================
Reproduce SmoothQuant O1/O2/O3 results and explore additional
weight/activation quantization granularity combinations.

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

# Add project root to path so we can import shared/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from shared.model_utils import load_model_and_tokenizer, get_gpu_info, resolve_model_name
from shared.save_utils import save_result, save_to_drive, print_result_summary
from shared.eval_utils import run_full_evaluation
from shared.quant_utils import get_act_scales, smooth_layer, get_scheme_info


# ─── SmoothQuant Integration ────────────────────────────────────────────────

def get_smooth_layer_pairs_opt(model):
    """
    Get (layernorm, [linear_layers]) pairs for OPT models.
    These are the pairs where smoothing is applied.
    
    For OPT, SmoothQuant smooths:
    - self_attn_layer_norm → q_proj, k_proj, v_proj
    - final_layer_norm → fc1
    """
    pairs = []
    for i, layer in enumerate(model.model.decoder.layers):
        # Attention block
        pairs.append((
            layer.self_attn_layer_norm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            f"layers.{i}.self_attn"
        ))
        # FFN block
        pairs.append((
            layer.final_layer_norm,
            [layer.fc1],
            f"layers.{i}.fc1"
        ))
    return pairs


def apply_smoothing(model, model_name, alpha, scales_dir=None):
    """
    Apply SmoothQuant smoothing to all layers of a model.
    
    Args:
        model: HuggingFace model.
        model_name: Full HF model name (e.g. 'facebook/opt-1.3b').
        alpha: Migration strength.
        scales_dir: Where to find .pt activation scale files.
    """
    act_scales = get_act_scales(model_name, scales_dir)
    
    # Determine model family and get layer pairs
    if "opt" in model_name.lower():
        pairs = get_smooth_layer_pairs_opt(model)
    else:
        raise NotImplementedError(
            f"Smoothing not yet implemented for {model_name}. "
            f"Add a get_smooth_layer_pairs_xxx() function for this architecture."
        )
    
    print(f"Applying smoothing: alpha={alpha}, {len(pairs)} layer pairs")
    
    for ln, linears, name in pairs:
        # Look up the activation scale for this layer
        # SmoothQuant scale keys look like: 'model.decoder.layers.0.self_attn.q_proj'
        # We need to find matching keys in act_scales
        scale_key = None
        for key in act_scales:
            if name.replace("layers.", "model.decoder.layers.") in key:
                scale_key = key
                break
        
        if scale_key is None:
            # Try a broader match
            layer_idx = name.split(".")[1]
            if "self_attn" in name:
                for key in act_scales:
                    if f"layers.{layer_idx}.self_attn" in key:
                        scale_key = key
                        break
            elif "fc1" in name:
                for key in act_scales:
                    if f"layers.{layer_idx}.fc1" in key:
                        scale_key = key
                        break
        
        if scale_key is not None:
            smooth_layer(ln, linears, act_scales[scale_key], alpha=alpha)
        else:
            print(f"  Warning: no activation scale found for {name}, skipping")
    
    print("Smoothing applied successfully.")


# ─── Quantized Linear Wrapper ────────────────────────────────────────────────

class QuantizedLinear(torch.nn.Module):
    """
    Wrapper that replaces nn.Linear with simulated quantization.
    Quantizes weights (offline) and activations (at forward time).
    """
    
    def __init__(self, original: torch.nn.Linear, w_quant_fn, a_quant_fn):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bias = original.bias
        
        # Pre-quantize weights (simulated: quantize + dequantize)
        self.weight = torch.nn.Parameter(w_quant_fn(original.weight.data), requires_grad=False)
        self.a_quant_fn = a_quant_fn
    
    def forward(self, x):
        # Quantize activation at runtime
        x_q = self.a_quant_fn(x)
        return torch.nn.functional.linear(x_q, self.weight, self.bias)


def quantize_model_weights_and_activations(model, scheme_name):
    """
    Replace all linear layers in a model with simulated quantized versions.
    
    This is simulated quantization: values are quantized then immediately
    dequantized back to float. This measures accuracy impact without
    needing actual INT8 kernels.
    """
    from shared.quant_utils import (
        quantize_tensor_absmax, 
        quantize_tensor_per_token,
        quantize_tensor_per_channel,
    )
    
    if scheme_name == "fp16":
        print("FP16 baseline — no quantization applied.")
        return
    
    scheme = get_scheme_info(scheme_name)
    print(f"Applying quantization scheme: {scheme['description']}")
    
    # Weight quantization function
    w_type = scheme["weight"]
    if w_type == "per_tensor":
        w_fn = quantize_tensor_absmax
    elif w_type == "per_channel":
        w_fn = quantize_tensor_per_channel
    else:
        raise ValueError(f"Unknown weight quant type: {w_type}")
    
    # Activation quantization function
    a_type = scheme["activation"]
    if a_type == "per_token_dynamic":
        a_fn = quantize_tensor_per_token
    elif a_type in ("per_tensor_dynamic", "per_tensor_static"):
        a_fn = quantize_tensor_absmax
    else:
        raise ValueError(f"Unknown activation quant type: {a_type}")
    
    # Replace all nn.Linear in the model
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                setattr(module, child_name, QuantizedLinear(child, w_fn, a_fn))
                replaced += 1
    
    print(f"  Replaced {replaced} linear layers with simulated quantization.")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 01: Quantization Scheme Exploration")
    p.add_argument("--model", type=str, default="opt-1.3b",
                   help="Model name (short or full HF ID)")
    p.add_argument("--scheme", type=str, default="O3",
                   help="Quantization scheme: fp16, O1, O2, O3, O1_pcw, O2_pcw, O3_pcw, naive_w8a8")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Smoothing migration strength (default: 0.5)")
    p.add_argument("--no_smooth", action="store_true",
                   help="Skip smoothing (for naive baselines)")
    p.add_argument("--scales_dir", type=str, default=None,
                   help="Directory with activation scale .pt files")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Local directory to save results (default: experiments/task01/results/)")
    p.add_argument("--drive_path", type=str, default=None,
                   help="Google Drive path to save results (e.g. thesis_results/task01)")
    p.add_argument("--skip_zeroshot", action="store_true",
                   help="Skip zero-shot eval (for quick testing)")
    p.add_argument("--skip_ppl", action="store_true",
                   help="Skip WikiText-2 perplexity eval")
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    hf_name = resolve_model_name(args.model)
    gpu = get_gpu_info()
    
    print("=" * 60)
    print("  TASK 01: Quantization Scheme Exploration")
    print(f"  Model:  {hf_name}")
    print(f"  Scheme: {args.scheme}")
    print(f"  Alpha:  {args.alpha}")
    print(f"  Smooth: {not args.no_smooth}")
    print(f"  GPU:    {gpu['name']} ({gpu['memory_gb']}GB)")
    print("=" * 60)
    
    # Step 1: Load model
    model, tokenizer = load_model_and_tokenizer(hf_name)
    
    # Step 2: Apply smoothing (unless fp16 baseline or --no_smooth)
    if args.scheme != "fp16" and not args.no_smooth:
        apply_smoothing(model, hf_name, args.alpha, args.scales_dir)
    
    # Step 3: Apply quantization
    if args.scheme != "fp16":
        quantize_model_weights_and_activations(model, args.scheme)
    
    # Step 4: Evaluate
    eval_results = run_full_evaluation(
        model, tokenizer,
        skip_zeroshot=args.skip_zeroshot,
        skip_ppl=args.skip_ppl,
    )
    
    elapsed = time.time() - start_time
    
    # Step 5: Assemble result
    result = {
        "task": "task01_scheme_exploration",
        "model": hf_name,
        "scheme": args.scheme,
        "alpha": args.alpha if not args.no_smooth else None,
        "smoothed": not args.no_smooth and args.scheme != "fp16",
        "metrics": eval_results.get("zeroshot", {}),
        "avg_zeroshot_acc": eval_results.get("avg_zeroshot_acc"),
        "wikitext2_ppl": eval_results.get("wikitext2_ppl"),
        "gpu": gpu["name"],
        "gpu_memory_gb": gpu["memory_gb"],
        "duration_seconds": round(elapsed, 1),
    }
    
    print_result_summary(result)
    
    # Step 6: Save results
    default_save = os.path.join(os.path.dirname(__file__), "results")
    save_dir = args.save_dir or default_save
    save_result(result, save_dir)
    
    if args.drive_path:
        save_to_drive(result, args.drive_path)
    
    print(f"\nDone in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
