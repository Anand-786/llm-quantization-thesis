"""
Quantization utilities.
Handles loading SmoothQuant activation scales, applying smoothing,
and wrapping models with quantized linear layers.
"""

import os
import torch
import torch.nn as nn
from functools import partial


# ─── Activation Scale Loading ────────────────────────────────────────────────

def get_act_scales(model_name: str, scales_dir: str = None) -> dict:
    """
    Load pre-computed activation scales for a model.
    
    SmoothQuant provides these in their repo under act_scales/.
    You should download them once and store on Drive.
    
    Args:
        model_name: HuggingFace model name (e.g. 'facebook/opt-1.3b').
        scales_dir: Directory containing .pt scale files.
                    If None, looks in smoothquant/act_scales/.
    
    Returns:
        Dict mapping layer names to scale tensors.
    """
    if scales_dir is None:
        # Default: look in cloned smoothquant repo
        scales_dir = "smoothquant/act_scales"
    
    # Convert model name to filename format
    fname = model_name.replace("/", "-") + ".pt"
    path = os.path.join(scales_dir, fname)
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Activation scales not found at {path}. "
            f"Either download from SmoothQuant repo or generate them. "
            f"Available files: {os.listdir(scales_dir) if os.path.exists(scales_dir) else 'dir missing'}"
        )
    
    scales = torch.load(path, map_location="cpu")
    print(f"Loaded activation scales: {len(scales)} layers from {path}")
    return scales


# ─── Smoothing ───────────────────────────────────────────────────────────────

@torch.no_grad()
def smooth_layer(
    ln: nn.Module,           # LayerNorm or RMSNorm before the linear layers
    linear_layers: list,     # List of nn.Linear modules that consume ln's output
    act_scales: torch.Tensor,  # Per-channel activation scales [C_i]
    alpha: float = 0.5,
):
    """
    Apply SmoothQuant smoothing to a single transformer sublayer.
    
    Divides activation channels by smoothing factor s and multiplies
    corresponding weight channels by s, preserving mathematical equivalence.
    
    s_j = act_scales_j^alpha / weight_scales_j^(1-alpha)
    
    Args:
        ln: The LayerNorm preceding the linear layers.
        linear_layers: Linear layers that take ln's output as input.
        act_scales: Pre-computed max absolute activation per channel.
        alpha: Migration strength. 0.5 = equal difficulty sharing.
    """
    device, dtype = ln.weight.device, ln.weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # Compute weight scales: max across all target linear layers
    weight_scales = torch.cat(
        [l.weight.abs().max(dim=0, keepdim=True).values for l in linear_layers],
        dim=0,
    ).max(dim=0).values.clamp(min=1e-5)
    
    # Compute smoothing factor
    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    
    # Apply to LayerNorm: absorb 1/s into ln weight and bias
    ln.weight.div_(s)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(s)
    
    # Apply to Linear layers: absorb s into weight columns
    for linear in linear_layers:
        linear.weight.mul_(s.unsqueeze(0))  # s along input dim


# ─── Simulated Quantization ──────────────────────────────────────────────────

def quantize_tensor_absmax(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """
    Simulate symmetric absmax quantization: quantize then immediately dequantize.
    Returns a float tensor with quantization noise but same dtype as input.
    """
    qmax = 2 ** (n_bits - 1) - 1  # 127 for 8-bit
    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-10)
    x_q = (x / scale).round().clamp(-qmax, qmax)
    return x_q * scale


def quantize_tensor_per_token(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Per-token (row-wise) symmetric absmax quantization."""
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)
    x_q = (x / scale).round().clamp(-qmax, qmax)
    return x_q * scale


def quantize_tensor_per_channel(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Per-channel (column-wise) symmetric absmax quantization."""
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().amax(dim=0, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)
    x_q = (x / scale).round().clamp(-qmax, qmax)
    return x_q * scale


# ─── Quantization Scheme Definitions ─────────────────────────────────────────

# Maps scheme names to (weight_quant_fn, activation_quant_fn) pairs.
# These match SmoothQuant Table 2 and extend beyond O1/O2/O3.
QUANT_SCHEMES = {
    # SmoothQuant original presets
    "O1": {
        "weight": "per_tensor",
        "activation": "per_token_dynamic",
        "description": "SmoothQuant-O1: per-tensor W, per-token dynamic A",
    },
    "O2": {
        "weight": "per_tensor",
        "activation": "per_tensor_dynamic",
        "description": "SmoothQuant-O2: per-tensor W, per-tensor dynamic A",
    },
    "O3": {
        "weight": "per_tensor",
        "activation": "per_tensor_static",
        "description": "SmoothQuant-O3: per-tensor W, per-tensor static A (fastest)",
    },
    # Additional schemes to explore in Task 1
    "O1_pcw": {
        "weight": "per_channel",
        "activation": "per_token_dynamic",
        "description": "Per-channel W, per-token dynamic A",
    },
    "O2_pcw": {
        "weight": "per_channel",
        "activation": "per_tensor_dynamic",
        "description": "Per-channel W, per-tensor dynamic A",
    },
    "O3_pcw": {
        "weight": "per_channel",
        "activation": "per_tensor_static",
        "description": "Per-channel W, per-tensor static A",
    },
    "naive_w8a8": {
        "weight": "per_tensor",
        "activation": "per_tensor_dynamic",
        "description": "Naive W8A8 (no smoothing applied)",
    },
}


def get_scheme_info(scheme_name: str) -> dict:
    """Get quantization scheme configuration."""
    if scheme_name not in QUANT_SCHEMES:
        raise ValueError(
            f"Unknown scheme '{scheme_name}'. "
            f"Available: {list(QUANT_SCHEMES.keys())}"
        )
    return QUANT_SCHEMES[scheme_name]
