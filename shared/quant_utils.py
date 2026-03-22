"""
Quantization utilities.
Uses SmoothQuant's own library functions where possible.
Our custom code only for things SmoothQuant doesn't provide.
"""

import os
import torch


def find_act_scales_file(model_name: str, scales_dir: str) -> str:
    """
    Find the activation scales .pt file for a given model.
    
    SmoothQuant names these files using the last part of the model name:
      'facebook/opt-1.3b' -> 'opt-1.3b.pt'
      'meta-llama/Llama-2-7b-hf' -> 'Llama-2-7b-hf.pt'
    """
    short_name = model_name.split("/")[-1]
    path = os.path.join(scales_dir, short_name + ".pt")
    
    if os.path.exists(path):
        return path
    
    # Try lowercase
    path_lower = os.path.join(scales_dir, short_name.lower() + ".pt")
    if os.path.exists(path_lower):
        return path_lower
    
    # List what's available to help debug
    available = []
    if os.path.isdir(scales_dir):
        available = [f for f in os.listdir(scales_dir) if f.endswith(".pt")]
    
    raise FileNotFoundError(
        f"No activation scales found for '{model_name}'.\n"
        f"  Looked for: {path}\n"
        f"  Available files: {available}"
    )


def load_act_scales(model_name: str, scales_dir: str) -> dict:
    """Load pre-computed activation scales."""
    path = find_act_scales_file(model_name, scales_dir)
    scales = torch.load(path, map_location="cpu")
    print(f"Loaded activation scales: {len(scales)} layers from {path}")
    return scales


def quantize_tensor_absmax(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Simulate symmetric per-tensor absmax quantization (quantize + dequantize)."""
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-10)
    return (x / scale).round().clamp(-qmax, qmax) * scale


def quantize_tensor_per_token(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Simulate symmetric per-token (row-wise) absmax quantization."""
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)
    return (x / scale).round().clamp(-qmax, qmax) * scale


def quantize_tensor_per_channel(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Simulate symmetric per-channel (column-wise) absmax quantization."""
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().amax(dim=0, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)
    return (x / scale).round().clamp(-qmax, qmax) * scale