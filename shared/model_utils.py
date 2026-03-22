"""
Model loading utilities shared across all experiments.
Handles OPT and LLaMA model families with proper device placement.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Map of short names to HuggingFace model IDs
MODEL_REGISTRY = {
    # OPT family
    "opt-125m":  "facebook/opt-125m",
    "opt-1.3b":  "facebook/opt-1.3b",
    "opt-2.7b":  "facebook/opt-2.7b",
    "opt-6.7b":  "facebook/opt-6.7b",
    "opt-13b":   "facebook/opt-13b",
    # LLaMA family (add your HF-authorized IDs)
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
}


def resolve_model_name(name: str) -> str:
    """Convert short name to full HuggingFace ID, or pass through if already full."""
    return MODEL_REGISTRY.get(name, name)


def get_gpu_info() -> dict:
    """Return basic GPU information."""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "memory_gb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "memory_gb": round(props.total_mem / 1e9, 1),
    }


def load_model_and_tokenizer(
    model_name: str,
    dtype=torch.float16,
    device_map: str = "auto",
):
    """
    Load a HuggingFace causal LM and its tokenizer.
    
    Args:
        model_name: Short name (e.g. 'opt-1.3b') or full HF ID.
        dtype: Model precision. Default FP16.
        device_map: HuggingFace Accelerate device map strategy.
    
    Returns:
        (model, tokenizer) tuple
    """
    hf_name = resolve_model_name(model_name)
    print(f"Loading model: {hf_name}")
    print(f"  dtype={dtype}, device_map={device_map}")
    
    gpu = get_gpu_info()
    print(f"  GPU: {gpu['name']} ({gpu['memory_gb']}GB)")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.2f}B parameters")

    return model, tokenizer


def can_fit_model(model_name: str) -> bool:
    """
    Quick check: can this model likely fit on the current GPU in FP16?
    Conservative estimates (need headroom for activations).
    """
    gpu = get_gpu_info()
    if not gpu["available"]:
        return False
    
    # Rough FP16 memory estimates (GB) including activation overhead
    estimates = {
        "opt-125m": 1, "opt-1.3b": 4, "opt-2.7b": 7,
        "opt-6.7b": 15, "opt-13b": 30,
        "llama2-7b": 16, "llama3-8b": 18,
    }
    
    needed = estimates.get(model_name, 999)
    return gpu["memory_gb"] >= needed
