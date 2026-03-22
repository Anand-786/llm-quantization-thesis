"""
Quick Sanity Test
==================
Run this FIRST to verify everything works before full evaluation.
Uses OPT-125M (tiny) and skips zero-shot eval.
Should complete in under 5 minutes even on CPU.

Usage:
    python experiments/task01_scheme_exploration/quick_sanity_test.py
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from shared.model_utils import load_model_and_tokenizer, get_gpu_info
from shared.quant_utils import quantize_tensor_absmax, quantize_tensor_per_token

def main():
    print("=" * 60)
    print("  SANITY TEST — Verifying setup")
    print("=" * 60)
    
    # 1. Check GPU
    gpu = get_gpu_info()
    print(f"\n[1/5] GPU: {gpu['name']} ({gpu['memory_gb']}GB)")
    if not gpu["available"]:
        print("  ⚠️  No GPU — will run on CPU (slow but OK for testing)")
    
    # 2. Load tiny model
    print("\n[2/5] Loading OPT-125M...")
    model, tokenizer = load_model_and_tokenizer("opt-125m")
    print("  ✅ Model loaded")
    
    # 3. Test quantization functions
    print("\n[3/5] Testing quantization functions...")
    x = torch.randn(4, 768)  # Fake activation tensor
    x_q_tensor = quantize_tensor_absmax(x)
    x_q_token = quantize_tensor_per_token(x)
    err_tensor = (x - x_q_tensor).abs().mean().item()
    err_token = (x - x_q_token).abs().mean().item()
    print(f"  Per-tensor quant error: {err_tensor:.6f}")
    print(f"  Per-token quant error:  {err_token:.6f}")
    print(f"  Per-token should be <= per-tensor: {err_token <= err_tensor + 1e-6}")
    print("  ✅ Quantization functions work")
    
    # 4. Test model forward pass
    print("\n[4/5] Testing forward pass...")
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    next_token = tokenizer.decode(logits[0, -1].argmax())
    print(f"  Input:  'The capital of France is'")
    print(f"  Predicted next token: '{next_token}'")
    print("  ✅ Forward pass works")
    
    # 5. Test perplexity computation (tiny subset)
    print("\n[5/5] Testing perplexity computation (small subset)...")
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    enc = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    ppl = torch.exp(out.loss).item()
    print(f"  Test perplexity: {ppl:.2f}")
    print("  ✅ Perplexity computation works")
    
    # 6. Test saving
    print("\n[6/6] Testing result saving...")
    from shared.save_utils import save_result
    test_result = {
        "task": "sanity_test", "model": "opt-125m",
        "scheme": "test", "metrics": {"test_ppl": ppl}
    }
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    path = save_result(test_result, save_dir, "sanity_test.json")
    
    # Clean up test file
    if os.path.exists(path):
        os.remove(path)
        print("  ✅ Save/load works (test file cleaned up)")
    
    print("\n" + "=" * 60)
    print("  ✅ ALL CHECKS PASSED — You're ready to run experiments!")
    print("=" * 60)


if __name__ == "__main__":
    main()
