"""
Quick sanity test — verifies setup works before full evaluation.
Run: python experiments/task01_scheme_exploration/quick_sanity_test.py

Redirects to colab_runner.py --sanity since the checks are the same.
"""
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from shared.model_utils import load_model_and_tokenizer, get_gpu_info
from shared.quant_utils import quantize_tensor_absmax, quantize_tensor_per_token
import torch


def main():
    print("=" * 60)
    print("  SANITY TEST")
    print("=" * 60)

    # 1. GPU
    gpu = get_gpu_info()
    print(f"\n[1/4] GPU: {gpu['name']} ({gpu['memory_gb']}GB)")
    if not gpu["available"]:
        print("  ⚠️  No GPU — will use CPU (slow but works for testing)")

    # 2. Load tiny model
    print("\n[2/4] Loading OPT-125M ...")
    model, tokenizer = load_model_and_tokenizer("opt-125m")
    print("  ✅ Model loaded")

    # 3. Test quantization functions
    print("\n[3/4] Testing quantization functions ...")
    x = torch.randn(4, 768)
    x_q1 = quantize_tensor_absmax(x)
    x_q2 = quantize_tensor_per_token(x)
    print(f"  Per-tensor error: {(x - x_q1).abs().mean():.6f}")
    print(f"  Per-token error:  {(x - x_q2).abs().mean():.6f}")
    print("  ✅ Quantization functions work")

    # 4. Test forward pass
    print("\n[4/4] Testing forward pass ...")
    device = next(model.parameters()).device
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    next_tok = tokenizer.decode(out.logits[0, -1].argmax())
    print(f"  'The capital of France is' → '{next_tok}'")
    print("  ✅ Forward pass works")

    print("\n" + "=" * 60)
    print("  ✅ ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()