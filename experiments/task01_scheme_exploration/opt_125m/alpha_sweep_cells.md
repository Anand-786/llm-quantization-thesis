# Task 01: Alpha Sweep — O1 vs C on OPT-125M

Sweeps alpha at 5 discrete levels (0.1, 0.3, 0.5, 0.7, 0.9) for both O1 and C configs.
Tests 2 values below and 2 above the paper's recommended alpha=0.5 at step size 0.2.
10 total runs (~10-15 min on T4). Run Cell 1 (setup), then Cell 2 (sweep) and walk away.

---

## Cell 1: Setup

```python
# Clone repos + install
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy activation scales from Drive
!mkdir -p smoothquant_repo/act_scales
!cp /content/drive/MyDrive/thesis_results/act_scales/*.pt smoothquant_repo/act_scales/

# Create Drive output folder
!mkdir -p /content/drive/MyDrive/thesis_results/task01_alphasweep

# Verify
!nvidia-smi
!ls -la smoothquant_repo/act_scales/
!python -c "from smoothquant.smooth import smooth_lm; print('smoothquant OK')"
```

---

## Cell 2: Run full alpha sweep (leave this running)

```python
import sys
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from datasets import load_dataset
import json, os, time, copy, tqdm

MODEL = "facebook/opt-125m"
SCALES = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/opt-125m.pt"
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]  # step=0.2, 2 below and 2 above paper's 0.5
SAVE_DIR = "/content/drive/MyDrive/thesis_results/task01_alphasweep"

CONFIGS = {
    "SQ-O1":     {"weight_quant": "per_tensor",  "act_quant": "per_token"},
    "SQ-PCW-PT": {"weight_quant": "per_channel", "act_quant": "per_token"},
}


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n), desc="Evaluating"):
            batch = self.dataset[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            nlls.append(loss.float() * 2048)
        return torch.exp(torch.stack(nlls).sum() / (n * 2048))


# --- Load once ---
print("Loading tokenizer + dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda")
act_scales = torch.load(SCALES)

print("Loading base model weights (will reload per run)...")
# We just verify it loads; actual loading happens in the loop
_ = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
del _
torch.cuda.empty_cache()

all_results = []

total_runs = len(ALPHAS) * len(CONFIGS)
run_num = 0

for alpha in ALPHAS:
    for config_label, qparams in CONFIGS.items():
        run_num += 1
        print(f"\n{'='*60}")
        print(f"  Run {run_num}/{total_runs}: {config_label} alpha={alpha}")
        print(f"  Weight: {qparams['weight_quant']}, Act: {qparams['act_quant']}")
        print(f"{'='*60}")

        start = time.time()

        # Fresh model load each run (smoothing modifies weights in-place)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Smooth
        smooth_lm(model, act_scales, alpha)

        # Quantize
        model = quantize_model(
            model,
            weight_quant=qparams["weight_quant"],
            act_quant=qparams["act_quant"],
            quantize_bmm_input=True,
        )

        # Evaluate
        ppl = evaluator.evaluate(model)
        elapsed = time.time() - start
        ppl_val = ppl.item()

        result = {
            "config_label": config_label,
            "model": MODEL,
            "alpha": alpha,
            "weight_quant": qparams["weight_quant"],
            "act_quant": qparams["act_quant"],
            "wikitext2_ppl": round(ppl_val, 4),
            "duration_seconds": round(elapsed, 1),
        }
        all_results.append(result)

        # Save individual result
        fname = f"opt-125m_{config_label}_a{alpha}.json"
        with open(os.path.join(SAVE_DIR, fname), "w") as f:
            json.dump(result, f, indent=2)

        print(f">>> {config_label} alpha={alpha}: PPL = {ppl_val:.4f} ({elapsed:.0f}s)")

        # Cleanup
        del model
        torch.cuda.empty_cache()

# --- Save combined results ---
with open(os.path.join(SAVE_DIR, "all_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# --- Print summary table ---
print(f"\n\n{'='*60}")
print(f"  ALPHA SWEEP RESULTS — OPT-125M")
print(f"{'='*60}")
print(f"\n{'Alpha':>6} {'SQ-O1 PPL':>12} {'SQ-PCW-PT PPL':>15} {'Diff (O1-C)':>12} {'C wins?':>8}")
print("-" * 55)

for alpha in ALPHAS:
    o1 = next((r for r in all_results if r["alpha"] == alpha and r["config_label"] == "SQ-O1"), None)
    c  = next((r for r in all_results if r["alpha"] == alpha and r["config_label"] == "SQ-PCW-PT"), None)
    if o1 and c:
        diff = o1["wikitext2_ppl"] - c["wikitext2_ppl"]
        wins = "YES" if diff > 0 else "no"
        print(f"{alpha:>6.1f} {o1['wikitext2_ppl']:>12.4f} {c['wikitext2_ppl']:>15.4f} {diff:>+12.4f} {wins:>8}")

print(f"\nPositive diff = C is better (lower PPL). Results saved to {SAVE_DIR}")
```
