# Task 01: Scheme Comparison — Colab Cells (OPT-125M)

Copy-paste each cell into Colab. Run in order.

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

# Download Pile validation set (needed if re-generating scales)
!mkdir -p smoothquant_repo/dataset
!wget -q -O smoothquant_repo/dataset/val.jsonl.zst \
    https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst

# Copy activation scales from Drive
!mkdir -p smoothquant_repo/act_scales
!cp /content/drive/MyDrive/thesis_results/act_scales/*.pt smoothquant_repo/act_scales/

# Verify
!nvidia-smi
!ls -la smoothquant_repo/act_scales/
!python -c "from smoothquant.smooth import smooth_lm; print('smoothquant OK')"
```

---

## Cell 2: Create the parameterized eval script

```python
%%writefile /content/llm-quantization-thesis/run_scheme_compare.py
"""
Parameterized scheme comparison script.
Based on smoothquant/ppl_eval.py but accepts --weight_quant and --act_quant args.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from datasets import load_dataset
import argparse
import json
import os
import time
import tqdm


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
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating"):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)
        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--act_scales_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--weight_quant", type=str, default="per_channel",
                        choices=["per_channel", "per_tensor"])
    parser.add_argument("--act_quant", type=str, default="per_token",
                        choices=["per_token", "per_tensor"])
    parser.add_argument("--quantize_bmm", action="store_true", default=True)
    parser.add_argument("--config_label", type=str, default="unknown",
                        help="Label for this config (e.g. O1, O2, C, D)")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Path to save result JSON")
    args = parser.parse_args()

    start = time.time()

    print("=" * 60)
    print(f"  Config: {args.config_label}")
    print(f"  Model:  {args.model_path}")
    print(f"  Smooth: {args.smooth} (alpha={args.alpha})")
    print(f"  Quant:  {args.quantize}")
    if args.quantize:
        print(f"  Weight: {args.weight_quant}")
        print(f"  Act:    {args.act_quant}")
        print(f"  BMM:    {args.quantize_bmm}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, "cuda")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if args.smooth:
        act_scales = torch.load(args.act_scales_path)
        smooth_lm(model, act_scales, args.alpha)
        print("Smoothing applied.")

    if args.quantize:
        model = quantize_model(
            model,
            weight_quant=args.weight_quant,
            act_quant=args.act_quant,
            quantize_bmm_input=args.quantize_bmm,
        )
        print("Quantization applied.")

    ppl = evaluator.evaluate(model)
    elapsed = time.time() - start
    ppl_val = ppl.item()

    print(f"\n>>> Config {args.config_label}: Perplexity = {ppl_val:.4f} ({elapsed:.0f}s)")

    # Save result
    result = {
        "config_label": args.config_label,
        "model": args.model_path,
        "smooth": args.smooth,
        "alpha": args.alpha if args.smooth else None,
        "quantize": args.quantize,
        "weight_quant": args.weight_quant if args.quantize else None,
        "act_quant": args.act_quant if args.quantize else None,
        "quantize_bmm": args.quantize_bmm if args.quantize else None,
        "wikitext2_ppl": round(ppl_val, 4),
        "duration_seconds": round(elapsed, 1),
    }

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.save_json}")

    return result


if __name__ == "__main__":
    main()
```

---

## Cell 3: FP16 Baseline

```python
%cd /content/llm-quantization-thesis

!python run_scheme_compare.py \
    --model_path facebook/opt-125m \
    --act_scales_path smoothquant_repo/act_scales/opt-125m.pt \
    --config_label FP16 \
    --save_json results/task01/opt-125m_FP16.json
```

---

## Cell 4: Naive W8A8 (no smoothing)

```python
!python run_scheme_compare.py \
    --model_path facebook/opt-125m \
    --act_scales_path smoothquant_repo/act_scales/opt-125m.pt \
    --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label W8A8-naive \
    --save_json results/task01/opt-125m_W8A8-naive.json
```

---

## Cell 5: Config A — SmoothQuant-O1 (per-tensor W, per-token A)

```python
!python run_scheme_compare.py \
    --model_path facebook/opt-125m \
    --act_scales_path smoothquant_repo/act_scales/opt-125m.pt \
    --smooth --alpha 0.5 --quantize \
    --weight_quant per_tensor --act_quant per_token \
    --config_label SQ-O1 \
    --save_json results/task01/opt-125m_SQ-O1.json
```

---

## Cell 6: Config B — SmoothQuant-O2 (per-tensor W, per-tensor A)

```python
!python run_scheme_compare.py \
    --model_path facebook/opt-125m \
    --act_scales_path smoothquant_repo/act_scales/opt-125m.pt \
    --smooth --alpha 0.5 --quantize \
    --weight_quant per_tensor --act_quant per_tensor \
    --config_label SQ-O2 \
    --save_json results/task01/opt-125m_SQ-O2.json
```

---

## Cell 7: Config C — SQ-PCW-PT (per-channel W, per-token A) — repo default

```python
!python run_scheme_compare.py \
    --model_path facebook/opt-125m \
    --act_scales_path smoothquant_repo/act_scales/opt-125m.pt \
    --smooth --alpha 0.5 --quantize \
    --weight_quant per_channel --act_quant per_token \
    --config_label SQ-PCW-PT \
    --save_json results/task01/opt-125m_SQ-PCW-PT.json
```

---

## Cell 8: Config D — SQ-PCW-TEN (per-channel W, per-tensor A)

```python
!python run_scheme_compare.py \
    --model_path facebook/opt-125m \
    --act_scales_path smoothquant_repo/act_scales/opt-125m.pt \
    --smooth --alpha 0.5 --quantize \
    --weight_quant per_channel --act_quant per_tensor \
    --config_label SQ-PCW-TEN \
    --save_json results/task01/opt-125m_SQ-PCW-TEN.json
```

---

## Cell 9: Backup results to Drive

```python
!mkdir -p /content/drive/MyDrive/thesis_results/task01
!cp results/task01/*.json /content/drive/MyDrive/thesis_results/task01/

# Print summary
import json, glob
print(f"\n{'Config':<15} {'PPL':>10} {'Time':>8}")
print("-" * 35)
for f in sorted(glob.glob("results/task01/opt-125m_*.json")):
    r = json.load(open(f))
    print(f"{r['config_label']:<15} {r['wikitext2_ppl']:>10.4f} {r['duration_seconds']:>7.1f}s")
```

---

## Expected output format

After all runs, Cell 9 should print something like:

```
Config              PPL     Time
-----------------------------------
FP16            xx.xxxx   xxx.xs
W8A8-naive      xx.xxxx   xxx.xs
SQ-O1           xx.xxxx   xxx.xs
SQ-O2           xx.xxxx   xxx.xs
SQ-PCW-PT       xx.xxxx   xxx.xs
SQ-PCW-TEN      xx.xxxx   xxx.xs
```

If C < O1 and D < O2 (lower PPL = better), the hypothesis holds.
