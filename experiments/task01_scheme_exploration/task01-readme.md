# Task 01: Quantization Scheme Exploration

**Goal:** Reproduce SmoothQuant's O1/O2/O3 results and explore additional
weight/activation quantization granularity combinations. Build a comprehensive
table mapping the full accuracy-vs-scheme tradeoff space.

**Reference:** SmoothQuant Table 2 (scheme definitions) and Table 11 (latency).

## What this does
- Loads a model (e.g., OPT-1.3B)
- Loads pre-computed activation scales from SmoothQuant
- Applies smoothing with a given alpha
- Applies quantization with a specified scheme (O1, O2, O3, or custom)
- Evaluates on zero-shot benchmarks and WikiText-2 perplexity
- Saves results as JSON

## Usage

```bash
# Single run:
python run_scheme_eval.py --model opt-1.3b --scheme O1 --alpha 0.5

# With Drive save (Colab):
python run_scheme_eval.py --model opt-1.3b --scheme O3 --alpha 0.5 \
    --drive_path thesis_results/task01

# FP16 baseline (no quantization):
python run_scheme_eval.py --model opt-1.3b --scheme fp16

# Naive W8A8 (quantize without smoothing):
python run_scheme_eval.py --model opt-1.3b --scheme naive_w8a8 --no_smooth

# Quick test (skip zero-shot, only perplexity):
python run_scheme_eval.py --model opt-125m --scheme O3 --alpha 0.5 --skip_zeroshot
```

## Schemes to run for a complete table

For each model (opt-1.3b, opt-6.7b), run:
1. `--scheme fp16` (baseline)
2. `--scheme naive_w8a8 --no_smooth` (no smoothing)
3. `--scheme O1 --alpha 0.5` 
4. `--scheme O2 --alpha 0.5`
5. `--scheme O3 --alpha 0.5`
6. `--scheme O1_pcw --alpha 0.5` (new: per-channel weight)
7. `--scheme O2_pcw --alpha 0.5`
8. `--scheme O3_pcw --alpha 0.5`

## Expected output

Each run produces a JSON like:
```json
{
  "task": "task01_scheme_exploration",
  "model": "facebook/opt-1.3b",
  "scheme": "O3",
  "alpha": 0.5,
  "smoothed": true,
  "metrics": {
    "lambada_openai": 0.5842,
    "hellaswag": 0.2934,
    "piqa": 0.7171,
    "winogrande": 0.5933
  },
  "avg_zeroshot_acc": 0.5470,
  "wikitext2_ppl": 14.62,
  "gpu": "Tesla T4",
  "duration_seconds": 1234
}
```
