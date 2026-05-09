# Task 04: Real INT8 Inference — Experiment Plan

## Goal

Demonstrate the **memory and latency advantages** of INT8 inference using the actual CUTLASS INT8 GEMM kernels from [torch-int](https://github.com/Guangxuan-Xiao/torch-int), matching the demo in [smoothquant_opt_real_int8_demo.ipynb](../../smoothquant_repo/examples/smoothquant_opt_real_int8_demo.ipynb).

Tasks 01–03 used **fake quantization** (round-trip in fp16) to study accuracy. Task 04 produces the headline "50% memory, faster latency, same accuracy" plot the thesis defense needs.

## Critical caveat — kernel limitation

The CUTLASS INT8 GEMM kernels exposed by torch-int (`W8A8B8O8Linear`, `W8A8BFP32OFP32Linear`, etc.) only support **per-tensor weight × per-tensor static activation** quantization. This corresponds to the paper's **O3** recipe (per-tensor W, per-tensor static A).

They do **not** support our winning **C scheme** (per-channel W + per-token A). Per-channel weight quant requires different kernel layouts; per-token activation quant requires dynamic scale computation per token, which CUTLASS INT8 GEMM does not expose.

Implications:
- The **weight memory** halving (fp16 → int8 weights) is unaffected by this — same storage regardless of granularity. The headline 50% memory reduction holds.
- The **activation behaviour** in real INT8 mode is necessarily O3 (per-tensor static), even when we apply our percentile-smoothing.
- For the thesis chapter, this is presented as an engineering boundary: the accuracy work in Tasks 01–03 motivates *why* the C scheme is best on paper, and Task 04 shows the *real-world deployment* path with the granularity the kernels actually support — with our smoothing improvement still applied.

## Configurations

Per model size, we run three configs:

| # | Label             | Model                                       | Source                       |
|---|-------------------|---------------------------------------------|------------------------------|
| 1 | FP16              | `facebook/opt-<size>` in fp16               | HuggingFace                  |
| 2 | INT8 paper (O3+max)| `mit-han-lab/opt-<size>-smoothquant`       | HuggingFace, prebuilt        |
| 3 | INT8 ours (O3+pct) | exported locally with `smooth_lm_pct` then static calibration | Path B below |

Config 2 is the demo from the SmoothQuant repo, used as a sanity check that real-INT8 works on our hardware. Config 3 is the contribution — same kernels, our smoothing.

## Method

For each config, measure:
- **Model size on GPU** (`param_size + buffer_size` in MB) — the headline memory metric.
- **WikiText-2 perplexity** at seq_len=2048 — same protocol as Task 01 for direct comparability.
- **Per-sample latency** on LAMBADA last-token (1000 samples, seq_len=512, padded) — same protocol as the upstream demo. This is a *latency proxy*, not a primary accuracy metric.
- **LAMBADA last-token accuracy** — sanity check that the int8 model didn't break.

## Two execution paths

### Path A — Pre-quantized HF model (sanity check)

Load `mit-han-lab/opt-<size>-smoothquant` directly via `Int8OPTForCausalLM.from_pretrained(...)`. Verifies the torch-int install and gives us the paper's O3+max numbers as a baseline. No calibration on our side.

### Path B — Export our percentile-smoothed INT8 model

1. Load fp16 model.
2. Apply our `smooth_lm_pct(model, pct_scales, alpha)` with the per-model winning (p, α) from Task 02.
3. Run `get_static_decoder_layer_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512)` on the Pile validation set to compute static per-tensor activation scales for each decoder layer site.
4. `int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)`; save to disk.
5. Reload and evaluate.

## Hardware

- **OPT-1.3B**: T4 (free Colab) — fits easily.
- **OPT-2.7B**: T4 should fit; A100 is faster.
- **OPT-6.7B and up**: A100 (per [CLAUDE.md](../../CLAUDE.md) challenge #2).

## Prerequisites

- `act_scales/opt-<size>.pt` on Drive (Task 01) — needed for max-smoothing fallback if testing.
- `act_percentiles/opt-<size>/p<p>.pt` on Drive (Task 02) — needed for Path B.
- Pile validation set `val.jsonl.zst` on Drive — needed for static calibration in Path B.
- A working **torch-int** install in the Colab session — see Cell 1 of `real_int8_cells.md`.

## Per-model winning (p, α) from Task 02

| Model     | p    | α   |
|-----------|------|-----|
| OPT-125M  | 0.999| 0.5 |
| OPT-1.3B  | 0.999| 0.9 |
| OPT-2.7B  | 0.995| 0.5 |
| OPT-6.7B  | TBD  | TBD |

Update Cell 2 of each model's cells file accordingly.
