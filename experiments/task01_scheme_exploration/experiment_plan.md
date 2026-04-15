# Task 01: Quantization Scheme Comparison — Experiment Plan

## Goal

Systematically compare the paper's named schemes (O1, O2) against per-channel weight variants (C, D) that the paper uses but never formally names or compares. Show that per-channel weight quantization consistently outperforms per-tensor weight after SmoothQuant smoothing, and quantify the gap across model scales.

## Configs to test

| Label | Short name | weight_quant | act_quant | quantize_bmm | smooth | Paper equivalent |
|-------|-----------|-------------|-----------|-------------|--------|-----------------|
| Baseline | FP16 | — | — | — | No | FP16 |
| Naive | W8A8-naive | per_tensor | per_tensor | Yes | No | W8A8 in Table 2 |
| A | SQ-O1 | per_tensor | per_token | Yes | Yes | SmoothQuant-O1 (Table 2) |
| B | SQ-O2 | per_tensor | per_tensor | Yes | Yes | SmoothQuant-O2 (Table 2) |
| C | SQ-PCW-PT | per_channel | per_token | Yes | Yes | Repo default / Table 7 config |
| D | SQ-PCW-TEN | per_channel | per_tensor | Yes | Yes | Not in paper |

Note: O3 (static activation quantization) is excluded — the repo code only supports dynamic quantization.

## Why per-channel weight is expected to outperform

Per-channel weight quantization (one scale per output channel) is finer-grained than per-tensor (one scale for entire matrix). It is hardware-friendly — scaling along outer dimension Co is compatible with INT8 GEMM (paper Figure 3, page 3). The paper's O1/O2/O3 all use per-tensor weight as part of an "efficiency level" hierarchy, but per-channel weight has negligible hardware overhead. The authors themselves switched to per-channel weight for Llama-2/Falcon/Mistral/Mixtral (Table 7) without discussing the tradeoff.

## Models

### Priority 1: OPT-1.3B
- Activation scales: already generated (on Drive)
- Fits T4: yes
- Why: fast iteration, validates methodology
- Paper comparison: no direct O1/O2 numbers in paper for this size, but Figure 7 shows O3 trend across OPT scales

### Priority 2: OPT-6.7B
- Activation scales: need to generate
- Fits T4: should fit (~13GB in FP16)
- Why: paper says outliers emerge at 6.7B+ (page 1). Table 1 (page 3) shows per-tensor activation drops to 39.9% accuracy at this scale. This is where scheme choice matters most.
- Paper comparison: Table 1 gives INT8 per-tensor (39.9%), per-token (42.5%), per-channel (64.8%) activation-only numbers. Table 4 gives O1/O2/O3 for 175B but not 6.7B.

### Priority 3: Llama-2-7B
- Activation scales: need to generate
- Fits T4: tight, may need careful memory management
- Why: different architecture (RMSNorm vs LayerNorm), Table 7 gives config C PPL = 5.515 with alpha=0.85 — direct comparison possible
- Paper comparison: only config C number available (5.515 PPL)

## Metrics

- WikiText-2 perplexity (primary, all runs)
- Zero-shot accuracy: LAMBADA, HellaSwag, PIQA, WinoGrande (if time/compute allows)

## How to run

All experiments use SmoothQuant's `ppl_eval.py` as the base, but with modified `quantize_model()` calls to vary weight_quant and act_quant parameters. The script needs to be called with different parameter combos since it currently hardcodes per_channel + per_token.

Options:
1. Modify `ppl_eval.py` to accept `--weight_quant` and `--act_quant` CLI args
2. Write a thin wrapper that patches the `quantize_model` call
3. Use the project's own `run_scheme_eval.py` (but it uses its own FakeQuantLinear, not the repo's W8A8Linear — different code path, less comparable)

Option 1 is cleanest for reproducibility.

## Expected results

- C (per-channel W + per-token A) should beat O1 (per-tensor W + per-token A)
- D (per-channel W + per-tensor A) should beat O2 (per-tensor W + per-tensor A)
- Gap should be larger on OPT-6.7B than OPT-1.3B (outlier effect)
- C should be the best overall config (matches what the authors implicitly chose)

## What we already have

From docs/task01_1.ipynb (2026-03-22), config C results on OPT-1.3B:
- FP16: 14.63 PPL
- W8A8 config C no smooth: 15.24 PPL
- W8A8 config C with smooth (alpha=0.5): 14.94 PPL

Still needed: O1, O2, D runs on the same model for comparison.

## Thesis framing

The contribution is:
1. First systematic side-by-side comparison of per-tensor vs per-channel weight quantization under SmoothQuant
2. Explains why the official implementation defaults to a config not defined in the paper
3. Quantifies the accuracy gap across model scales
4. Provides guidance on which scheme to use in practice
