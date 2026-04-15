# Task 01: Quantization Scheme Exploration — Progress

## Goal

Compare the paper's named schemes (O1, O2) against per-channel weight variants (C, D) that the paper uses in code but never formally names or compares. Show that per-channel weight quantization is both more accurate and more robust to alpha selection.

## Scheme definitions

Paper Table 2:
- **O1**: per-tensor weight, per-token activation (dynamic)
- **O2**: per-tensor weight, per-tensor activation (dynamic)
- **O3**: per-tensor weight, per-tensor activation (static) — not implementable with repo code

Our additional configs:
- **C (SQ-PCW-PT)**: per-channel weight, per-token activation — the repo's hardcoded default
- **D (SQ-PCW-TEN)**: per-channel weight, per-tensor activation

The repo's `ppl_eval.py` hardcodes config C. The paper never compares it against O1/O2.

## Experiment 1: OPT-1.3B scheme comparison (2026-04-14)

Cells: `experiments/task01_scheme_exploration/opt_1_3b/scheme_comparison_cells.md`
Results: `results/task01/opt_1_3b/scheme_comparison_results.ipynb`

All configs at alpha=0.5:

| Config | Weight | Activation | PPL | Time |
|--------|--------|-----------|-----|------|
| FP16 | — | — | 14.469 | 161s |
| SQ-O1 | per-tensor | per-token | 14.686 | 182s |
| SQ-PCW-PT (C) | per-channel | per-token | 14.702 | 179s |
| SQ-O2 | per-tensor | per-tensor | 14.827 | 180s |
| SQ-PCW-TEN (D) | per-channel | per-tensor | 14.835 | 179s |
| W8A8-naive | per-tensor | per-tensor | 15.649 | 182s |

Findings: C beats O2 by 0.126 PPL but loses to O1 by 0.015 PPL at alpha=0.5. This led to the alpha sweep.

## Experiment 2: OPT-1.3B alpha sweep (2026-04-15)

Cells: `experiments/task01_scheme_exploration/opt_1_3b/alpha_sweep_cells.md`
Results: `results/task01/opt_1_3b/alpha_sweep_results.ipynb`

O1 vs C at 5 discrete alpha levels (step 0.2: 2 below and 2 above the paper's 0.5). This is the **standard experiment template** to repeat for each model going forward.

| Alpha | O1 PPL | C PPL | Diff (O1-C) | C wins? |
|-------|--------|-------|-------------|---------|
| 0.1 | 16.018 | 15.646 | +0.372 | YES |
| 0.3 | 14.933 | 14.900 | +0.033 | YES |
| 0.5 | 14.686 | 14.702 | -0.015 | no |
| 0.7 | 14.778 | 14.642 | +0.135 | YES |
| 0.9 | 14.754 | 14.617 | +0.137 | YES |

Key findings:
- C's best (14.617 at alpha=0.9) beats O1's best (14.686 at alpha=0.5)
- C's PPL spread: 1.03 vs O1's 1.33 — C is more stable across alpha choices
- C wins at 4 of 5 levels; O1 only wins at alpha=0.5 (the paper's tuned value)
- At low alpha: C wins because per-channel handles harder weight quantization
- At high alpha: C wins because per-channel absorbs migrated difficulty better

## Thesis framing

Per-channel weight (config C) is both more accurate at its optimal alpha AND more robust to alpha selection than per-tensor weight (O1). The paper's alpha=0.5 recommendation is tuned for per-tensor weight; per-channel weight benefits from higher alpha values. This alpha-scheme interaction is a novel finding not explored in the paper.

Discrete alpha levels (0.1, 0.3, 0.5, 0.7, 0.9) chosen to test 2 below and 2 above the paper's recommended 0.5, saving compute while capturing the trend.

## Standard experiment template for scaling to other models

For each new model, run the alpha sweep cells with only MODEL and SCALES changed:
1. Generate activation scales if not already available
2. Run the alpha sweep (O1 vs C, alpha 0.1/0.3/0.5/0.7/0.9) — 10 runs
3. Record results in the same table format
4. Compare: does C beat O1? At which alphas? Is C more stable?

## Next steps

- OPT-6.7B: outliers more severe at this scale (paper Table 1 shows per-tensor activation drops to 39.9% accuracy). Need to generate scales first.
- Llama-2-7B: different architecture (RMSNorm), paper Table 7 gives config C PPL=5.515 for direct comparison. Need to generate scales first.
