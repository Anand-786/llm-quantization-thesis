# Task 01: Quantization Scheme Exploration — Progress

## Goal

Compare the paper's named schemes (O1, O2) against per-channel weight variants (C, D) that the paper uses in code but never formally names or compares. Show that per-channel weight quantization is both more accurate and more robust to alpha selection across a scaling ladder of OPT models.

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

## Experiment 3: OPT-125M scheme comparison (2026-04-21)

Cells: `experiments/task01_scheme_exploration/opt_125m/scheme_comparison_cells.md`
Results: `results/task01/opt_125m/opt_125m_task_1.ipynb`

All configs at alpha=0.5:

| Config | Weight | Activation | PPL | Δ vs FP16 |
|--------|--------|-----------|-----|-----------|
| FP16 | — | — | 27.570 | — |
| SQ-PCW-PT (C) | per-channel | per-token | 27.601 | +0.03 (+0.1%) |
| SQ-PCW-TEN (D) | per-channel | per-tensor | 28.267 | +0.70 |
| SQ-O1 | per-tensor | per-token | 28.298 | +0.73 |
| SQ-O2 | per-tensor | per-tensor | 29.164 | +1.59 |
| W8A8-naive | per-tensor | per-tensor | 30.163 | +2.59 |

Findings: at alpha=0.5, C is effectively lossless (0.1% over FP16) and beats O1 by 0.70 PPL, D by 0.67 PPL, O2 by 1.56 PPL. Both granularity axes (W per-channel vs per-tensor; A per-token vs per-tensor) independently contribute ~0.7–0.9 PPL each, and they stack — C vs O2 closes essentially all the quantization gap.

## Experiment 4: OPT-125M alpha sweep (2026-04-21)

Cells: `experiments/task01_scheme_exploration/opt_125m/alpha_sweep_cells.md`
Results: `results/task01/opt_125m/opt_125m_task1_2.ipynb`

Because 125M is small and each run took ~21s on T4, we swept **9 alpha values at step 0.1** (0.1 through 0.9) instead of the standard 5-level sweep. 18 runs total.

| Alpha | O1 PPL | C PPL | Diff (O1-C) | C wins? |
|-------|--------|-------|-------------|---------|
| 0.1 | 28.303 | 27.861 | +0.442 | YES |
| 0.2 | 28.327 | 27.749 | +0.578 | YES |
| 0.3 | 28.167 | 27.715 | +0.452 | YES |
| 0.4 | 28.144 | 27.746 | +0.397 | YES |
| 0.5 | 28.298 | 27.600 | +0.697 | YES |
| 0.6 | 28.190 | 27.631 | +0.559 | YES |
| 0.7 | 28.573 | 27.681 | +0.892 | YES |
| 0.8 | 29.708 | 27.679 | +2.028 | YES |
| 0.9 | 30.654 | 27.662 | +2.992 | YES |

Key findings:
- **C wins at all 9/9 alphas** — stronger than 1.3B (4/5).
- **C is nearly alpha-invariant**: range 27.60 – 27.86, spread of only 0.26 PPL. Best alpha=0.5, but any alpha in [0.1, 0.9] stays within 0.3 PPL of FP16.
- **O1 has a high-alpha cliff**: flat-ish in [0.1, 0.7] (28.14 – 28.57), then collapses to 29.71 at alpha=0.8 and 30.65 at alpha=0.9 — worse than naive W8A8 (30.16) at alpha=0.9.
- **O1's optimum is alpha=0.4 (28.14)**, not the paper's 0.5, but the difference is small.
- **O1 spread is ~2.5 PPL vs C's 0.26 PPL** — C is ~10× more stable across alpha.
- **Mechanism exposed**: the O1↔C gap widens from ~0.4 PPL at low alpha to 3.0 PPL at alpha=0.9. This isolates per-tensor W quantization as the failure point — as alpha pushes difficulty onto the weights, C handles it gracefully while O1 falls off a cliff.

## Thesis framing

Per-channel weight (config C) is both more accurate at its optimal alpha AND more robust to alpha selection than per-tensor weight (O1). The paper's alpha=0.5 recommendation is tuned for per-tensor weight; per-channel weight benefits from higher alpha values. This alpha-scheme interaction is a novel finding not explored in the paper.

125M strengthens the robustness claim: C's variance across alpha is ~10× smaller than O1's, and C wins uniformly (9/9 vs 1.3B's 4/5). But 125M is the most permissive setting — activation outliers are an emergent property of scale (Dettmers et al., LLM.int8()), so 125M has less headroom for any scheme to fail. The robustness claim needs 6.7B to confirm it at outlier-heavy scale.

Discrete alpha levels (0.1, 0.3, 0.5, 0.7, 0.9) are the standard template; 125M used the finer step=0.1 version because each run is only ~21s.

## Standard experiment template for scaling to other models

For each new model, run the alpha sweep cells with only MODEL and SCALES changed:
1. Generate activation scales if not already available
2. Run the alpha sweep (O1 vs C, alpha 0.1/0.3/0.5/0.7/0.9) — 10 runs
3. Record results in the same table format
4. Compare: does C beat O1? At which alphas? Is C more stable?

## Next steps

- **OPT-2.7B**: fills the 1.3B → 6.7B gap in the OPT ladder. Standard architecture, fits on T4 in bf16 (~5.4 GB). Need to generate scales first.
- **OPT-6.7B**: outliers more severe at this scale (paper Table 1 shows per-tensor activation drops to 39.9% accuracy). Scales already generated. Needs A100 at seq_len=2048.
- OPT-350M skipped: non-standard architecture (post-LN with project_in/project_out) — smooth_lm likely requires patching, not worth the engineering cost.
- Llama-2-7B (later task): different architecture (RMSNorm), paper Table 7 gives config C PPL=5.515 for direct comparison. Need to generate scales first.

Status summary: 125M ✅ (9/9 C wins) · 1.3B ✅ (4/5 C wins) · 2.7B ⏳ · 6.7B ⏳
