# Task 03: Zero-Shot Benchmark Evaluation — Experiment Plan

## Goal

Compare the **paper's recipe** (max-based smoothing on the O1 and O2 schemes) against **our proposed recipe** (percentile-based smoothing on the C scheme) on the 7 zero-shot tasks the SmoothQuant paper uses for OPT-175B:

LAMBADA (lambada_openai), HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA.

This adds task-level signal on top of the WikiText-2 PPL evidence from Task 01 (scheme contribution) and Task 02 (smoothing-statistic contribution).

## Configurations

We run 6 configs per model. The two anchors (FP16, W8A8-naive) frame the ceiling and floor; the four middle rows are the actual comparison.

| # | Label | Scheme | Smoothing | α | p_w | Source of act scales |
|---|-------|--------|-----------|---|-----|----------------------|
| 1 | FP16 | — | — | — | — | — |
| 2 | W8A8-naive | per-tensor W, per-tensor A | none | — | — | — |
| 3 | SQ-O1 (paper) | per-tensor W, per-token A | max | 0.5 | 1.0 | Task 01 `act_scales/opt-1.3b.pt` |
| 4 | SQ-O2 (paper) | per-tensor W, per-tensor A | max | 0.5 | 1.0 | Task 01 `act_scales/opt-1.3b.pt` |
| 5 | SQ-C-pct (ours) | per-channel W, per-token A | percentile | 0.9 | 0.95 | Task 02 `act_percentiles/opt-1.3b/p0.95.pt` |
| 6 | SQ-C-max (reference) | per-channel W, per-token A | max | 0.9 | 1.0 | Task 01 `act_scales/opt-1.3b.pt` |

**Notes on choices:**

- **α and p for SQ-C-pct are placeholders.** Task 02's percentile sweep on OPT-1.3B will pick the winning (α, p) for the C scheme on WikiText-2 PPL; whatever wins there gets locked in here. The cells file makes the (α, p) selection a single variable so updating after Task 02 is trivial.
- **α=0.5 for O1/O2/max** is the paper's default and matches Task 01's confirmed PPL optimum on 1.3B for O1 (14.686 at α=0.5). O2 collapses at high α (per-tensor W), so 0.5 is the safe choice.
- **SQ-C-max is included** as an internal reference: it isolates "what does percentile-smoothing add *on top of* the C scheme" by holding the scheme fixed and only flipping the smoothing statistic. Without it, any C-pct gain over O1/O2 would conflate two independent contributions (Task 01's scheme switch + Task 02's smoothing switch). With it, we get a clean factorisation.
- **Per the user's instruction**, the max-smoothing rows reuse Task 01's `act_scales/opt-1.3b.pt` directly — *not* the `p1.0.pt` file from Task 02's percentile calibration (even though they should be numerically equal — Task 02 verifies this). Task 01's scales are the ground-truth baseline for the paper's recipe.

## Method

- **Tool:** `lm-evaluation-harness` v0.4.4, zero-shot (`num_fewshot=0`). This is the standard the SmoothQuant authors use and a published version pin keeps the numbers reproducible.
- **Tasks:** the seven listed above. Default lm-eval task configs (no custom prompt templates). Primary metric per task as reported by the harness:
    - LAMBADA → `acc`
    - HellaSwag, PIQA, OpenBookQA, WinoGrande → `acc_norm`
    - RTE, COPA → `acc`
- **Eval suite is fixed across all 6 configs;** only the model state differs.
- **Smoothing implementation:**
    - max-smoothing path → `smoothquant.smooth.smooth_lm` (upstream).
    - percentile-smoothing path → `experiments.task02_percentile_smoothing.percentile_smooth.smooth_lm_pct`.
- **Quantization implementation:** `smoothquant.fake_quant.quantize_model` for all four quantized rows. We rely on its `weight_quant`/`act_quant` flags + `quantize_bmm_input=True` to pick the per-channel/per-tensor and per-token/per-tensor axes — same as Task 01.
- **Reproducibility:** seed lm-eval's RNGs; pin `lm-eval==0.4.4`, `transformers`, `accelerate`. Record exact versions in the saved JSON output of each run.
- **Outputs:** one JSON per config under `results/task03/opt-1.3b_zeroshot_<label>.json`, plus a final results notebook (`results/task03/opt_1_3b/zero_shot_results.ipynb`) that loads them and prints the comparison table.

## Models

### Priority 1 — OPT-1.3B (T4 sufficient; A100 nice-to-have)

- Model fits T4 comfortably; 6 configs × ~30–45 min ≈ 3–4 hours on T4, ~1 hour on A100.
- **Alpha values (from Task 01 PPL sweep on 1.3B):** O1 / O2 / C-max all at α=0.5 (paper default; Task 01 1.3B used step=0.2 sweep, O1 best at 0.5). C-pct placeholder α=0.9, p=0.95.
- **Prerequisite:** Task 02 has produced `act_percentiles/opt-1.3b/p<chosen>.pt` on Drive. Until then, leave the percentile p as a placeholder (we use `p0.95.pt`).
- **Prerequisite:** Task 01's `act_scales/opt-1.3b.pt` already on Drive (it is).

### Priority 2 — OPT-125M (T4)

- Tiny model; 6 configs × ~5–10 min ≈ 30–60 min on T4. Sanity / spectrum-completion data point.
- **Alpha values (from Task 01 step=0.1 sweep on 125M):** O1 α=0.5 (paper default; near-best at 28.298), O2 α=0.5 (per-tensor W safe choice), C-max α=0.5 (PPL min 27.600). C-pct placeholder α=0.5, p=0.95.
- **Prerequisite:** `act_scales/opt-125m.pt` on Drive (it is) and `act_percentiles/opt-125m/p<chosen>.pt` (pending Task 02 125M run).
- **Caveat:** 125M is the most permissive scale — outliers are weakest, so the absolute accuracy gap between schemes will be smaller than at 1.3B / 6.7B. Don't read a small Δ here as evidence the method doesn't scale.

### Priority 3 — OPT-2.7B, OPT-6.7B, OPT-13B

Run after the 1.3B story is locked. ≥6.7B requires A100 (CLAUDE.md challenge #2). Cells per model land alongside each model's Task 02 percentile sweep output.

## What we expect

If the Task 01 + Task 02 stories carry over to task accuracy:

- **W8A8-naive** drops noticeably on LAMBADA and HellaSwag (the most outlier-sensitive tasks). RTE/COPA are small and noisy.
- **SQ-O1** recovers most of the W8A8-naive drop. **SQ-O2** less, because per-tensor activations clip residual outliers.
- **SQ-C-max** matches or slightly exceeds SQ-O1 — Task 01's PPL story.
- **SQ-C-pct** is the headline: should match or beat SQ-C-max if percentile-smoothing makes the smoothing factor more outlier-stable. Task 02 PPL evidence will tell us by how much before we run this.

If SQ-C-pct beats both SQ-O1 (paper recipe) and SQ-C-max (Task 01 recipe), the thesis comparison story is clean: scheme contribution + smoothing-statistic contribution stack to a configuration the paper does not consider.

## Risks / open questions

- **lm-eval task definitions drift across versions.** Pinning v0.4.4 keeps us aligned with Task 01's existing zero-shot run.
- **COPA has 100 items;** ±2–3% deltas are within noise. Don't read too much into it. LAMBADA + HellaSwag are the reliable signals.
- **HellaSwag is the slowest task** (~10k val items). If T4 OOMs, drop `--batch_size` from 8 → 4 → 2.
- **(α, p) for SQ-C-pct is a placeholder.** Final values come from Task 02's WikiText-2 PPL winner for OPT-1.3B. If Task 02 picks a winner that's not in the placeholder set, update the cells file's two variables and re-run.

## Status

- experiment_plan.md ✅ (this file)
- `opt_1_3b/zero_shot_eval_cells.md` ✅
- `opt_1_3b/run_zero_shot_t3.py` ✅ (supports both max and percentile smoothing via flag, with post-quant assert)
- `opt_125m/zero_shot_eval_cells.md` ✅
- `opt_125m/run_zero_shot_t3.py` ✅
- OPT-1.3B run: ⏳ (awaiting Task 02 1.3B winner)
- OPT-125M run: ⏳ (awaiting Task 02 125M winner)
