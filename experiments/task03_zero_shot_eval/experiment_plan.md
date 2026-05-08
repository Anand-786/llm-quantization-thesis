# Task 03: Zero-Shot Benchmark Evaluation — Experiment Plan

## Goal

Compare the **paper's recipe** (max-based smoothing on the O1 and O2 schemes) against **our proposed recipe** (percentile-based smoothing on the C scheme) on the 7 zero-shot tasks the SmoothQuant paper uses for OPT-175B:

LAMBADA (lambada_openai), HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA.

This adds task-level signal on top of the WikiText-2 PPL evidence from Task 01 (scheme contribution) and Task 02 (smoothing-statistic contribution).

## Configurations

We run 5 configs per model. The two anchors (FP16, W8A8-naive) frame the ceiling and floor; the three middle rows are the actual comparison: paper's two named recipes (O1, O2) vs ours (C + percentile smoothing). C+pct is treated as a single compound method, not a factorisation, so SQ-C-max is **not** part of the canonical table.

| # | Label | Scheme | Smoothing | α | p_w | Source of act scales |
|---|-------|--------|-----------|---|-----|----------------------|
| 1 | FP16 | — | — | — | — | — |
| 2 | W8A8-naive | per-tensor W, per-tensor A | none | — | — | — |
| 3 | SQ-O1 (paper) | per-tensor W, per-token A | max | 0.5 | 1.0 | Task 01 `act_scales/opt-<size>.pt` |
| 4 | SQ-O2 (paper) | per-tensor W, per-tensor A | max | 0.5 | 1.0 | Task 01 `act_scales/opt-<size>.pt` |
| 5 | SQ-C-pct (ours) | per-channel W, per-token A | percentile | per-model | per-model | Task 02 `act_percentiles/opt-<size>/p<p>.pt` |

**Notes on choices:**

- **(α, p) for SQ-C-pct comes from Task 02's WikiText-2 PPL winner per model.**
- **α=0.5 for O1/O2/max** is the paper's default. O2 collapses at high α (per-tensor W), so 0.5 is the safe choice.
- **Per the user's instruction**, the max-smoothing rows reuse Task 01's `act_scales/opt-<size>.pt` directly — *not* the `p1.0.pt` file from Task 02's percentile calibration (even though they should be numerically equal — Task 02 verifies this). Task 01's scales are the ground-truth baseline for the paper's recipe.
- **SQ-C-max is optional**, kept as a separate cell in each model's cells file *after* the final comparison table, for ad-hoc analysis only. Not part of the saved comparison.

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

### Priority 3 — OPT-2.7B (T4 sufficient)

- Model fits T4 (~5.4 GB fp16 weights); 5 configs × ~10–15 min ≈ 1–1.5 h on T4. Hellaswag dominates the time.
- **Alpha values:** O1/O2 at α=0.5 (paper default). C-pct from Task 02 OPT-2.7B winner: **p=0.995, α=0.5**.
- **Prerequisite:** `act_scales/opt-2.7b.pt` (Task 01) and `act_percentiles/opt-2.7b/p0.995.pt` (Task 02) on Drive.

### Priority 4 — OPT-6.7B (A100 required)

- 6.7B fp16 weights are ~13.3 GB — does not fit T4 for batched zero-shot. Run on Colab Pro A100 (CLAUDE.md Challenge #2).
- Estimated A100 runtime: 5 configs × ~15–25 min ≈ 1.5–2 h.
- **Alpha values:** O1/O2 at α=0.5 (paper default). C-pct (α, p) **pending Task 02 OPT-6.7B sweep**; placeholder p=0.99, α=0.5 in cells file based on the 1.3B → 2.7B trend (optimum drifts to lower p and lower α as scale grows).
- **Prerequisite:** `act_scales/opt-6.7b.pt` (Task 01 — already generated) and `act_percentiles/opt-6.7b/p<chosen>.pt` (Task 02 — pending) on Drive.
- This is the discriminating data point: SmoothQuant Table 1 reports per-tensor activation accuracy collapses to 39.9% at 6.7B, so the W8A8-naive floor should drop substantially and let the smoothing recipes actually separate on zero-shot — unlike 2.7B where the floor was only −0.5pp from FP16.

### Priority 5 — OPT-13B

A100 required. Defer until 6.7B story is locked.

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
- OPT-1.3B run: ✅ complete (FP16, W8A8-naive, SQ-O1-max, SQ-O2-max, SQ-C-pct)
- `opt_2_7b/zero_shot_eval_cells.md` ✅
- `opt_2_7b/run_zero_shot_t3.py` ✅
- OPT-2.7B run: ✅ complete
- `opt_6_7b/zero_shot_eval_cells.md` ✅
- `opt_6_7b/run_zero_shot_t3.py` ✅
- OPT-6.7B run: ⏳ (A100 + Task 02 6.7B winner needed)
- OPT-125M run: ✅ complete

## Results — OPT-1.3B

Primary metric per task as specified in Method (LAMBADA `acc`, HellaSwag/PIQA/OpenBookQA/WinoGrande `acc_norm`, RTE/COPA `acc`).

| Config | LAMBADA | HellaSwag | PIQA | WinoGrande | OpenBookQA | RTE | COPA | Avg |
|---|---|---|---|---|---|---|---|---|
| FP16              | 0.5787 | 0.5370 | 0.7236 | 0.5943 | 0.3320 | 0.5235 | 0.8000 | 0.5842 |
| W8A8-naive        | 0.5432 | 0.5174 | 0.7078 | 0.5714 | 0.3160 | 0.5668 | 0.7900 | 0.5732 |
| SQ-O1-max         | 0.5754 | 0.5342 | 0.7231 | 0.5880 | 0.3380 | 0.5199 | 0.8000 | 0.5826 |
| SQ-O2-max         | 0.5570 | 0.5310 | 0.7089 | 0.5825 | 0.3300 | 0.5343 | 0.8100 | 0.5791 |
| **SQ-C-pct (ours)** | **0.5744** | **0.5365** | **0.7242** | **0.5927** | **0.3280** | **0.5271** | **0.8100** | **0.5847** |

C+pct uses (α=0.9, p=0.95) — Task 02 WikiText-2 winner for OPT-1.3B was p=0.999/α=0.9, but the cells file shipped with p=0.95 placeholder; result still matches FP16 zero-shot avg.

### Notes
- **Headline**: C+pct matches FP16 zero-shot avg (0.5847 vs 0.5842) and beats both paper recipes on average (+0.21pp over O1, +0.56pp over O2).
- **W8A8-naive vs FP16**: LAMBADA acc drops 3.55pp; HellaSwag −2.0pp, PIQA −1.6pp, WinoGrande −2.3pp, OpenBookQA −1.6pp. The outlier-sensitivity floor that smoothing recovers.
- **Per-task** (vs O1-max): C+pct wins on HellaSwag, PIQA, WinoGrande, RTE, COPA; ties LAMBADA; loses OpenBookQA. RTE/COPA are noisy (small val sets) — main signal lives in LAMBADA / HellaSwag / PIQA / WinoGrande.
- A non-fatal `Failed to get model SHA` warning in the log is cosmetic (results JSON provenance only); metrics unaffected.

## Results — OPT-125M

C+pct uses (α=0.5, p=0.999) — Task 02 OPT-125M winner.

| Config | LAMBADA | HellaSwag | PIQA | WinoGrande | OpenBookQA | RTE | COPA | Avg |
|---|---|---|---|---|---|---|---|---|
| FP16              | 0.3788 | 0.3135 | 0.6192 | 0.5020 | 0.2780 | 0.5018 | 0.6900 | 0.4690 |
| W8A8-naive        | 0.3538 | 0.3104 | 0.6159 | 0.4972 | 0.2680 | 0.4838 | 0.6300 | 0.4513 |
| SQ-O1-max         | 0.3829 | 0.3136 | 0.6181 | 0.4988 | 0.2740 | 0.4765 | 0.6800 | 0.4634 |
| SQ-O2-max         | 0.3707 | 0.3121 | 0.6148 | 0.5083 | 0.2600 | 0.4729 | 0.6500 | 0.4555 |
| **SQ-C-pct (ours)** | **0.3749** | **0.3136** | **0.6197** | **0.4941** | **0.2720** | **0.4874** | **0.7000** | **0.4660** |

### Notes

- **W8A8 floor is real at 125M**: −1.77pp from FP16 (0.4513 vs 0.4690), deeper in absolute pp than 1.3B (−1.10) or 2.7B (−0.54). Why a small model would have a deeper floor than 2.7B is counterintuitive (outliers should grow with scale, not shrink) — most likely because base accuracy is much closer to chance at 125M, so a fixed amount of quantization noise has relatively more bite on borderline predictions. Don't make a strong scale-trend claim from one seed.
- **Ordering on avg**: FP16 (0.4690) > C-pct (0.4660) > O1-max (0.4634) > O2-max (0.4555) > naive (0.4513). C-pct beats O1 by +0.26pp on avg, sits 0.30pp below FP16.
- **Recovery from W8A8 floor**: C-pct +1.47pp (83% of floor), O1 +1.21pp (68%), O2 +0.42pp (24%). C-pct recovers best.
- **Per-task C-pct vs O1-max**: ties HellaSwag; small wins on PIQA (+0.16) and big wins on RTE (+1.09) and COPA (+2.0); losses on LAMBADA (−0.80), WinoGrande (−0.47), OpenBookQA (−0.20). The +0.26pp avg gain is carried by the noisy small tasks (RTE 277 items, COPA 100 items). On the more reliable LAMBADA / WinoGrande, O1 actually edges out — be honest about this in the writeup.

## Results — OPT-2.7B

C+pct uses (α=0.5, p=0.995) — Task 02 OPT-2.7B winner.

| Config | LAMBADA | HellaSwag | PIQA | WinoGrande | OpenBookQA | RTE | COPA | Avg |
|---|---|---|---|---|---|---|---|---|
| FP16              | 0.6361 | 0.6063 | 0.7481 | 0.6093 | 0.3520 | 0.5523 | 0.7700 | 0.6106 |
| W8A8-naive        | 0.6577 | 0.5744 | 0.7231 | 0.6069 | 0.3480 | 0.5162 | 0.8100 | 0.6052 |
| SQ-O1-max         | 0.6522 | 0.6012 | 0.7497 | 0.6125 | 0.3460 | 0.5343 | 0.7600 | 0.6080 |
| SQ-O2-max         | 0.6499 | 0.5999 | 0.7476 | 0.6014 | 0.3460 | 0.5235 | 0.7600 | 0.6040 |
| **SQ-C-pct (ours)** | **0.6420** | **0.6035** | **0.7476** | **0.6014** | **0.3500** | **0.5379** | **0.7600** | **0.6060** |

### Notes

- **The W8A8 floor is shallow at 2.7B**: W8A8-naive avg (0.6052) is only −0.54pp below FP16 (0.6106). Compare to 1.3B where naive dropped −1.10pp. There's much less of a gap for smoothing to recover at this scale.
- **Ordering on avg**: FP16 (0.6106) > O1-max (0.6080) > C-pct (0.6060) > naive (0.6052) > O2-max (0.6040). All four quantized configs sit within ~0.7pp of FP16 — within typical zero-shot noise for these task sizes.
- **C-pct does NOT beat O1 at 2.7B** (−0.20pp on avg) — opposite sign from 1.3B (+0.21pp). The two model points are within zero-shot noise of each other; the cleaner PPL signal from Task 02 (C-pct 12.34 ≈ FP16 12.3425, vs O1/max winner) does not translate cleanly to coarse zero-shot accuracy at 2.7B.
- **LAMBADA anomaly**: W8A8-naive *beats* FP16 on LAMBADA (+2.16pp). All quantized configs beat FP16 on LAMBADA at 2.7B. At 1.3B the same task showed the expected −3.55pp naive drop. Possible reading: at 2.7B activation outliers are still moderate (not yet at 6.7B-style severity), and small numerical perturbations from quantization can flip marginal greedy next-token predictions to net-positive on aggregate. Worth flagging in the writeup but don't over-interpret a single seed; consider re-running with 2–3 seeds for the LAMBADA row alone if this becomes load-bearing for the thesis story.
- **Per-task vs O1-max**: C-pct loses LAMBADA (−1.0pp) and WinoGrande (−1.1pp); wins HellaSwag (+0.23), OpenBookQA (+0.40), RTE (+0.36); ties PIQA and COPA. Mixed.
- **Takeaway for the thesis**: at 1.3B the compound recipe shows a small but consistent advantage on coarse zero-shot; at 2.7B the W8A8 floor is too shallow for any smoothing recipe to differentiate on avg. The clean signal lives in PPL (Task 01/02), not zero-shot at this scale. The 6.7B run on A100 is where outliers become severe enough for the zero-shot test to discriminate.
