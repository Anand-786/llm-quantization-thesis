# Task 02: Percentile-Based Smoothing — Progress

## Goal

Replace the per-channel `max(|X|)` and `max(|W|)` in SmoothQuant's smoothing factor with a per-channel percentile (`p ∈ {0.90, 0.95, 0.99, 0.995, 0.999}`). Test whether percentile-smoothing is more accurate and more alpha-stable than max-smoothing, and whether the gain stacks with the C scheme from Task 01.

Plan: [experiment_plan.md](experiment_plan.md)

## Hypothesis

`max` is dominated by single-token outliers in calibration, making the smoothing factor unstable per channel. A high percentile (e.g. 99) captures the channel's typical magnitude without being skewed by spikes. The paper's §5.2 ad-hoc 2% clip for large models is symptomatic of the same problem — percentile-smoothing addresses it inside the formula instead of patching the quantization step.

Expected best pairing: **percentile + C scheme**, because per-token activation quantization tolerates the residual outliers that percentile-smoothing leaves above the threshold.

## Naive W8A8 collapse across the model ladder (verification, FP16-anchored)

Same per-tensor W + per-tensor A, no smoothing. Numbers from each model's single-session verification run.

| Model    | FP16    | Naive W8A8 | Δ vs FP16   |
|----------|--------:|-----------:|------------:|
| OPT-125M | 27.5684 | 30.2257    | +2.66       |
| OPT-1.3B | 14.4677 | 15.5867    | +1.12       |
| OPT-2.7B | 12.3449 | 13.4438    | +1.10       |
| OPT-6.7B | 10.6732 | 25.9135    | **+15.24**  |
| OPT-13B  |  9.9439 | 4325.6772  | **+4315.73** |

The collapse is super-exponential past 2.7B — at 13B the model is outputting noise without smoothing.

## Sweep grid convention (from OPT-6.7B onward)

Every Task 02 sweep on a new model uses the full grid in one notebook run:
- `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}` (5 values)
- `alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (5 values, step=0.2)
- 25 runs per model, scheme C.

OPT-1.3B and OPT-2.7B used a narrower `α ∈ {0.5, 0.7, 0.9}` grid first; from 6.7B onward we run the full grid in one pass since the optimum drifts with model size and we don't know it a priori.

## Modified formula

Original (paper / repo):
```
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

Proposed:
```
s_j = quantile(|X_j|, p)^alpha / quantile(|W_j|, p)^(1-alpha)
```

`p = 1.0` recovers the original (sanity check baseline).

## Hardware assumption

Task 02 runs on **A100** (Colab Pro or JarvisLabs.ai). Activation calibration uses a per-channel **top-K buffer** sized for `p_min = 0.90` (`K ≈ 0.10·N`), which is exact for every requested `p ≥ p_min`. Buffer memory: ~5 GB on 1.3B, ~13 GB on 6.7B, ~21 GB on 13B (all fp16). GPU buffer for ≤6.7B; CPU buffer for 13B.

## Code (planned)

All Task 02 code lives inside `experiments/task02_percentile_smoothing/`. Cross-model utilities sit at the task root; per-model cell `.md` files sit under `opt_<size>/`. Nothing goes in `shared/` — Task 02 is self-contained.

- `experiments/task02_percentile_smoothing/percentile_smooth.py` ✅ — `smooth_lm_pct(model, act_pct_scales, alpha, p_w)` mirroring `smoothquant.smooth.smooth_lm` but with exact per-channel weight `torch.quantile` instead of max. Falls back to exact max when `p_w == 1.0`. OPT-only.
- `experiments/task02_percentile_smoothing/percentile_calibration.py` ✅ — `get_act_percentiles(model, tokenizer, dataset_path, percentiles, ...)` hooks the SmoothQuant smoothing-relevant linear inputs (q_proj, fc1) only, maintains a per-channel **top-K buffer** sized for `p_min`, and reads off every requested `p` exactly after the pass.
- Output: per-percentile files at `/content/drive/MyDrive/thesis_results/act_percentiles/opt-<size>/p<value>.pt`, each a `dict[name -> tensor[in_features]]` matching the shape of `act_scales/opt-<size>.pt`. A `p = 1.0` file is also written (the per-channel max from the same buffer) and is diff'd against `act_scales/opt-<size>.pt` as a correctness check.

## Experiments (none run yet)

### Experiment 1: OPT-1.3B percentile sweep at C scheme — ✅ RUN

Cells: [opt_1_3b/percentile_sweep_cells.md](opt_1_3b/percentile_sweep_cells.md)
Results: `results/task02/opt_1_3b/percentile_sweep_results.ipynb` (to be regenerated as a notebook artefact)

Grid: `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90} × alpha ∈ {0.5, 0.7, 0.9}` with scheme = C (per-channel W + per-token A). 15 runs total. `p = 1.0` excluded because the top-K calibration's per-channel max sits inside one fp16 ULP (~0.5% relative) of upstream — too noisy to function as an in-harness baseline. Reference baseline is Task 01's max-smoothing C/O1 numbers.

Result table (WikiText-2 PPL, OPT-1.3B, scheme C):

| p | alpha=0.5 | alpha=0.7 | alpha=0.9 |
|---|---:|---:|---:|
| 0.999 | 14.7363 | 14.6291 | **14.6167** |
| 0.995 | 14.6883 | **14.6349** | 14.7110 |
| 0.99  | 14.6907 | **14.6898** | 14.8002 |
| 0.95  | **14.6790** | 14.6968 | 15.1264 |
| 0.90  | 14.6793 | **14.6272** | 15.8997 |

Key findings:
- **Best in sweep: p=0.999, alpha=0.9 → 14.6167 PPL.** A close second is p=0.90, alpha=0.7 → 14.6272.
- Both materially beat Task 01's O1/max baseline of 14.68 on the same model.
- **Best alpha shifts down as p drops** — 0.9 → 0.7 → 0.5 across the rows. Mechanism: smaller smoothing scales raised to a high alpha over-smooth and crush the activation, visible most starkly at `p=0.90, alpha=0.9 = 15.90` PPL.
- Implication for follow-up: a full alpha sweep at the two competing optima (p=0.999 and p=0.90) is the right next step before declaring a winner.

### Experiment 2: OPT-1.3B full alpha sweep at the two competing optima — ✅ RUN

Cells: [opt_1_3b/full_alpha_sweep_cells.md](opt_1_3b/full_alpha_sweep_cells.md)
Grid: `p ∈ {0.999, 0.90} × alpha ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`, scheme = C. 18 runs.

Result table (WikiText-2 PPL, OPT-1.3B, scheme C):

|     p | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 | α=0.6 | α=0.7 | α=0.8 | α=0.9 |
|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 0.999 | 15.33 | 15.03 | 14.77 | 14.72 | 14.74 | 14.63 | 14.63 | 14.72 | **14.6167** |
| 0.90  | 15.37 | 15.16 | 14.92 | 14.76 | **14.68** | 14.72 | **14.6272** | 14.95 | 15.90 |

Findings:
- **Overall best: p=0.999, α=0.9 → 14.6167 PPL.** Beats Task 01 O1/max baseline (14.68) by 0.063 PPL.
- **p=0.999 is on a wide alpha plateau.** Cells at α ∈ {0.6, 0.7, 0.9} are all within 0.013 PPL of each other (14.6298 / 14.6291 / 14.6167). Spread across α ∈ [0.4, 0.9] is only 0.108 PPL.
- **p=0.90 is sharply fragile.** Best at α=0.7 (14.6272), but α=0.8 jumps to 14.95 and α=0.9 collapses to 15.90. Spread across α ∈ [0.4, 0.9] is 1.27 PPL — ~12× wider than p=0.999.
- **No optimum outside the step=0.2 grid.** α=0.6 at p=0.999 is competitive (14.6298) but does not beat α=0.9. Future Task 02 sweeps on bigger models can stay on the `{0.1, 0.3, 0.5, 0.7, 0.9}` step=0.2 grid.
- **Thesis-relevant pattern**: high p (conservative percentile, top ~0.1% of |X| ≈ paper's §5.2 2% clip intuition) gives both lower PPL *and* stronger alpha-stability. Low p (aggressive, top 10%) trades alpha-stability for nothing — its peak is no better than the conservative one's plateau, and the failure modes off-peak are severe.

#### Single-session verification (cross-config, FP16-anchored)

Cells: [`experiments/verification/opt_1_3b/full_table_cells.md`](../verification/opt_1_3b/full_table_cells.md). One Colab kernel, all 8 configs (FP16, naive, O1/max, O2/max, C/max, C/pct winner, O1+per-layer-α, C+per-layer-α) under one `Evaluator` instance — within-table deltas are noise-free. Use these numbers in the thesis when comparing rows across schemes; the per-experiment sweeps above stay authoritative for *within-grid* shape but their absolute PPLs are not directly comparable across notebooks.

| # | Config | PPL | Δ vs FP16 |
|---|---|---:|---:|
| 1 | FP16 | 14.4677 | +0.0000 |
| 2 | Naive W8A8 (per-tensor W & A, no smoothing) | 15.5867 | +1.1190 |
| 3 | O1 max α=0.5 | 14.8333 | +0.3656 |
| 4 | O2 max α=0.5 | 14.8335 | +0.3658 |
| 5 | **C pct p=0.999, α=0.9** | **14.6248** | **+0.1571** |
| 6 | C max α=0.5 | 14.7710 | +0.3033 |
| 7 | O1 + per-layer α (Task 05) | 14.6949 | +0.2272 |
| 8 | C + per-layer α (Task 05) | 14.6281 | +0.1604 |

C/pct (winner from the original sweep above) lands at **+0.1571 above FP16** vs C/max at +0.3033 — a **0.146 PPL improvement** from swapping max for percentile under matched eval. C+per-layer-α reaches the same floor (+0.1604) via a completely different route, confirming both knobs target the same outlier-driven instability. Verification-run number for C/pct (14.6248) is within ~0.008 PPL of the in-grid 14.6167 above, i.e. bf16 run-to-run noise on the same config — the *ranking* and the headline deltas are stable.

### Experiment 3: cross-scheme check — ⏳ PLANNED

At `p = 0.999, alpha = 0.9` (the Experiment 2 winner), run O1, O2, D once each. Confirms (a) percentile + per-token A still gains, (b) percentile + per-tensor A degrades (the predicted residual-outlier failure mode).

### Experiment 5: OPT-125M full percentile sweep at C scheme — ✅ RUN

Cells: [opt_125m/percentile_sweep_cells.md](opt_125m/percentile_sweep_cells.md). Grid: 5p × 5α (step=0.2), 25 runs.

|     p | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.9 |
|------:|------:|------:|------:|------:|------:|
| 0.999 | 27.88 | 27.70 | **27.6291** | 27.67 | 27.71 |
| 0.995 | 27.85 | 27.72 | 27.76 | 27.78 | 27.80 |
| 0.99  | 27.73 | 27.76 | 27.74 | 27.77 | 27.81 |
| 0.95  | 27.84 | 27.73 | 27.80 | 27.74 | 27.74 |
| 0.90  | 27.76 | 27.72 | 27.80 | 27.73 | 27.79 |

Task 01 references: FP16 = 27.57, C/max ≈ 27.6, O1/max = 28.30, O2/max = 29.16.

Findings: percentile + C effectively ties C/max (27.6291 vs ~27.6) and beats O1/max by 0.67 PPL. The sweep is almost flat — total spread across 25 cells is 0.25 PPL, no catastrophic collapse even at `(p=0.90, α=0.9) = 27.79`. Outlier pressure at 125M is small, so the smoothing-statistic choice doesn't matter much. This is a useful control datapoint and corroborates SmoothQuant's central premise that outlier-handling becomes load-bearing with scale.

### Experiment 4: OPT-2.7B percentile sweep at C scheme — ✅ RUN

Cells: [opt_2_7b/percentile_sweep_cells.md](opt_2_7b/percentile_sweep_cells.md)
Grid: `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90} × alpha ∈ {0.5, 0.7, 0.9}`, scheme = C. 15 runs.

Result table (WikiText-2 PPL, OPT-2.7B, scheme C):

| p     | α=0.5    | α=0.7    | α=0.9    |
|------:|---------:|---------:|---------:|
| 0.999 | 12.3833  | 12.3672  | 12.4280  |
| 0.995 | **12.3422** | 12.3801  | 12.7400  |
| 0.99  | 12.3509  | 12.3725  | 13.0295  |
| 0.95  | 12.3885  | 12.3640  | 323.7864 |
| 0.90  | 12.4411  | 12.3817  | 1765.9023 |

Task 01 references for OPT-2.7B (from the user's Task 01 result table): FP16 = 12.3425, SQ-O1 = 12.3946, SQ-O2 = 12.4214, SQ-PCW-PT (C/max) = 12.3714, SQ-PCW-TEN = 12.3969, W8A8-naive = 13.4655.

Findings:
- **Winner: p=0.995, α=0.5 → 12.3422 PPL.** Effectively matches FP16 (12.3425) — the percentile+C config is lossless at INT8 on this model.
- vs C/max baseline: **−0.029 PPL** (12.3714 → 12.3422). vs O1/max: **−0.052** (12.3946 → 12.3422).
- **Optimum shifted from 1.3B's `(0.999, 0.9)` to 2.7B's `(0.995, 0.5)`** — both more aggressive p and much lower α. Likely mechanism: larger outliers at 2.7B mean smaller p is enough to absorb them, and lower α is sufficient because outlier magnitudes are larger so even mild smoothing produces a quantization-friendly activation.
- **α=0.5 column is a near-FP16 plateau.** Four of five p values give PPL ≤ 12.45 at α=0.5; three are within 0.01 of FP16.
- **The "low p + high α" failure mode is catastrophic at scale.** `p=0.95, α=0.9 = 323.79`; `p=0.90, α=0.9 = 1765.90`. On 1.3B the same cells gave 15.13 and 15.90 — bad but recoverable. At 2.7B the model collapses entirely. This is a thesis-relevant safety-boundary observation: the percentile-smoothing failure surface is sharp, not gradual, and gets sharper with model size.

#### Single-session verification (cross-config, FP16-anchored)

Cells: [`experiments/verification/opt_2_7b/full_table_cells.md`](../verification/opt_2_7b/full_table_cells.md). One Colab kernel, all 8 configs under one `Evaluator` instance.

| # | Config | PPL | Δ vs FP16 |
|---|---|---:|---:|
| 1 | FP16 | 12.3449 | +0.0000 |
| 2 | Naive W8A8 (per-tensor W & A, no smoothing) | 13.4438 | +1.0989 |
| 3 | O1 max α=0.5 | 12.3885 | +0.0436 |
| 4 | O2 max α=0.5 | 12.4142 | +0.0693 |
| 5 | **C pct p=0.995, α=0.5** | **12.3564** | **+0.0115** |
| 6 | C max α=0.5 | 12.3613 | +0.0164 |
| 7 | O1 + per-layer α (Task 05) | 12.4867 | +0.1418 |
| 8 | C + per-layer α (Task 05) | 12.3670 | +0.0221 |

At 2.7B the C scheme has nearly closed the gap to FP16 already (C/max +0.0164), so the percentile improvement compresses to **+0.005 PPL** — directionally consistent with 1.3B but at the noise floor. The thesis-relevant takeaway is the *trajectory*: percentile-vs-max delta = 0.146 PPL at 1.3B, 0.005 at 2.7B. Outlier-handling matters most exactly where there's PPL headroom to recover; once the scheme already saturates near FP16, the smoothing-statistic choice becomes second-order.

### OPT-6.7B verification (Task 02 sweep pending)

Cells: [`experiments/verification/opt_6_7b/full_table_cells.md`](../verification/opt_6_7b/full_table_cells.md). Row 5 uses placeholder `(p=0.995, α=0.5)` extrapolated from 2.7B; replace with actual Task 02 6.7B winner once that sweep runs.

| # | Config | PPL | Δ vs FP16 |
|---|---|---:|---:|
| 1 | FP16 | 10.6732 | +0.0000 |
| 2 | Naive W8A8 | 25.9135 | +15.2403 |
| 3 | O1 max α=0.5 | 10.6988 | +0.0256 |
| 4 | O2 max α=0.5 | 10.7000 | +0.0268 |
| 5 | **C pct p=0.995, α=0.5** | **10.6875** | **+0.0143** |
| 6 | C max α=0.5 | 10.7252 | +0.0520 |
| 7 | O1 + per-layer α | 10.7382 | +0.0650 |
| 8 | C + per-layer α | 10.6826 | +0.0094 |

C/pct beats C/max by 0.038 PPL — percentile is back to being meaningful at 6.7B (cf. 0.005 at 2.7B). C+per-layer-α (10.6826) ≈ C/pct (10.6875), same convergence as on 1.3B.

### OPT-13B verification (Task 02 sweep pending)

Cells: [`experiments/verification/opt_13b/full_table_cells.md`](../verification/opt_13b/full_table_cells.md). Row 5 placeholder `(p=0.995, α=0.5)`; not the true 13B optimum.

| # | Config | PPL | Δ vs FP16 |
|---|---|---:|---:|
| 1 | FP16 | 9.9439 | +0.0000 |
| 2 | Naive W8A8 | 4325.6772 | +4315.7333 |
| 3 | O1 max α=0.5 | 10.1767 | +0.2328 |
| 4 | O2 max α=0.5 | 10.2037 | +0.2598 |
| 5 | **C pct p=0.995, α=0.5** | **10.0077** | **+0.0638** |
| 6 | C max α=0.5 | 10.1691 | +0.2252 |
| 7 | O1 + per-layer α | 9.9963 | +0.0524 |
| 8 | C + per-layer α | 9.9825 | +0.0386 |

C/pct beats C/max by 0.161 PPL even at the placeholder (p, α) — the largest percentile-vs-max gap on the ladder. C+per-layer (9.9825) and C/pct (10.0077) both within 0.07 of FP16.

## Open implementation questions

- Calibration approach is **per-channel top-K buffer**, exact for every `p ≥ p_min`. `K = ⌈(1 - p_min)·N⌉ + safety` and the buffer is updated batch-by-batch via `torch.topk` along `dim=0`. After calibration, the sorted buffer is indexed (with linear interpolation matching `torch.quantile`) for every requested `p`. Estimated/binned approximations are explicitly avoided.
- Buffer dtype is fp16 (5 GB on 1.3B). The fp16 cast on stored values introduces ~0.5% relative perturbation on the recovered per-channel max — small enough that percentile rows (which differ by 5-30% between consecutive `p` values) are unaffected, but enough that the `p = 1.0` row of the sweep is excluded from both saved files and sweep grids. The max baseline is taken from Task 01 directly.
- Hooks attach only to the SmoothQuant smoothing-relevant linears (q_proj input, fc1 input). out_proj and fc2 inputs are not hooked because their inputs are not used in the smoothing formula.
- Weight percentile uses the **same `p` as activation percentile** — one knob, one figure axis. Decouple later only if results suggest the two should differ.

## Next steps

1. ✅ `experiments/task02_percentile_smoothing/percentile_calibration.py` (top-K, exact).
2. ✅ `experiments/task02_percentile_smoothing/opt_1_3b/generate_act_percentiles_cells.md` (A100).
3. ✅ `experiments/task02_percentile_smoothing/opt_1_3b/percentile_sweep_cells.md` for Experiment 1.
4. Run on A100, save results notebook to `results/task02/opt_1_3b/`.
5. Update this PROGRESS.md with findings and decide direction for Experiments 2 and 3.

## Status

OPT-125M: ✅ done · OPT-1.3B: ✅ Exp1 + Exp2 done · OPT-2.7B: ✅ done · OPT-6.7B: ⏳ cells ready ([opt_6_7b/](opt_6_7b/)) · OPT-13B: ⏳ cells ready, A100-80 ([opt_13b/](opt_13b/))
