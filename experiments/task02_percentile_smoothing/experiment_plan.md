# Task 02: Percentile-Based Smoothing — Experiment Plan

## Goal

Modify SmoothQuant's smoothing factor to use a per-channel **percentile** of `|X|` and `|W|` instead of the per-channel **max**. Test whether this is more accurate and/or more robust than max-based smoothing, and whether the gain stacks with the per-channel weight scheme (C) from Task 01.

## Background — the formula we are changing

SmoothQuant computes the per-channel smoothing factor as

```
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

implemented in `smoothquant_repo/smoothquant/smooth.py` (`smooth_ln_fcs`):

- `act_scales` is the per-channel running max of `|X|` over calibration data, computed in `smoothquant_repo/smoothquant/calibration.py:get_act_scales`.
- `weight_scales` is `torch.cat([fc.weight.abs().max(dim=0)]).max(dim=0)` over the fused linears that share the LayerNorm.

Our proposed replacement:

```
s_j = quantile(|X_j|, p)^alpha / quantile(|W_j|, p)^(1-alpha)
```

where `p ∈ {0.90, 0.95, 0.99, 0.995, 0.999}`. `p = 1.0` recovers the original max-based formula and serves as the baseline (already on Drive as `act_scales/opt-1.3b.pt`).

## Why this is worth testing

1. `max` is dominated by a single token's outlier in any channel. One spike defines the entire channel's smoothing factor — `s_j` is unstable to the calibration set.
2. The paper itself admits this failure mode: §5.2 notes that for OPT-175B and similar models they clip the top 2% of activation magnitudes during quantization. That clip is a post-hoc patch; percentile-smoothing addresses the same instability principled-ly inside the smoothing factor.
3. The fix is local — two functions, ~10 lines each. The framing for the thesis is "modify the smoothing operator, not just the quantization scheme."
4. It composes naturally with the C scheme (per-channel W + per-token A) from Task 01, because per-token activation quantization absorbs residual outliers above the percentile gracefully.

## Risks and tradeoffs

- **Residual outliers post-smoothing.** Values above `quantile(|X_j|, p)` are not folded into the smoothing factor. After smoothing they remain in the activation and are *relatively* larger than under max-smoothing. Per-token activation quant tolerates this; per-tensor activation quant (O2, D) will likely degrade. Hypothesis: percentile helps for C and O1 but hurts for O2 and D.
- **Calibration cost.** Exact per-channel quantile only needs the **top-K largest values** per channel, where `K = ⌈(1 - p_min)·N⌉ + safety` and `N = num_samples · seq_len`. For our sweep `p_min = 0.90` so `K ≈ 0.10·N`. One sorted top-K buffer per smoothing site covers every requested `p ≥ p_min`. Memory in fp16: ~5 GB on OPT-1.3B, ~13 GB on 6.7B, ~21 GB on 13B — all fit on a single A100 (GPU buffer for ≤6.7B; CPU buffer for 13B).
- **Alpha will re-tune.** Percentile-based scales are numerically smaller than max-based scales, so the same alpha redistributes a different magnitude of difficulty to weights. We should not assume the task01 best-alpha (~0.5–0.9 for C) carries over. Each percentile setting needs its own alpha sweep.
- **Comparability.** Calibration set (Pile val), `num_samples=512`, `seq_len=512` must stay identical to the max-baseline so the only changing axis is the statistic.

## Scope of Task 02

We only modify the *smoothing* statistic. We do **not** modify the quantization step (no clipping of outliers, no different bit widths). This isolates the contribution of the smoothing change.

## Reference baseline

The user's reference baseline for the thesis comparison is **O1 + max smoothing** (the paper's recipe). The reference number for this on OPT-1.3B already exists in Task 01 (`results/task01/opt_1_3b/alpha_sweep_results.ipynb`); no need to re-run inside Task 02.

## Configs to test

The smoothing statistic axis (new) is orthogonal to the scheme axis (Task 01). The Task 02 primary contribution is **C scheme + percentile smoothing**, so the OPT-1.3B sweep concentrates there. Optional cross-scheme runs are deferred to a follow-up only if the C+percentile result motivates them.

**Default sweep grid (from OPT-6.7B onward; retroactive for OPT-125M):**

| Smoothing percentile p | Scheme | Alpha | Purpose |
|-----------------------|--------|-------|---------|
| 0.999 | C | 0.1, 0.3, 0.5, 0.7, 0.9 | Conservative percentile |
| 0.995 | C | 0.1, 0.3, 0.5, 0.7, 0.9 | Between 0.99 and 0.999 |
| 0.99 | C | 0.1, 0.3, 0.5, 0.7, 0.9 | Matches paper's §5.2 2% clip intuition |
| 0.95 | C | 0.1, 0.3, 0.5, 0.7, 0.9 | Aggressive percentile |
| 0.90 | C | 0.1, 0.3, 0.5, 0.7, 0.9 | Stress test |

25 runs per model, all in one notebook. Justification: OPT-1.3B winner sat at α=0.9 but OPT-2.7B winner shifted to α=0.5 — best-α drifts strongly with model size, so a narrow grid would miss the optimum at scale.

**Historical note**: OPT-1.3B was run with the older narrow `α ∈ {0.5, 0.7, 0.9}` grid plus a separate full-alpha follow-up at the two competing optima ([opt_1_3b/full_alpha_sweep_cells.md](opt_1_3b/full_alpha_sweep_cells.md)). OPT-2.7B used the narrow grid and the (p=0.995, α=0.5) winner happened to land on the grid corner, but adjacent cells `(α=0.4)` etc. were never measured. From OPT-6.7B forward the full grid is run in a single pass.

**`p = 1.0` is not in the sweep grid.** The top-K calibration's `p = 1.0` row sits inside one fp16 ULP of the upstream max scales (~0.5% relative), which is enough noise to make an in-harness baseline row unreliable. The max-smoothing baseline comes from Task 01's `alpha_sweep_results.ipynb` (C/max and O1/max numbers) instead.

Follow-up if the C+percentile sweep motivates it:
- Full alpha sweep at the winning `p` (alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}).
- Cross-scheme spot checks (percentile + O1, O2, D at the winning alpha).

## Implementation approach

All Task 02 code lives **inside** `experiments/task02_percentile_smoothing/`. Cross-model utilities (calibration, smoothing) sit at the task root next to this plan; per-model cell `.md` files sit in `opt_<size>/`. We do not put anything under `shared/` — Task 02 owns its own code so the change set is self-contained for the thesis chapter.

### 1. Activation-side: exact percentile via top-K buffers — `percentile_calibration.py` (to be written)

The p-th percentile of `N` per-channel samples equals the value at sorted-rank `p·(N-1)`, which is in the top `(1 - p)·N` largest values. Keeping a per-channel top-K buffer with `K = ⌈(1 - p_min)·N⌉ + safety_margin` is therefore sufficient to read off **every** requested `p ≥ p_min` exactly — no histogram, no disk staging, no approximation.

**Calibration flow (`get_act_percentiles`):**

1. Identify the smoothing-relevant linear inputs in OPT — the inputs to `q_proj` (shared with k/v) and to `fc1`. That's `2 × num_layers` sites (48 for 1.3B, 64 for 6.7B). We deliberately do *not* hook every linear; only sites whose inputs feed the SmoothQuant smoothing operator are needed.
2. Compute `N = num_samples · seq_len = 262 144`, `K = ⌈(1 - p_min)·N⌉ + 16` (safety margin for ties + interpolation lookahead). For `p_min = 0.90`, `K ≈ 26 230`.
3. Pre-allocate one buffer per site, shape `[K, in_features]`, dtype fp16, on **GPU for models ≤ 6.7B** and on **CPU for 13B** (where the model itself eats most of the A100). Initialise with `-inf` so the first batch dominates.
4. Forward-pass hook on each tracked Linear:
   ```python
   new = x.reshape(-1, in_features).abs().to(buffer.dtype, copy=False)   # [B*T, in]
   combined = torch.cat([buffer, new], dim=0)                            # [K + B*T, in]
   buffer.copy_(torch.topk(combined, k=K, dim=0, largest=True).values)   # [K, in]
   ```
   `torch.topk` along `dim=0` is one GPU kernel covering all channels; sub-millisecond per call on A100 for a 1.3B-sized site.
5. Run the standard 512-sample × 512-token Pile calibration in one pass — same dataset, same shuffle seed (42), same `num_samples` and `seq_len` as Task 01's max-scale generation, so comparability is preserved.
6. After the pass, sort each site's buffer ascending along `dim=0`. For every requested `p` (including `p = 1.0`), index the sorted buffer with linear interpolation between adjacent ranks to match `torch.quantile`'s convention exactly. The resulting `dict[name -> tensor[in_features]]` per `p` is what gets saved.
7. **Save one file per p < 1.0** to Drive — see "Storage layout" below. `p = 1.0` is computed in-memory from the same buffer for a one-shot pipeline-correctness diff against `act_scales/opt-<size>.pt` and then dropped without saving (the fp16 storage in the top-K buffer perturbs the per-channel max by up to ~0.5%, which is too noisy to be useful as a sweep baseline row).

**Why per-p files instead of one combined file:**
- Each sweep run loads exactly one percentile dict; per-p files keep load times and memory minimal.
- Future experiments can add a new `p` without rewriting the existing files.
- Mirrors the existing `act_scales/opt-1.3b.pt` shape exactly: each per-p file is a drop-in replacement, so `smooth_lm_pct` stays signature-compatible.

**Storage layout on Drive:**

```
/content/drive/MyDrive/thesis_results/act_percentiles/
    opt-1.3b/
        p0.999.pt
        p0.995.pt
        p0.99.pt
        p0.95.pt
        p0.90.pt
    opt-2.7b/  ...
    opt-6.7b/  ...
```

Each `p<value>.pt` is a `dict[name -> tensor[in_features]]`, identical in shape to `act_scales/opt-<size>.pt`. The pipeline-correctness check (in-memory diff of `p = 1.0` against `act_scales/opt-<size>.pt`) runs inside the calibration cell and gates the save. Sweep runs reference the Task 01 max-smoothing numbers as the baseline rather than re-running max in-harness.

### 2. Weight-side and smoothing call — `percentile_smooth.py` (already written, kept as-is)

Forks `smooth_ln_fcs` and `smooth_lm` from `smoothquant.smooth` into `smooth_ln_fcs_pct` / `smooth_lm_pct` that:
- Replace the per-channel weight `max` with per-channel exact `torch.quantile(p_w)` along the `out_features` dim (weights are small enough that exact quantile is cheap directly on the parameter tensor).
- Accept the activation per-channel percentile tensor as `act_scales` (signature-compatible with the upstream `smooth_lm`).
- Fall back to exact `max` when `p == 1.0` so the `p=1.0` row of the sweep is bit-identical to the upstream `smooth_lm`.

Only OPT support is needed for Task 02 (1.3B → 13B); we do not port the BLOOM/LLaMA/Falcon/Mistral/Mixtral branches.

## Models

### Priority 1: OPT-1.3B (A100)
- Activation max-scales already on Drive (`act_scales/opt-1.3b.pt`) — used as the ground truth to validate the top-K pipeline's `p=1.0` output.
- Generate `act_percentiles/opt-1.3b/p<value>.pt` for `p ∈ {1.0, 0.999, 0.995, 0.99, 0.95, 0.90}` from a single calibration pass.
- GPU buffer: ~5 GB fp16 + 2.6 GB fp16 model + activations ≈ 12-14 GB peak — easy on A100-40.
- Each PPL eval ~3 min; the full `6 p × 3 alphas = 18 runs` block ≈ 55 min after scales are generated.
- Why first: matches Task 01's primary model so we can directly compare gains over Task 01's C-best.

### Priority 2: OPT-125M (T4 / A100)
- Cheap sanity check (~21s per run). Run after 1.3B confirms the methodology.
- Useful for fine-grained alpha × percentile heatmap if we want one figure for the thesis.

### Priority 3: OPT-2.7B, OPT-6.7B, OPT-13B (A100)
- Outlier-heavy regime — this is where percentile-smoothing should pay off the most. Defer until 1.3B confirms direction.
- 6.7B requires A100 (see CLAUDE.md challenge #2). 13B will too.
- 6.7B fits with GPU-side top-K buffer (~13 GB fp16 + ~13 GB fp16 model + activations ≈ 30 GB peak on A100-40).
- 13B uses CPU-side top-K buffer (~21 GB fp16) so the GPU can hold the model and activations alone. JarvisLabs A100-80 is the cleanest option; A100-40 also works because the buffer is on host memory.

## Metrics

- WikiText-2 perplexity at `seq_len=2048` (primary).
- Same evaluator as Task 01 — `n_samples=40` chunks, identical to `alpha_sweep_cells.md`.
- Optional later: zero-shot accuracy on LAMBADA / HellaSwag / PIQA / WinoGrande, mirroring `zero_shot_eval_cells.md`.

## Expected results

- p=0.99 with C should beat p=1.0 (max) with C at the C-best alpha. Magnitude unknown — hopeful range 0.05–0.3 PPL on 1.3B; could be larger on 6.7B where outliers are worse.
- p=0.95 and p=0.90 will probably degrade (cutting off too much information from the smoothing statistic).
- Percentile-smoothing should make C *more* alpha-stable, not less, because the smoothing factor itself is more stable.
- Percentile-smoothing combined with O2/D (per-tensor activation quant) will likely degrade — residual outliers above the percentile will clip during per-tensor activation quantization.

## Thesis framing

Task 01 contribution: scheme choice matters, the paper's per-tensor weight quantization is a fragile choice, per-channel weight (C) is uniformly better and more alpha-stable.

Task 02 contribution: the smoothing *statistic* itself is a degree of freedom the paper did not explore. Replacing `max` with a high-percentile makes the smoothing factor robust to single-token outliers in calibration, addresses the same root cause as the paper's ad-hoc 2% clip in §5.2 but in a principled way, and stacks with per-channel weight quantization to give a configuration (`percentile + C`) that the original paper does not consider.

If the gain holds on 6.7B, the combined story is: SmoothQuant's two implicit choices (per-tensor weights, max-statistic) are both suboptimal in the outlier regime, and fixing both gives a strictly better quantization recipe at no hardware cost.

## Status

- experiment_plan.md ✅ (this file)
- PROGRESS.md ✅
- `percentile_smooth.py` ✅ (exact `torch.quantile` on weights, max fallback at p=1.0)
- `percentile_calibration.py` ⏳ (to be written — disk-staged exact quantile)
- `opt_1_3b/generate_act_percentiles_cells.md` ⏳ (to be written for A100)
- `opt_1_3b/percentile_sweep_cells.md` ⏳ (to be written for A100)
- OPT-1.3B percentile sweep: ⏳ (awaiting code + Colab run)
