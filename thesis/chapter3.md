# Chapter 3 — Proposed Work

This chapter describes the methodology in detail. The starting point is the SmoothQuant recipe of [5]: a one-shot offline calibration step that builds a per-channel smoothing factor `s`, a transformation that folds `s` into the preceding LayerNorm or linear layer, and an outer quantization step that converts both weights and activations to INT8. The proposed work modifies two parts of this pipeline. The calibration step replaces the per-channel `max(|X|)` summary statistic with an exact per-channel quantile, and lets the migration strength `α` vary across layers according to a per-layer severity profile. The outer step is fixed at per-channel weight together with per-token activation, which the original paper does not adopt for its OPT and BLOOM headline numbers but which pairs naturally with smoothed weights. Everything else — the mathematical identity that makes smoothing exact, the choice of calibration corpus, the use of INT8 GEMM kernels at inference — is kept identical to the upstream recipe so that the comparison stays clean.

The chapter is organised in five parts. Section 3.1 lays out the smoothing identity and the place in the pipeline where each modification sits. Section 3.2 describes the quantile-based calibration in detail, including the exact (non-binned) per-channel quantile estimator. Section 3.3 describes the outer quantization scheme and explains why per-channel weights are the right partner for smoothed activations. Section 3.4 describes the per-layer migration strength, the severity profile that drives it, and the parametric form used to map severity onto `α(l)`. Section 3.5 describes how the calibrated configuration is moved from a fake-quant simulation onto real INT8 kernels for end-to-end deployment, including the regime shift that the deployment step forces on the optimal `α`.

## 3.1 The smoothing identity and where each modification sits

The proposed method operates on the same linear-layer identity that SmoothQuant introduced. For an input activation `X ∈ R^{T × C_i}` and a weight `W ∈ R^{C_i × C_o}`, the matmul `Y = X W` is invariant under any positive per-channel rescaling `s ∈ R_{>0}^{C_i}`:

$$
Y = X W = \big(X \, \mathrm{diag}(s)^{-1}\big) \cdot \big(\mathrm{diag}(s) \, W\big) = \hat X \, \hat W.
$$

The transformation is mathematically exact; nothing about the matmul output changes. What changes is the *distribution* of the operands. If `s` is chosen so that the channels of `\hat X` are flatter than those of `X`, then a coarse quantizer applied to `\hat X` loses less information than the same quantizer applied to `X`. The matching shift on the weight side, `\hat W = \mathrm{diag}(s) \, W`, makes some channels of `\hat W` larger than the corresponding channels of `W`, so the weight tensor becomes a little harder to quantize. The trade-off is profitable because activations have heavy per-channel outliers and weights do not [4, 5]: redistributing some difficulty from the harder operand onto the easier one improves the joint quantization error.

The factor `s` itself is fixed at a single calibration step and never recomputed. In SmoothQuant's recipe [5], the `j`-th component of `s` is

$$
s_j = \frac{\max_t |X_{t,j}|^{\alpha}}{\max_i |W_{i,j}|^{1-\alpha}}, \qquad j = 1, \ldots, C_i,
$$

with the activation maximum taken over the tokens `t` in a small calibration buffer (the SmoothQuant calibration uses 512 sentences of length 512 from the Pile validation set [7]). Here `α ∈ [0, 1]` is a single migration-strength scalar that controls how the per-channel difficulty is split between `\hat X` and `\hat W`. The vector `s` is then folded into the parameters of the linear layer (and its preceding LayerNorm) so that no scaling kernel runs at inference time.

The proposed method changes two things about this construction.

The first change is the choice of summary statistic. The `max` is replaced by a per-channel quantile `Q_p`, with the same `p` used for both activation and weight, giving

$$
s_j = \frac{Q_p\big(|X_{:,j}|\big)^{\alpha}}{Q_p\big(|W_{:,j}|\big)^{1-\alpha}}, \qquad j = 1, \ldots, C_i, \quad p \in (0, 1].
$$

`p = 1` recovers the upstream `max`-based formula exactly. For `p` slightly below 1, the statistic is robust to single-token spikes that dominate `\max`. The value of `p` is one new hyperparameter, swept jointly with `α`. The reason this change is principled, rather than just empirical, is given in §3.2.

The second change is to let `α` vary across the layers of the network rather than being fixed globally. Concretely, layer `l` uses its own migration strength `α(l)`, and its smoothing factor becomes

$$
s_j^{(l)} = \frac{Q_p\big(|X^{(l)}_{:,j}|\big)^{\alpha(l)}}{Q_p\big(|W^{(l)}_{:,j}|\big)^{1-\alpha(l)}}.
$$

`α(l)` is determined by a per-layer severity profile measured on the same calibration buffer. This change is described in §3.4.

The outer quantization scheme — the granularity at which `\hat W` and `\hat X` are converted to INT8 — is fixed throughout to per-channel on the weight side and per-token on the activation side. This is the partner that the smoothing direction asks for, and the reason is given in §3.3.

The two modifications are *orthogonal* in the sense that they change different parts of the pipeline. Quantile-based smoothing changes how `s` is constructed at calibration time; per-layer `α` changes the exponent that the same statistic is raised to. Either can be enabled without the other, and the experiments later in the thesis report both individually and combined. They also reach a similar accuracy floor through different routes — a fact that the verification runs (Chapter 5) confirm and that this chapter motivates structurally.

## 3.2 Quantile-based calibration

### 3.2.1 Why the maximum is the wrong statistic on activations

The smoothing factor enters the network multiplicatively and is fixed offline. Any noise or single-sample bias in the per-channel statistic that defines `s` is therefore baked into every subsequent forward pass for the rest of the model's deployment lifetime. On weights this is benign: trained weight distributions are flat and approximately uniform across channels [4, 6], so `\max_i |W_{i,j}|` is a stable summary of channel `j`. On activations it is not. A single anomalously large token in the 512×512 calibration buffer sets `\max_t |X_{t,j}|` for the entire deployment, and the resulting `s_j` may overfit to that one token rather than reflecting the channel's actual operating range.

This is not a hypothetical concern. The SmoothQuant authors run into the same phenomenon explicitly when applying their most aggressive O3 configuration to GLM-130B (§5.2 of [5]), and patch it by clipping the top 2% of tokens before computing the static quantization step sizes, citing [8]. The clip is effective but ad hoc: it lives at the *quantization* step rather than the *smoothing* step, it has to be tuned per model, and it is presented as a workaround rather than as a principled component of the recipe. The phenomenon it patches — a few extreme tokens dominating a magnitude estimate that is supposed to summarise an entire channel — is exactly the same phenomenon that affects `\max_t |X_{t,j}|` inside the smoothing formula in the first place.

The diagnostic from Chapter 4 makes this concrete. On OPT-2.7B, OPT-6.7B, and OPT-13B, the per-channel `max` already encodes most of the *inter-channel* outlier structure that SmoothQuant was designed to absorb. But within many layers — particularly the middle layers of larger models — the ratio `Q_{0.99}(|X_{:,j}|) / \mathrm{median}(|X_{:,j}|)` is close to one even for channels whose `\max(|X_{:,j}|) / \mathrm{median}(|X_{:,j}|)` is in the tens. That is to say, for these channels, the bulk of the distribution is tame and only one or two tokens drive the maximum upward. Using `\max` as the summary statistic at exactly these channels means `s_j` is set by the spike rather than by the typical magnitude. A high quantile sees the typical magnitude.

### 3.2.2 The quantile-based formula

The replacement is to take the quantile of the absolute values along the token axis (for activations) or the input-feature axis (for weights), at a fixed probability `p`:

$$
s_j = \frac{Q_p\big(|X_{:,j}|\big)^{\alpha}}{Q_p\big(|W_{:,j}|\big)^{1-\alpha}}, \qquad p \in (0, 1].
$$

`Q_p(\cdot)` is the empirical quantile of the input set, defined here in the same convention used by `torch.quantile` — the value `v` such that a fraction `p` of the elements are `≤ v`, with linear interpolation between the two adjacent ranked elements when `p \cdot (N - 1)` is not an integer. The same `p` is used for both the activation and the weight side; decoupling the two would add a second probability hyperparameter without obvious motivation, since the goal of the substitution is to push the same robustness property onto both summary statistics.

`p = 1` is equal to `\max` by construction, and the calibrator is verified to reproduce the upstream `act_scales` files bit-for-bit at this probability (modulo a known fp16-storage perturbation discussed in §3.2.4). The proposed method is therefore a strict generalisation of the upstream recipe, not a competing alternative. For `p` slightly below 1 — values in `{0.999, 0.995, 0.99, 0.95, 0.90}` are swept in the experiments — the formula ignores the very top of each channel's distribution. This is a smoothing-side analogue of the GLM-130B top-2% clip from [5], applied at the right step in the pipeline.

The activation side of the formula calls `Q_p` on an empirical sample drawn from the calibration corpus, so its value depends on how many tokens have been seen. The weight side calls `Q_p` directly on the weight tensor, which is a fixed-size population once the model has been loaded; no calibration corpus is involved. This asymmetry is intentional and matches how `\max` is used in the original recipe: weights are static and their statistics are exact, activations are dynamic and their statistics are estimated.

### 3.2.3 Exact per-channel quantile estimation

The activation side requires a per-channel quantile across the entire calibration corpus. A naive implementation would store every absolute activation value seen during calibration and then call `torch.quantile` per channel at the end. With 512 sentences of 512 tokens and OPT-13B's hidden size of 5120, the buffer for a single layer's activation is `512 × 512 × 5120 = 1.34 × 10^9` fp16 values, or 2.5 GB; multiplied by the number of smoothing-relevant linear inputs in the network, the total is well into the hundreds of GB. Storing it is not feasible on the available hardware.

A histogram-based estimator was considered as an alternative. The idea is to use the per-channel `max` (already computed by the upstream `act_scales` script) as the upper edge of a fixed number of bins per channel, and accumulate counts in a single forward pass. The total memory drops from "all tokens" to `num_bins · C_i` per layer, which fits easily even on a T4. The problem is that the resulting estimate disagrees with the upstream `max` baseline at `p = 1`, because the histogram's top bin reports a midpoint rather than the actual peak. The disagreement is small in absolute terms but large enough that the within-sweep deltas — which are themselves of the order of 0.01–0.1 PPL — become unreliable. This option was rejected.

The estimator used in this work is exact for every requested `p` above a fixed lower bound `p_min`. For each smoothing-relevant linear input, the calibrator maintains a per-channel **top-K buffer** of size `K = ⌈(1 - p_min) \cdot N_{\text{total}}⌉` where `N_{\text{total}}` is the total number of tokens that will be observed during calibration. After every forward pass on a calibration batch, the buffer is updated: the new batch's absolute values are concatenated with the existing buffer along the token axis, and `torch.topk` is taken along that axis to retain only the top `K` per channel. At the end of calibration, the buffer for each channel contains exactly the `K` largest absolute values that were seen across the entire corpus, in sorted order. The empirical quantile at any `p ≥ p_min` can then be read off directly by indexing into the sorted buffer at rank `\lceil p \cdot (N_{\text{total}} - 1) \rfloor`, with linear interpolation between the two adjacent ranks to match the `torch.quantile` convention.

This procedure is exact in the sense that the value returned at probability `p ≥ p_min` is identical to the value that a full-storage quantile call would have returned. No approximation is introduced. The `p_min` lower bound is set to 0.90 in the implementation, which gives `K \approx 0.10 \cdot N_{\text{total}}` and therefore admits every `p \in [0.90, 1]` exactly. With 512 sentences of 512 tokens, `N_{\text{total}} = 262{,}144` per linear input, so `K \approx 26{,}214` per channel. The total buffer memory is around 5 GB on OPT-1.3B, 13 GB on OPT-6.7B, and 21 GB on OPT-13B in fp16. This fits in GPU memory on an A100-40GB up to 6.7B; for 13B the buffer is moved to CPU and the `torch.topk` call is performed there, at the cost of an additional host–device transfer per batch.

### 3.2.4 Sanity check against the upstream max

The same buffer that is used to compute every `Q_p` for `p < 1` also contains the per-channel maximum, in element 0 of the sorted buffer. Comparing this max against the upstream `act_scales/opt-<size>.pt` file from [5] gives a direct correctness check on the calibrator. On OPT-1.3B and OPT-2.7B the maximum relative difference between the two is below `10^{-3}`, well within fp16 numerical noise. On OPT-6.7B and OPT-13B the worst-case channel reaches around 1–2% relative difference. This is consistent with two sources of perturbation that scale with model size. The first is that buffer storage is fp16, and the per-channel max sits at the high end of channel magnitude where one fp16 unit-in-the-last-place is already 1–2% of the value. The second is that OPT's attention path is not bit-deterministic in fp16 on CUDA, so the same calibration on the same data can land on slightly different fp16-representable max values across runs. Neither perturbation affects the lower-quantile rows that the experiments actually use, since those rows are read off rank `p \cdot (N - 1) \ll N`, far from the noisy top of the buffer. To avoid the noise on the top row contaminating the sweep grid, the row at `p = 1` is excluded from both saved files and the sweep; the `\max`-baseline numbers come from the upstream `act_scales` file directly, computed once with full fp32 precision in the original SmoothQuant calibration script.

### 3.2.5 Where the calibration is hooked

The original SmoothQuant smoothing transformation only touches two linear inputs per Transformer block: the input to the attention `q_proj` (which is shared with `k_proj` and `v_proj` because all three see the same hidden state coming out of LayerNorm) and the input to the FFN `fc1`. The other two linear layers in a block — the attention output projection `out_proj` and the FFN second projection `fc2` — see post-attention and post-activation tensors, respectively, which are not used in the smoothing formula. The calibrator therefore attaches forward hooks to exactly those two inputs per block, ignoring the rest. This matches the upstream `generate_act_scales.py` behaviour and keeps the buffer footprint bounded.

## 3.3 Outer quantization scheme: per-channel weight, per-token activation

### 3.3.1 The natural partner to smoothed activations

The SmoothQuant transformation deliberately moves variation from `X` into `W`. After calibration, the smoothed activation `\hat X` is much flatter across its channels than `X` was, but the smoothed weight `\hat W` is more variable across its rows than `W` was. Higher values of `α` push more variation onto the weight side, by construction.

The three configurations defined in [5] — O1, O2, and O3 — all use **per-tensor** weight quantization. A single step size `\Delta_W` is computed from the global maximum magnitude of the entire `\hat W` matrix, and every channel is then represented relative to that one step. Channels whose typical magnitude is much smaller than the global maximum get squeezed into a small number of effective integer levels. This is the same effective-bits problem that originally afflicted activations and that motivated smoothing in the first place; the migration has now reproduced it on the weight side.

The natural countermeasure is per-channel weight granularity, where the step size is not one global scalar but a vector with one entry per output channel:

$$
\Delta_W \in \mathbb{R}^{C_o}, \qquad \Delta_{W, k} = \frac{\max_i |\hat W_{i, k}|}{2^{N-1} - 1}, \quad k = 1, \ldots, C_o.
$$

Each output channel of `\hat W` then uses its own range, and the residual variation introduced by smoothing is absorbed channel by channel rather than averaged into a single global scale. The effective quantization levels for a channel of typical magnitude `m_k` is now `2^N` rather than `2^N \cdot m_k / m_{\text{global}}`.

This is fully compatible with the same INT8 GEMM kernels that the paper's O1–O3 schemes target. The dimensionality argument is the same one the paper itself uses to justify per-token activation scaling along the token dimension `T` (Figure 3 of [5]). In a linear matmul `Y = X W` of shape `(T, C_i) \times (C_i, C_o) = (T, C_o)`, the contraction is over the inner dimension `C_i`. The token dimension `T` and the output-channel dimension `C_o` are *outer* dimensions of the contraction; scaling factors along either can be applied as a vector multiply on the GEMM output without disturbing the integer arithmetic inside the kernel. Per-token activation scaling along `T` — used by the paper's O1 — applies a scaling vector of length `T` to the rows of the output. Per-channel weight scaling along `C_o` applies a scaling vector of length `C_o` to the columns. Both are post-GEMM vector multiplies and both add only negligible runtime overhead.

The full output of a per-channel-weight, per-token-activation INT8 GEMM is

$$
Y_{t, k} = \Delta_{X, t} \cdot \Delta_{W, k} \cdot \sum_{i=1}^{C_i} \bar X_{t, i} \cdot \bar W_{i, k},
$$

where `\bar X` and `\bar W` are the INT8-quantized operands, `\Delta_{X, t}` is the per-token activation step (vector of length `T`), and `\Delta_{W, k}` is the per-channel weight step (vector of length `C_o`). The sum is the integer GEMM that the kernel actually runs; both rescalings are outer-product fix-ups on the result.

### 3.3.2 Why the original paper does not use this scheme on OPT

The original SmoothQuant paper does not adopt this scheme for its OPT or BLOOM headline numbers. The three configurations it benchmarks across the OPT 1.3B–175B ladder (Figure 7 and Table 1 of [5]) all use per-tensor weights. Per-channel weight quantization is only used implicitly in the paper's later results on Llama-2, Falcon, Mistral, and Mixtral (Table 7 of [5]), and there it is mentioned in passing — "we used per-token activation quantization and per-channel weight quantization for SmoothQuant" — without being identified as a deliberate design choice or compared directly against the per-tensor schemes that dominate the rest of the paper. To the best of this thesis's knowledge, no paper in the SmoothQuant family has explicitly named the per-channel-weight + per-token-activation configuration on OPT, swept the migration strength under it, or compared its alpha-stability against O1 and O2 across model scales. The first part of the proposed work makes that comparison explicit. The configuration is named "C" throughout the experiments in this thesis, alongside the existing O1 and O2 labels for the per-tensor-weight schemes from [5].

### 3.3.3 Static versus dynamic activation quantization

Per-token activation quantization can be either dynamic — `\Delta_{X, t}` is recomputed at every forward pass from the running activation — or static, with `\Delta_X` calibrated once on the same Pile buffer used for `s`. The fake-quant experiments in Chapter 4 use dynamic per-token quantization, which keeps the activation step always faithful to the data. Real-INT8 deployment in Chapter 5 uses static per-tensor activation quantization, because the CUTLASS INT8 GEMM kernels available through the torch-int library [referenced in the original SmoothQuant repository] only support per-tensor static activations. The shift in regime from "fake-quant per-channel-weight + per-token-activation" to "real-INT8 per-tensor-weight + per-tensor-static-activation" is documented in §3.5.

## 3.4 Per-layer migration strength

### 3.4.1 The severity profile

The SmoothQuant paper picks one global value of `α` per model: 0.5 for OPT and BLOOM, 0.75 for GLM-130B [5]. The choice is made by a small grid search on the Pile validation set. This collapses a network of dozens of layers down to a single scalar.

A diagnostic on the OPT family (Chapter 4) shows that the outlier severity of the activation is *not* uniform across layers, and the non-uniformity gets sharper with model size. The diagnostic uses the same per-channel `max(|X|)` data that the SmoothQuant calibrator already produces, and computes for each smoothing-relevant linear input the *severity ratio*

$$
\sigma(l) = \frac{\max_j \, \big[\max_t |X^{(l)}_{t,j}|\big]}{\mathrm{median}_j \, \big[\max_t |X^{(l)}_{t,j}|\big]},
$$

which is the ratio of the most outlier-heavy channel to the median channel within layer `l`. Across OPT-2.7B, OPT-6.7B, and OPT-13B, the spread of `σ(l)` from the calmest layer to the most outlier-heavy layer grows from roughly 13× at 2.7B to over 46× at 13B. The shape of `σ(l)` as a function of `l` is also strikingly consistent across model sizes:

- Layer 0 has low severity (`σ(0) ≈ 1.5–3`), because the embedding feed has not yet developed channel-level outliers.
- Layers 1–3 ramp sharply to a peak (`σ ≈ 36, 60, 82` at the peak for 2.7B, 6.7B, 13B respectively).
- Layers 3 through the end decay smoothly and near-monotonically to `σ ≈ 10–15`.
- The `q_proj` and `fc1` traces almost overlap, which means a single per-layer `α` is sufficient and per-site `α` is not needed.

A single global `α` cannot accommodate this. A layer with `σ ≈ 80` needs aggressive smoothing — a large `α` that pushes most of the per-channel difficulty onto the weight side — while a layer with `σ ≈ 5` is best served by a small `α` that leaves the weights largely untouched. Fixing `α = 0.5` for the whole network is a compromise that under-smooths the hard layers and over-smooths the easy ones.

### 3.4.2 A parametric `α(l)` tied to the severity profile

A free per-layer search would treat `α(l)` as 32 (or 40, or 80) independent hyperparameters and grid-search over them. That is intractable and would also defeat the point: the severity profile is structured and predictable, so the right granularity is a parametric form with two or three free parameters that follows the shape of the profile.

The form used in this work is

$$
\alpha(l) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \tilde \sigma(l), \qquad \tilde \sigma(l) = \frac{\sigma(l)}{\max_l \sigma(l)},
$$

where `\tilde \sigma(l) \in [0, 1]` is the normalised severity and `α_{\min}, α_{\max}` are two scalar hyperparameters that bound the migration strength below and above. Layers near the peak of the severity profile (`\tilde \sigma \to 1`) get `α(l) \to α_{\max}`; layers in the calm tail (`\tilde \sigma \to 0`) get `α(l) \to α_{\min}`. The specific values used for the verification runs are `α_{\min} = 0.5` and `α_{\max} = 0.9`, which gives a fixed schedule with zero free parameters — the entire `α(l)` curve is determined by the severity profile measured on the Pile buffer, so no further tuning is performed on the evaluation set.

The severity itself is computed from the same `act_scales` data used by the upstream SmoothQuant calibration, so this modification adds no new pass over the calibration corpus. `α(l)` is a property of layer `l` only; it is fixed offline and folded into `s^{(l)}` exactly as the original `α` would have been.

### 3.4.3 How per-layer `α` composes with quantile smoothing

The quantile statistic and the per-layer migration strength change different parts of the smoothing factor. The full form, with both modifications enabled, is

$$
s_j^{(l)} = \frac{Q_p\big(|X^{(l)}_{:,j}|\big)^{\alpha(l)}}{Q_p\big(|W^{(l)}_{:,j}|\big)^{1-\alpha(l)}}.
$$

Setting `p = 1` and `α(l) = α` for all `l` recovers the upstream `max`-based, global-`α` formula. Setting `p < 1` and `α(l) = α` for all `l` gives quantile smoothing alone. Setting `p = 1` and `α(l)` non-constant gives per-layer smoothing alone. Setting `p < 1` and `α(l)` non-constant gives the combined recipe. All four configurations are well-defined and can be evaluated against the same FP16 ceiling and W8A8-naive floor.

The structural reason these two modifications attack the same problem from two different angles deserves a sentence. The quantile statistic is an *intra-channel* fix: it removes the influence of the few extreme tokens within a channel, so that `s_j` reflects the channel's typical magnitude rather than its peak. Per-layer `α` is an *inter-layer* fix: it lets layers with wildly different outlier severities use different migration strengths. Both are responses to the same observation — that the activation outlier distribution in real LLMs is more structured than a single global statistic can capture — but they act at different scales. They reach a similar accuracy floor through different routes, and the combined recipe keeps both fixes active at once.

## 3.5 From fake quantization to real INT8 deployment

### 3.5.1 Two evaluation regimes

The recipe above is evaluated in two regimes. The first is a *fake-quant* regime, in which all linear layers and attention BMMs run as FP16 matmuls with the operands rounded to their INT8 grid before the multiply. This is a faithful simulation of the integer arithmetic that a real INT8 kernel would perform, and is the regime in which the original SmoothQuant paper reports its accuracy results [5]. It is also the only regime in which fine-grained schemes such as per-token activation quantization can be benchmarked, because not every fine-grained combination has a corresponding fused integer kernel available. All accuracy comparisons across smoothing variants in this thesis — quantile vs. max, per-layer vs. global `α`, per-channel vs. per-tensor weights — are run under fake quantization.

The second regime is *real INT8*, in which the linear layers of the smoothed network are replaced with fused INT8 GEMM kernels from the torch-int library, the activation tensors are converted to INT8 at the boundary of each linear layer, and the matmuls run on the GPU's integer Tensor Cores. This is the regime in which the memory and latency benefits of INT8 are actually realised. The torch-int kernels available for OPT only support the most aggressive paper configuration, O3: per-tensor weight quantization, per-tensor static activation quantization, with the activation step `\Delta_X` calibrated once on the same Pile buffer and folded into the kernel's bias. The deployment validation in this thesis therefore moves from fake-quant `C` (per-channel W + per-token A) onto real-INT8 O3 (per-tensor W + per-tensor static A) for the final memory measurement.

### 3.5.2 The α-regime split

Moving from one regime to the other has a non-obvious consequence for the optimal migration strength. In the fake-quant regime with the `C` scheme, per-channel weight quantization can absorb the variation that high-`α` smoothing pushes onto the weight side; aggressive smoothing actively helps, and the optimal `α` for OPT-1.3B in this regime sits at 0.9. In the real-INT8 O3 regime, per-tensor weight quantization cannot absorb the same shift; the largest channels of the smoothed weight blow up the global `\Delta_W`, and rounding error increases across all weights. Aggressive smoothing actively hurts, and the optimal `α` falls back to 0.5.

The quantile probability `p`, by contrast, is regime-independent. The mechanism that makes a high quantile a more stable summary statistic than the max — robustness to single-token spikes in the calibration buffer — has nothing to do with whether the downstream weight quantization is per-channel or per-tensor. The Task-02 winner `p` therefore transfers from the fake-quant `C` regime onto the real-INT8 O3 regime unchanged; only the migration strength has to be retuned.

The deployment recipe in this thesis is therefore:

$$
s_j = \frac{Q_p\big(|X_{:,j}|\big)^{0.5}}{Q_p\big(|W_{:,j}|\big)^{0.5}}, \qquad p = p^\star_{\text{Task 02}},
$$

with `p^\star_{\text{Task 02}}` taken as the per-model winner from the fake-quant sweep (for example `p = 0.999` on OPT-1.3B), `α = 0.5` fixed by the O3 regime constraint, and the resulting smoothed model run through torch-int's CUTLASS kernels on a single A100. The memory and accuracy of this configuration are reported in Chapter 5 against the matched paper-recipe baseline (`s_j = \max^{0.5} / \max^{0.5}`, also at `α = 0.5`, also under O3), so the comparison isolates the smoothing-statistic change.

### 3.5.3 What is actually measured at deployment

Three quantities are measured on the real-INT8 deployment. The first is **model size on disk**, which is the sum of the INT8 weights, the per-tensor weight scales, the per-tensor static activation scales, and the FP16 biases. This is the size the deployed model occupies in storage and the size that has to be transferred to GPU memory at load time. The second is **peak GPU memory** during a forward pass at sequence length 2048, measured via the CUDA memory allocator's peak-tracking interface and including weights, KV-cache, attention buffers, and any temporaries. This is the operational quantity that decides whether the model fits on a given device. The third is **WikiText-2 perplexity** at the same sequence length, evaluated on the standard 287k-token validation set, which closes the loop between the memory claim and the accuracy claim — there is no point in halving the memory if the perplexity has gone up by an order of magnitude. The same three quantities are reported for the FP16 baseline and the upstream-recipe INT8 baseline, so all three numbers — proposed, paper, and FP16 — are directly comparable on the same machine and the same evaluation harness.

## 3.6 Summary of the proposed pipeline

The full pipeline can be stated end to end in five steps.

1. **Per-channel quantile calibration.** Run one offline pass over the Pile validation set (512 sentences × 512 tokens) with forward hooks on every smoothing-relevant linear input (`q_proj` input, `fc1` input). Maintain a per-channel exact top-K buffer of size `K = ⌈(1 - p_{\min}) \cdot N_{\text{total}}⌉` per layer, with `p_{\min} = 0.90`. Read off `Q_p(|X_{:,j}|)` for every requested `p \in \{0.999, 0.995, 0.99, 0.95, 0.90\}` and every channel `j`. Save one file per `p` value to disk.

2. **Per-layer severity profile.** From the same calibration, compute the per-layer severity ratio `σ(l) = \max_j \mathrm{stat}_j / \mathrm{median}_j \mathrm{stat}_j`, where `\mathrm{stat}_j` is the per-channel statistic used for smoothing (here `\max_t |X_{t,j}|`, but the choice is regime-independent). Define the parametric `α(l) = α_{\min} + (α_{\max} - α_{\min}) \cdot \sigma(l) / \max_l σ(l)` with `α_{\min} = 0.5`, `α_{\max} = 0.9`.

3. **Smoothing factor.** For each layer `l` and each input channel `j`, build

   $$
   s_j^{(l)} = \frac{Q_p\big(|X^{(l)}_{:,j}|\big)^{\alpha(l)}}{Q_p\big(|W^{(l)}_{:,j}|\big)^{1-\alpha(l)}},
   $$

   and fold `s^{(l)}` into the parameters of the preceding LayerNorm or linear layer so that no scaling kernel runs at inference.

4. **Outer quantization.** Quantize the smoothed weights `\hat W^{(l)}` per output channel and the smoothed activations `\hat X^{(l)}` per token in the fake-quant regime. In the real-INT8 deployment regime, fall back to the O3 configuration (per-tensor W, per-tensor static A) and re-tune `α` to 0.5 while keeping `p` fixed at the fake-quant winner.

5. **Evaluation.** Report WikiText-2 perplexity at sequence length 2048 against the FP16 ceiling, the W8A8-naive floor, and the upstream `max`-based recipes O1 and O2. For the deployment configuration, additionally report model size on disk and peak GPU memory during the forward pass.

The pipeline makes no architectural changes to the model. It does not retrain or fine-tune any weights. It does not introduce any new operations at inference time — all changes are absorbed into the same offline calibration step that the upstream SmoothQuant recipe already uses. It is a pure post-training calibration intervention, layered on top of the same INT8 GEMM kernels and the same hardware that the upstream method targets. The accuracy and memory consequences of these changes, on the OPT scaling ladder from 125M to 13B, are the subject of the next two chapters.
