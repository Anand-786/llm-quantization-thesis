# Chapter 3 — Proposed Work

This chapter describes the proposed method in detail. The starting point is SmoothQuant [5]. Section 3.1 reviews it. The proposed method makes three changes to that construction. The first change is the outer quantization scheme. It is fixed to per-channel weights and per-token activations. The second change is the summary statistic that defines the smoothing factor. The per-channel maximum is replaced by a per-channel quantile. The third change is the migration strength. It is allowed to vary across the layers of the transformer. The three changes are orthogonal. Each can be enabled or disabled without the others.

## 3.1 The SmoothQuant foundation

SmoothQuant [5] is the foundation of this thesis. It is a post-training INT8 quantization method for large language models. The method addresses the central difficulty of LLM quantization. Activations carry heavy per-channel outliers; weights do not [4, 5]. A few outlier channels stretch the per-tensor quantization range. This leaves only 2–3 effective levels for the non-outlier channels [5, §3]. Per-channel activation quantization solves the problem statistically. But it maps poorly to INT8 GEMM kernels [5, Figure 3, Table 1]. SmoothQuant takes a different route. It migrates the per-channel difficulty from the activation onto the weight before quantization.

The construction relies on a mathematically equivalent transformation. For a linear layer `Y = X W` with `X ∈ R^{T × C_i}` and `W ∈ R^{C_i × C_o}`, the output is invariant under any positive per-channel rescaling `s ∈ R_{>0}^{C_i}`:

$$
Y = X W = \big(X \, \mathrm{diag}(s)^{-1}\big) \cdot \big(\mathrm{diag}(s) \, W\big) = \hat X \, \hat W.
$$

The output of the linear layer does not change. What changes is the distribution of the two operands. If `s` is chosen so that the channels of `\hat X` are flatter than those of `X`, a coarse quantizer applied to `\hat X` loses less information. The matching shift on the weight side makes some channels of `\hat W` larger. The weight tensor becomes a little harder to quantize. The trade-off is profitable because weights are flat to begin with [4, 5, Figure 4]. The redistribution improves the joint quantization error.

The smoothing factor `s` is fixed once at calibration and never recomputed. SmoothQuant defines the j-th component of `s` by

$$
s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}, \qquad j = 1, \ldots, C_i,
$$

where the activation maximum is taken over the tokens of the calibration sample. The calibration sample is 512 random sentences of length 512 from the Pile validation set [7]. The factor `s` is then fused into the parameters of the preceding LayerNorm or linear layer. No extra scaling kernel runs at inference [5, §4].

The exponent `α ∈ [0, 1]` is the migration strength. It controls how much of the per-channel difficulty is moved from `\hat X` onto `\hat W`. At `α = 0` no difficulty is moved. At `α = 1` all of it is moved to the weight side. The authors of [5] find a global `α` per model by a small grid search on the Pile validation set. They report `α = 0.5` for the OPT and BLOOM families and `α = 0.75` for GLM-130B. Table 7 of [5] lists the per-model values for the more recent architectures: 0.85 for Llama-2-7B and 13B, 0.9 for Llama-2-70B, 0.6 for Falcon-7B, 0.7 for Falcon-40B, 0.8 for Mistral-7B, and 0.8 for Mixtral-8x7B.

The transformation only touches two linear inputs per Transformer block. The first is the input to the attention `q_proj`. This input is shared with `k_proj` and `v_proj` because all three see the same hidden state coming out of LayerNorm. The second is the input to the FFN `fc1`. The other two linear layers in a block are the attention output projection and the FFN second projection. These see post-attention and post-activation tensors. They are not used in the smoothing formula. The smoothed weights `\hat W` and activations `\hat X` are then quantized to INT8. All GEMMs in self-attention and the feed-forward block run on hardware INT8 kernels.

## 3.2 Per-channel weight, per-token activation quantization

The smoothing transformation deliberately moves variation from `X` into `W`. After calibration, the smoothed activation `\hat X` is much flatter across its channels than `X` was, but the smoothed weight `\hat W` is more variable across its rows than `W` was. Larger values of the migration strength `α` push more variation onto the weight side, by construction.

The three quantization schemes defined in [5] — O1, O2 and O3 — all use **per-tensor** weight quantization. A single step size `\Delta_W` is computed from the global maximum magnitude of the entire `\hat W` matrix, and every channel of `\hat W` is then represented relative to that one step. Channels whose typical magnitude is much smaller than the global maximum get squeezed into a small number of effective integer levels — this is exactly the *low-effective-quantization-bits* phenomenon that [5] uses to motivate smoothing on the activation side in the first place (§3 and Figure 2 of [5]), now reproduced on the weight side because the smoothing has redistributed the variation there.

The natural countermeasure is per-channel weight quantization, in which the step size is not a single scalar but a vector with one entry per output channel:

$$
\Delta_W \in \mathbb{R}^{C_o}, \qquad \Delta_{W, k} = \frac{\max_i |\hat W_{i, k}|}{2^{N-1} - 1}, \quad k = 1, \ldots, C_o.
$$

Each output channel of `\hat W` then uses its own range, and the additional variation introduced by smoothing is absorbed channel by channel rather than averaged into one global scale. As [5] itself shows in Table 1, simulated per-channel activation quantization is the only scheme that closes the gap to FP16 across the OPT 6.7B–175B scale; the same argument applies symmetrically to the weight side once smoothing has pushed difficulty onto it.

Per-channel weight quantization remains fully compatible with the same INT8 GEMM kernels that [5] targets. The argument is the one [5] makes in §3 and Figure 3: in a linear matmul `Y = X W` of shape `(T, C_i) \times (C_i, C_o) = (T, C_o)`, scaling factors can be applied along the *outer* dimensions of the matrix multiplication — the token dimension `T` of the activation and the output-channel dimension `C_o` of the weight — without being inserted into the integer accumulation that runs inside the kernel. Per-token activation quantization along `T` is the choice already adopted by SmoothQuant-O1; per-channel weight quantization along `C_o` is its symmetric counterpart on the weight side. Both rescalings are post-GEMM vector multiplies on the output and add only negligible runtime overhead. The full output of a per-channel-weight, per-token-activation INT8 GEMM is

$$
Y_{t, k} = \Delta_{X, t} \cdot \Delta_{W, k} \cdot \sum_{i=1}^{C_i} \bar X_{t, i} \cdot \bar W_{i, k},
$$

where `\bar X` and `\bar W` are the INT8-quantized operands, `\Delta_{X, t}` is the per-token activation step (a vector of length `T`), and `\Delta_{W, k}` is the per-channel weight step (a vector of length `C_o`). The summation is the integer GEMM that the kernel runs; both rescalings are outer fix-ups applied to the result.

The original SmoothQuant paper does not adopt this scheme for its OPT or BLOOM headline numbers. The three schemes it benchmarks across the OPT 1.3B–175B scaling ladder (Table 2, Figure 7 and Table 3 of [5]) all use per-tensor weight quantization. Per-channel weight quantization is used only implicitly in the paper's later results on Llama-2, Falcon, Mistral and Mixtral (Table 7 of [5]), where it is mentioned in passing — "we used per-token activation quantization and per-channel weight quantization for SmoothQuant" — without being identified as a deliberate design choice or compared directly against the per-tensor schemes that dominate the rest of the paper. To the best of this thesis's knowledge, no previous work in the SmoothQuant family has explicitly named the per-channel-weight + per-token-activation configuration on OPT, swept the migration strength `α` under it, or compared its `α`-stability against O1 and O2 across the OPT scaling ladder. The first part of the proposed work makes that comparison explicit. The configuration is named "C" throughout the experiments in this thesis, alongside the existing O1 and O2 labels for the per-tensor-weight schemes from [5].

Per-token activation quantization can be either dynamic — `\Delta_{X, t}` is recomputed at every forward pass from the running activation — or static, with `\Delta_X` calibrated once on the same Pile sample used for `s`. The fake-quantization experiments in Chapter 4 use dynamic per-token quantization, which keeps the activation step always faithful to the data and matches the SmoothQuant-O1 setting in Table 2 of [5]. Real-INT8 deployment in Chapter 5 uses static per-tensor activation quantization, because the CUTLASS INT8 GEMM kernels exposed through the torch-int library [referenced in the SmoothQuant repository] only support per-tensor static activations and per-tensor weights on OPT. The change in regime from "fake-quant per-channel-weight + per-token-activation" to "real-INT8 per-tensor-weight + per-tensor-static-activation" is documented in §3.5.

## 3.3 Quantile-based calibration

SmoothQuant solves the inter-channel outlier problem. A few activation channels are much larger than the rest, and these dominate the per-tensor quantization range [5, §3]. The smoothing factor `s` flattens the channels and the problem is absorbed into the weights. But within a single channel, the activation distribution is itself not uniform. A handful of tokens can be much larger than the typical magnitude of that channel. The original method does not handle this intra-channel outlier problem in its mathematical formulation. The per-channel maximum `\max_t |X_{t,j}|` that defines `s_j` is fixed once on the calibration sample. One anomalously large token in that sample can set `s_j` for the entire deployment.

The SmoothQuant authors run into this themselves on GLM-130B (§5.2 of [5]). They patch it by clipping the top 2% of tokens before computing the static quantization step sizes, citing [8]. The clip works but it is ad hoc. It lives at the quantization step, not at the smoothing step. It has to be tuned per model. It is presented in the paper as a workaround, not as part of the method.

The same fix can be built directly into the smoothing formula. If the summary statistic that defines `s_j` is itself robust to a handful of extreme tokens, no separate clipping step is needed. The natural replacement for the per-channel maximum is a per-channel quantile. A quantile at a probability `p` slightly below 1 ignores the very top of the channel's distribution and reports the typical magnitude instead of the spike.

The upstream formula in [5] uses the per-channel maximum on both sides of the ratio:

$$
s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}.
$$

The proposed formula replaces both maxima with a per-channel quantile `Q_p` at the same fixed probability `p`:

$$
s_j = \frac{Q_p\big(|X_{:,j}|\big)^{\alpha}}{Q_p\big(|W_{:,j}|\big)^{1-\alpha}}, \qquad p \in (0, 1].
$$

`Q_p(\cdot)` is the empirical quantile of the input set, defined in the same convention as `torch.quantile`. It is the value `v` such that a fraction `p` of the elements are `≤ v`, with linear interpolation between the two adjacent ranked elements when `p \cdot (N - 1)` is not an integer. `\max(|X_j|)` is the per-channel maximum over all tokens of the calibration sample. `\max(|W_j|)` is the per-channel maximum over the input-feature dimension of the weight. `α ∈ [0, 1]` is the migration strength from [5]. At `p = 1` the proposed formula reduces to the upstream formula exactly. For `p < 1` it ignores the very top of each channel's distribution. The same `p` is used on both sides of the ratio.

Computing `Q_p` on activations naively is not feasible. It would require storing every absolute activation value seen during calibration and calling `torch.quantile` per channel at the end. For OPT-13B at 512 sentences of 512 tokens, the buffer for a single layer alone would be 2.5 GB in fp16. The network has dozens of such layers and the total runs into hundreds of GB.

The approach used here is a per-channel top-K buffer. For each smoothing-relevant linear input, the calibrator keeps the `K = \lceil (1 - p_{\min}) \cdot N_{\text{total}} \rceil` largest absolute values seen so far for each channel, in sorted order. `N_{\text{total}}` is the total number of tokens observed during calibration. The buffer is updated after every forward pass. The new batch's absolute values are concatenated with the buffer along the token axis. `torch.topk` is then taken to retain only the top `K` per channel. At the end of calibration, the empirical quantile at any `p \geq p_{\min}` can be read off directly by indexing into the sorted buffer at rank `\lceil p \cdot (N_{\text{total}} - 1) \rfloor`. The result is exact: it is identical to what a full-storage `torch.quantile` call would have returned.

This is done in the same calibration pass that already generates the upstream `act_scales` files. No extra forward pass over the calibration corpus is needed. Forward hooks are attached to the input of `q_proj` and the input of `fc1` in every Transformer block. These are exactly the two linear inputs that the upstream `generate_act_scales.py` script already hooks [5, §4]. The hooks write one `.pt` file per requested `p` value at the end of calibration. The cost is the buffer memory during calibration: around 5 GB on OPT-1.3B, 13 GB on OPT-6.7B and 21 GB on OPT-13B in fp16. This is far less than storing every token would cost. No additional cost is incurred at inference. The smoothing factor `s` is fused into the preceding LayerNorm exactly as in [5]. The real-INT8 deployment runs in Chapter 5 confirm this directly.

The lower bound `p_{\min}` is set to 0.90 in the implementation. This gives `K \approx 0.10 \cdot N_{\text{total}}` and admits every `p \in [0.90, 1]` exactly from the same buffer. One calibration pass therefore produces five `.pt` files at `p \in \{0.90, 0.95, 0.99, 0.995, 0.999\}` from a single top-K buffer.

A histogram-based estimator was considered as an alternative. It was rejected in early tests because the top bin reports a midpoint instead of the actual peak. The resulting estimate is too noisy to resolve the within-sweep PPL deltas of order 0.01–0.1.

These quantile-based activation-scale files are the input to the next step of the proposed method. The smoothing factor `s` is rebuilt from `Q_p` instead of `\max`, and the resulting smoothed model is evaluated against the upstream `\max`-based baseline in Chapter 4. The goal is to improve PPL and accuracy over naive W8A8 quantization [5, Table 3].

## 3.4 Per-layer migration strength

The migration strength `α` is the central knob of SmoothQuant. The original paper picks one global `α` per model by a grid search on the Pile validation set [5, §5.1]. The per-model values are listed in §3.1. Each value closes the gap from naive W8A8 to FP16 on that one model. The search has to be rerun for every new model. This is an extra computation cost on top of the calibration pass itself.

This cost can be removed by looking at the outlier severity across the layers of a model. The severity is already exposed in the per-channel `max(|X|)` data that the calibration pass produces. No new forward pass is needed. The proposed method derives `α` per layer from this severity, with no grid search over the model as a whole.

The figure in Chapter 4 plots the per-layer severity ratio

$$
\sigma(l) = \frac{\max_j \, \big[\max_t |X^{(l)}_{t,j}|\big]}{\mathrm{median}_j \, \big[\max_t |X^{(l)}_{t,j}|\big]}
$$

for OPT-2.7B, OPT-6.7B and OPT-13B. `σ(l)` is the ratio of the most outlier-heavy channel to the median channel within layer `l`. The shape of `σ(l)` is consistent across the three model sizes. Layer 0 has low severity. Layers 1–3 ramp sharply to a peak. The peak grows with model size: roughly 36 at 2.7B, 60 at 6.7B and 82 at 13B. The remaining layers decay smoothly and near-monotonically to a tail value of 10–15. The profile is clearly not flat. A layer at the peak needs aggressive smoothing. A layer at the tail does not. A single global `α` under-smooths the hard layers and over-smooths the easy ones. This is what motivates a variable migration strength across the layers of the model.

The proposed method maps the severity to a per-layer `α(l)` through a simple linear equation:

$$
\alpha(l) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \tilde \sigma(l), \qquad \tilde \sigma(l) = \frac{\sigma(l)}{\max_l \sigma(l)}.
$$

`\tilde \sigma(l) \in [0, 1]` is the normalised severity within the model. `α_{\min}` and `α_{\max}` bound the migration strength below and above. The layer with the highest severity gets `α(l) = α_{\max}`. The layer with the lowest severity gets `α(l) = α_{\min}`. All other layers interpolate linearly. The values used in this work are `α_{\min} = 0.5` and `α_{\max} = 0.9`. These two scalars replace the per-model grid search of [5]. No tuning is performed on the evaluation set.

The per-layer values `α(l)` are computed once at the end of the calibration pass and saved as a single vector on disk, alongside the activation-scale files. The vector has one entry per Transformer block, keyed by layer index `l`. At inference time, the smoothing factor for layer `l` is built as

$$
s_j^{(l)} = \frac{\max(|X^{(l)}_j|)^{\alpha(l)}}{\max(|W^{(l)}_j|)^{1-\alpha(l)}},
$$

using the per-layer `α(l)` from the stored vector. The factor `s^{(l)}` is fused into the parameters of the preceding LayerNorm of layer `l`, exactly as in the upstream method [5, §4]. No extra operation is added to the forward pass. The per-layer granularity of `α` lives entirely in the offline calibration step. The inference path is unchanged.

## 3.5 Real INT8 deployment

Simulated quantization is the standard way to evaluate a quantization method. The operands are rounded to their INT8 grid before each matmul. The matmul itself still runs in FP16. This lets a new method be tested for its effect on PPL and accuracy. No fused INT8 kernel has to be built for every scheme variation first. This is the setting in which [5] reports its accuracy numbers [5, §5.2]. It is also the setting used for every accuracy comparison in this thesis.

Real INT8 deployment is what realises the memory and latency benefit. The linear layers are replaced with fused INT8 GEMM kernels. The activations are converted to INT8 at the boundary of each linear layer. The weights are stored in INT8. The matmul runs on the GPU's integer Tensor Cores. The paper reports up to 1.96× memory saving and 1.51× speedup at this step [5, §5.3, Figure 8]. We expect to measure close to 50% memory savings on the OPT models deployed in Chapter 5.

The torch-int CUTLASS INT8 GEMM kernels available for OPT only support the O3 scheme from [5]: per-tensor weight quantization and per-tensor static activation quantization, with the activation step folded into the kernel's bias [5, Table 2]. The proposed quantile method is plugged into O3 by rebuilding `s` with `Q_p` instead of `\max` at the calibration step. The matched paper baseline uses the upstream `\max`-based `s` under the same O3 scheme. The comparison isolates the smoothing-statistic change.

A small extra accuracy drop is expected here compared to the simulated setting. Per-tensor static activation scales cannot adapt to the token distribution at runtime. The rounding error accumulates across the matmul. The gap to FP16 should be larger than under simulated quantization. It should still be much smaller than the gap from naive W8A8.

## 3.6 Summary of the proposed method

The full method can be stated end to end in five steps.

1. **Per-channel quantile calibration.** Run one offline pass over the Pile validation set (512 sentences × 512 tokens) with forward hooks on every smoothing-relevant linear input (`q_proj` input, `fc1` input). Maintain a per-channel exact top-K buffer of size `K = ⌈(1 - p_{\min}) \cdot N_{\text{total}}⌉` per layer, with `p_{\min} = 0.90`. Read off `Q_p(|X_{:,j}|)` for every requested `p \in \{0.999, 0.995, 0.99, 0.95, 0.90\}` and every channel `j`. Save one file per `p` value to disk.

2. **Per-layer severity profile.** From the same calibration, compute the per-layer severity ratio `σ(l) = \max_j \mathrm{stat}_j / \mathrm{median}_j \mathrm{stat}_j`, where `\mathrm{stat}_j` is the per-channel statistic used for smoothing (here `\max_t |X_{t,j}|`, but the choice is regime-independent). Define the parametric `α(l) = α_{\min} + (α_{\max} - α_{\min}) \cdot \sigma(l) / \max_l σ(l)` with `α_{\min} = 0.5`, `α_{\max} = 0.9`.

3. **Smoothing factor.** For each layer `l` and each input channel `j`, build

   $$
   s_j^{(l)} = \frac{Q_p\big(|X^{(l)}_{:,j}|\big)^{\alpha(l)}}{Q_p\big(|W^{(l)}_{:,j}|\big)^{1-\alpha(l)}},
   $$

   and fold `s^{(l)}` into the parameters of the preceding LayerNorm or linear layer so that no scaling kernel runs at inference.

4. **Outer quantization.** Quantize the smoothed weights `\hat W^{(l)}` per output channel and the smoothed activations `\hat X^{(l)}` per token in the fake-quant regime. In the real-INT8 deployment regime, fall back to the O3 scheme (per-tensor W, per-tensor static A) and re-tune `α` to 0.5 while keeping `p` fixed at the fake-quant winner.

5. **Evaluation.** Report WikiText-2 perplexity at sequence length 2048 against the FP16 ceiling, the W8A8-naive floor, and the upstream `max`-based schemes O1 and O2. For the deployment configuration, additionally report model size on disk and peak GPU memory during the forward pass.

The proposed method makes no architectural changes to the model. It does not retrain or fine-tune any weights. It does not introduce any new operations at inference time — all changes are absorbed into the same offline calibration step that the upstream SmoothQuant method already uses. It is a pure post-training calibration intervention, layered on top of the same INT8 GEMM kernels and the same hardware that [5] targets. The accuracy and memory consequences of these changes, on the OPT scaling ladder from 125M to 13B and on Llama-2 7B/13B and Falcon-7B, are the subject of the next two chapters.
