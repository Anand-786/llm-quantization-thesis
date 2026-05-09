# Chapter 1 — Introduction

## 1.1 Background

The last five years have seen language models grow from hundreds of millions to hundreds of billions of parameters, accompanied by a corresponding leap in their capabilities on language understanding, reasoning, and code generation tasks. GPT-3 (Brown et al., 2020) crossed the 100B mark and demonstrated emergent few-shot learning; OPT (Zhang et al., 2022), BLOOM (Scao et al., 2022), GLM-130B (Zeng et al., 2022), and MT-NLG 530B (Smith et al., 2022) followed in the same regime, and the Llama (Touvron et al., 2023) and Mistral (Jiang et al., 2023) families have since pushed open-weight access to comparable scales. Capability has scaled with parameter count, but so has the cost of using these models — and that cost has not scaled with the memory of the GPUs that have to host them. Figure 1 of Xiao et al. (2023) makes the point quantitatively: between 2018 and 2022 model size grew by more than three orders of magnitude while accelerator memory grew by less than one. A GPT-3-class model in FP16 occupies roughly 350 GB of weights alone, requiring eight A100-80GB or A6000-48GB cards just to fit, before any activations, KV-cache, or batching headroom is taken into account.

This work approaches the problem from a hardware–software co-design perspective: rather than building larger models or faster accelerators, the question is how to bring the inference cost of *existing* models down so that they can be served on the hardware that already exists, including resource-constrained edge devices such as mobile phones and embedded SoCs where memory budgets are measured in single-digit gigabytes. **Quantization** — replacing high-precision floating-point tensors with low-bit integer counterparts — is one of the few techniques that simultaneously addresses memory footprint, memory bandwidth, and arithmetic throughput, and does so by exploiting hardware that is already deployed at scale: NVIDIA Tensor Cores, Intel AMX units, ARM dot-product instructions, and Qualcomm DSPs all expose native INT8 GEMM kernels that deliver roughly twice the throughput of FP16 at half the memory traffic.

Within quantization, two broad families exist. **Quantization-aware training (QAT)** retrains or fine-tunes the model with simulated low-precision arithmetic in the loop and tends to recover accuracy well, but at a cost that is often comparable to pre-training itself — prohibitive for models with tens or hundreds of billions of parameters. **Post-training quantization (PTQ)**, by contrast, freezes the trained model and quantizes it in a single calibration pass on a small held-out corpus. PTQ is therefore the only practical option at LLM scale, and is the setting of this thesis.

The standard form of integer quantization used throughout this work, including by SmoothQuant, is symmetric uniform quantization (Jacob et al., 2018):

$$
\bar X^{\text{INT8}} = \left\lceil \frac{X^{\text{FP16}}}{\Delta} \right\rfloor, \qquad \Delta = \frac{\max(|X|)}{2^{N-1} - 1}
$$

with `N = 8` bits and a step size `Δ` derived from the maximum absolute magnitude of the tensor. The granularity at which `Δ` is computed defines the *scheme*: per-tensor uses one step for the entire matrix, per-token uses one per row of the activation, and per-channel uses one per output channel of the weight. Coarser schemes are cheaper to implement and map most cleanly to vendor INT8 GEMM kernels; finer schemes give each channel or token its own range and can therefore tolerate heterogeneous distributions, at the cost of additional scaling work that must be fused into the matmul. Calibration may be **static** (statistics collected once on a held-out set) or **dynamic** (recomputed at every forward pass).

### 1.1.1 The activation-memory bottleneck

A naive expectation when quantizing weights from FP16 to INT8 is a 50 % reduction in peak GPU memory at inference, since weights are halved in storage. Empirical inspection of LLM inference quickly contradicts this. Across model scales — and increasingly so as the model grows — the peak memory of an inference forward pass is dominated not by the weights but by **activation memory**: the intermediate tensors materialised inside attention and feed-forward layers, including the query/key/value projections, the attention probability matrix, the FFN hidden state at four times the model dimension, and the KV-cache that grows with sequence length and batch size. Reducing only the precision of the *weights* therefore yields a peak-memory saving that shrinks as scale grows, because the unquantized activations come to dominate the budget. This observation is the empirical foundation of the **W8A8** setting — quantizing both weights and activations to INT8 — which is the setting of LLM.int8() (Dettmers et al., 2022), SmoothQuant (Xiao et al., 2023), and this thesis. Quantizing activations is also what unlocks the use of INT8 GEMM kernels at all: the matrix-multiply operands must both be INT8 for the integer Tensor Cores to be invoked, otherwise the kernel falls back to FP16 with on-the-fly dequantization.

### 1.1.2 Activation outliers and the failure of naive W8A8

Unfortunately, activations in LLMs are far harder to quantize than weights. Whereas trained weight distributions are flat and approximately uniform across channels — and tolerate INT8 or even INT4 representation with negligible accuracy loss — activations exhibit a small number of **outlier channels** whose magnitudes are systematically about two orders of magnitude larger than the rest. These outliers were systematically characterised by Dettmers et al. (2022) in the LLM.int8() paper and have two properties that are central to the rest of this thesis:

1. **Outliers are confined to a small fraction of channels.** At 6.7B parameters and beyond they emerge in roughly 0.1 % of feature dimensions, but those channels carry information that is critical to model accuracy — clipping or zeroing them destroys downstream performance.
2. **Outliers are persistent within a channel across tokens.** If a channel is an outlier channel, it is large for *every* token; the variance along the token axis within a single channel is small, while the variance across channels for a single token is enormous.

Under per-tensor activation quantization, the maximum magnitude entering `Δ` is set by these outliers. If a channel's typical magnitude is `m_i` and the global maximum is `m`, the effective number of quantization levels available to that channel reduces to `2^8 · m_i / m`, which for non-outlier channels can fall to two or three levels. The result is the catastrophic accuracy collapse reported by Dettmers et al. and reproduced by Xiao et al. for OPT-175B: zero-shot average accuracy falls from 71.6 % (FP16) to 32.3 % (naive W8A8 per-tensor), close to the random-guessing floor.

### 1.1.3 Prior approaches: LLM.int8() and ZeroQuant

**LLM.int8()** (Dettmers et al., 2022) addresses the outlier problem by mixed-precision decomposition: it identifies the outlier channels at runtime and routes them through an FP16 sub-matmul, while the remaining channels are computed in INT8. This preserves accuracy across all tested scales, but the decomposition introduces a non-uniform compute path that does not map well to monolithic INT8 GEMM kernels — in practice, LLM.int8() inference is reported to be slower than the FP16 baseline it was meant to accelerate. **ZeroQuant** (Yao et al., 2022) instead uses very fine-grained quantization — per-token activations and group-wise weights — implemented through custom CUDA kernels. It works for models up to roughly 20B parameters but cannot maintain accuracy at OPT-175B scale, where the outlier magnitudes are most severe. Neither approach, then, achieves the desirable point: training-free, fully INT8 on the compute-intensive operators, accuracy-preserving across scales, and faster than FP16.

### 1.1.4 SmoothQuant: migrating quantization difficulty

SmoothQuant (Xiao et al., 2023), which forms the basis of this thesis, achieves that point by reframing the outlier problem as one of *distribution shape* rather than *value range*. The key observation, building on Dettmers' channel-persistence finding, is that activation outliers are stable per channel but variable across channels — exactly the pattern that per-channel scaling can normalise. Per-channel *activation* quantization is, however, infeasible in INT8 GEMM kernels because scaling along the inner contraction dimension of the matmul cannot be fused with the integer arithmetic.

Instead, SmoothQuant migrates the per-channel scale variance from the activations into the weights, offline, before quantization. For a linear layer `Y = X W`, the identity

$$
Y = (X \, \mathrm{diag}(s)^{-1}) \cdot (\mathrm{diag}(s) \, W) = \hat X \, \hat W
$$

holds for any non-zero per-channel vector `s ∈ R^{C_i}`. Choosing `s` so that `\hat X` is much flatter than `X` makes the smoothed activation easy to quantize at coarse granularity, while the adjusted weight `\hat W` becomes only modestly harder to quantize than the original `W`. Concretely, SmoothQuant uses

$$
s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}, \qquad j = 1, \dots, C_i,
$$

with `\max(|X_j|)` estimated on roughly 512 calibration sentences from the Pile validation set, and a single migration-strength hyperparameter `α ∈ [0, 1]` controlling how much of the per-channel difficulty is pushed from activations onto weights. The smoothing factor `s` is folded offline into the preceding LayerNorm or linear layer, so it incurs no kernel-call overhead at inference. The paper reports that `α = 0.5` is a sweet spot for the OPT and BLOOM families, while GLM-130B requires `α = 0.75` due to more pronounced activation outliers.

On top of this transformation the paper defines three quantization configurations of progressively higher efficiency:

- **O1**: per-tensor weight, per-token dynamic activation;
- **O2**: per-tensor weight, per-tensor dynamic activation;
- **O3**: per-tensor weight, per-tensor static activation.

All three configurations use per-tensor weights and quantize the linear layers and attention BMMs to INT8, leaving only LayerNorm, softmax, and ReLU in FP16. Across OPT, BLOOM, GLM-130B, and the 530B-parameter MT-NLG, SmoothQuant matches FP16 perplexity and zero-shot accuracy to within a fraction of a percentage point while delivering a measured 1.5× end-to-end speedup and a 2× memory saving on PyTorch and FasterTransformer. It is the strongest training-free, fully-INT8 baseline available at the time of writing, and is therefore the natural starting point for the present work.

## 1.2 Motivation

SmoothQuant's design has two clearly separated stages. The **outer stage** is the W8A8 quantization itself — the choice of granularity for the weight tensor and the activation tensor. The **inner stage**, which precedes the outer one, is a one-shot *calibration*: a per-channel smoothing factor `s` is computed from a small Pile sample and folded into the network so that the activations the outer stage sees are already flattened. This factorisation is convenient because it means each stage can be examined independently, and the rest of the thesis is organised around two principled improvements — one to each stage — that follow from the structure of the outlier problem itself.

### 1.2.1 Finer-grained weight quantization on the smoothed weights

The first opportunity sits at the outer stage. The whole point of the smoothing transformation is to *migrate* per-channel difficulty from activations onto weights: after smoothing, the weight tensor `\hat W` carries variation it did not previously carry, and at higher values of `α` it carries proportionally more of it. The natural countermeasure to per-channel variation is per-channel granularity, and Dettmers' observation that outliers persist along channels — the same observation that originally motivated the smoothing direction — applies just as forcefully to the migrated channels of `\hat W`.

The three configurations defined in the SmoothQuant paper (O1, O2, O3) all quantize weights at **per-tensor** granularity. A single step size `Δ_W` is computed from the global maximum magnitude of `\hat W`, and every channel of the weight matrix is then represented relative to that one step size. Channels whose typical magnitude is much smaller than the global maximum are forced into a small number of effective integer levels — the same effective-bits problem that originally afflicted the activations, now reappearing on the weight side because the migration has concentrated more variation there. **Per-channel** weight quantization assigns one step size to each output channel and so eliminates this internal mismatch entirely: each channel uses its own range, and the residual variation introduced by smoothing is absorbed channel by channel rather than averaged into a single global scale.

Per-channel weight quantization is also fully compatible with the same INT8 GEMM kernels that the paper's O1–O3 schemes target. In a linear matmul `Y = X W` of dimension `T × C_i × C_o`, the output-channel dimension `C_o` is an *outer* dimension of the contraction; per-channel weight scaling along `C_o` can be applied as a vector multiply on the output of the integer GEMM, in exactly the same way per-token activation scaling is applied along `T`. This is the same "outer-dimension scaling" argument made in Figure 3 of Xiao et al. (2023) for per-token activation quantization, and it is the reason that per-channel weights add only negligible runtime overhead. The paper itself implicitly endorses this: the recipes it later reports for Llama-2, Falcon, Mistral, and Mixtral (Table 7) all use per-channel weight + per-token activation. The first level of modification proposed in this thesis simply makes that choice explicit, names it as a distinct configuration ("config C": per-channel weight + per-token activation), and studies how its accuracy and its sensitivity to the migration strength `α` compare to the per-tensor-weight schemes O1 and O2 across an OPT scaling ladder.

### 1.2.2 Outlier-robust calibration: from `max` to a percentile

The second opportunity sits at the inner stage. The smoothing factor

$$
s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}
$$

uses the **maximum absolute value** of each channel as its summary statistic. This choice is the simplest possible per-channel magnitude estimate, and on flat weight distributions it is harmless. On *activations*, however, `max` is exactly the statistic that the outlier problem was defined to break: a single anomalously large token in the calibration sample sets `\max(|X_j|)` for the entire deployment of the model. Because `s_j` enters the network multiplicatively and is fixed offline, any noise or single-sample bias in `\max(|X_j|)` is baked into every subsequent forward pass. The smoothing transformation is mathematically equivalent for any positive `s`, but its *quantization-friendliness* — the actual quantity it is supposed to optimise — is sensitive to whether the magnitudes used to construct `s` are representative of the channel's real distribution or of one extreme calibration token.

The SmoothQuant paper itself acknowledges this fragility in its experimental section. In §5.2, when applying the most aggressive O3 configuration to GLM-130B, the authors report that they "clip the top 2 % tokens when calibrating the static quantization step sizes", citing Wei et al. (2022). This top-2 % clip is an effective patch but an *ad hoc* one: it is applied at the quantization step rather than at the smoothing step, it requires per-model tuning, and it is presented as a workaround rather than as a principled component of the recipe. The phenomenon it patches — a few extreme tokens dominating a magnitude estimate that was supposed to summarise an entire channel — is precisely the same phenomenon that affects `max(|X_j|)` inside the smoothing formula.

A principled mathematical replacement is to substitute `max` with a high **per-channel quantile**:

$$
s_j = \frac{Q_p(|X_j|)^{\alpha}}{Q_p(|W_j|)^{1-\alpha}},
$$

where `Q_p(|X_j|)` denotes the empirical quantile of the absolute values of channel `j` at probability `p ∈ (0, 1]`. Setting `p = 1` recovers the original `max`-based formula exactly, so the percentile variant is a strict generalisation rather than a competing alternative. For `p` just below 1 (for example `p = 0.999` or `p = 0.99`), the statistic ignores the very top of the distribution — by construction the same kind of single-token spike that the GLM-130B clip was designed to neutralise — while still tracking the channel's true scale. Because `Q_p` is an offline calibration statistic computed in exactly the same one-shot pass as `max`, this modification adds nothing to inference cost and is fully orthogonal to the choice of outer scheme; the two improvements compose.

### 1.2.3 Where the contribution lives: mid-range models and edge deployment

A final point about scope. The headline results in Xiao et al. (2023) are reported at OPT-175B, BLOOM-176B, GLM-130B, and MT-NLG 530B — scales at which a 1.5× speedup translates into substantial absolute savings in datacentre inference cost. At those scales, however, the model still fundamentally requires a multi-GPU server to run at all; quantization changes the GPU count from sixteen to eight, but does not bring the model within reach of a single consumer device.

The deployment regime where W8A8 quantization changes the *qualitative* deployment story is the **mid-range** of model scales — OPT-1.3B through OPT-13B, and the modern equivalents at the 7B–13B mark. A 7B FP16 model needs roughly 14 GB of weight memory plus activation overhead, which sits awkwardly above the budget of consumer GPUs, smartphone NPUs, and laptop integrated graphics. The same model in INT8 fits comfortably in 7–8 GB and becomes runnable on hardware that ordinary users actually own. Improving the accuracy–memory tradeoff at this scale therefore has direct deployment value, even if the absolute model sizes are smaller than the ones the SmoothQuant paper used to demonstrate its method. This thesis works on the OPT family from 125M to 6.7B (with 13B as a stretch target) for two reasons: it is the family on which the SmoothQuant paper reports its primary OPT scaling results (Figure 7, Table 1), giving a clean basis for comparison; and it is also exactly the deployment regime where the proposed modifications are most useful in practice. Resource constraints during this work — single-GPU Colab T4 and Pro A100 instances — make the multi-hundred-billion-parameter scale infeasible, but the choice of the mid-range is one of relevance, not just of necessity.

## 1.3 Objectives

The thesis has three concrete objectives, each scoped to what can be defended on the OPT scaling ladder (125M, 1.3B, 2.7B, 6.7B, with 13B as a stretch target) using single-GPU Colab T4 and Pro A100 hardware:

1. **Compare quantization schemes on smoothed OPT models, including a per-channel-weight configuration alongside the paper's per-tensor-weight schemes O1 and O2.** Run the SmoothQuant transformation with each configuration under a common migration-strength sweep, report WikiText-2 perplexity across the model ladder, and characterise how the relative ordering of the schemes — and the value of `α` at which each is optimal — evolves with model size.

2. **Replace the `max`-based smoothing statistic with a per-channel percentile, and evaluate the resulting recipe.** Implement an exact (non-binned) per-channel quantile calibrator, sweep the percentile probability `p` jointly with `α` on the same OPT models, and quantify whether percentile-based smoothing reduces perplexity and broadens the stable region in `α` relative to the original `max`-based recipe. Verify that `p = 1` reproduces the upstream `max`-based behaviour, so that the modification is a strict generalisation of the paper's method.

3. **Validate the most promising compound configuration on standard zero-shot benchmarks.** Run the seven zero-shot tasks used in the SmoothQuant paper (LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA) for the FP16 ceiling, the W8A8-naive floor, the paper's O1 and O2 recipes, and the proposed percentile-on-per-channel-weight recipe across the OPT models that fit the available hardware, and report task-level evidence to complement the perplexity comparison.

The aim of the thesis is not to claim a new state-of-the-art across all model families, nor to push to the largest possible model scale. It is to take a careful, reproducible look at the SmoothQuant recipe on a fixed mid-range model family, identify two principled places where the recipe can be improved without leaving the training-free, fully-INT8, hardware-friendly envelope it was designed to enforce, and quantify the resulting accuracy gain.
