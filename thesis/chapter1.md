# Chapter 1 — Introduction

## 1.1 Background

Large language models have grown faster than the hardware that runs them. In the last five years, model sizes have moved from a few hundred million parameters to several hundred billion, while GPU memory has barely doubled in the same window [1, 2]. A GPT-3 scale model in FP16 needs around 350 GB just to hold its weights [1]. Serving it requires multiple high-end accelerators per replica, before any room is left for activations, KV-cache, or batching. The same gap shows up at smaller scales too: a 7B model in FP16 already exceeds the memory of most consumer GPUs and almost all mobile NPUs.

This makes inference, not training, the practical bottleneck for deploying these models. Training is a one-time cost paid by a few labs. Inference is paid every time a user sends a query, and it is what decides whether a model can run on a given device at all. Reducing the compute and memory cost of inference is therefore central to making these models usable outside large data centres — on laptops, phones, and embedded hardware where the memory budget is in single-digit gigabytes.

This thesis approaches the problem from the systems side. Rather than building larger models or waiting for larger GPUs, the goal is to reduce the inference cost of *existing* models so that they fit on hardware that already exists. **Quantization** — replacing high-precision floating-point tensors with low-bit integers — is one of the few techniques that addresses memory footprint, memory bandwidth, and arithmetic throughput at the same time. It also maps to hardware that is already deployed at scale: NVIDIA Tensor Cores, Intel AMX, ARM dot-product instructions, and Qualcomm DSPs all provide native INT8 GEMM kernels with roughly 2× the throughput and half the memory traffic of FP16.

There are two broad ways to quantize a model. **Quantization-aware training (QAT)** retrains the model with simulated low precision and recovers accuracy well, but its cost is comparable to pre-training itself. At LLM scale this is not a practical option. **Post-training quantization (PTQ)** keeps the trained weights frozen and quantizes them in a single calibration pass on a small held-out corpus. PTQ is the only practical setting at this scale, and it is the setting of this thesis.

The standard form of integer quantization used throughout this work is symmetric uniform quantization [3]:

$$
\bar X^{\text{INT8}} = \left\lceil \frac{X^{\text{FP16}}}{\Delta} \right\rfloor, \qquad \Delta = \frac{\max(|X|)}{2^{N-1} - 1}
$$

with `N = 8` bits. The *granularity* at which `Δ` is computed defines the scheme. Per-tensor uses one step size for the whole matrix. Per-token uses one per row of the activation. Per-channel uses one per output channel of the weight. Coarser schemes are cheaper and map directly to vendor INT8 GEMM kernels; finer schemes give each row or channel its own range but add scaling work that has to be fused into the matmul.

A naive expectation when moving weights from FP16 to INT8 is a 50% drop in peak GPU memory at inference. In practice this does not hold. As the model grows, the peak memory of a forward pass is dominated not by the weights but by the *activations* — the intermediate tensors materialised inside attention and feed-forward layers, including the Q/K/V projections, the attention probability matrix, the FFN hidden state, and the KV-cache. Quantizing only the weights gives a memory saving that gets smaller as the model gets larger, because the un-quantized activations grow to dominate the budget. This is the **W8A8** setting — both weights and activations in INT8 — and it is the setting of LLM.int8() [4], SmoothQuant [5], and this work. It is also what unlocks INT8 GEMM kernels at all: both operands have to be INT8 for the integer Tensor Cores to be invoked.

The difficulty with W8A8 is that activations are much harder to quantize than weights. Trained weights are flat and roughly uniform across channels, and tolerate INT8 (or even INT4) representation with little accuracy loss [4, 6]. Activations do not. Beyond around 6.7B parameters, a small number of **outlier channels** emerge whose magnitudes are roughly two orders of magnitude larger than the rest of the tensor [4]. Two properties of these outliers are central to the rest of this thesis [4]:

1. **Outliers are confined to a small fraction of channels.** Roughly 0.1% of feature dimensions, but those channels carry information that the model needs.
2. **Outliers are persistent within a channel.** If a channel is an outlier channel, it is large for *every* token. The variance along the token axis inside one channel is small, while the variance across channels for one token is huge.

Under per-tensor activation quantization, a single global `Δ` is set by the largest outlier in the tensor. If a non-outlier channel has typical magnitude `m_i` and the global maximum is `m`, the effective number of integer levels available to that channel is only `2^8 · m_i / m`, which can fall to two or three levels [5]. The result is the accuracy collapse reported by [4] and reproduced in [5]: on OPT-175B, naive W8A8 drops zero-shot average accuracy from 71.6% to 32.3%, near the random-guessing floor.

Two prior approaches tried to fix this. **LLM.int8()** [4] keeps the outlier channels in FP16 and routes the rest of the matmul through INT8. This preserves accuracy but breaks the monolithic INT8 GEMM path; the resulting mixed-precision kernel is in practice slower than the FP16 baseline it was meant to accelerate [5]. **ZeroQuant** [6] uses very fine-grained quantization — per-token activations and group-wise weights — through custom CUDA kernels. It works up to around 20B parameters but does not hold up at 175B scale.

**SmoothQuant** [5] is the method this thesis builds on, and it solves the problem differently. The key observation, building on the channel-persistence finding from [4], is that the outliers are stable *per channel* but variable *across* channels. That is exactly the pattern that per-channel scaling can normalise. Per-channel activation quantization itself is not feasible inside an INT8 GEMM, because scaling along the inner contraction dimension cannot be fused with the integer arithmetic [5]. So SmoothQuant migrates the per-channel difficulty *off the activations and into the weights*, offline, before quantization.

For a linear layer `Y = X W`, the identity

$$
Y = (X \, \mathrm{diag}(s)^{-1}) \cdot (\mathrm{diag}(s) \, W) = \hat X \, \hat W
$$

holds for any positive per-channel vector `s ∈ R^{C_i}`. Choose `s` so that `\hat X` is much flatter than `X`, and the smoothed activation becomes easy to quantize at coarse granularity. The adjusted weight `\hat W` becomes a little harder than the original `W`, but only a little, because the weight distribution started out flat. SmoothQuant uses

$$
s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}, \qquad j = 1, \dots, C_i,
$$

with `\max(|X_j|)` estimated on roughly 512 calibration sentences from the Pile validation set [7], and a single migration-strength `α ∈ [0, 1]` that controls how much of the per-channel difficulty is pushed from activations onto weights. The vector `s` is folded offline into the preceding LayerNorm or linear layer, so it costs nothing at inference time. The paper reports `α = 0.5` as a sweet spot for OPT and BLOOM, and `α = 0.75` for GLM-130B.

On top of this transformation, the paper defines three configurations of progressively higher efficiency [5]:

- **O1**: per-tensor weight, per-token dynamic activation;
- **O2**: per-tensor weight, per-tensor dynamic activation;
- **O3**: per-tensor weight, per-tensor static activation.

All three use per-tensor weights. Across OPT, BLOOM, GLM-130B, and the 530B-parameter MT-NLG, SmoothQuant matches FP16 perplexity and zero-shot accuracy to within a fraction of a percent, and reports up to 1.5× speedup and roughly 2× memory saving [5]. It is the strongest training-free, fully-INT8 baseline available, and is the natural starting point for the present work.

## 1.2 Motivation

SmoothQuant is built around a single per-channel calibration statistic — the maximum absolute value of each input channel — and a single migration strength `α` that is shared across the whole network. Both of those design choices are clean, but both are also coarser than what the underlying outlier structure would suggest. The motivation for this thesis comes from looking carefully at what those outliers actually look like, and noticing that the paper's own recipe leaves room for two principled refinements.

### 1.2.1 The shape of the outlier problem

Dettmers et al. [4] established that outliers are concentrated in a small set of channels and that, within such a channel, they are present for every token. This is what makes per-channel smoothing work in the first place: the variation between channels is large, but the variation along the token axis within a single channel is small, so a single per-channel scale is enough to flatten the activation tensor.

Two observations in the present work qualify that picture. The first is on the calibration side. When the per-channel maximum is computed not from a handful of tokens but from the whole calibration buffer, a non-trivial fraction of the channels show a clear gap between their typical magnitude and their top-most few activations. Inside a channel that is *not* an outlier channel, most tokens sit near the channel's median, but a few isolated tokens spike well above. These intra-channel spikes are not the inter-channel outliers that motivated SmoothQuant; they are a secondary effect, but they enter the smoothing factor through `\max(|X_j|)` and end up dictating `s_j` for every subsequent forward pass. The SmoothQuant authors themselves run into this on GLM-130B and patch it by clipping the top 2% of tokens before computing static quantization step sizes, citing [8] (§5.2 of [5]). The clip is effective but ad hoc: it lives at the quantization step rather than the smoothing step, it is tuned per model, and it is presented as a workaround rather than as a principled part of the recipe. The phenomenon it patches is exactly the same one that affects `\max(|X_j|)` inside the smoothing formula.

The second observation is on the network side. Outlier severity is not constant across the depth of the model. Across OPT-2.7B, OPT-6.7B, and OPT-13B, the ratio between the most outlier-heavy layer and the cleanest layer of the same model grows from roughly 13× at 2.7B to over 40× at 13B. Different layers therefore live in different regimes: a layer with mild activation outliers is best served by a small `α` that leaves the weights largely untouched, while a layer with very heavy outliers benefits from a much larger `α` that shifts more of the difficulty onto its weights. A single global `α`, the configuration that all of the SmoothQuant headline numbers use, has to compromise between these two regimes.

These two observations point to two distinct gaps in the recipe — one at the calibration step that produces `s`, and one at the per-layer level above it. Both relate to the same underlying fact: the outlier distribution in real LLM activations is more structured than a single global statistic can capture.

### 1.2.2 Quantile-based smoothing

The first refinement is on the calibration statistic. The smoothing formula

$$
s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}
$$

uses the maximum absolute value of each channel as its summary. This is the simplest possible per-channel magnitude estimate, and on flat weight distributions it works fine. On activations it does not, for the reason described above: a single anomalously large token in the calibration buffer sets `\max(|X_j|)` for the entire deployment of the model, and `s_j` enters the network multiplicatively and is fixed offline. Any noise or single-sample bias in `\max(|X_j|)` is therefore baked into every subsequent forward pass.

A principled replacement is to substitute `\max` with a high per-channel **quantile**:

$$
s_j = \frac{Q_p(|X_j|)^{\alpha}}{Q_p(|W_j|)^{1-\alpha}},
$$

where `Q_p(|X_j|)` is the empirical quantile of the absolute values of channel `j` at probability `p ∈ (0, 1]`. Setting `p = 1` recovers the original `\max`-based formula exactly, so the quantile-based variant is a strict generalisation rather than a competing alternative. For `p` slightly below 1 (for example 0.999 or 0.99), the statistic ignores the very top of the distribution — the same kind of single-token spike that the GLM-130B clip was designed to neutralise — while still tracking the channel's true scale. The calibration cost is the same as for `\max`: a single offline pass over the Pile validation set. There is no extra cost at inference.

### 1.2.3 Per-layer migration strength

The second refinement is at the per-layer level. The migration strength `α` controls how the per-channel difficulty is split between activations and weights, and the paper recommends a single value (0.5 for OPT and BLOOM, 0.75 for GLM-130B) for the entire network. As discussed above, the severity of the outlier problem varies sharply across the layers of a single model, especially at scale. Fixing `α` globally therefore over-smooths the easy layers — pushing more difficulty than necessary onto their weights — and under-smooths the hard ones.

A natural alternative is to let `α` vary with layer index `l`:

$$
s_j^{(l)} = \frac{Q_p(|X_j^{(l)}|)^{\alpha(l)}}{Q_p(|W_j^{(l)}|)^{1-\alpha(l)}},
$$

with `α(l)` chosen from a per-layer outlier severity profile measured on the same calibration buffer. Layers with mild outliers get a small `α(l)`; layers with heavy outliers get a larger one. Like quantile smoothing, this is an offline, one-shot modification of the calibration step — `α(l)` is fixed once and folded into `s` along with the rest of the smoothing factor — and adds nothing to the inference path.

### 1.2.4 Per-channel weight quantization

Both refinements above act on the *inner* calibration step: they change how `s` is computed. There is one further change on the *outer* quantization step that follows from the same outlier picture. The whole point of SmoothQuant is to migrate per-channel difficulty *into* the weights. After smoothing, the weight tensor `\hat W` carries variation that the original `W` did not, and at higher `α` it carries proportionally more of it.

The natural countermeasure to per-channel variation is per-channel granularity. The three configurations in the original paper (O1, O2, O3) all use per-tensor weights — a single `Δ_W` for the whole matrix. Channels of `\hat W` whose typical magnitude is much smaller than the global maximum then get squeezed into a small number of integer levels. This is the same effective-bits problem that originally afflicted activations, reappearing on the weight side because the migration has concentrated more variation there. Per-channel weight quantization, with one `Δ_W` per output channel, eliminates this internal mismatch. It is also fully compatible with INT8 GEMM kernels: the output-channel dimension `C_o` is an *outer* dimension of the contraction, so per-channel weight scaling along `C_o` reduces to a vector multiply on the GEMM output, exactly like per-token activation scaling along the token dimension `T` (Figure 3 of [5]). The paper itself implicitly endorses this choice in its later results (Llama-2, Falcon, Mistral, Mixtral all use per-channel weights in Table 7 of [5]); this thesis adopts it explicitly and pairs it with the two calibration-side refinements above.

### 1.2.5 Where this contribution fits

The headline numbers in [5] are reported at OPT-175B, BLOOM-176B, GLM-130B, and MT-NLG 530B. At those sizes, a 1.5× speedup translates into substantial datacentre savings. But the model itself still needs a multi-GPU server to run at all, so quantization at that scale changes the GPU count rather than the deployability story.

The regime where W8A8 quantization changes the deployment story qualitatively is the **mid-range** of model scales: roughly 1.3B through 13B. A 7B FP16 model needs around 14 GB just for weights, which sits awkwardly above the budget of consumer GPUs, smartphone NPUs, and laptop integrated graphics. The same model in INT8 fits in 7-8 GB and becomes runnable on hardware that ordinary users actually own. Improvements at this scale therefore have direct deployment value, even if the absolute model size is smaller than the ones [5] used. This thesis works on the OPT family from 125M up to 13B for two reasons. It is the family on which the SmoothQuant paper reports its primary OPT scaling results (Figure 7, Table 1 of [5]), which gives a clean basis for comparison. And it is also the deployment regime where the proposed modifications are most useful in practice.

## 1.3 Objectives

The thesis has three concrete objectives, each scoped to what can be defended on the OPT scaling ladder (125M, 1.3B, 2.7B, 6.7B, 13B) using single-GPU Colab T4 and Pro A100 hardware.

1. **Replace the `max`-based smoothing statistic with a per-channel quantile, paired with per-channel weight quantization, and show that this combination improves on the upstream recipe.** The pairing is deliberate: smoothing migrates per-channel difficulty into the weight tensor, so the natural outer scheme on the weight side is the one that gives each output channel its own step size. Per-channel weights are also fully compatible with INT8 GEMM kernels, since `C_o` is an outer dimension of the matmul. The original SmoothQuant paper does not report this scheme for OPT — the OPT and BLOOM headline numbers in [5] all use per-tensor weights (O1, O2, O3) — and only adopts per-channel weights implicitly in its later Llama-2, Falcon, Mistral, and Mixtral results (Table 7 of [5]), without identifying it as a deliberate design choice. This thesis names the configuration explicitly, sweeps the quantile probability `p` jointly with the migration strength `α`, and quantifies whether the resulting recipe reduces perplexity and broadens the stable region in `α` relative to the `max`-based, per-tensor-weight schemes O1 and O2 across the OPT ladder. The implementation uses an exact (non-binned) per-channel quantile calibrator, and `p = 1` is verified to reproduce the upstream `max`-based behaviour bit-for-bit, so the modification is a strict generalisation rather than a competing alternative.

2. **Let `α` vary across layers according to a per-layer outlier severity profile.** Measure the severity profile across the OPT ladder, define a parametric `α(l)` derived from it, and evaluate whether per-layer `α` improves on the best global `α` from objective 1. Combine this with the same per-channel weight scheme and the quantile statistic.

3. **Demonstrate that the final recipe pays no extra memory cost and delivers close to the theoretical 50% reduction in real INT8 inference.** Validate the most promising compound configuration on the seven zero-shot tasks used in the SmoothQuant paper (LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA) against the FP16 ceiling, the W8A8-naive floor, and the paper's O1 and O2 recipes. Also report measured peak VRAM usage during real INT8 inference, to show that the gains in objectives 1 and 2 do not come at the cost of additional runtime memory and that the deployed quantized model approaches the theoretical 2× memory reduction over FP16.

The aim of the thesis is not to claim a new state-of-the-art across all model families, nor to push to the largest possible scale. It is to take a careful, reproducible look at the SmoothQuant recipe on a fixed mid-range model family, identify the places where the recipe can be tightened without leaving the training-free, fully-INT8, hardware-friendly envelope it was designed to enforce, and quantify the resulting accuracy and memory gain.

---

**References (placeholder numbering — to be matched in the bibliography):**

[1] Brown et al., *Language Models are Few-Shot Learners*, NeurIPS 2020.
[2] Zhang et al., *OPT: Open Pre-trained Transformer Language Models*, 2022.
[3] Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018.
[4] Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, NeurIPS 2022.
[5] Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML 2023.
[6] Yao et al., *ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers*, NeurIPS 2022.
[7] Gao et al., *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, 2020.
[8] Wei et al., *Outlier Suppression: Pushing the Limit of Low-Bit Transformer Language Models*, NeurIPS 2022.
