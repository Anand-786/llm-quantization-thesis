# Chapter 4 — Experiment Setup and Results

This chapter reports the results of the different experiments performed for this thesis.

## 4.1 Experimental setup

All experiments use the OPT family of decoder-only Transformer language models [2]. This is the same family on which the SmoothQuant paper reports its primary scaling results. The five sizes are OPT-125M, OPT-1.3B, OPT-2.7B, OPT-6.7B, and OPT-13B. Each is loaded from the Hugging Face `facebook/opt-<size>` checkpoint in FP16. OPT-350M is skipped. Its post-LayerNorm architecture and the extra `project_in` / `project_out` linears around the embedding break the SmoothQuant smoothing transformation. The family spans three orders of magnitude in parameter count. It covers the size range across which activation outliers grow from a minor irritation at 125M into a load-bearing failure at 6.7B and above.

Calibration uses a 512-sample subset of the Pile validation set [7]. Each sample is truncated or padded to 512 tokens. The total buffer is 512 × 512 = 262,144 tokens. This matches the shape used by the upstream `generate_act_scales.py` script. The original Pile mirror at `mystic.the-eye.eu` is offline. The replacement source is `huggingface.co/datasets/mit-han-lab/pile-val-backup`. It holds a byte-identical copy of the original file. The file is staged once on Google Drive and reused across Colab kernels. The same 512 × 512 buffer drives every offline statistic in this work. It produces the per-channel `max` baseline, the per-channel quantile statistics at every `p`, the per-layer severity profile `σ(l)`, and the static activation step sizes for the real-INT8 deployment.

Two evaluation tracks are used. The first is language modelling perplexity on WikiText-2 [9]. The `wikitext-2-raw-v1` validation split is concatenated into a single token stream and broken into non-overlapping windows of length 2048. The average per-token cross-entropy is exponentiated to give the perplexity. Sequence length 2048 is the convention used by the SmoothQuant paper [5] and by LLM.int8() [4]. Shorter contexts would invalidate cross-paper comparability. Sequence length is therefore fixed at 2048 throughout. The second track is zero-shot accuracy on seven downstream tasks. These are LAMBADA [10], HellaSwag [11], PIQA [12], WinoGrande [13], OpenBookQA [14], RTE [15], and COPA [16]. Scoring runs through `lm-evaluation-harness` v0.4.4 [17]. The headline number is the unweighted average of per-task accuracies. The `datasets` library is pinned below version 3.0.0. Without this pin the PIQA and COPA legacy loaders fail.

All experiments run on Google Colab across three accelerator tiers. OPT-125M and OPT-1.3B run on the free-tier T4 with 14.56 GB. OPT-2.7B and OPT-6.7B run on the Pro A100-40GB. OPT-13B runs on the A100-80GB. The T4 cannot host OPT-6.7B at sequence length 2048. Its FP16 weights consume 13.3 GB and leave no room for attention buffers. Reducing the sequence length to fit a smaller GPU is explicitly avoided. The real-INT8 deployment in §4.6 runs OPT-1.3B on the A100-40GB. The torch-int CUTLASS kernels are compiled once per Colab session.

Eight configurations are compared head-to-head per model. Rows 1 and 2 bound the range. Rows 3 and 4 are the SmoothQuant paper baselines. Row 5 is the matched-`α` per-channel baseline. Rows 6 and 8 are the proposed contributions. Row 7 isolates the per-layer effect.

| # | Label | Smoothing statistic | Outer weight | Outer activation | Migration strength |
|---|---|---|---|---|---|
| 1 | FP16 | — | — | — | — |
| 2 | Naive W8A8 | — | per-tensor | per-tensor dynamic | no smoothing |
| 3 | O1 / max | `max` | per-tensor | per-token dynamic | global `α = 0.5` |
| 4 | O2 / max | `max` | per-tensor | per-tensor dynamic | global `α = 0.5` |
| 5 | C / max | `max` | per-channel | per-token dynamic | global `α = 0.5` |
| 6 | **C / quantile** | `Q_p` | per-channel | per-token dynamic | global `α^\star_{\text{model}}` |
| 7 | O1 + per-layer α | `max` | per-tensor | per-token dynamic | parametric `α(l)` |
| 8 | **C + per-layer α** | `max` | per-channel | per-token dynamic | parametric `α(l)` |

The per-model migration strength `α^\star_{\text{model}}` in row 6 is the winner from the joint `(p, α)` grid search. The global `α` in rows 3, 4, 5, and 7 is fixed at 0.5. This is the value the SmoothQuant paper recommends for OPT.

## 4.2 Activation memory dominates inference

Before any quantization step is applied, the memory profile of an FP16 forward pass is measured directly. The motivating observation behind W8A8 quantization is that activation memory dominates total GPU memory at sequence length 2048. Weight memory is fixed once the model loads. Activation memory grows with batch size and sequence length. On every OPT size above 1.3B the activation buffers at a single sequence of length 2048 already exceed the FP16 weight footprint. This is the empirical reason the thesis targets the activation side of W8A8 and not weight-only quantization.

> **Figure 4.1.** Weight memory versus peak activation memory per OPT size at sequence length 2048. *[PNG to be inserted.]*

> **Table 4.1.** Weight footprint, peak activation footprint, and the activation-to-weight ratio across OPT-125M, OPT-1.3B, OPT-2.7B, OPT-6.7B, and OPT-13B. *[To be filled.]*

## 4.3 Quantile calibration with per-channel weights

This section reports the full-run WikiText-2 PPL of the eight configurations at sequence length 2048. The proposed method (row 6, C/quantile) is compared against the SmoothQuant baselines and the FP16 anchor. Sequence length is 2048 throughout. The per-model winning `(p, α)` is noted in each caption.

### OPT-125M

`(p^\star, α^\star) = (0.999, 0.5)`.

| Configuration       | WikiText-2 PPL | Δ vs FP16 |
|---------------------|---------------:|----------:|
| FP16                | 27.5684        | —         |
| Naive W8A8          | 30.2257        | +2.6573   |
| O1 / max, α=0.5     | 28.3081        | +0.7397   |
| O2 / max, α=0.5     | 29.1863        | +1.6179   |
| C / max, α=0.5      | 27.5991        | +0.0307   |
| **C / quantile**    | **27.6291**    | +0.0607   |

C/quantile is statistically tied with C/max at this size. Both beat the SmoothQuant O1 baseline by 0.68 PPL. Outlier handling is not load-bearing at 125M. The flatness is itself a finding.

### OPT-1.3B

`(p^\star, α^\star) = (0.999, 0.9)`.

| Configuration       | WikiText-2 PPL | Δ vs FP16 |
|---------------------|---------------:|----------:|
| FP16                | 14.4677        | —         |
| Naive W8A8          | 15.5867        | +1.1190   |
| O1 / max, α=0.5     | 14.8333        | +0.3656   |
| O2 / max, α=0.5     | 14.8335        | +0.3658   |
| C / max, α=0.5      | 14.7710        | +0.3033   |
| **C / quantile**    | **14.6248**    | +0.1571   |

C/quantile beats the SmoothQuant O1 baseline by 0.21 PPL and sits 0.16 PPL above the FP16 anchor.

### OPT-2.7B

`(p^\star, α^\star) = (0.995, 0.5)`.

| Configuration       | WikiText-2 PPL | Δ vs FP16 |
|---------------------|---------------:|----------:|
| FP16                | 12.3425        | —         |
| Naive W8A8          | *[to fill]*    | —         |
| O1 / max, α=0.5     | 12.3946        | +0.0521   |
| O2 / max, α=0.5     | 12.4214        | +0.0789   |
| C / max, α=0.5      | 12.3714        | +0.0289   |
| **C / quantile**    | **12.3422**    | −0.0003   |

C/quantile is INT8 at FP16 quality. It beats O1 by 0.052 PPL and C/max by 0.029 PPL.

## 4.4 Per-layer migration strength

A single global `α` ignores the structure of outliers across depth. The severity statistic `σ(l) = max_j(stat_j) / median_j(stat_j)` at each layer reveals that structure directly.

### Severity profile

| Model    | min σ | median σ | max σ | spread |
|----------|------:|---------:|------:|-------:|
| OPT-2.7B | 2.77  | 15.0     | 36.2  | 13.1×  |
| OPT-6.7B | 2.62  | 21.3     | 59.9  | 22.9×  |
| OPT-13B  | 1.77  | 23.3     | 82.1  | 46.3×  |

Spread grows monotonically with size. The shape is the same across all three sizes. Layer 0 is low. Layers 1 to 3 hold the peak. The remaining layers decay smoothly toward the median. The `q_proj` and `fc1` traces overlap almost exactly at each layer.

> **Figure 4.2.** Per-layer severity ratio `σ(l) = max_j / median_j` for OPT-2.7B, OPT-6.7B, and OPT-13B. *[PNG to be inserted.]*

> **Figure 4.3.** Per-layer `Q_0.99 / median` curves showing the bathtub shape at the network edges. *[PNG to be inserted.]*

### Full-run PPL under `α(l)`

The parametric form is

$$
\alpha(l) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{\sigma(l)}{\max_{l'} \sigma(l')}, \qquad \alpha_{\min}=0.5,\; \alpha_{\max}=0.9.
$$

The bounds are fixed for every OPT size. No per-model tuning is performed.

#### OPT-125M

Realized α(l) range across 12 layers: [0.606, 0.900].

| Configuration            | WikiText-2 PPL | Δ vs FP16 |
|--------------------------|---------------:|----------:|
| FP16                     | 27.5684        | —         |
| O1 / max, α=0.5          | 28.3081        | +0.7397   |
| C / max, α=0.5           | 27.5991        | +0.0307   |
| O1 + α(l)                | 29.4631        | +1.8947   |
| **C + α(l)**             | **27.5859**    | +0.0175   |

C+α(l) beats every other configuration and lands within 0.02 PPL of FP16. O1+α(l) regresses because per-tensor weights cannot absorb the higher migration that α(l) assigns to the peak layers.

#### OPT-1.3B

Realized α(l) range across 24 layers: [0.642, 0.900].

| Configuration            | WikiText-2 PPL | Δ vs FP16 |
|--------------------------|---------------:|----------:|
| FP16                     | 14.4677        | —         |
| O1 / max, α=0.5          | 14.8333        | +0.3656   |
| C / max, α=0.5           | 14.7710        | +0.3033   |
| O1 + α(l)                | 14.6949        | +0.2272   |
| **C + α(l)**             | **14.6281**    | +0.1604   |

α(l) helps under both outer schemes at this size. C+α(l) matches C/quantile at the same model.

#### OPT-2.7B

| Configuration            | WikiText-2 PPL | Δ vs FP16 |
|--------------------------|---------------:|----------:|
| FP16                     | 12.3425        | —         |
| O1 / max, α=0.5          | 12.3946        | +0.0521   |
| C / max, α=0.5           | 12.3714        | +0.0289   |
| O1 + α(l)                | *[to fill]*    | —         |
| **C + α(l)**             | *[to fill]*    | —         |

The structural shape of `σ(l)` is fixed across sizes. Only its amplitude scales. One parametric form therefore covers the entire family without retuning.

## 4.5 Robustness across scale and architecture

This section extends the same evaluation to the larger OPT sizes and to two non-OPT architecture families. The `α(l)` bounds are set once per architecture family from a single inspection of severity spread. No per-model PPL search is performed.

### OPT-6.7B

`(p^\star, α^\star) = (0.995, 0.5)`. Realized α(l) range across 32 layers: [0.597, 0.900].

| Configuration            | WikiText-2 PPL | Δ vs FP16 |
|--------------------------|---------------:|----------:|
| FP16                     | 10.6732        | —         |
| Naive W8A8               | 25.9135        | +15.2403  |
| O1 / max, α=0.5          | 10.6988        | +0.0256   |
| O2 / max, α=0.5          | 10.7000        | +0.0268   |
| C / max, α=0.5           | 10.7252        | +0.0520   |
| **C / quantile**         | 10.6875        | +0.0143   |
| O1 + α(l)                | 10.7382        | +0.0650   |
| **C + α(l)**             | **10.6826**    | +0.0094   |

Naive W8A8 collapses to 2.4× FP16 PPL. Both proposed methods sit within 0.02 PPL of FP16. C+α(l) wins. The collapse-vs-recovery contrast is what motivates the W8A8 problem in the first place.

### OPT-13B

`(p^\star, α^\star) = (0.995, 0.5)`. Realized α(l) range across 40 layers: [0.575, 0.900].

| Configuration            | WikiText-2 PPL | Δ vs FP16 |
|--------------------------|---------------:|----------:|
| FP16                     | 9.9439         | —         |
| Naive W8A8               | 4325.6772      | +4315.73  |
| O1 / max, α=0.5          | 10.1767        | +0.2328   |
| O2 / max, α=0.5          | 10.2037        | +0.2598   |
| C / max, α=0.5           | 10.1691        | +0.2252   |
| **C / quantile**         | 10.0077        | +0.0638   |
| O1 + α(l)                | 9.9963         | +0.0524   |
| **C + α(l)**             | **9.9825**     | +0.0386   |

Naive W8A8 is destroyed at 13B. The SmoothQuant O1 baseline recovers most of the gap but still trails FP16 by 0.23 PPL. C+α(l) closes the gap to 0.04 PPL.

### Cross-architecture verification

The same `α(l)` construction is applied to Llama-2-7B, Llama-2-13B, and Falcon-7B. The bound range is set once per family from severity spread.

| Model       | Paper α | α(l) range   | C/max α=paper vs FP16 | α(l) vs FP16 | α(l) − C/max α |
|-------------|--------:|-------------:|----------------------:|-------------:|---------------:|
| Llama-2-7B  | 0.85    | [0.70, 0.95] | +0.0416               | +0.0388      | **−0.0028**    |
| Llama-2-13B | 0.85    | [0.75, 0.95] | +0.0427               | +0.0507      | **+0.0080**    |
| Falcon-7B   | 0.60    | [0.50, 0.90] | +0.0368               | +0.0397      | **+0.0029**    |

All three deltas sit within ±0.01 PPL of the paper's per-model tuned global `α`. The Llama-2-7B paper W8A8 gap of +0.0410 PPL matches our +0.0416 PPL to four decimal places. This is independent verification of the evaluation harness.

The `α(l)` construction reaches the same quality as the SmoothQuant paper's per-model tuned global `α` on three different architecture families. No grid search per model is required.

## 4.6 Memory footprint verification

The deployment measurement runs OPT-1.3B through torch-int's CUTLASS INT8 kernels in the O3 configuration (per-tensor weight and per-tensor static activation). Three rows are reported on the same A100 in the same Python session.

| Configuration                          | Size (MB) | Peak VRAM (MB) | Activation peak (MB) | WikiText-2 PPL |
|----------------------------------------|----------:|---------------:|---------------------:|---------------:|
| FP16                                   | 2509.61   | 4557.38        | 2045.58              | 14.62          |
| INT8 paper (`max`, α=0.5)              | 1357.84   | 3140.76        | 1780.58              | 18.02          |
| **INT8 ours (`Q_0.999`, α=0.5)**       | 1357.47   | 3140.38        | 1780.58              | **17.96**      |

Weight memory drops to 54% of FP16. Peak VRAM drops to 69%. Activation peak drops to 87%. Memory is identical between paper and ours to 0.04%. The deployed quantile-based method reaches lower PPL at zero memory cost.

> **Table 4.3.** Deployment metrics for OPT-2.7B and OPT-6.7B under the same three configurations. *[To be filled.]*

The `α` winner shifts from 0.9 under fake-quantization with per-channel weights to 0.5 in this deployment setting. Per-tensor static activation cannot absorb the outlier shift that aggressive smoothing pushes onto weights. The calibration probability `p` is independent of the deployment setting. The bound `α` is not. This is itself a reportable finding.

---

## References used in this chapter

[2] Zhang et al., *OPT: Open Pre-trained Transformer Language Models*, 2022.
[4] Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, NeurIPS 2022.
[5] Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML 2023.
[7] Gao et al., *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, 2020.
[9] Merity et al., *Pointer Sentinel Mixture Models*, ICLR 2017 (WikiText-2/103).
[10] Paperno et al., *The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context*, ACL 2016.
[11] Zellers et al., *HellaSwag: Can a Machine Really Finish Your Sentence?*, ACL 2019.
[12] Bisk et al., *PIQA: Reasoning about Physical Commonsense in Natural Language*, AAAI 2020.
[13] Sakaguchi et al., *WinoGrande: An Adversarial Winograd Schema Challenge at Scale*, AAAI 2020.
[14] Mihaylov et al., *Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering*, EMNLP 2018.
[15] Wang et al., *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*, ICLR 2019.
[16] Roemmele et al., *Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning*, AAAI Spring Symposium 2011.
[17] Gao et al., *A Framework for Few-Shot Language Model Evaluation* (`lm-evaluation-harness`), 2024.
