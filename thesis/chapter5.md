# Chapter 5 — Conclusion and Future Work

This thesis investigates the calibration step of the SmoothQuant method on the OPT family across five model sizes from 125M to 13B parameters. Two specific choices in the construction of the per-channel smoothing factor are examined in detail. The per-channel summary statistic and the migration strength. For both choices a more principled alternative is proposed and verified empirically. The resulting method is training-free, INT8-compatible, and requires no modification to the inference path.

The first change replaces the per-channel `max` with a per-channel quantile `Q_p`. Setting `p = 1` gives back the original formula. The smoothing factor no longer depends on the single most extreme token in the calibration data. The same outlier problem the SmoothQuant authors patched on GLM-130B with a top-2% clip is solved at the smoothing step itself. The quantile statistic is paired with per-channel weight quantization. Per-channel weights absorb the variation that smoothing pushes toward the weights.

The second change lets the migration strength `α` vary across layers. A per-layer outlier severity profile drives the choice. The profile is sharply structured. Layer 0 is low. Layers 1 to 3 hold the peak. The remaining layers decay smoothly. The same shape repeats across OPT-2.7B, OPT-6.7B, and OPT-13B. The spread between the calmest and most outlier-heavy layer grows from 13× to 46× with size. A single global `α` cannot fit this. A parametric `α(l)` with two scalar bounds and no further degrees of freedom can.

The two changes are orthogonal. Quantile smoothing handles intra-channel outliers. Per-layer `α` handles inter-layer severity drift. Both are pure offline calibration. Nothing changes at inference time. The same INT8 GEMM kernels the upstream method uses are reused unchanged. The torch-int deployment measurement confirms this. The proposed method runs in real INT8 at zero extra memory cost over the upstream baseline. Weight memory drops to 54% of FP16. Peak VRAM drops to 69%. The same `α(l)` construction reaches paper-tuned quality on Llama-2-7B, Llama-2-13B, and Falcon-7B with no per-model tuning.

The contribution is narrow but clean. It is not a new architecture. It is not a new training procedure. It is a reproducible fix for two places where the SmoothQuant calibration leaves accuracy on the table on the OPT family.

Several follow-ups stand out. The first is W4A8. Combining quantile-based smoothing with INT4 weight quantizers such as GPTQ [21] and AWQ [22] is a clean extension. The smoothing step is independent of the bit-width. The second is KV-cache quantization. The KV cache is the last large FP16 memory consumer in the inference path and the same offline calibration philosophy applies. The third is scaling beyond 30B parameters. Models such as OPT-30B and Llama-2-70B require multi-GPU inference and a tensor-parallel evaluation harness. Verifying the per-layer severity profile and the parametric `α(l)` at these sizes is the natural extension. The fourth is kernel development. The current torch-int kernels support only the paper's O3 setting with per-tensor weights and per-tensor static activations. The per-channel weight (PCW) scheme proposed in this thesis has no end-to-end INT8 deployment path today. Writing a CUTLASS or CUDA INT8 GEMM kernel for PCW would close the gap between the fake-quantization accuracy reported here and the real deployed accuracy of the same method.

---

## References newly cited in this chapter

[21] Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023.
[22] Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024.
