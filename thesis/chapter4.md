# Chapter 4 — Experiment Setup and Results

This chapter has two halves. The first half describes the experimental setup in detail: the model family, the calibration and evaluation corpora, the hardware, the evaluation harness, the configurations under comparison, and the engineering decisions that were forced by the available hardware. The aim is that anyone reading this chapter should be able to reproduce the numbers in the second half exactly, given a clone of the public repository and a Colab Pro account. The second half reports the results in the same order in which the corresponding objectives were laid out in §1.3 — scheme exploration, quantile-based smoothing, per-layer migration strength, the combined recipe, zero-shot evaluation, and the real-INT8 deployment measurement.

## 4.1 Models

All experiments use the OPT family of decoder-only Transformer language models [2]. The OPT family is the same family on which the original SmoothQuant paper reports its primary scaling results (Figure 7 and Table 1 of [5]), which gives a clean basis for direct comparison and lets the proposed recipe be evaluated against the upstream baseline at every scale at which the paper reports numbers.

The five model sizes used in this work are OPT-125M, OPT-1.3B, OPT-2.7B, OPT-6.7B, and OPT-13B, with parameter counts 125 million, 1.3 billion, 2.7 billion, 6.7 billion, and 13 billion respectively. All models are loaded directly from the Hugging Face `facebook/opt-<size>` checkpoints in FP16, which is the precision in which OPT was released by Meta and the precision against which both the SmoothQuant paper and this thesis anchor their accuracy comparisons. OPT-350M is deliberately skipped: its architecture differs from the rest of the OPT family in using post-LayerNorm together with separate `project_in` and `project_out` linears around the embedding, which makes the SmoothQuant smoothing transformation in `smooth_lm` not directly applicable without code changes. The resulting code path divergence is not worth the engineering for a single intermediate datapoint that the rest of the family already brackets cleanly.

The OPT family covers three orders of magnitude in parameter count and spans the regime in which activation outliers transition from a minor irritation (125M) into a load-bearing problem that destroys naive INT8 quantization (6.7B and beyond), so it is well suited to studying how the proposed modifications scale with model size.

## 4.2 Calibration corpus

The calibration corpus for the smoothing transformation is a 512-sentence subset of the Pile validation set [7], with each sentence truncated or padded to 512 tokens. This is the same corpus and the same shape (512 × 512 = 262,144 tokens) used by the upstream `generate_act_scales.py` script that ships with the SmoothQuant repository, and is therefore the corpus against which the upstream `act_scales/opt-<size>.pt` files were originally produced. Holding the corpus fixed at exactly the upstream choice is what lets the per-channel-max sanity check (§3.2.4) directly compare this work's calibrator output against the released reference data.

The Pile validation file used by the upstream repo was originally hosted at `https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst`, but that mirror has been offline for some time and the script's `wget` against it hangs indefinitely. The mirror used instead in this work is `huggingface.co/datasets/mit-han-lab/pile-val-backup`, which contains a byte-identical copy of the original file. After download the file is staged once on Google Drive at a fixed path so that subsequent Colab kernels can pick it up directly without re-downloading.

The same 512×512 calibration buffer drives every offline statistic used in the proposed pipeline: the per-channel `max` baseline against which the implementation is checked, the per-channel quantile statistics at every requested `p`, the per-layer severity profile `σ(l)`, and (for the real-INT8 deployment in §4.10) the static activation step sizes used by torch-int's CUTLASS kernels. No part of the proposed method requires a second calibration pass on a different corpus.

## 4.3 Evaluation corpora and tasks

Two complementary evaluation regimes are used.

The first is **language modelling perplexity on WikiText-2** [9]. The standard `wikitext-2-raw-v1` validation split is concatenated into a single token stream, broken into non-overlapping windows of length 2048, and the average per-token cross-entropy is computed across all windows; the perplexity is the exponential of that average. Sequence length 2048 is the convention used by the SmoothQuant paper for OPT [5] and by LLM.int8() before it [4]; reducing the sequence length would cut attention memory quadratically and therefore reduce the operational cost of the eval, but it would also invalidate cross-paper comparability, because perplexity is defined relative to the eval protocol — a shorter context window gives every token less history to condition on, raises the cold-start penalty, and produces numbers that are not on the same scale as the published baselines. Sequence length is therefore held fixed at 2048 for every PPL number reported in this thesis, even on hardware where shorter contexts would be more comfortable.

Perplexity on WikiText-2 is the headline metric for the calibration sweeps in §4.6 and §4.7 because (a) it is the only metric the original SmoothQuant paper reports on OPT-1.3B through OPT-13B for its W8A8 ablations, and (b) it is sensitive enough to a 0.05 PPL shift that the within-grid deltas of the proposed sweeps register clearly above evaluation noise.

The second regime is **zero-shot evaluation on the seven downstream tasks** used in the SmoothQuant paper: LAMBADA [10], HellaSwag [11], PIQA [12], WinoGrande [13], OpenBookQA [14], RTE [15], and COPA [16]. These are run through `lm-evaluation-harness` v0.4.4 [17] using the same scoring conventions that the SmoothQuant paper used (loglikelihood-based ranking on the canonical answer set per task), and the headline number reported in §4.9 is the unweighted average of the per-task accuracies, again following [5]. Per-task numbers are also reported so that any single-task regression is visible.

A small but practically important detail: as of `datasets ≥ 3.0`, the legacy dataset-script loaders for PIQA and COPA were removed, and the harness fails on those two tasks with `RuntimeError: Dataset scripts are no longer supported`. The fix used here is to pin `datasets < 3.0.0` in the evaluation environment before any zero-shot run; this is documented as Challenge 3 in the project's CLAUDE.md and is mentioned here because the pin is part of the reproducible setup, not a workaround that lives outside it.

## 4.4 Hardware

All experiments run on Google Colab. Three accelerator tiers are used.

OPT-125M and OPT-1.3B fit comfortably on the **free-tier T4 (14.56 GB)**, which is therefore the default hardware for the smaller two ladder rungs. OPT-2.7B in bf16 (~5.4 GB of weights) also fits on a T4 with no code changes. OPT-6.7B and beyond require an A100, for two reasons. First, the FP16 weights of OPT-6.7B occupy 13.3 GB out of the T4's 14.56 GB of memory, leaving no headroom for the attention buffers at sequence length 2048; an SDPA forward pass on a T4 OOMs while asking for an additional ~128 MB. Second, the calibration top-K buffer for the quantile pass scales with both model dimension and number of smoothing-relevant linear inputs, and reaches ~13 GB at 6.7B and ~21 GB at 13B; the latter exceeds the 40 GB A100's free memory once the model is also on device, and is therefore moved to host memory.

The two A100 tiers used are **Colab Pro A100-40GB** (for OPT-2.7B and OPT-6.7B) and **A100-80GB** (for OPT-13B). Reducing the sequence length to fit a smaller GPU is explicitly avoided, for the comparability reason given in §4.3.

For real-INT8 deployment in §4.10, OPT-1.3B runs on the Pro A100-40GB, with the torch-int CUTLASS kernels compiled once per Colab session and staged on disk as a binary egg at `/usr/local/lib/python3.12/dist-packages/torch_int-0.0.0-py3.12-linux-x86_64.egg`. The compile itself requires several patches to torch-int's build chain to make it work on the current Colab toolchain (Python 3.12, PyTorch 2.x, CUDA 12.8, NumPy 2.x); these are documented as Challenge 4 in CLAUDE.md and are not repeated here, but they are part of the reproducible setup.

## 4.5 Configurations under comparison

Eight configurations are evaluated head-to-head per model. The first two are the FP16 ceiling and the W8A8-naive floor; everything else lives between them. The remaining six split into "upstream-style" recipes (rows 3–4 below) and the two proposed modifications and their combinations (rows 5–8).

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

Rows 3 and 4 are the two configurations the original SmoothQuant paper reports on OPT (O1 and O2 from Table 2 of [5]). Row 5 is the per-channel-weight scheme that the paper does not report on OPT; it is included as the matched-`α` baseline against which the proposed quantile and per-layer modifications are measured, so that the difference between rows 5 and 6/8 isolates *only* the modification being studied. Row 6 is the calibration-side modification of §3.2 (quantile-based smoothing); row 8 is the per-layer-`α` modification of §3.4 stacked on top of the same per-channel-weight scheme. Row 7 — per-layer `α` paired with the upstream per-tensor weight scheme — is included to test whether per-layer `α` requires per-channel weights to deliver its gain, or whether it works on its own.

The migration strength `α^\star_{\text{model}}` for row 6 is the per-model winner from the joint `(p, α)` sweep described in §4.7. For rows 3, 4, 5, and 7 the global `α` is fixed at 0.5, which is the value the SmoothQuant paper recommends for OPT [5] and which gives the cleanest matched-condition comparison against the proposed recipes.

The full sweep grids for rows 5, 6, and 8 are described in §4.6, §4.7, and §4.8 respectively. The eight-row table above is reused in §4.9 as the single-session verification table per model — one Colab kernel, one shared evaluator, all eight configurations evaluated under matched conditions so that within-table deltas are noise-free against the FP16 anchor.

## 4.6 Setup: scheme exploration sweep

The first sweep tests how the three outer schemes — O1 (per-tensor W + per-token A), O2 (per-tensor W + per-tensor A), and C (per-channel W + per-token A) — compare under matched smoothing. The smoothing statistic is held fixed at the upstream `max`-based formula, the migration strength is swept across a fixed grid, and the WikiText-2 perplexity is reported per cell.

The default grid is `α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (five values, step 0.2), giving 15 evaluations per model when run across O1, O2, and C. On OPT-125M the same grid is run at finer step (`α ∈ {0.1, 0.2, …, 0.9}`, nine values) because each evaluation completes in under thirty seconds and the additional resolution helps characterise the alpha-stability of each scheme on a model where outlier pressure is mild. On OPT-2.7B and beyond the step-0.2 grid is the default.

The activation-scale files used as input to the smoothing transformation are the upstream `act_scales/opt-<size>.pt` files released alongside the SmoothQuant code. These files are themselves the per-channel `max(|X_j|)` of the same Pile 512×512 buffer described in §4.2, so the sweep is performed against the exact statistic the original paper used.

## 4.7 Setup: quantile-based smoothing sweep

The second sweep replaces the per-channel `max` in the smoothing factor with a per-channel quantile `Q_p` and sweeps the probability `p` jointly with the migration strength `α`. The outer scheme is fixed at C (per-channel W + per-token A) for the reasons given in §3.3.

Two grid resolutions are used. On OPT-1.3B and OPT-2.7B, where the optimum is not yet known, the grid is `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90} × α ∈ {0.5, 0.7, 0.9}` (15 cells). On OPT-1.3B a follow-up sweep over the full `α ∈ {0.1, 0.2, …, 0.9}` grid at the two competing `p` values from the initial sweep (`p = 0.999` and `p = 0.90`) is run to characterise alpha-stability. On OPT-125M, OPT-6.7B, and OPT-13B the full `5 × 5` grid (`p × α` both at five values, 25 cells) is run in one notebook pass, since by that point the cost of one cell is small enough to evaluate the entire grid without prior narrowing.

The quantile statistics at all five `p` values are produced by a single calibration pass per model — the exact per-channel top-K calibrator described in §3.2.3, with `p_{\min} = 0.90` so every `p \ge 0.90` is exact. Output is one `.pt` file per `(model, p)` pair, each a dictionary mapping linear-input name to a tensor of shape `[in_features]`, and each saved to Drive at `act_percentiles/opt-<size>/p<value>.pt`. The schema of these files matches `act_scales/opt-<size>.pt` exactly, so the same `smooth_lm` driver can consume either file with a single argument change.

The `p = 1` row is excluded from both saved files and from the sweep grid, for the fp16-buffer reason given in §3.2.4. The `\max`-based reference baseline used in cross-method comparisons is the upstream `act_scales/opt-<size>.pt` file directly, computed once with full fp32 precision in the original SmoothQuant calibration script. This avoids letting fp16 ULP noise on the recovered max contaminate the within-sweep deltas.

## 4.8 Setup: per-layer migration strength

The per-layer migration strength experiments use the parametric form

$$
\alpha(l) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{\sigma(l)}{\max_{l'} \sigma(l')}, \qquad \sigma(l) = \frac{\max_j \, \mathrm{stat}_j^{(l)}}{\mathrm{median}_j \, \mathrm{stat}_j^{(l)}},
$$

with `\mathrm{stat}_j^{(l)} = \max_t |X^{(l)}_{t, j}|` taken from the upstream `act_scales/opt-<size>.pt` file directly. The two scalar bounds are fixed at `α_{\min} = 0.5` and `α_{\max} = 0.9` for every model — no per-model retuning of these bounds is performed, so the entire `α(l)` schedule is determined once the severity profile is measured.

`σ(l)` is computed as the average of the per-channel-max-to-per-channel-median ratio across the two smoothing-relevant linear inputs at layer `l` (`q_proj` input and `fc1` input). The diagnostic in §4.8 reports the resulting `σ(l)` curves for OPT-2.7B, OPT-6.7B, and OPT-13B, and the structural observations from those curves were already summarised in §3.4.1: a low-severity layer 0, a sharp peak at layers 1–3, and a smooth monotonic decay through the rest of the network.

The per-layer experiments are run under both the O1 outer scheme (row 7 of §4.5) and the C outer scheme (row 8 of §4.5), so that the dependence of per-layer `α`'s gain on the choice of outer scheme can be characterised. The smoothing statistic in both rows is held at the upstream `max`, so that the contrast against the corresponding global-`α = 0.5` baseline (rows 3 and 5 respectively) isolates the per-layer-`α` effect from any quantile-based effect.

## 4.9 Verification protocol

Cross-cell perplexity comparisons across separate Colab notebooks pick up small amounts of noise from non-deterministic kernel choices, version drift in `transformers` and `accelerate`, and minor differences in evaluator configuration. To make the eight-row comparison in §4.5 noise-free, every model has a dedicated *single-session verification notebook* that loads exactly one shared `Evaluator` instance, runs all eight configurations against it back-to-back in the same kernel, and prints the resulting eight-row table. Within that table the FP16 row is fixed at the model's measured FP16 PPL and every other row's `Δ vs FP16` is computed against that exact anchor, so the deltas reported in the verification table are noise-free relative to each other.

The within-grid sweeps in §4.6, §4.7, and §4.8 remain authoritative for *shape* (relative ordering inside a grid, alpha-stability, the location of the optimum). The verification table is authoritative for *cross-method comparisons* (does C/quantile beat O1/max? by how much?). When the two disagree by more than ~0.01 PPL, the verification number is the one quoted in the headline.

## 4.10 Setup: real-INT8 deployment

The deployment experiment moves the calibrated configuration from fake quantization onto torch-int's CUTLASS INT8 GEMM kernels, in the most aggressive paper configuration: O3 (per-tensor weight, per-tensor static activation). The rationale and the consequence for `α` were given in §3.5.

The static activation step sizes are calibrated on a separate Pile-derived corpus, since the upstream torch-int code path uses the the-eye.eu Pile mirror (which is offline). The replacement corpus is a `val.jsonl.zst` file built from `NeelNanda/pile-10k` on Hugging Face — 10k Pile samples in the same JSONL.zst format the upstream calibration script expects, of which only the first 512 are actually consumed. This file is staged once on Drive and reused across deployment runs.

Three quantities are measured and reported per configuration:

1. **Model size on disk** (`size_mb`): sum of INT8 weights, per-tensor weight scales, per-tensor static activation scales, and FP16 biases.
2. **Peak GPU memory during forward pass** (`peak_vram_mb`): measured via `torch.cuda.max_memory_allocated()` across a full WikiText-2 PPL evaluation at sequence length 2048, including weights, KV-cache, attention buffers, and any temporaries.
3. **WikiText-2 perplexity** at sequence length 2048, evaluated on the same window protocol as the fake-quant numbers, on the same shared `Evaluator`.

Three rows are populated for the deployment table per model: FP16, INT8 with the upstream `max`-based recipe at `α = 0.5` ("INT8-paper"), and INT8 with the proposed quantile-based recipe at `(p^\star, α = 0.5)` ("INT8-ours"). All three rows run on the same A100, in the same Colab session, within the same Python process. Differences in the third column therefore measure exactly what they appear to measure.

---

## 4.11 Results

The remainder of the chapter reports the numbers produced by the setup above. Each subsection corresponds to one of the experiments described in §4.6 through §4.10. Tables and figures are placeholders to be filled in once the corresponding sweep has finished running on every ladder rung.

### 4.11.1 Naive W8A8 collapse across the OPT ladder

Sanity-check baseline showing how naive per-tensor W + per-tensor A quantization degrades with model size, anchored against the model's own FP16 perplexity. This is the floor that every smoothed configuration has to clear, and its scaling shape — gentle on small models, catastrophic at 6.7B and beyond — is the empirical motivation for the rest of the chapter.

> **Table 4.1.** WikiText-2 perplexity at sequence length 2048: FP16 vs naive W8A8 across the OPT ladder.

> *[Table to be filled: rows = OPT-125M, OPT-1.3B, OPT-2.7B, OPT-6.7B, OPT-13B; columns = FP16, Naive W8A8, Δ vs FP16.]*

### 4.11.2 Scheme exploration: O1, O2, and C under matched `max`-based smoothing

Outer-scheme comparison from §4.6, with the smoothing statistic held at the upstream `max`. One subsection per model, with the alpha sweep table for each. Headline plot: best-PPL-per-scheme as a function of model size.

#### OPT-125M (9-level alpha sweep, step 0.1)

> *[Table to be filled: rows = α ∈ {0.1, …, 0.9}; columns = O1, C; final row = "best per scheme". Annotate the high-α cliff on O1.]*

#### OPT-1.3B (5-level alpha sweep, step 0.2)

> *[Table to be filled: rows = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; columns = O1, C; final row = "best per scheme".]*

#### OPT-2.7B (5-level alpha sweep, step 0.2)

> *[Table to be filled: rows = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; columns = O1, O2, C; final row = "best per scheme".]*

#### OPT-6.7B (5-level alpha sweep, step 0.2)

> *[Table to be filled: rows = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; columns = O1, O2, C; final row = "best per scheme".]*

#### OPT-13B (5-level alpha sweep, step 0.2)

> *[Table to be filled: rows = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; columns = O1, O2, C; final row = "best per scheme".]*

#### Cross-scale summary

> **Figure 4.1.** Best WikiText-2 PPL per outer scheme (O1, O2, C) as a function of OPT model size, anchored against FP16. *[To be filled.]*

> **Figure 4.2.** Alpha-spread (max−min PPL across the α grid) per outer scheme, as a measure of alpha-stability, plotted against OPT model size. *[To be filled.]*

### 4.11.3 Quantile-based smoothing

Joint `(p, α)` sweep from §4.7, with the outer scheme fixed at C and the smoothing statistic varying. One subsection per model. The headline observation expected from the diagnostic in Chapter 3 is that high `p` (conservative quantile, close to the max) gives a wide, stable α-plateau, while low `p` (aggressive quantile, top 10%) gives sharp high-α failure modes that get worse with model size.

#### OPT-125M (5 × 5 grid)

> *[Table to be filled: rows = p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}; columns = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; cells = WikiText-2 PPL. Annotate the per-row best.]*

#### OPT-1.3B (initial 5 × 3 grid plus full 2 × 9 follow-up)

> *[Two tables to be filled: (a) initial sweep at α ∈ {0.5, 0.7, 0.9}; (b) full alpha sweep at p ∈ {0.999, 0.90}. Highlight the alpha-stability contrast between the two p values.]*

#### OPT-2.7B (5 × 3 grid)

> *[Table to be filled: rows = p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}; columns = α ∈ {0.5, 0.7, 0.9}; cells = WikiText-2 PPL. Flag the catastrophic cells at low p, high α as a safety-boundary observation.]*

#### OPT-6.7B (5 × 5 grid)

> *[Table to be filled: rows = p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}; columns = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.]*

#### OPT-13B (5 × 5 grid)

> *[Table to be filled: rows = p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}; columns = α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.]*

#### How the optimal `(p, α)` drifts with scale

> **Figure 4.3.** Per-model winner `(p^\star, α^\star)` plotted on the `(p, α)` plane across the OPT ladder. *[To be filled.]*

> **Figure 4.4.** Quantile-sweep alpha-spread vs `p` across the OPT ladder, as a measure of how much alpha-stability quantile-smoothing buys back. *[To be filled.]*

### 4.11.4 Per-layer migration strength

Per-layer-`α` results from §4.8. The diagnostic plots come first (the severity profile that drives `α(l)`), then the verification numbers under both outer schemes.

#### Severity profile across the OPT ladder

> **Figure 4.5.** Per-layer severity ratio `σ(l) = \max_j \mathrm{stat}_j / \mathrm{median}_j \mathrm{stat}_j` for OPT-2.7B, OPT-6.7B, and OPT-13B. Panel (a): `max/median`; panel (b): `Q_{0.99}/median` (the bathtub curve from §3.4.1). *[To be filled.]*

> **Table 4.2.** Per-model severity statistics: `(min σ, median σ, max σ, spread)` and the layer index at which the peak is reached, for OPT-2.7B / 6.7B / 13B. *[To be filled.]*

#### Per-layer `α` results under O1 and C

> **Table 4.3.** Per-model verification rows for global `α = 0.5` versus per-layer `α(l)`, under both O1 and C outer schemes. Columns: model, scheme, global α=0.5 PPL, per-layer α PPL, Δ. *[To be filled.]*

> Comments to add inline once the table is populated: which scales benefit from per-layer `α` under each scheme, and where the per-layer-`α` and quantile-smoothing recipes converge to the same accuracy floor through different routes.

### 4.11.5 The combined recipe (verification table)

The eight-row verification table from §4.5 and §4.9, one per model, evaluated in a single Colab kernel against a shared `Evaluator` so that every Δ is noise-free against the FP16 anchor. This is the headline accuracy table of the chapter.

#### OPT-125M

> *[Table to be filled: 8 rows as in §4.5; columns = Config, PPL, Δ vs FP16. Note that quantile-smoothing improvement is expected to be small at this scale, since outlier pressure is mild.]*

#### OPT-1.3B

> *[Table to be filled.]*

#### OPT-2.7B

> *[Table to be filled.]*

#### OPT-6.7B

> *[Table to be filled.]*

#### OPT-13B

> *[Table to be filled.]*

#### Cross-scale summary

> **Figure 4.6.** Δ vs FP16 of the best proposed configuration (C/quantile or C+per-layer α, whichever wins per model) versus the upstream-paper configuration (O1/max at α=0.5) across the OPT ladder. *[To be filled.]*

### 4.11.6 Zero-shot evaluation

Seven-task zero-shot results on LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, and COPA, scored through `lm-evaluation-harness` v0.4.4. Reported per model.

> **Table 4.4.** Zero-shot accuracy per task and unweighted average for FP16, naive W8A8, O1/max, and the best proposed configuration. *[To be filled, one block per OPT size.]*

> **Figure 4.7.** Average zero-shot accuracy as a function of model size, for FP16, the upstream paper recipe, and the best proposed configuration. *[To be filled.]*

The thesis's accuracy claim is closed by this table: any per-task regression that the perplexity number could have hidden has to show up here, and the unweighted average is the comparable metric to the SmoothQuant paper's reported zero-shot numbers in Table 1 and Figure 7 of [5].

### 4.11.7 Real-INT8 deployment

End-to-end memory and accuracy measurement under torch-int's CUTLASS kernels in the O3 regime, from §4.10. Three rows per model: FP16, INT8-paper (max-smoothing, `α = 0.5`), and INT8-ours (quantile-smoothing at the per-model `p^\star`, `α = 0.5`).

#### OPT-1.3B

> **Table 4.5.** Real-INT8 deployment metrics on OPT-1.3B: model size on disk, peak GPU memory at sequence length 2048, and WikiText-2 PPL. *[To be filled. Headline ratios — weight memory / peak VRAM / activation peak as a fraction of FP16 — to be added underneath.]*

#### OPT-2.7B

> *[Table to be filled if the deployment ladder extends past 1.3B.]*

#### OPT-6.7B

> *[Table to be filled if the deployment ladder extends past 1.3B.]*

#### Discussion

The two claims to close once the table is populated: (a) the proposed quantile-based recipe deploys in real INT8 at zero extra memory cost relative to the upstream `max`-based recipe — same kernels, same INT8 storage, same activation behaviour — so any accuracy difference is a clean isolation of the smoothing statistic; (b) the deployed quantized model approaches the theoretical 2× memory reduction over FP16, with weight memory at roughly 50% of FP16 and peak VRAM at roughly 70% of FP16, the gap between the two arising from the FP16-resident attention buffers and KV cache that are not affected by INT8 weight storage.

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
