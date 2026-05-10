# Task 05: Per-Layer Alpha — Motivation via Outlier-Severity Profile

## Why this notebook

SmoothQuant smooths every layer with a **single global α**. The smoothing factor is

```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
```

with the same α reused for every decoder layer in the model. This implicitly assumes that all layers face roughly the same activation-quantization difficulty. If they don't — i.e. if some layers carry far more severe per-channel outliers than others — then a single α is necessarily a compromise: too aggressive on easy layers (over-smooths, hurts weight quantization) and too gentle on hard ones (under-smooths, leaves activation outliers).

This notebook does the simplest possible empirical check before we touch any code. For each smoothing site (`q_proj` input and `fc1` input of every decoder layer), we measure outlier severity as

```
severity_layer = max_j(s_j) / median_j(s_j)
       where s_j = max(|X_j|) over calibration data (the per-channel act-scale)
```

`max/median` is the cleanest one-number summary of "how spiky is this layer's per-channel activation distribution": a layer where every channel has similar magnitude scores ~1, a layer with a few extreme outlier channels scores high. We do **not** use `max/mean` because the mean is itself pulled up by the outliers.

If severity varies materially across layers (say, >2× spread), then a global α cannot be optimal for all of them, and a per-layer α has clear motivation. If severity is nearly flat across layers, the global-α choice is already well-matched to the problem and the motivation collapses — useful negative evidence too.

We profile **OPT-2.7B, OPT-6.7B, OPT-13B** because (a) the act-scales `.pt` files for these are already on Drive, and (b) outliers grow with scale, so we want to see whether the per-layer spread also grows with scale.

This is a **read-only** notebook: it loads existing `act_scales/opt-*.pt` files, computes summary stats and plots, and saves figures + a CSV to `results/task05/`. No model loading, no GPU needed. Can run on Colab CPU runtime in seconds.

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!pip install -q matplotlib pandas

from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs("/content/llm-quantization-thesis/results/task05", exist_ok=True)

# Sanity: confirm the three .pt files are present on Drive
SCALES_DIR = "/content/drive/MyDrive/thesis_results/act_scales"
for size in ["2.7b", "6.7b", "13b"]:
    p = f"{SCALES_DIR}/opt-{size}.pt"
    assert os.path.exists(p), f"missing: {p}"
    print(f"  ok  {p}  ({os.path.getsize(p)/1024**2:.1f} MB)")
```

---

## Cell 2: Compute per-site severity for one model

A site is one smoothed linear: either `model.decoder.layers.<i>.self_attn.q_proj` or `model.decoder.layers.<i>.fc1`. Each entry in the `.pt` file is a 1D tensor of length `in_features` holding the per-channel `max(|X|)` over the calibration data — that's exactly the `s_j` distribution we want to summarise.

We extract the layer index and the site type (`q_proj` / `fc1`) from each name and record:

- `max`: largest per-channel activation magnitude in the layer
- `median`: median per-channel magnitude
- `mean`: mean per-channel magnitude (for context, not used as severity)
- `severity = max / median`
- `top1pct_over_median = quantile(s, 0.99) / median`  — robust variant in case a single channel is freakishly large; if `top1pct/median` also varies a lot, the spread is real, not a single-channel artifact

```python
import re
import torch
import pandas as pd

LAYER_RE = re.compile(r"model\.decoder\.layers\.(\d+)\.(.+)")

def site_type(suffix: str) -> str:
    if suffix.startswith("self_attn.q_proj"):
        return "q_proj"
    if suffix == "fc1" or suffix.startswith("fc1"):
        return "fc1"
    return "other"

def severity_table(scales_path: str, model_tag: str) -> pd.DataFrame:
    raw = torch.load(scales_path, map_location="cpu")
    rows = []
    for name, vec in raw.items():
        m = LAYER_RE.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        st = site_type(m.group(2))
        if st == "other":
            continue  # only the smoothed sites — q_proj input and fc1 input
        v = vec.float().abs()
        s_max = v.max().item()
        s_med = v.median().item()
        s_mean = v.mean().item()
        s_p99 = torch.quantile(v, 0.99).item()
        rows.append(dict(
            model=model_tag,
            layer=layer_idx,
            site=st,
            n_channels=v.numel(),
            max=s_max,
            median=s_med,
            mean=s_mean,
            p99=s_p99,
            severity=s_max / max(s_med, 1e-12),
            p99_over_median=s_p99 / max(s_med, 1e-12),
        ))
    return pd.DataFrame(rows).sort_values(["site", "layer"]).reset_index(drop=True)

# Quick test on 2.7B
df_test = severity_table(f"{SCALES_DIR}/opt-2.7b.pt", "opt-2.7b")
print(df_test.head(8))
print("\nlayers found:", df_test["layer"].nunique(), " sites per layer:", df_test.groupby("layer").size().unique())
```

Sanity expectation: 32 layers × 2 sites = 64 rows for OPT-2.7B; 32×2 = 64 for OPT-6.7B; 40×2 = 80 for OPT-13B.

---

## Cell 3: Build the combined table for all three models, save CSV

```python
frames = []
for size, tag in [("2.7b", "opt-2.7b"), ("6.7b", "opt-6.7b"), ("13b", "opt-13b")]:
    df = severity_table(f"{SCALES_DIR}/opt-{size}.pt", tag)
    print(f"{tag:>10}: {len(df)} sites, severity median={df['severity'].median():.2f}, "
          f"min={df['severity'].min():.2f}, max={df['severity'].max():.2f}, spread={df['severity'].max()/df['severity'].min():.2f}x")
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
csv_path = "/content/llm-quantization-thesis/results/task05/outlier_severity.csv"
all_df.to_csv(csv_path, index=False)
print(f"\nsaved -> {csv_path}  ({len(all_df)} rows)")
```

The "spread" number (max severity / min severity within a model) is the key one for motivation. If it's ~1.5×, a global α is fine. If it's >3×, per-layer α has obvious headroom.

---

## Cell 4: Per-layer severity plot (one panel per model)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=False)
for ax, (tag, df) in zip(axes, [(t, all_df[all_df.model == t]) for t in ["opt-2.7b", "opt-6.7b", "opt-13b"]]):
    for site, marker, color in [("q_proj", "o", "#1f77b4"), ("fc1", "s", "#d62728")]:
        sub = df[df.site == site].sort_values("layer")
        ax.plot(sub["layer"], sub["severity"], marker=marker, color=color,
                label=site, linewidth=1.2, markersize=5)
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("severity = max(|X|) / median(|X|)  [per channel]")
    ax.set_title(tag)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

fig.suptitle("Per-layer outlier severity across OPT scales (calibration: Pile val, 512×512)", y=1.02)
fig.tight_layout()
out_png = "/content/llm-quantization-thesis/results/task05/outlier_severity_per_layer.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
plt.show()
print(f"saved -> {out_png}")
```

Reading guide:

- **Flat line** → outlier severity uniform across layers → global α is well-matched, per-layer α has little room.
- **Trend with depth** (rises or falls monotonically) → there is a *structural* layer-position effect a per-layer α could exploit.
- **Spiky / non-monotonic** → some layers are individually much harder than their neighbours, which is the strongest possible motivation: no global α can handle both a 2× layer and a 100× layer well at once.
- **`q_proj` vs `fc1` differ in their pattern** → motivates not just per-layer but per-site α (the fused-linear group inside a layer can have its own α).

---

## Cell 5: Severity distribution + summary stats

A second view: ignore layer order, just look at the spread of severities as a histogram per model. Tells us the magnitude of variation independent of where the hard layers sit.

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
for tag, color in [("opt-2.7b", "#1f77b4"), ("opt-6.7b", "#2ca02c"), ("opt-13b", "#d62728")]:
    sub = all_df[all_df.model == tag]
    ax.hist(sub["severity"], bins=25, alpha=0.45, label=f"{tag} (n={len(sub)})", color=color)
ax.set_xscale("log")
ax.set_xlabel("severity = max / median (log scale)")
ax.set_ylabel("count of sites")
ax.set_title("Distribution of per-site outlier severity across OPT scales")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_png = "/content/llm-quantization-thesis/results/task05/outlier_severity_histogram.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
plt.show()

# Per-model summary
summary = (
    all_df.groupby("model")["severity"]
    .agg(["min", "median", "max", lambda s: s.max() / s.min()])
    .rename(columns={"<lambda_0>": "spread_max_over_min"})
)
print("\nSeverity summary per model:")
print(summary.round(2))
summary.to_csv("/content/llm-quantization-thesis/results/task05/severity_summary.csv")
```

---

## Cell 6: Robustness check — does p99/median tell the same story?

`severity = max/median` can in principle be dominated by one freak channel. If we replace `max` with `quantile(s, 0.99)` (i.e. ignore the top 1% of channels) and the across-layer pattern still shows a wide spread, the variation is *structural* in the layer's distribution, not an artifact of one outlier channel.

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=False)
for ax, tag in zip(axes, ["opt-2.7b", "opt-6.7b", "opt-13b"]):
    df = all_df[all_df.model == tag]
    for site, marker, color in [("q_proj", "o", "#1f77b4"), ("fc1", "s", "#d62728")]:
        sub = df[df.site == site].sort_values("layer")
        ax.plot(sub["layer"], sub["p99_over_median"], marker=marker, color=color,
                label=site, linewidth=1.2, markersize=5)
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("p99(|X|) / median(|X|)")
    ax.set_title(tag)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

fig.suptitle("Per-layer severity using p99/median (robust variant)", y=1.02)
fig.tight_layout()
out_png = "/content/llm-quantization-thesis/results/task05/outlier_severity_p99_per_layer.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
plt.show()
print(f"saved -> {out_png}")
```

---

## What we expect to see (and what each outcome would mean)

| Pattern | Interpretation for per-layer α |
| --- | --- |
| Flat severity, all layers within ~1.5× of each other | Global α already well-matched; per-layer α has little room — drop the idea or pivot. |
| Monotonic trend (e.g. severity rises with depth) | A *parametric* per-layer α (e.g. `α_l = α_0 + β · l/L`) is a strong candidate — only one extra knob over the global recipe. |
| Spiky, layer-specific outlier hotspots | Full per-layer α is justified; consider whether the hot layers correlate with known SmoothQuant failure points (large outlier channels in early/late layers). |
| `q_proj` and `fc1` follow different patterns | Per-site α (a single α per fused-linear group) gives more headroom than per-layer α. |
| Severity spread grows with model size (2.7B → 13B) | Strongest motivation: the limitation of global α is exactly the regime that matters for deployment-scale models. |

After running the notebook, write findings into `experiments/task05_per_layer_alpha/PROGRESS.md` (per-model spread numbers + a one-line read of the plots) before deciding whether to proceed to a per-layer α PPL sweep.
