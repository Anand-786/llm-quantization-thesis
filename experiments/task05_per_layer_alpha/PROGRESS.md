# Task 05: Per-Layer Alpha — Progress

## Motivation diagnostic (done — 2026-05-10)

Loaded existing `act_scales/opt-{2.7b,6.7b,13b}.pt` (per-channel `max(|X|)` over Pile-val 512×512) and computed two per-site severity measures: `max/median` and `p99/median`. Plots and CSV at [results/task05/](../../results/task05/).

### Key numbers

| Model  | sites | min severity | median | max  | spread (max/min) |
| ------ | ----- | ------------ | ------ | ---- | ---------------- |
| 2.7B   | 64    | 2.77         | 15.0   | 36.2 | **13.1×**        |
| 6.7B   | 64    | 2.62         | 21.3   | 59.9 | **22.9×**        |
| 13B    | 80    | 1.77         | 23.3   | 82.1 | **46.3×**        |

Spread **grows monotonically with model scale** — the regime where deployment matters is exactly the regime where global α is most compromised.

### Per-layer pattern (`max/median`) — same template across all three scales

- Layer 0: severity ~1.5–3 (embedding feed, distribution still flat).
- Layers 1–3: sharp ramp to a peak (≈36 / 60 / 82 for 2.7B / 6.7B / 13B).
- Layers 3–end: smooth, near-monotonic decay to ~10–15.
- **`q_proj` and `fc1` trace each other almost exactly** — one α-per-layer can serve both sites; we do not need per-site α.
- No spikes, no chaos. Shape is **structured and predictable** → a parametric per-layer α (2–3 hyperparameters) should be enough; a 32/40-knob grid search is overkill.

### Per-layer pattern (`p99/median`) — different shape, bathtub

- Layer 0 high; drops fast through layers 1–5; flat through the middle; **rises again at the last 3–4 layers** in 2.7B and 6.7B (weaker / absent in 13B).
- Combined with `max/median`, this resolves into three regimes:
  - **Early (0–3):** both `max/median` and `p99/median` elevated → outliers are present *and* spread across many channels.
  - **Middle (4–25):** `max/median` still 15–25 but `p99/median` ≈ 1.3–1.4 → outliers are *concentrated in 1–2 freak channels*; the rest of the distribution is tame.
  - **Late (26–end):** `max/median` settles low but `p99/median` rises → distribution *broadens*; outliers are spread, not freak.

### What this implies

1. A global α is provably suboptimal: variation is 13–46× depending on scale.
2. The structure is so consistent across scales that a parametric `α(l)` (e.g. `α(l) = α_low + (α_high − α_low) · g(l/L)` with a bump-then-decay `g`) is the right shape — not a free per-layer search.
3. The middle-layer pattern (high `max/median`, low `p99/median`) is *secondary motivation for percentile-based smoothing* (Task 02): max-based smoothing in those layers is dominated by 1–2 channels that don't reflect the layer's real activation distribution. This re-frames Task 02 as already addressing one of the two axes of difficulty exposed here.

## Results — single-session FP16-anchored verification

Source: [`experiments/verification/opt_{1_3b,2_7b}/full_table_cells.md`](../verification/). One Colab kernel per model, 8 configs sharing one `Evaluator` so the per-layer-α delta is noise-free against the matched global-α=0.5 row.

α(l) schedule used (zero free parameters): `α(l) = 0.5 + 0.4 · sev(l) / max_l sev(l)` with `sev(l) = mean over {q_proj, fc1} of max(|X|) / median(|X|)`.

| Model    | Scheme | global α=0.5 | per-layer α(l) | Δ (positive ⇒ per-layer wins) |
|----------|--------|-------------:|---------------:|------------------------------:|
| OPT-125M | O1     | 28.3081      | 29.4631        | −1.1550                       |
| OPT-125M | C      | 27.5991      | 27.5859        | +0.0132                       |
| OPT-1.3B | O1     | 14.8333      | 14.6949        | **+0.1384**                   |
| OPT-1.3B | C      | 14.7710      | 14.6281        | **+0.1429**                   |
| OPT-2.7B | O1     | 12.3885      | 12.4867        | −0.0982                       |
| OPT-2.7B | C      | 12.3613      | 12.3670        | −0.0057                       |
| OPT-6.7B | O1     | 10.6988      | 10.7382        | −0.0394                       |
| OPT-6.7B | C      | 10.7252      | 10.6826        | **+0.0426**                   |
| OPT-13B  | O1     | 10.1767      |  9.9963        | **+0.1804**                   |
| OPT-13B  | C      | 10.1691      |  9.9825        | **+0.1866**                   |

### Interpretation

- **C-scheme per-layer wins at every scale except 2.7B**, where global α=0.5 already lands within 0.016 of FP16 — no room left to recover.
- **O1 per-layer wins at 1.3B and 13B, loses at 125M / 2.7B / 6.7B.** Per-tensor weights penalise layer-varying α; the win only appears when the activation-side gain is large enough to outweigh it (1.3B has headroom; 13B has very sharp outliers).
- **Pattern**: per-layer α pays off when global α=0.5 is leaving PPL on the table for that scheme. That is *not* a monotonic function of scale.
- **Two-routes-same-floor with Task 02 percentile**:
  - 1.3B: C/per-layer 14.6281 ≈ C/pct (0.999, 0.9) 14.6248
  - 6.7B: C/per-layer 10.6826 ≈ C/pct (0.995, 0.5) 10.6875
  - 13B: C/per-layer 9.9825 vs C/pct (0.995, 0.5) 10.0077 (C/pct uses placeholder; true Task 02 13B winner not yet known)

## Next steps (planned)

See discussion thread in conversation 2026-05-10. Headline plan:
1. Lock percentile p at Task 02's winner per model, vary α per-layer.
2. Start with a **two-region α**: one for the peak band (layers 1–4), one for the rest. Two knobs total. Grid-search on OPT-1.3B with the C-pct config from Task 02.
3. If two-region beats global, fit the parametric `α(l)` form to the severity curve and check that it generalises to 2.7B / 6.7B without per-model retuning.
4. Defer the `p99/median` axis (per-layer p) to a follow-up; address one axis at a time.
