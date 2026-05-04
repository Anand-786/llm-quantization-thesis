# Task 02: Percentile Sweep — C scheme on OPT-6.7B (A100)

Sweeps the smoothing percentile `p` and SmoothQuant `alpha` against scheme C (per-channel W + per-token A — the Task 01 winner).

Grid (full Task 02 sweep convention from OPT-6.7B onward):
- `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}` (5 values)
- `alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (5 values, step=0.2)
- 25 runs total. On A100, OPT-6.7B PPL eval takes ~2× the 2.7B time (~4-5 min/run including model reload), so total ≈ 1.5-2 hours.

α=0.9 + low-p cells will likely give very high PPL on OPT-6.7B too (the OPT-2.7B sweep gave 323.79 at p=0.95 and 1765.90 at p=0.90 for that α). They're included for completeness of the figure rather than as candidate optima.

`p = 1.0` is intentionally excluded (top-K calibration's per-channel max is fp16-noisy by ~one ULP). The max-smoothing reference baseline is Task 01's `alpha_sweep_results.ipynb` for OPT-6.7B (FP16 ceiling, SQ-O1, SQ-PCW-PT, etc.).

Prereq: run [generate_act_percentiles_cells.md](generate_act_percentiles_cells.md) once first to produce `act_percentiles/opt-6.7b/p<value>.pt` files on Drive.

---

## Cell 1: Setup

```python
# Clone repos + install
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Drive output dir for sweep results
!mkdir -p /content/drive/MyDrive/thesis_results/task02_percentile_sweep/opt-6.7b

# Verify all per-p scale files are present
!nvidia-smi
!ls -la /content/drive/MyDrive/thesis_results/act_percentiles/opt-6.7b/
!python -c "from smoothquant.fake_quant import quantize_model; print('smoothquant OK')"
```

---

## Cell 2: Write the percentile-smoothing helper

Self-contained `%%writefile` so the notebook does not depend on whether `experiments/task02_percentile_smoothing/percentile_smooth.py` has been pushed to GitHub yet.

```python
%%writefile /content/percentile_smooth.py
"""Percentile-based smoothing for SmoothQuant Task 02 (OPT only).

Mirrors `smoothquant.smooth.smooth_lm` but replaces both activation- and
weight-side per-channel `max(|.|)` with per-channel `quantile(., p)`. When
`p == 1.0` it falls back to exact `max` so the baseline row of any sweep is
bit-identical to the upstream `smooth_lm`.
"""
import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer


@torch.no_grad()
def _per_channel_weight_stat(fcs, p_w, dtype):
    stacked = torch.cat([fc.weight.abs() for fc in fcs], dim=0)
    if p_w >= 1.0:
        scales = stacked.max(dim=0).values
    else:
        scales = stacked.float().quantile(p_w, dim=0).to(stacked.dtype)
    return scales.to(dtype).clamp_(min=1e-5)


@torch.no_grad()
def smooth_ln_fcs_pct(ln, fcs, act_scales, alpha=0.5, p_w=1.0):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype).clamp_(min=1e-5)
    weight_scales = _per_channel_weight_stat(fcs, p_w, dtype).to(device)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm_pct(model, scales, alpha=0.5, p_w=1.0):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_pct(attn_ln, qkv, qkv_input_scales, alpha, p_w)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs_pct(ffn_ln, fc1, fc1_input_scales, alpha, p_w)
```

---

## Cell 3: Run full 5p × 5α sweep (25 runs)

```python
import sys
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")
sys.path.insert(0, "/content")  # for the %%writefile'd percentile_smooth.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.fake_quant import quantize_model
from percentile_smooth import smooth_lm_pct
from datasets import load_dataset
import json, os, time, tqdm

MODEL = "facebook/opt-6.7b"
PCT_DIR = "/content/drive/MyDrive/thesis_results/act_percentiles/opt-6.7b"
SAVE_DIR = "/content/drive/MyDrive/thesis_results/task02_percentile_sweep/opt-6.7b"

# Sweep grid. Each p maps to its own .pt file under PCT_DIR.
# p=1.0 is intentionally excluded — see the note in the doc header.
PERCENTILES = [0.999, 0.995, 0.99, 0.95, 0.90]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Scheme C — per-channel W + per-token A (Task 01 winner).
WEIGHT_QUANT = "per_channel"
ACT_QUANT = "per_token"


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n), desc="Evaluating"):
            batch = self.dataset[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            nlls.append(loss.float() * 2048)
        return torch.exp(torch.stack(nlls).sum() / (n * 2048))


# --- Load tokenizer + dataset once; lazy-load each per-p file when needed ---
print("Loading tokenizer + dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda")


def scales_path_for(p):
    return os.path.join(PCT_DIR, f"p{p:g}.pt")


# Sanity-load every required scale file before starting the sweep, so we fail
# fast if any are missing rather than ~2 hours in.
for p in PERCENTILES:
    path = scales_path_for(p)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scale file for p={p}: {path}")
print(f"All {len(PERCENTILES)} per-p scale files present.")


# Resume support: skip cells whose per-run JSON is already saved. Useful if the
# Colab runtime disconnects mid-sweep — re-running the cell picks up where it
# left off rather than restarting from scratch.
def already_done(p, alpha):
    fname = f"opt-6.7b_C_p{p:g}_a{alpha}.json"
    return os.path.exists(os.path.join(SAVE_DIR, fname))


all_results = []
total_runs = len(PERCENTILES) * len(ALPHAS)
run_num = 0

for p in PERCENTILES:
    print(f"\n[loading scales] p={p:g} from {scales_path_for(p)}")
    scales_for_p = torch.load(scales_path_for(p))

    for alpha in ALPHAS:
        run_num += 1
        config_label = f"C-pct{p:g}-a{alpha}"

        if already_done(p, alpha):
            with open(os.path.join(SAVE_DIR, f"opt-6.7b_C_p{p:g}_a{alpha}.json")) as f:
                cached = json.load(f)
            all_results.append(cached)
            print(f"\n[{run_num}/{total_runs}] {config_label}: cached PPL = {cached['wikitext2_ppl']:.4f} (skipped)")
            continue

        print(f"\n{'='*60}")
        print(f"  Run {run_num}/{total_runs}: {config_label}")
        print(f"  Scheme C (per_channel W, per_token A)")
        print(f"  p={p}, alpha={alpha}")
        print(f"{'='*60}")

        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )

        smooth_lm_pct(model, scales_for_p, alpha=alpha, p_w=p)

        model = quantize_model(
            model,
            weight_quant=WEIGHT_QUANT,
            act_quant=ACT_QUANT,
            quantize_bmm_input=True,
        )

        ppl = evaluator.evaluate(model)
        elapsed = time.time() - start
        ppl_val = ppl.item()

        result = {
            "config_label": config_label,
            "model": MODEL,
            "scheme": "C",
            "weight_quant": WEIGHT_QUANT,
            "act_quant": ACT_QUANT,
            "p": p,
            "alpha": alpha,
            "wikitext2_ppl": round(ppl_val, 4),
            "duration_seconds": round(elapsed, 1),
        }
        all_results.append(result)

        fname = f"opt-6.7b_C_p{p:g}_a{alpha}.json"
        with open(os.path.join(SAVE_DIR, fname), "w") as f:
            json.dump(result, f, indent=2)

        print(f">>> {config_label}: PPL = {ppl_val:.4f} ({elapsed:.0f}s)")

        del model
        torch.cuda.empty_cache()

    del scales_for_p

# --- Save combined results ---
with open(os.path.join(SAVE_DIR, "all_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
```

---

## Cell 4: Summary table

```python
print(f"\n{'='*90}")
print(f"  FULL PERCENTILE × ALPHA SWEEP — OPT-6.7B, Scheme C")
print(f"{'='*90}")
print(f"\n{'p':>8} " + " ".join(f"{'alpha=' + str(a):>12}" for a in ALPHAS))
print("-" * (10 + 13 * len(ALPHAS)))

by_pa = {(r["p"], r["alpha"]): r["wikitext2_ppl"] for r in all_results}
for p in PERCENTILES:
    row = f"{p:>8g} "
    for a in ALPHAS:
        ppl = by_pa.get((p, a))
        row += f" {ppl:>12.4f}" if ppl is not None else f" {'--':>12}"
    print(row)

best = min(all_results, key=lambda r: r["wikitext2_ppl"])
print(f"\nBest in sweep: p={best['p']:g}, alpha={best['alpha']}, PPL={best['wikitext2_ppl']:.4f}")

# Best PPL per alpha — useful to see how the optimal alpha shifts as p drops.
print("\nBest p per alpha:")
for a in ALPHAS:
    rows = [r for r in all_results if r["alpha"] == a]
    br = min(rows, key=lambda r: r["wikitext2_ppl"])
    print(f"  alpha={a}: best p={br['p']:g} -> PPL={br['wikitext2_ppl']:.4f}")

worst = max(all_results, key=lambda r: r["wikitext2_ppl"])
print(f"\nWorst in sweep: p={worst['p']:g}, alpha={worst['alpha']}, PPL={worst['wikitext2_ppl']:.4f}")
```

To compare against max smoothing, look up the corresponding number from Task 01 (`results/task01/opt_6_7b/alpha_sweep_results.ipynb` once that exists) — both the C/max row at the same alpha and the paper's O1/max row.

---

## Notes

- **A100 required** for any OPT-6.7B work at seq_len=2048 (CLAUDE.md challenge #2). T4 will OOM.
- **Resume support in Cell 3**: a per-run JSON marker is written per cell. If Colab disconnects, re-running Cell 3 picks up from the last incomplete cell. The 2.7B run lost the summary cell to a runtime disconnect; the resume logic eliminates that risk on the longer 6.7B sweep.
- **No `p=1.0` row** in this sweep (deliberate). The fp16 buffer makes it ~one-ULP noisy on per-channel max. Reference baseline is Task 01.
- **Low-p + α=0.9 cells are kept** for grid completeness even though they're expected to score very high PPL based on the 2.7B run.
- **Per-p .pt files are loaded one at a time** (then freed before the next `p`) so peak CPU memory stays small. Each file is ~20-30 MB for OPT-6.7B.
