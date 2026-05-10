# Task 02: Percentile Sweep — C scheme on OPT-13B (A100-80)

Sweeps the smoothing percentile `p` and SmoothQuant `alpha` against scheme C (per-channel W + per-token A — the Task 01 winner).

Grid (full Task 02 sweep convention):
- `p ∈ {0.999, 0.995, 0.99, 0.95, 0.90}` (5 values)
- `alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (5 values, step=0.2)
- 25 runs total. On A100, OPT-13B PPL eval is ~3× the 6.7B time (~12-15 min/run including model reload), so total ≈ **5-6 hours**.

`p = 1.0` is intentionally excluded (top-K calibration's per-channel max is fp16-noisy). The max-smoothing reference baseline is Task 01's `alpha_sweep_results.ipynb` for OPT-13B once that exists.

Hardware:
- **GPU**: A100-80. Each sweep run loads OPT-13B in **bfloat16** for evaluation (matching the smaller-model sweeps), so peak GPU during eval is ~26 GB model + ~3-5 GB activations at seq_len=2048 ≈ 30 GB. Plenty of headroom on the 80 GB card.
- **CPU RAM**: Colab Pro High-RAM (~51 GB) is enough. Model reloads between cells briefly peak host RAM at ~25 GB.

Prereq: run [generate_act_percentiles_cells.md](generate_act_percentiles_cells.md) once first to produce `act_percentiles/opt-13b/p<value>.pt` files on Drive.

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
!mkdir -p /content/drive/MyDrive/thesis_results/task02_percentile_sweep/opt-13b

# Verify all per-p scale files are present
!nvidia-smi
!cat /proc/meminfo | head -3
!ls -la /content/drive/MyDrive/thesis_results/act_percentiles/opt-13b/
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

Includes resume support via per-run JSON markers. Given the 5-6 hour total wall time on OPT-13B, a Colab disconnect mid-sweep is plausible — re-running this cell picks up where it left off.

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

MODEL = "facebook/opt-13b"
PCT_DIR = "/content/drive/MyDrive/thesis_results/act_percentiles/opt-13b"
SAVE_DIR = "/content/drive/MyDrive/thesis_results/task02_percentile_sweep/opt-13b"

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


print("Loading tokenizer + dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda")


def scales_path_for(p):
    return os.path.join(PCT_DIR, f"p{p:g}.pt")


for p in PERCENTILES:
    path = scales_path_for(p)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scale file for p={p}: {path}")
print(f"All {len(PERCENTILES)} per-p scale files present.")


def already_done(p, alpha):
    fname = f"opt-13b_C_p{p:g}_a{alpha}.json"
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
            with open(os.path.join(SAVE_DIR, f"opt-13b_C_p{p:g}_a{alpha}.json")) as f:
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
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
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

        fname = f"opt-13b_C_p{p:g}_a{alpha}.json"
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
print(f"  FULL PERCENTILE × ALPHA SWEEP — OPT-13B, Scheme C")
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

print("\nBest p per alpha:")
for a in ALPHAS:
    rows = [r for r in all_results if r["alpha"] == a]
    br = min(rows, key=lambda r: r["wikitext2_ppl"])
    print(f"  alpha={a}: best p={br['p']:g} -> PPL={br['wikitext2_ppl']:.4f}")

worst = max(all_results, key=lambda r: r["wikitext2_ppl"])
print(f"\nWorst in sweep: p={worst['p']:g}, alpha={worst['alpha']}, PPL={worst['wikitext2_ppl']:.4f}")
```

---

## Notes

- **Same A100-40-with-CPU-buffer setup** as the calibration cell. The eval phase doesn't need the calibration buffers — those have been written to Drive and freed. So during the sweep, the GPU only holds the model (~26 GB bf16) plus activations.
- **Resume support**: per-run JSON markers; re-running the cell picks up where it left off if Colab disconnects.
- **No `p=1.0` row** in this sweep (deliberate). Reference baseline lives in Task 01.
- **Per-p .pt files are loaded one at a time** (then freed before the next `p`) so peak CPU memory stays small. Each file is ~30-50 MB for OPT-13B.
- **Total wall time ≈ 5-6 hours**. Run on a stable A100 instance (JarvisLabs is more reliable than Colab for runs of this length).
