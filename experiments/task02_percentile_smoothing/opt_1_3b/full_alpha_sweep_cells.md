# Task 02 Experiment 2: Full Alpha Sweep at the Two Competing Optima — OPT-1.3B (A100)

Experiment 1 left us with two candidate optima on OPT-1.3B (scheme C):
- `p = 0.999, alpha = 0.9 → 14.6167 PPL`
- `p = 0.90, alpha = 0.7 → 14.6272 PPL`

Experiment 2 separates them with a full alpha sweep at each `p`. Per the user's request this round, alpha is swept across **all 9 values** `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` — one-off, to confirm the optimum is not hiding outside the previously-tested `{0.5, 0.7, 0.9}` grid.

Grid:
- `p ∈ {0.999, 0.90}` (2 values)
- `alpha ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` (9 values)
- Scheme: C (per-channel W + per-token A)
- 18 runs total, ~3 min/run on A100 → ~55 min end-to-end.

`p = 1.0` is intentionally excluded (top-K calibration's per-channel max is fp16-noisy). The max-smoothing baseline is the Task 01 `alpha_sweep_results.ipynb` for OPT-1.3B; compare against that.

Prereq: run [generate_act_percentiles_cells.md](generate_act_percentiles_cells.md) once first to produce `act_percentiles/opt-1.3b/p<value>.pt` files on Drive. This experiment reuses the existing `p0.999.pt` and `p0.9.pt`; nothing new to calibrate.

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

# Drive output dir for this experiment's results
!mkdir -p /content/drive/MyDrive/thesis_results/task02_full_alpha_sweep/opt-1.3b

# Verify per-p scale files we need are on Drive
!nvidia-smi
!ls -la /content/drive/MyDrive/thesis_results/act_percentiles/opt-1.3b/p0.999.pt
!ls -la /content/drive/MyDrive/thesis_results/act_percentiles/opt-1.3b/p0.9.pt
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

## Cell 3: Run the full alpha sweep at p=0.999 and p=0.90

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

MODEL = "facebook/opt-1.3b"
PCT_DIR = "/content/drive/MyDrive/thesis_results/act_percentiles/opt-1.3b"
SAVE_DIR = "/content/drive/MyDrive/thesis_results/task02_full_alpha_sweep/opt-1.3b"

# Two competing optima from Experiment 1.
PERCENTILES = [0.999, 0.90]
# Full alpha range — one-off check that the optimum isn't outside {0.5, 0.7, 0.9}.
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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


# Fail fast if any required scale file is missing.
for p in PERCENTILES:
    path = scales_path_for(p)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scale file for p={p}: {path}")
print(f"All {len(PERCENTILES)} per-p scale files present.")


all_results = []
total_runs = len(PERCENTILES) * len(ALPHAS)
run_num = 0

for p in PERCENTILES:
    print(f"\n[loading scales] p={p:g} from {scales_path_for(p)}")
    scales_for_p = torch.load(scales_path_for(p))

    for alpha in ALPHAS:
        run_num += 1
        config_label = f"C-pct{p:g}-a{alpha}"
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

        fname = f"opt-1.3b_C_p{p:g}_a{alpha}.json"
        with open(os.path.join(SAVE_DIR, fname), "w") as f:
            json.dump(result, f, indent=2)

        print(f">>> {config_label}: PPL = {ppl_val:.4f} ({elapsed:.0f}s)")

        del model
        torch.cuda.empty_cache()

    del scales_for_p

with open(os.path.join(SAVE_DIR, "all_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
```

---

## Cell 4: Summary table and per-p best

```python
print(f"\n{'='*80}")
print(f"  FULL ALPHA SWEEP — OPT-1.3B, Scheme C, p ∈ {PERCENTILES}")
print(f"{'='*80}")
print(f"\n{'p':>8} " + " ".join(f"{'a=' + str(a):>9}" for a in ALPHAS))
print("-" * (10 + 10 * len(ALPHAS)))

by_pa = {(r["p"], r["alpha"]): r["wikitext2_ppl"] for r in all_results}
for p in PERCENTILES:
    row = f"{p:>8g} "
    for a in ALPHAS:
        ppl = by_pa.get((p, a))
        row += f" {ppl:>9.4f}" if ppl is not None else f" {'--':>9}"
    print(row)

# Per-p best
print("\nBest alpha per p:")
for p in PERCENTILES:
    rows = [r for r in all_results if r["p"] == p]
    br = min(rows, key=lambda r: r["wikitext2_ppl"])
    print(f"  p={p:g}: best alpha={br['alpha']} -> PPL={br['wikitext2_ppl']:.4f}")

best = min(all_results, key=lambda r: r["wikitext2_ppl"])
print(f"\nOverall best: p={best['p']:g}, alpha={best['alpha']}, PPL={best['wikitext2_ppl']:.4f}")
```

To compare against max smoothing, look up the corresponding number from Task 01's `results/task01/opt_1_3b/alpha_sweep_results.ipynb` — both the C/max row at the same alpha and the paper's O1/max number (14.68 reference).

---

## Notes

- **Why both `p` together**: a single notebook run keeps eval setup fixed across the 18 runs (tokenizer, evaluator dataset, dtype). Splitting into two notebooks risks tiny harness differences that would muddy comparisons.
- **Per-p .pt files are loaded one at a time** (then freed before the next `p`) so peak CPU memory stays small.
- **One-time full alpha grid**: future Task 02 sweeps on bigger models should go back to the `{0.1, 0.3, 0.5, 0.7, 0.9}` step=0.2 grid — running a fine alpha grid for every p × scheme combination is wasteful. We only fine-grain here because the 1.3B sweep showed the optimal alpha drifts with p, and we want an unambiguous reading on this particular model before scaling up.
- **Nothing new to calibrate**: scales come from the existing `p0.999.pt` and `p0.9.pt` on Drive.
