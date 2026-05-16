# Verification: WikiText-2 PPL table — Falcon-7B (paper-comparable)

Single-session run producing a **3-row table** against SmoothQuant paper Table 7 for Falcon-7B.

**Reference (paper Table 7):**

| Method | PPL    | α    |
|--------|--------|------|
| FP16   | 6.590  | —    |
| W8A8 SQ (per-token A, per-channel W) | 6.629 | 0.60 |

Same C scheme (per-token activations, per-channel weights) as Llama. The interesting bit: Falcon's paper-tuned α is **0.60**, much lower than Llama-2's 0.85. This is a good test of whether α(l) can land near-lossless across a wide regime of paper-optimal α values **without retuning the range**.

**Runs (3 total):**

| # | Label              | Smoothing                       | Quantizer (W / A)             | Purpose                                |
|---|--------------------|---------------------------------|-------------------------------|-----------------------------------------|
| 1 | FP16               | —                               | none (fp16)                   | Baseline; verify protocol vs paper      |
| 2 | C max α=0.60       | max,   α=0.60                   | per_channel / per_token       | Paper's exact config (their 6.629 row)  |
| 3 | C per-layer α      | max,   α(l) ∈ [0.5, 0.9]        | per_channel / per_token       | Our recipe — original OPT-style range   |

**Prereqs on Drive:**
- `act_scales/falcon-7b.pt` — from [`generate_act_scales_cells.md`](generate_act_scales_cells.md).

**Architectural notes:**
- Falcon-7B has `parallel_attn=True` and `new_decoder_architecture=False`. A single `input_layernorm` per layer absorbs into **both** `self_attention.query_key_value` and `mlp.dense_h_to_4h`. The smoothquant repo's `smooth_lm` handles this — our per-layer applier replicates that same dispatch logic.
- 32 decoder layers, hidden=4544, LayerNorm (not RMSNorm).
- Module names use `transformer.h.<i>.*`, **not** `model.layers.<i>.*` like Llama.

**Hardware:** Falcon-7B fp16 ≈ 13.5 GB. **A100-40GB recommended.** Each PPL run ~5-8 min on A100; 3-run sweep ≈ 25-35 min plus loading time.

---

## Cell 1: Setup

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm einops

from google.colab import drive
drive.mount('/content/drive')

import os, shutil
SAVE_DIR = "/content/drive/MyDrive/thesis_results/verification/falcon-7b"
os.makedirs(SAVE_DIR, exist_ok=True)

DRIVE_SCALES = "/content/drive/MyDrive/thesis_results/act_scales/falcon-7b.pt"
REPO_SCALES  = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/falcon-7b.pt"
assert os.path.exists(DRIVE_SCALES), f"missing: {DRIVE_SCALES} — run generate_act_scales_cells.md first."
os.makedirs(os.path.dirname(REPO_SCALES), exist_ok=True)
shutil.copy2(DRIVE_SCALES, REPO_SCALES)

print("max scales :", REPO_SCALES)
!nvidia-smi
```

---

## Cell 2: Per-layer α schedule (severity → α(l))

Severity = `max / median` of per-channel input scales at `self_attention.query_key_value` and `mlp.dense_h_to_4h`, averaged per layer. **Linear normalisation** (same as the working Llama-2 versions), with the **original OPT-style `[0.5, 0.9]` range** — the paper's Falcon-7B optimum α=0.60 sits well inside this range so no shifting needed.

```python
import re, torch, json

SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/falcon-7b.pt"
SAVE_DIR    = "/content/drive/MyDrive/thesis_results/verification/falcon-7b"
ALPHA_MIN, ALPHA_MAX = 0.5, 0.9   # original range — paper α=0.6 sits inside this

raw = torch.load(SCALES_PATH, map_location="cpu")
LAYER_RE = re.compile(r"transformer\.h\.(\d+)\.(.+)")

sev_by_layer = {}
for name, vec in raw.items():
    m = LAYER_RE.match(name)
    if not m:
        continue
    suffix = m.group(2)
    if suffix not in ("self_attention.query_key_value", "mlp.dense_h_to_4h"):
        continue
    layer = int(m.group(1))
    v = vec.float().abs()
    s = (v.max() / v.median().clamp(min=1e-12)).item()
    sev_by_layer.setdefault(layer, []).append(s)

layers = sorted(sev_by_layer.keys())
sev = torch.tensor([sum(sev_by_layer[l]) / len(sev_by_layer[l]) for l in layers])

# Linear normalisation (same as Llama-2 final versions).
sev_norm = sev / sev.max()
alpha_per_layer = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sev_norm).tolist()

assert len(alpha_per_layer) == 32, f"expected 32 layers, got {len(alpha_per_layer)}"
print(f"per-layer α range: [{min(alpha_per_layer):.3f}, {max(alpha_per_layer):.3f}]  (32 layers)")
print(f"severity spread (max/min): {sev.max().item() / sev.min().item():.2f}×")
print()
print("layer  severity   α(l)")
for l, s, a in zip(layers, sev.tolist(), alpha_per_layer):
    print(f"  {l:2d}    {s:7.2f}    {a:.3f}")

with open(f"{SAVE_DIR}/falcon-7b_alpha_schedule.json", "w") as f:
    json.dump({
        "layers": layers,
        "severity": sev.tolist(),
        "alpha_per_layer": alpha_per_layer,
        "alpha_range": [ALPHA_MIN, ALPHA_MAX],
        "normalisation": "linear",
    }, f, indent=2)
print(f"\nschedule saved -> {SAVE_DIR}/falcon-7b_alpha_schedule.json")
```

---

## Cell 3: Per-layer α applier (Falcon)

Replicates the upstream `smooth_lm` dispatch logic for `FalconDecoderLayer` but with a per-layer α. For Falcon-7B (parallel_attn=True, new_decoder_arch=False) the single `input_layernorm` absorbs into both QKV and FFN paths via one `smooth_ln_fcs(ln, [qkv, fc1], ...)` call; for Falcon-40B-style configs (new_decoder_arch or non-parallel) it falls back to the two-LN branch. Both branches matter because we may run this on 40B later.

```python
%%writefile /content/smooth_per_layer_falcon.py
"""Per-layer α extension of smoothquant.smooth.smooth_lm (Falcon)."""
import re
import torch
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from smoothquant.smooth import smooth_ln_fcs

LAYER_RE = re.compile(r"transformer\.h\.(\d+)$")

@torch.no_grad()
def smooth_lm_per_layer_falcon(model, scales, alpha_schedule):
    if not isinstance(alpha_schedule, (list, tuple)):
        raise TypeError("alpha_schedule must be a list/tuple of floats")
    for name, module in model.named_modules():
        if not isinstance(module, FalconDecoderLayer):
            continue
        m = LAYER_RE.match(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        alpha = float(alpha_schedule[layer_idx])

        qkv = module.self_attention.query_key_value
        fc1 = module.mlp.dense_h_to_4h
        qkv_input_scales = scales[name + ".self_attention.query_key_value"]
        fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]

        if (not module.config.new_decoder_architecture
                and module.config.parallel_attn):
            # Falcon-7B path: one input_layernorm absorbs into both qkv and fc1.
            attn_ln = module.input_layernorm
            smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
        else:
            # Falcon-40B-style path: separate norms for attn and ffn.
            attn_ln = (module.ln_attn if module.config.new_decoder_architecture
                       else module.input_layernorm)
            ffn_ln  = (module.ln_mlp  if module.config.new_decoder_architecture
                       else module.post_attention_layernorm)
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            smooth_ln_fcs(ffn_ln,  fc1, fc1_input_scales, alpha)
```

---

## Cell 4: Shared setup (tokenizer, dataset, evaluator, helpers)

```python
import sys
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")

import torch, torch.nn as nn, json, time, tqdm, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from smooth_per_layer_falcon import smooth_lm_per_layer_falcon

MODEL = "tiiuae/falcon-7b"
SCALES_PATH = "/content/llm-quantization-thesis/smoothquant_repo/act_scales/falcon-7b.pt"
SAVE_DIR    = "/content/drive/MyDrive/thesis_results/verification/falcon-7b"


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
        self.n_samples = n_samples
    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n = self.n_samples
        for i in tqdm.tqdm(range(n), desc="PPL"):
            batch = self.dataset[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * 2048)
        return torch.exp(torch.stack(nlls).sum() / (n * 2048))


print("Loading tokenizer + dataset + scales...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL)

# Tokenizer sanity check — Falcon uses HF byte-BPE, no SentencePiece quirks expected.
_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids[0].tolist()
_dec = tokenizer.decode(_ids)
print(f"tokenizer class: {type(tokenizer).__name__}")
print(f"sanity decode  : {_dec}")
assert "<unk>" not in _dec, "tokenizer is producing <unk> for spaces — bail out before wasting GPU time."

dataset    = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator  = Evaluator(dataset, tokenizer, "cuda")
act_scales = torch.load(SCALES_PATH)

C_QPARAMS   = dict(weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
PAPER_ALPHA = 0.60  # SmoothQuant paper Table 7 — Falcon-7B row

results = []  # appended to by each run cell below

def _run_config(label, smooth_spec, qparams):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    if smooth_spec == "none":
        pass
    elif smooth_spec[0] == "max":
        smooth_lm(model, act_scales, smooth_spec[1])
    elif smooth_spec[0] == "perlayer":
        smooth_lm_per_layer_falcon(model, act_scales, smooth_spec[1])
    else:
        raise ValueError(f"unknown smooth spec: {smooth_spec}")

    if qparams is not None:
        model = quantize_model(model, **qparams)

    ppl = evaluator.evaluate(model).item()
    elapsed = time.time() - t0
    print(f">>> {label}: PPL = {ppl:.4f}  ({elapsed:.0f}s)")

    rec = {
        "model": MODEL,
        "label": label,
        "smooth": (smooth_spec if isinstance(smooth_spec, str)
                   else (smooth_spec[0] if smooth_spec[0] != "perlayer" else "perlayer")),
        "qparams": qparams,
        "ppl": round(ppl, 4),
        "seconds": round(elapsed, 1),
    }
    results.append(rec)
    with open(f"{SAVE_DIR}/falcon-7b_{label}.json", "w") as f:
        json.dump(rec, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return rec

print("setup ready — run the three config cells below in order.")
```

---

## Cell 5: Run 1 — FP16 baseline

```python
_run_config("1_FP16", "none", None)
```

---

## Cell 6: Run 2 — C + max α=0.60 (paper config)

```python
_run_config(f"2_C_max_a{PAPER_ALPHA}", ("max", PAPER_ALPHA), C_QPARAMS)
```

---

## Cell 7: Run 3 — C + per-layer α (ours)

```python
_run_config("3_C_perlayer", ("perlayer", alpha_per_layer), C_QPARAMS)
```

---

## Cell 8: Final results table — paper comparison

```python
def ppl(label):
    return next(r for r in results if r["label"] == label)["ppl"]

PAPER_FP16, PAPER_SQ, PAPER_ALPHA = 6.590, 6.629, 0.60
our_fp16    = ppl("1_FP16")
our_paper   = ppl(f"2_C_max_a{PAPER_ALPHA}")
our_alpha_l = ppl("3_C_perlayer")

print(f"\n{'='*78}")
print(f"  Falcon-7B — WikiText-2 PPL — paper-comparable table")
print(f"{'='*78}")
print(f"\n{'config':<30} {'ours':>10} {'paper':>10} {'Δ ours−paper':>14}")
print("-" * 70)
print(f"{'FP16':<30} {our_fp16:>10.4f} {PAPER_FP16:>10.4f} {our_fp16 - PAPER_FP16:>+14.4f}")
print(f"{'C + max α=0.60 (paper cfg)':<30} {our_paper:>10.4f} {PAPER_SQ:>10.4f} {our_paper - PAPER_SQ:>+14.4f}")
print(f"{'C + per-layer α (ours)':<30} {our_alpha_l:>10.4f} {'—':>10} {'—':>14}")

print(f"\nDeltas vs OUR FP16 (within-session, noise-free):")
print(f"  C max α=0.60       − FP16  = {our_paper   - our_fp16:+.4f}")
print(f"  C per-layer α      − FP16  = {our_alpha_l - our_fp16:+.4f}")
print(f"  C per-layer        − C max = {our_alpha_l - our_paper:+.4f}  (negative → α(l) wins)")

print(f"\nPaper-gap reference:")
print(f"  paper W8A8 − paper FP16    = {PAPER_SQ - PAPER_FP16:+.4f}  (Table 7)")

shift = our_fp16 - PAPER_FP16
print(f"\nProtocol-shift diagnostic:")
print(f"  our FP16 − paper FP16  = {shift:+.4f}")
if abs(shift) < 0.05:
    print(f"  → protocols match; can cite paper numbers directly.")
else:
    print(f"  → protocols differ by ~{shift:+.3f} PPL; compare *within-session* deltas only.")

with open(f"{SAVE_DIR}/falcon-7b_summary.json", "w") as f:
    json.dump({"results": results, "alpha_per_layer": alpha_per_layer}, f, indent=2)
print(f"\nsaved -> {SAVE_DIR}/falcon-7b_summary.json")
```

---

## Notes

- **α range stays [0.5, 0.9]** — the original OPT-style range. Paper Falcon-7B optimum is α=0.60, comfortably inside this range, so no shifting is needed. This is the strongest version of the "no per-model tuning" claim: same range that worked for OPT/Llama-1, applied to a model whose paper-optimum α is 0.25 lower than Llama-2's.
- **Linear normalisation** — same choice that worked on Llama-2 final runs. The log-norm variant was tried on Llama-2 earlier and performed *worse* than linear; we're not re-exploring that path here.
- **Cell 4's tokenizer sanity check** — Falcon uses a byte-BPE tokenizer (no SentencePiece), so we don't expect the Llama-2-13B-style failure. The assert in Cell 4 catches the regression early if anything breaks.
- **Honest framing**: target is for the C+per-layer row to land within ±0.01 PPL of C+max α=0.60. That's "matches without grid search," consistent with the Llama-2 result framing.
- Results land at `/content/drive/MyDrive/thesis_results/verification/falcon-7b/`.
