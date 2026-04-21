# CLAUDE.md

MTech thesis (IIT Guwahati, 2026) on activation quantization for efficient LLM inference. Builds on SmoothQuant (Xiao et al., ICML 2023) and LLM.int8() (Dettmers et al., NeurIPS 2022).

## Execution environment

- All GPU experiments run on Google Colab (free-tier T4 or Pro A100)
- Local machine is for editing and git only — code is NOT run locally
- Results save to Google Drive at `/content/drive/MyDrive/thesis_results/`

## Colab setup (proven working)

```python
!git clone https://github.com/Anand-786/llm-quantization-thesis.git
%cd /content/llm-quantization-thesis
!git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo
!pip uninstall smoothquant -y
!cd smoothquant_repo && pip install -e .
!pip install -q transformers accelerate datasets zstandard tqdm
import sys; sys.path.insert(0, "/content/llm-quantization-thesis/smoothquant_repo")
from google.colab import drive
drive.mount('/content/drive')
```

Key: install smoothquant from the cloned repo, NOT from PyPI. Add `sys.path.insert` for notebook kernel.

## Activation scales

Generated via `smoothquant_repo/examples/generate_act_scales.py` with Pile validation set.
On Drive at `/content/drive/MyDrive/thesis_results/act_scales/`. Generated so far: `opt-125m.pt`, `opt-1.3b.pt`

## Project structure

```
experiments/          — cell .md files (reproducible instructions) per task/model
  task01_scheme_exploration/
    PROGRESS.md       — detailed findings, data tables, thesis framing
    experiment_plan.md
    opt_125m/         — cells for scheme comparison and alpha sweep (step=0.1, 9 levels)
    opt_1_3b/         — cells for scheme comparison and alpha sweep (step=0.2, 5 levels)
    opt_6_7b/         — cells + scales generation for A100
results/              — .ipynb notebooks with actual outputs
  task01/opt_125m/    — scheme_comparison, alpha_sweep
  task01/opt_1_3b/    — initial_validation, scheme_comparison, alpha_sweep
docs/                 — reference material (paper PDF)
shared/               — reusable Python utilities
smoothquant_repo/     — cloned official SmoothQuant repo (not tracked in git)
```

## Progress overview

### Task 01 — Scheme exploration

Detailed progress: [experiments/task01_scheme_exploration/PROGRESS.md](experiments/task01_scheme_exploration/PROGRESS.md)

**Status**: OPT-125M and OPT-1.3B complete. On 125M, config C (per-channel W + per-token A) wins at all 9/9 alpha levels (step=0.1 sweep) and is ~10× more alpha-stable than O1 (spread 0.26 vs 2.5 PPL). O1 collapses at alpha≥0.8 to worse-than-naive-W8A8, exposing per-tensor weight quantization as the failure point. On 1.3B, C wins 4/5 alphas. Next: OPT-2.7B (fills the ladder), then OPT-6.7B on A100.

## Challenges faced (for thesis reporting)

Running log of concrete obstacles hit during the work and how they were resolved. Useful material for the "Challenges" / "Engineering considerations" section of the thesis.

**IMPORTANT for Claude**: whenever a new obstacle is hit during experiments (OOM, numerical issues, repo incompatibilities, tooling breakage, reproducibility problems, etc.), append a new entry here with a short title, what went wrong, why it mattered, and how it was resolved (or "unresolved, pending X"). Keep entries tight — one short paragraph each.

1. **SmoothQuant repo ships no pre-computed activation scales.** The `smoothquant_repo/act_scales/` folder referenced by the paper's eval scripts is empty. Every model we evaluate needs its own scales generated from scratch via `examples/generate_act_scales.py` on the Pile validation set (512 samples × 512 seq_len). On Colab T4 this takes ~15-25 min per model and has to be done before any scheme comparison can run. Resolved by building a separate `generate_act_scales_cells.md` per model and caching outputs to Drive at `/content/drive/MyDrive/thesis_results/act_scales/`.

2. **OPT-6.7B does not fit on Colab free-tier T4 for PPL eval at seq_len=2048.** Weights alone occupy ~13.3 GiB of the T4's 14.56 GiB; a single SDPA attention forward at seq_len=2048 then OOMs asking for ~128 MiB more. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did not close the gap. Reducing seq_len would cut attention memory quadratically but would **invalidate comparability**: PPL is defined relative to the eval protocol — shorter context gives each prediction less history and multiplies the cold-start penalty, so numbers at seq_len=1024 are not comparable to the 1.3B results at 2048 or to the SmoothQuant paper's 2048 convention. CPU offload via `accelerate`'s `max_memory` is viable but would make 16 runs take a full evening. Resolved by moving to Colab Pro A100 (13.3 GiB model fits comfortably in 40 GiB with headroom for 2048-token attention). Keep this in mind for any future model ≥ 7B — T4 is not an option at seq_len=2048.
