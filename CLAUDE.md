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
    opt_1_3b/         — cells for scheme comparison and alpha sweep
results/              — .ipynb notebooks with actual outputs
  task01/opt_1_3b/    — initial_validation, scheme_comparison, alpha_sweep
docs/                 — reference material (paper PDF)
shared/               — reusable Python utilities
smoothquant_repo/     — cloned official SmoothQuant repo (not tracked in git)
```

## Progress overview

### Task 01 — Scheme exploration

Detailed progress: [experiments/task01_scheme_exploration/PROGRESS.md](experiments/task01_scheme_exploration/PROGRESS.md)

**Status**: OPT-1.3B complete. Compared paper schemes (O1, O2) against per-channel weight variants (C, D). Alpha sweep shows config C (per-channel W + per-token A) beats O1 at 4 of 5 tested alpha levels and is more robust to alpha selection. Next: OPT-6.7B.
