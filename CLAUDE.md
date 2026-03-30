# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MTech thesis (IIT Guwahati, 2026) investigating alternative approaches to activation quantization for efficient LLM inference. Builds on SmoothQuant (Xiao et al., ICML 2023) and LLM.int8() (Dettmers et al., NeurIPS 2022).

## Setup & Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install smoothquant

# Clone SmoothQuant repo (needed for activation scales and smooth_lm function)
git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo

# Sanity check (verifies GPU, model loading, quantization functions)
python experiments/task01_scheme_exploration/quick_sanity_test.py

# Run an experiment
python experiments/task01_scheme_exploration/run_scheme_eval.py \
    --model opt-1.3b --scheme O3 --alpha 0.5 --save_dir ./results

# Run via Colab wrapper (handles Drive paths automatically)
python colab/colab_runner.py --sanity
python colab/colab_runner.py --model opt-1.3b --scheme O3 --alpha 0.5
```

## Architecture

### Quantization Pipeline Flow
1. Load model via `shared/model_utils.py` (FP16, auto device mapping)
2. Optionally apply SmoothQuant smoothing (`smooth_lm()` from smoothquant_repo)
3. Replace all `nn.Linear` modules with `FakeQuantLinear` wrappers (quantize→dequantize simulation, no actual INT8 kernels)
4. Evaluate: WikiText-2 perplexity + zero-shot tasks (LAMBADA, HellaSwag, PIQA, WinoGrande) via `shared/eval_utils.py`
5. Save results as JSON via `shared/save_utils.py`

### Key Modules
- **shared/model_utils.py**: `MODEL_REGISTRY` maps short names (opt-1.3b, opt-6.7b, llama2-7b, llama3-8b) to HuggingFace IDs
- **shared/quant_utils.py**: Three quantization granularities — `quantize_tensor_absmax` (per-tensor), `quantize_tensor_per_token` (row-wise), `quantize_tensor_per_channel` (column-wise). All symmetric.
- **shared/eval_utils.py**: `run_full_evaluation()` orchestrates perplexity + zero-shot benchmarks
- **experiments/task01_scheme_exploration/run_scheme_eval.py**: `SCHEME_MAP` defines all quantization scheme configurations; `FakeQuantLinear` is the drop-in nn.Linear replacement

### Quantization Schemes (SCHEME_MAP)
- **O1/O2/O3**: SmoothQuant original schemes (absmax weights with varying activation granularity)
- **O1_pcw/O2_pcw/O3_pcw**: Extended schemes with per-channel weight quantization
- **naive_w8a8**: W8A8 without smoothing
- **fp16**: No quantization baseline

### External Dependencies
- **smoothquant_repo/act_scales/**: Pre-computed activation scales (.pt files) per model, loaded by `shared/quant_utils.py`
- **smoothquant_repo/smoothquant/**: Library providing `smooth_lm()` function imported in experiment scripts

## Conventions

- Each experiment lives in its own `experiments/taskNN_*/` directory with its own README
- All experiment results are JSON files with consistent schema (task, model, scheme, alpha, metrics, gpu, duration_seconds)
- Model weights and caches are gitignored; result JSONs are tracked
- Designed for Colab-first execution with Google Drive integration, but works locally with GPU

## Experiment execution
- All GPU experiments run on Google Colab (free tier T4 or Pro A100)
- Code is NOT run locally — local machine is for editing and git only
- SmoothQuant repo is cloned inside Colab at smoothquant_repo/
- Activation scales must be generated per model using their generate_act_scales.py
- Results save to Google Drive at /content/drive/MyDrive/thesis_results/

## Key files in SmoothQuant (cloned separately, not in this repo)
- smoothquant/smooth.py → smooth_lm() applies the core smoothing transformation
- smoothquant/fake_quant.py → quantize_model() replaces Linear with W8A8Linear
- smoothquant/ppl_eval.py → end-to-end evaluation script we use for experiments
- smoothquant/calibration.py → get_act_scales() generates activation scale .pt files