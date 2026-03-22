# LLM Quantization Thesis

**MTech Thesis — IIT Guwahati, 2026**

Investigating alternative approaches to activation quantization for efficient LLM inference, building on SmoothQuant (Xiao et al., ICML 2023) and LLM.int8() (Dettmers et al., NeurIPS 2022).

## Project Structure

```
shared/          → Reusable utilities (quantization, evaluation, model loading)
experiments/     → One folder per approach/task (numbered sequentially)
analysis/        → Cross-experiment comparison scripts and plots
colab/           → Template scripts for running on Google Colab
docs/            → Research plan, notes, paper summaries
```

## Experiments

| Task | Name | Status |
|------|------|--------|
| 01 | Quantization Scheme Exploration | 🔄 In Progress |
| 02 | Task-Specific Calibration | ⬜ Not Started |
| 03 | Edge-Scale Model Analysis | ⬜ Not Started |
| 04 | Percentile-Based Smoothing | ⬜ Not Started |
| 05 | Per-Layer Adaptive α | ⬜ Not Started |
| 06 | Combined Metric Smoothing | ⬜ Not Started |
| 07 | Percentile + Adaptive α Combined | ⬜ Not Started |
| baseline | FP16 Baselines | ⬜ Not Started |

## Quick Start (Colab)

```bash
# In a Colab cell:
!git clone https://github.com/YOUR_USERNAME/llm-quantization-thesis.git
%cd llm-quantization-thesis
!pip install -r requirements.txt
!pip install smoothquant

# Run Task 1 for a specific model and scheme:
!python experiments/task01_scheme_exploration/run_scheme_eval.py \
    --model facebook/opt-1.3b \
    --scheme O1 \
    --alpha 0.5 \
    --save_dir /content/drive/MyDrive/thesis_results/task01/
```

## Reference Papers
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., ICML 2023
- [LLM.int8()](https://arxiv.org/abs/2208.07339) — Dettmers et al., NeurIPS 2022
