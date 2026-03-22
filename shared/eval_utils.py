"""
Evaluation utilities for benchmarking quantized models.
Uses lm-evaluation-harness for zero-shot tasks and manual WikiText-2 perplexity.
"""

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


# The four zero-shot tasks matching SmoothQuant's evaluation
ZEROSHOT_TASKS = ["lambada_openai", "hellaswag", "piqa", "winogrande"]


def evaluate_zeroshot(model, tokenizer, tasks=None, batch_size=1, num_fewshot=0):
    """
    Run zero-shot evaluation using lm-evaluation-harness.
    
    Args:
        model: HuggingFace model (already on device).
        tokenizer: Corresponding tokenizer.
        tasks: List of task names. Default: SmoothQuant's 4 tasks.
        batch_size: Eval batch size (keep 1 for memory safety).
        num_fewshot: Number of few-shot examples (0 for zero-shot).
    
    Returns:
        Dict of {task_name: accuracy} plus 'average' key.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if tasks is None:
        tasks = ZEROSHOT_TASKS

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    # Extract accuracies
    accs = {}
    for task in tasks:
        task_res = results["results"].get(task, {})
        # lm-eval uses 'acc' or 'acc,none' depending on version
        acc = task_res.get("acc,none", task_res.get("acc", None))
        if acc is not None:
            accs[task] = round(float(acc), 6)
        else:
            print(f"  Warning: no accuracy found for {task}")
            accs[task] = None

    valid = [v for v in accs.values() if v is not None]
    accs["average"] = round(np.mean(valid), 6) if valid else None

    return accs


@torch.no_grad()
def evaluate_perplexity_wikitext2(model, tokenizer, max_length=2048):
    """
    Compute WikiText-2 perplexity using sliding window.
    This matches the standard evaluation used in quantization papers.
    
    Args:
        model: HuggingFace model.
        tokenizer: Corresponding tokenizer.
        max_length: Maximum sequence length for evaluation.
    
    Returns:
        Perplexity (float).
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    nlls = []
    stride = max_length // 2  # 50% overlap

    for begin in tqdm(range(0, seq_len, stride), desc="WikiText-2 PPL"):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(model.device)
        
        target = chunk.clone()
        # Only compute loss on the non-overlapping part (except first window)
        if begin > 0:
            target[:, :stride] = -100

        outputs = model(chunk, labels=target)
        nll = outputs.loss.float() * (target != -100).sum()
        nlls.append(nll.item())

        if end == seq_len:
            break

    total_tokens = (input_ids.size(1) - 1)  # approximate
    ppl = float(np.exp(sum(nlls) / total_tokens))

    return round(ppl, 2)


def run_full_evaluation(model, tokenizer, skip_zeroshot=False, skip_ppl=False):
    """
    Run the complete evaluation suite.
    
    Returns:
        Dict with all metrics.
    """
    results = {}

    if not skip_zeroshot:
        print("Running zero-shot evaluation...")
        zs = evaluate_zeroshot(model, tokenizer)
        results["zeroshot"] = zs
        results["avg_zeroshot_acc"] = zs.get("average")
        print(f"  Average zero-shot accuracy: {zs.get('average', '?')}")

    if not skip_ppl:
        print("Computing WikiText-2 perplexity...")
        ppl = evaluate_perplexity_wikitext2(model, tokenizer)
        results["wikitext2_ppl"] = ppl
        print(f"  WikiText-2 perplexity: {ppl}")

    return results
