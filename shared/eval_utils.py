"""
Evaluation utilities for benchmarking quantized models.
WikiText-2 perplexity + zero-shot tasks via lm-evaluation-harness.
"""

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


ZEROSHOT_TASKS = ["lambada_openai", "hellaswag", "piqa", "winogrande"]


def get_model_device(model) -> torch.device:
    """
    Get the device of a model, handling device_map='auto' where 
    model.device raises an error because parameters are on multiple devices.
    """
    try:
        return model.device
    except (AttributeError, ValueError):
        # Model is split across devices — get the device of the first parameter
        return next(model.parameters()).device


@torch.no_grad()
def evaluate_perplexity_wikitext2(model, tokenizer, max_length=2048):
    """
    Compute WikiText-2 perplexity.
    Uses the standard approach from quantization papers: 
    encode full test set, process in non-overlapping chunks.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # [1, seq_len]
    seq_len = input_ids.size(1)
    
    device = get_model_device(model)
    
    nlls = []
    n_tokens = 0
    
    for begin in tqdm(range(0, seq_len - 1, max_length), desc="WikiText-2 PPL"):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        targets = chunk.clone()
        targets[:, 0] = -100  # don't compute loss on first token of chunk
        
        outputs = model(chunk, labels=targets)
        
        # Count non-ignored tokens
        valid_tokens = (targets != -100).sum().item()
        nlls.append(outputs.loss.float().item() * valid_tokens)
        n_tokens += valid_tokens
        
        if end >= seq_len:
            break
    
    ppl = float(np.exp(sum(nlls) / n_tokens))
    return round(ppl, 2)


def evaluate_zeroshot(model, tokenizer, tasks=None, batch_size=1):
    """
    Run zero-shot evaluation using lm-evaluation-harness.
    Handles API differences across lm-eval versions.
    """
    import lm_eval
    
    if tasks is None:
        tasks = ZEROSHOT_TASKS
    
    # lm-eval >= 0.4 API
    try:
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    except (ImportError, TypeError):
        # Older API fallback
        from lm_eval.models.huggingface import HuggingFaceModel
        lm = HuggingFaceModel(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
    )
    
    accs = {}
    for task in tasks:
        task_res = results.get("results", {}).get(task, {})
        # lm-eval v0.4+ uses 'acc,none'; older uses 'acc'
        acc = task_res.get("acc,none", task_res.get("acc"))
        if acc is not None:
            accs[task] = round(float(acc), 6)
        else:
            print(f"  Warning: no accuracy found for {task}, keys={list(task_res.keys())}")
            accs[task] = None
    
    valid = [v for v in accs.values() if v is not None]
    accs["average"] = round(np.mean(valid), 6) if valid else None
    
    return accs


def run_full_evaluation(model, tokenizer, skip_zeroshot=False, skip_ppl=False):
    """Run complete evaluation suite. Returns dict with all metrics."""
    results = {}
    
    if not skip_ppl:
        print("Computing WikiText-2 perplexity...")
        ppl = evaluate_perplexity_wikitext2(model, tokenizer)
        results["wikitext2_ppl"] = ppl
        print(f"  WikiText-2 perplexity: {ppl}")
    
    if not skip_zeroshot:
        print("Running zero-shot evaluation...")
        zs = evaluate_zeroshot(model, tokenizer)
        results["zeroshot"] = zs
        results["avg_zeroshot_acc"] = zs.get("average")
        print(f"  Average zero-shot accuracy: {zs.get('average', '?')}")
    
    return results