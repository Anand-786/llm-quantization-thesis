"""Real-INT8 evaluation runner for Task 04.

Modes:
  --mode fp16         : load fp16 HF model, eval.
  --mode int8_hf      : load Int8OPTForCausalLM from HF (paper's O3+max).
  --mode int8_local   : load Int8OPTForCausalLM from a local export dir
                        (produced by export_our_int8.py).

Metrics:
  - GPU model size (params + buffers) in MB.
  - WikiText-2 PPL @ seq_len=2048 (Task 01 protocol).
  - LAMBADA last-token accuracy + per-sample latency (paper demo protocol).

Saves a JSON to --save_json.
"""
import argparse
import gc
import json
import time

import torch
from torch.nn.functional import pad


def model_size_mb(model):
    p = sum(x.nelement() * x.element_size() for x in model.parameters())
    b = sum(x.nelement() * x.element_size() for x in model.buffers())
    return (p + b) / (1024 ** 2)


@torch.no_grad()
def eval_wikitext_ppl(model, tokenizer, seq_len=2048, device="cuda"):
    """Task 01-style WikiText-2 PPL: concat test split, slide non-overlapping windows."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n_tokens = enc.shape[1]
    n_windows = n_tokens // seq_len
    nlls = []
    model.eval()
    for i in range(n_windows):
        ids = enc[:, i * seq_len : (i + 1) * seq_len]
        out = model(ids, labels=ids)
        # out.loss is mean NLL over the window
        nlls.append(out.loss.float() * seq_len)
    ppl = torch.exp(torch.stack(nlls).sum() / (n_windows * seq_len))
    return ppl.item()


@torch.no_grad()
def eval_lambada(model, tokenizer, n_samples=1000, pad_to=512, device="cuda"):
    from datasets import load_dataset
    ds = load_dataset("lambada", split=f"validation[:{n_samples}]")

    def tok(ex):
        return tokenizer(ex["text"])
    ds = ds.map(tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids"])

    total = hit = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latency_ms = 0.0
    model.eval()
    for batch in ds:
        ids = batch["input_ids"].to(device).unsqueeze(0)
        label = ids[:, -1]
        pad_len = pad_to - ids.shape[1]
        if pad_len < 0:
            ids = ids[:, :pad_to]
            pad_len = 0
        ids = pad(ids, (0, pad_len), value=1)
        torch.cuda.synchronize()
        start.record()
        out = model(ids)
        end.record()
        torch.cuda.synchronize()
        latency_ms += start.elapsed_time(end)
        last_logits = out.logits[:, -2 - pad_len, :]
        pred = last_logits.argmax(dim=-1)
        total += label.size(0)
        hit += (pred == label).sum().item()
    return hit / total, latency_ms / len(ds)


def load_model(args):
    if args.mode == "fp16":
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        return OPTForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto"
        )
    elif args.mode == "int8_hf":
        from smoothquant.opt import Int8OPTForCausalLM
        return Int8OPTForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto"
        )
    elif args.mode == "int8_local":
        from smoothquant.opt import Int8OPTForCausalLM
        return Int8OPTForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto"
        )
    raise ValueError(args.mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fp16", "int8_hf", "int8_local"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", required=True,
                    help="HF tokenizer name; usually facebook/opt-<size>")
    ap.add_argument("--config_label", required=True)
    ap.add_argument("--save_json", required=True)
    ap.add_argument("--seq_len_ppl", type=int, default=2048)
    ap.add_argument("--lambada_samples", type=int, default=1000)
    ap.add_argument("--skip_ppl", action="store_true")
    ap.add_argument("--skip_lambada", action="store_true")
    args = ap.parse_args()

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)

    t0 = time.time()
    model = load_model(args)
    load_s = time.time() - t0

    size_mb = model_size_mb(model)
    print(f"[{args.config_label}] model size: {size_mb:.2f} MB  (load {load_s:.1f}s)")

    result = {
        "config_label": args.config_label,
        "mode": args.mode,
        "model_path": args.model_path,
        "size_mb": size_mb,
        "seq_len_ppl": args.seq_len_ppl,
    }

    if not args.skip_ppl:
        ppl = eval_wikitext_ppl(model, tokenizer, seq_len=args.seq_len_ppl)
        print(f"[{args.config_label}] wikitext-2 PPL @ {args.seq_len_ppl}: {ppl:.4f}")
        result["wikitext2_ppl"] = ppl

    if not args.skip_lambada:
        acc, lat = eval_lambada(model, tokenizer, n_samples=args.lambada_samples)
        print(f"[{args.config_label}] LAMBADA last-token acc: {acc:.4f}  latency: {lat:.3f} ms/sample")
        result["lambada_last_token_acc"] = acc
        result["lambada_latency_ms_per_sample"] = lat

    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {args.save_json}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
