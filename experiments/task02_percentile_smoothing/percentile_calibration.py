"""Per-channel exact-percentile calibration for SmoothQuant Task 02.

Replaces SmoothQuant's per-channel `max(|X|)` activation statistic with a
per-channel exact `quantile(|X|, p)` for each `p` in a configured set. The
implementation keeps a per-channel top-K buffer per smoothing site (the input
of `q_proj` and the input of `fc1` in OPT) where
    K = ceil((1 - p_min) * N) + safety_margin,
    N = num_samples * seq_len.
The p-th percentile of `N` samples is the value at sorted-rank `p * (N - 1)`,
which sits inside the top `(1 - p) * N` largest values for any `p >= p_min`.
A single sorted top-K buffer therefore yields every requested `p >= p_min`
*exactly*, matching `torch.quantile`'s linear-interpolation convention.

Compared to a histogram (estimator) or full-sample retention (~48 GB for OPT-
1.3B), the top-K buffer is ~10x smaller and gives bit-exact quantiles. Hooks
attach only to the linears whose inputs feed the SmoothQuant smoothing
operator, not every linear in the model.
"""

import functools
import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def _smoothing_site_names(model: nn.Module) -> List[str]:
    """Qualified names of the OPT linears whose inputs feed `smooth_lm`.

    OPT's `smooth_lm` reads scales for `<layer>.self_attn.q_proj` (input shared
    with k/v through self_attn_layer_norm) and `<layer>.fc1` (input from
    final_layer_norm). Those are the only sites we need to calibrate.
    """
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    sites: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            sites.append(f"{name}.self_attn.q_proj")
            sites.append(f"{name}.fc1")
    return sites


@torch.no_grad()
def get_act_percentiles(
    model: nn.Module,
    tokenizer,
    dataset_path: str,
    percentiles: Iterable[float] = (1.0, 0.999, 0.995, 0.99, 0.95, 0.90),
    num_samples: int = 512,
    seq_len: int = 512,
    buffer_device: str = "cuda",
    buffer_dtype: torch.dtype = torch.float16,
    safety_margin: int = 16,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute exact per-channel percentiles of |X| for the OPT smoothing sites.

    Args:
        model: an OPT causal LM, already on `cuda` and in eval mode.
        tokenizer: matching tokenizer.
        dataset_path: path to Pile validation `.jsonl.zst` (same as the
            upstream `generate_act_scales.py`).
        percentiles: iterable of `p` values in (0, 1]. Must include the
            smallest as `p_min` of the application; all entries are read from
            one buffer.
        num_samples: calibration samples (paper default 512).
        seq_len: max tokens per sample (paper default 512).
        buffer_device: where the per-site top-K buffers live. "cuda" for
            models <= 6.7B on A100; "cpu" for 13B.
        buffer_dtype: dtype of stored values. fp16 is plenty (we only care
            about magnitudes, not signed precision).
        safety_margin: extra slots on top of `ceil((1 - p_min) * N)` to give
            interpolation room and absorb ties.

    Returns:
        `dict[p_str -> dict[name -> tensor[in_features]]]`. `p_str` is
        `f"{p:g}"` (e.g. "1", "0.99", "0.999"); the inner dict mirrors the
        layout of `smoothquant`'s `act_scales`.
    """
    p_list: List[float] = sorted(set(float(p) for p in percentiles))
    if not p_list:
        raise ValueError("`percentiles` must be non-empty")
    for p in p_list:
        if not (0.0 < p <= 1.0):
            raise ValueError(f"percentile {p} must be in (0, 1]")

    p_min = p_list[0]
    n_total = num_samples * seq_len
    k = math.ceil((1.0 - p_min) * n_total) + safety_margin
    k = max(k, 2)
    k = min(k, n_total)

    model.eval()
    device = next(model.parameters()).device

    site_names = _smoothing_site_names(model)
    if not site_names:
        raise RuntimeError("No OPT decoder layers found — is this an OPT model?")
    sites = {name: None for name in site_names}  # name -> nn.Linear, filled below
    for name, module in model.named_modules():
        if name in sites and isinstance(module, nn.Linear):
            sites[name] = module
    missing = [n for n, m in sites.items() if m is None]
    if missing:
        raise RuntimeError(f"Could not resolve smoothing-site modules: {missing[:3]}...")

    cpu_buffer = (str(buffer_device) == "cpu")
    buffers: Dict[str, torch.Tensor] = {}
    for name, lin in sites.items():
        in_features = lin.in_features
        buf = torch.full(
            (k, in_features),
            fill_value=float("-inf"),
            dtype=buffer_dtype,
            device=buffer_device,
        )
        if cpu_buffer and torch.cuda.is_available():
            # Pinned memory makes the per-site H2D copy on each batch faster
            # when we page the buffer up to GPU for the topk update.
            buf = buf.pin_memory()
        buffers[name] = buf

    def update_buffer(name: str, x: torch.Tensor) -> None:
        in_features = x.shape[-1]
        new = x.reshape(-1, in_features).abs().to(buffer_dtype)
        if cpu_buffer:
            # Buffer lives in host RAM (e.g. OPT-13B where the GPU is full of
            # weights). Page just this site's buffer up to GPU for the topk —
            # it's only ~270 MB per site at 13B and ~10x faster than a CPU
            # topk over the same shape.
            if new.device.type != "cuda" and torch.cuda.is_available():
                new = new.cuda(non_blocking=True)
            buf_gpu = buffers[name].to("cuda", non_blocking=True) if torch.cuda.is_available() else buffers[name]
            combined = torch.cat([buf_gpu, new], dim=0)
            topk = torch.topk(combined, k=k, dim=0, largest=True, sorted=False).values
            buffers[name].copy_(topk.cpu(), non_blocking=True)
            del buf_gpu, combined, topk
        else:
            if new.device.type != "cuda":
                new = new.to(buffer_device)
            combined = torch.cat([buffers[name], new], dim=0)
            topk = torch.topk(combined, k=k, dim=0, largest=True, sorted=False).values
            buffers[name].copy_(topk)

    def stat_input_hook(_m, inputs, _output, name):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(x, tuple):
            x = x[0]
        update_buffer(name, x)

    hooks = []
    for name, lin in sites.items():
        hooks.append(lin.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.shuffle(seed=42)
        for i in tqdm(range(num_samples), desc="Top-K calibration"):
            input_ids = tokenizer(
                dataset[i]["text"],
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
            ).input_ids.to(device)
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return _percentiles_from_topk_buffers(buffers, p_list, n_total)


def _percentiles_from_topk_buffers(
    buffers: Dict[str, torch.Tensor],
    p_list: List[float],
    n_total: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Read each requested percentile from the per-site sorted top-K buffers.

    Matches `torch.quantile`'s default linear-interpolation convention:
    sorted ascending, position = p * (N - 1), interpolate between floor and
    ceil ranks.
    """
    out: Dict[str, Dict[str, torch.Tensor]] = {f"{p:g}": {} for p in p_list}

    for name, buf in buffers.items():
        # Sort ascending. After sort, sorted_buf[-1] is the per-channel max.
        sorted_buf = torch.sort(buf, dim=0).values  # [K, in_features]
        k = sorted_buf.shape[0]

        for p in p_list:
            pos = p * (n_total - 1)  # exact position in a length-N sorted array
            # Rank from the top: top_rank = (n_total - 1) - pos
            # Index into our top-K (which is the top of the global sort, also
            # ascending now so its last row is the global max).
            # Position inside top-K, ascending: idx = k - 1 - top_rank
            #                                       = k - 1 - ((n_total - 1) - pos)
            #                                       = k - n_total + pos
            idx_f = float(k) - float(n_total) + pos
            if idx_f < 0:
                # `p` is below `p_min`: we don't have enough buffer to answer.
                # In practice we only call with p >= p_min so this is unreachable.
                raise ValueError(
                    f"percentile p={p} requires more retained samples than K={k} "
                    f"(N={n_total}); increase `percentiles` floor or buffer size"
                )
            lo = int(math.floor(idx_f))
            hi = int(math.ceil(idx_f))
            lo = max(0, min(k - 1, lo))
            hi = max(0, min(k - 1, hi))
            if hi == lo:
                val = sorted_buf[lo].clone()
            else:
                w = idx_f - lo
                val = sorted_buf[lo].float() * (1.0 - w) + sorted_buf[hi].float() * w
                val = val.to(sorted_buf.dtype)
            out[f"{p:g}"][name] = val.detach().cpu()

    return out
