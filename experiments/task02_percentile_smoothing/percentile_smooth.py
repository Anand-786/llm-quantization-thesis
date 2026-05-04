"""Percentile-based smoothing for SmoothQuant Task 02.

Mirrors `smoothquant.smooth.smooth_lm` / `smooth_ln_fcs` but replaces both the
activation-side and weight-side per-channel `max(|.|)` with a per-channel
`quantile(., p)`. The activation percentile comes from
`percentile_calibration.get_act_percentiles`; the weight percentile is computed
on the fly from the model parameters.

When `p == 1.0` we fall back to exact `max` so the baseline row of any sweep
is bit-identical to the upstream `smoothquant.smooth.smooth_lm`.

Only the OPT branch is implemented (the only family used in this thesis).
"""

import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer


@torch.no_grad()
def _per_channel_weight_stat(
    fcs, p_w: float, dtype: torch.dtype
) -> torch.Tensor:
    """Per-input-channel statistic over the fused linears sharing a LayerNorm.

    Mirrors smoothquant's `weight_scales` computation but with `quantile(p_w)`
    instead of `max` (when p_w < 1.0). Output shape: [in_features].
    """
    # Stack |W| from each fused linear along the row (out_features) dim, then
    # reduce along that dim to get one statistic per in_feature column.
    stacked = torch.cat([fc.weight.abs() for fc in fcs], dim=0)  # [sum_out, in]
    if p_w >= 1.0:
        scales = stacked.max(dim=0).values
    else:
        # torch.quantile only supports float32/float64; cast then come back.
        scales = stacked.float().quantile(p_w, dim=0).to(stacked.dtype)
    return scales.to(dtype).clamp_(min=1e-5)


@torch.no_grad()
def smooth_ln_fcs_pct(ln, fcs, act_scales, alpha=0.5, p_w=1.0):
    """Drop-in replacement for `smoothquant.smooth.smooth_ln_fcs`.

    `act_scales` is the per-channel activation percentile (or max when
    p_w==1.0) from calibration. `p_w` is the weight-side percentile.
    """
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
    """Drop-in replacement for `smoothquant.smooth.smooth_lm` (OPT only).

    Args:
        model: an OPT-family causal LM.
        scales: dict[name -> tensor[in_features]] of per-channel activation
            percentile (or max, if p_w==1.0). Same key structure as the
            standard `act_scales` dict — keys are the qualified module names
            of the q_proj / fc1 inputs.
        alpha: SmoothQuant alpha.
        p_w: weight-side percentile in (0, 1]. When 1.0, recovers the exact
            upstream smoothing formula.
    """
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
