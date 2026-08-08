"""The single Qwen GQA Taylor matrix-state plus prefix-mean initializer."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from .gdn2rwkv import (
    W_SCALE,
    _low_rank_target,
    _native_decay_inverse,
    fit_native_gate,
    fit_time_mix,
    refit_native_output,
)

CLOSURE = 0.25


def _previous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), 1)


@torch.no_grad()
def initialize_gqa_layer(
    source,
    target,
    normalized_hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Initialize two D256 states for every one of Qwen's eight query heads."""
    x = normalized_hidden
    batch, length, channels = x.shape
    q, source_gate = torch.chunk(source.q_proj(x).view(*x.shape[:-1], 8, 512), 2, -1)
    q = source.q_norm(q).transpose(1, 2)
    k = source.k_norm(source.k_proj(x).view(batch, length, 2, 256)).transpose(1, 2)
    v = source.v_proj(x).view(batch, length, 2, 256).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k = repeat_kv(k, 4)
    v = repeat_kv(v, 4)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / 256**0.5
    causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
    probabilities = scores.masked_fill(causal, -torch.inf).softmax(-1)
    exact_attention = torch.matmul(probabilities, v.float())
    hazard = probabilities.diagonal(dim1=-2, dim2=-1)

    # The first state stores the centered affine operator.  The second stores
    # the prefix value mean in one non-RoPE DC column.  Keep the physical
    # [query_head, state, channel] order used by Qwen2RWKVTimeMix._pair_sum().
    scale = 256**-0.25
    q_prime = q.float().transpose(1, 2) * scale
    k_prime = k.float().transpose(1, 2) * scale
    source_value = v.float().transpose(1, 2)
    steps = torch.arange(1, length + 1, device=x.device, dtype=torch.float32)
    rho = steps.reciprocal().view(1, length, 1, 1)
    mean_key = k_prime.cumsum(1) * rho
    mean_value = source_value.cumsum(1) * rho
    previous_mean_value = torch.cat(
        (torch.zeros_like(mean_value[:, :1]), mean_value[:, :-1]), dim=1
    )
    centered_key = k_prime - mean_key
    value_innovation = source_value - previous_mean_value
    centered_norm = centered_key.square().sum(-1, keepdim=True).sqrt()
    centered_direction = centered_key / centered_norm.clamp_min(1e-6)
    dc = torch.zeros(256, dtype=torch.float32, device=x.device)
    dc[-1] = 1
    dc = dc.view(1, 1, 1, 256).expand(batch, length, 8, -1)

    q_target = torch.stack((q_prime, dc), dim=3).reshape(batch, length, 4096)
    k_target = torch.stack((centered_direction, dc), dim=3).reshape(
        batch, length, 4096
    )
    previous = _previous(x)
    for name, wanted, module in (
        ("r", q_target, target.receptance),
        ("k", k_target, target.key),
    ):
        ratio, fitted = fit_time_mix(x, previous, wanted)
        getattr(target, f"x_{name}").copy_(ratio.view(1, 1, channels))
        module.weight.copy_(fitted.T.to(module.weight))

    matrix_write = rho * centered_norm * value_innovation
    mean_write = rho * source_value
    value_target = torch.stack((matrix_write, mean_write), dim=3).reshape(
        batch, length, 4096
    )
    ratio, fitted = fit_time_mix(x, previous, value_target)
    target.x_v.copy_(ratio.view(1, 1, channels))
    target.value_base.weight.copy_(
        torch.eye(channels, dtype=target.value_base.weight.dtype, device=x.device)
    )
    target.value_expand.weight.copy_(fitted.T.to(target.value_expand.weight))

    requested_decay = (1 - rho).expand(batch, length, 8, 256)
    log_decay = requested_decay.clamp_min(1e-6).log()
    realized_log_decay = log_decay.clamp_min(W_SCALE + 1e-6)
    realized_decay = realized_log_decay.exp()
    matrix_erase = CLOSURE * rho * centered_norm.square()
    mean_erase = (realized_decay - requested_decay).clamp_min(1e-6)
    erase = torch.stack((matrix_erase.expand_as(requested_decay), mean_erase), dim=3)
    decay_target = _native_decay_inverse(
        torch.stack((log_decay, log_decay), dim=3).reshape(batch, length, 4096)
    )
    erase_target = torch.logit(erase.clamp(1e-6, 1 - 1e-6).reshape(batch, length, 4096))
    for stem, wanted in (("w", decay_target), ("a", erase_target)):
        ratio, _ = fit_time_mix(x, previous, wanted)
        ratio_parameter = getattr(target, f"x_{stem}")
        ratio_parameter.copy_(ratio.view(1, 1, -1).to(ratio_parameter))
        mixed = x + (previous - x) * ratio
        bias, first, second = _low_rank_target(
            mixed, wanted, 128, nonlinear=stem == "w"
        )
        getattr(target, f"{stem}0").copy_(bias.view(1, 1, -1).to(ratio_parameter))
        getattr(target, f"{stem}1").copy_(first.to(ratio_parameter))
        getattr(target, f"{stem}2").copy_(second.to(ratio_parameter))

    source_gate = torch.sigmoid(source_gate.reshape(batch, length, channels).float())
    fit_native_gate(target, x, previous, source_gate)
    target.ln_x.weight.fill_(1)
    target.ln_x.bias.zero_()
    target.k_k.fill_(1)
    target.k_a.zero_()
    target.r_k.zero_()
    target.value_residual_scale.zero_()
    source_pre_output = exact_attention.transpose(1, 2).reshape(batch, length, channels)
    source_output = source.o_proj(source_pre_output * source_gate).float()
    native_output_fit_nmse = refit_native_output(target, x, source_output)
    return {
        "mean_exact_hazard": float(hazard.mean()),
        "matrix_state_key_rms": float(centered_key.square().mean().sqrt()),
        "matrix_state_write_rms": float(matrix_write.square().mean().sqrt()),
        "mean_state_write_rms": float(mean_write.square().mean().sqrt()),
        "requested_decay_below_native_floor_fraction": float(
            (log_decay < W_SCALE).float().mean()
        ),
        "native_output_fit_nmse": native_output_fit_nmse,
        "trace_tokens": float(batch * length),
    }


__all__ = ["initialize_gqa_layer"]
