"""The single Qwen GQA Taylor matrix-state plus prefix-mean initializer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from .gdn2rwkv import _native_decay_inverse, fit_native_gate, fit_time_mix, refit_native_output


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
    positions = torch.arange(length, device=x.device)
    lag = (positions[:, None] - positions[None, :]).clamp_min(0)
    mean_lag = (probabilities * lag).sum(-1).mean((0, 2)).clamp_min(1)
    concentration = probabilities.square().sum(-1).mean((0, 2)).clamp(1e-4, 1 - 1e-4)

    q_target = (F.normalize(q.float(), dim=-1) / 256**0.5).transpose(1, 2)
    k_target = F.normalize(k.float(), dim=-1).transpose(1, 2)
    q_target = q_target[:, :, :, None].expand(-1, -1, -1, 2, -1)
    k_target = k_target[:, :, :, None].expand_as(q_target)
    q_target = q_target.reshape(batch, length, 4096)
    k_target = k_target.reshape(batch, length, 4096)
    previous = _previous(x)
    for name, wanted, module in (
        ("r", q_target, target.receptance),
        ("k", k_target, target.key),
    ):
        ratio, fitted = fit_time_mix(x, previous, wanted)
        getattr(target, f"x_{name}").copy_(ratio.view(1, 1, channels))
        module.weight.copy_(fitted.T.to(module.weight))

    source_value = v.transpose(1, 2).reshape(batch, length, 2048)
    ratio, fitted = fit_time_mix(x, previous, source_value)
    target.x_v.copy_(ratio.view(1, 1, channels))
    target.value_base.weight.copy_(fitted.T.to(target.value_base.weight))
    target.value_expand.weight.zero_()
    eye = torch.eye(256, dtype=target.value_expand.weight.dtype, device=x.device) * 0.5
    for head in range(8):
        source_slice = slice(head * 256, (head + 1) * 256)
        for state in range(2):
            target_slice = slice((head * 2 + state) * 256, (head * 2 + state + 1) * 256)
            target.value_expand.weight[target_slice, source_slice].copy_(eye)

    log_decay = (-1 / mean_lag).repeat_interleave(2).repeat_interleave(256)
    target.w0.copy_(_native_decay_inverse(log_decay).view(1, 1, -1))
    target.w1.zero_()
    target.w2.zero_()
    target.a0.copy_(
        torch.logit(concentration).repeat_interleave(2).repeat_interleave(256).view(1, 1, -1)
    )
    target.a1.zero_()
    target.a2.zero_()

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
        "mean_attention_lag": float(mean_lag.mean()),
        "mean_attention_concentration": float(concentration.mean()),
        "mean_exact_hazard": float(hazard.mean()),
        "native_output_fit_nmse": native_output_fit_nmse,
        "trace_tokens": float(batch * length),
    }


__all__ = ["initialize_gqa_layer"]
