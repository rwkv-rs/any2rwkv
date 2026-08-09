"""Initialize and diagnose the source-shell Qwen3.5 GDN/WKV hybrid."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

from .transformers.modeling_qwen2rwkv import CLAMP_W_EPSILON, W_SCALE


def _nmse(actual: torch.Tensor, wanted: torch.Tensor) -> float:
    actual = actual.float()
    wanted = wanted.float()
    value = (actual - wanted).square().mean() / (wanted.square().mean() + 1e-12)
    return float(value.detach())


def _rwkv_trace(
    read: torch.Tensor,
    key: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
    log_decay: torch.Tensor,
) -> torch.Tensor:
    """Reference RWKV state update in the kernel's [key, value] orientation."""
    batch, length, heads, dim = read.shape
    state = read.new_zeros(batch, heads, dim, dim, dtype=torch.float32)
    outputs = []
    for token in range(length):
        key_t = key[:, token].float()
        memory = torch.einsum("bhk,bhkv->bhv", key_t, state)
        state = (
            state * log_decay[:, token].float().exp()[..., None, None]
            - erase[:, token].float()[..., None, None]
            * torch.einsum("bhk,bhv->bhkv", key_t, memory)
            + torch.einsum("bhk,bhv->bhkv", key_t, write[:, token].float())
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", read[:, token].float(), state))
    return torch.stack(outputs, 1)


def _gdn_trace(source, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, length, _ = hidden.shape
    mixed_qkv = causal_conv1d_fn(
        source.in_proj_qkv(hidden).transpose(1, 2),
        source.conv1d.weight.squeeze(1),
        source.conv1d.bias,
        activation=source.activation,
    )[:, :, :length].transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, (source.key_dim, source.key_dim, source.value_dim), dim=-1
    )
    query = query.view(batch, length, source.num_k_heads, source.head_k_dim)
    key = key.view(batch, length, source.num_k_heads, source.head_k_dim)
    value = value.view(batch, length, source.num_v_heads, source.head_v_dim)
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    if source.num_v_heads // source.num_k_heads > 1:
        repeats = source.num_v_heads // source.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    beta = torch.sigmoid(source.in_proj_b(hidden)).float()
    log_decay = -source.A_log.float().exp() * F.softplus(
        source.in_proj_a(hidden).float() + source.dt_bias.float()
    )
    retention = log_decay.exp()
    read = query.float() / math.sqrt(source.head_k_dim)
    write = beta[..., None] * value.float()
    erase = beta * retention
    oracle_raw = _rwkv_trace(read, key, write, erase, log_decay)

    captured: dict[str, torch.Tensor] = {}

    def capture_boundary(_module, arguments, output):
        captured["raw"] = (
            arguments[0].detach().view(batch, length, source.num_v_heads, source.head_v_dim)
        )
        captured["gate"] = (
            arguments[1].detach().view(batch, length, source.num_v_heads, source.head_v_dim)
        )
        captured["pre_output"] = output.detach().view(batch, length, source.value_dim)

    hook = source.norm.register_forward_hook(capture_boundary)
    try:
        source_output = source(hidden).float()
    finally:
        hook.remove()
    if set(captured) != {"raw", "gate", "pre_output"}:
        raise RuntimeError("failed to capture the source GDN output boundary")

    trace_pre_output = source.norm(
        oracle_raw.reshape(-1, source.head_v_dim).to(captured["raw"]),
        captured["gate"].reshape(-1, source.head_v_dim),
    ).view(batch, length, source.value_dim)
    trace_output = source.out_proj(trace_pre_output).float()

    clamp_ratio = (log_decay / W_SCALE).clamp(CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON)
    realized_log_decay = W_SCALE * clamp_ratio
    realized_erase = beta * realized_log_decay.exp()
    clamped_raw = _rwkv_trace(read, key, write, realized_erase, realized_log_decay)
    clamped_pre_output = source.norm(
        clamped_raw.reshape(-1, source.head_v_dim).to(captured["raw"]),
        captured["gate"].reshape(-1, source.head_v_dim),
    ).view(batch, length, source.value_dim)
    clamped_output = source.out_proj(clamped_pre_output).float()
    return {
        "raw": captured["raw"].float(),
        "oracle_raw": oracle_raw,
        "source_pre_output": captured["pre_output"].float(),
        "trace_pre_output": trace_pre_output.float(),
        "source_output": source_output,
        "trace_output": trace_output,
        "log_decay": log_decay,
        "realized_log_decay": realized_log_decay,
        "clamped_raw": clamped_raw,
        "clamped_output": clamped_output,
    }


def _trace_metrics(prefix: str, trace: dict[str, torch.Tensor]) -> dict[str, float]:
    ratio = trace["log_decay"] / W_SCALE
    outside = (ratio < CLAMP_W_EPSILON) | (ratio > 1 - CLAMP_W_EPSILON)
    return {
        f"{prefix}_source_trace_output_nmse": _nmse(trace["trace_output"], trace["source_output"]),
        f"{prefix}_source_trace_pre_output_nmse": _nmse(
            trace["trace_pre_output"], trace["source_pre_output"]
        ),
        f"{prefix}_source_recurrence_raw_nmse": _nmse(trace["oracle_raw"], trace["raw"]),
        f"{prefix}_clamp_w_outside_fraction": float(outside.float().mean()),
        f"{prefix}_clamp_w_log_decay_nmse": _nmse(trace["realized_log_decay"], trace["log_decay"]),
        f"{prefix}_clamp_w_raw_nmse": _nmse(trace["clamped_raw"], trace["oracle_raw"]),
        f"{prefix}_clamp_w_mixer_nmse": _nmse(trace["clamped_output"], trace["source_output"]),
    }


@torch.no_grad()
def initialize_gdn_layer(
    source,
    target,
    normalized_hidden: torch.Tensor,
    selection_hidden: torch.Tensor | None = None,
) -> dict[str, float]:
    """Strict-copy one source GDN shell and report recurrence/Clamp-W diagnostics."""
    target.load_state_dict(source.state_dict(), strict=True)
    source_state = source.state_dict()
    target_state = target.state_dict()
    unequal = [
        name for name, value in source_state.items() if not torch.equal(value, target_state[name])
    ]
    if unequal:
        raise RuntimeError(f"GDN source-shell copy mismatch: {unequal}")

    calibration_trace = _gdn_trace(source, normalized_hidden)
    metrics = {
        "source_shell_parameter_tensors": float(len(source_state)),
        "source_shell_strict_copy": 1.0,
        "calibration_trace_tokens": float(normalized_hidden.shape[0] * normalized_hidden.shape[1]),
        **_trace_metrics("calibration", calibration_trace),
    }
    if selection_hidden is not None:
        selection_trace = _gdn_trace(source, selection_hidden)
        metrics.update(
            {
                "selection_trace_tokens": float(
                    selection_hidden.shape[0] * selection_hidden.shape[1]
                ),
                **_trace_metrics("selection", selection_trace),
            }
        )
    return metrics


__all__ = ["initialize_gdn_layer"]
