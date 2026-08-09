"""Hedgehog/LoLCATs initialization and exact GQA teachers.

The product recurrence and bounded sidecar live in modeling_qwen2rwkv so
training, reference evaluation, and inference share one implementation. This
module only owns source-attention extraction and the fail-closed layer-3
initializer.
"""

from __future__ import annotations

import math

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

REFERENCE_NMSE_GATE = 1e-6
ORACLE_BLOCK_NMSE_GATE = 1.5e-3


def _nmse(actual: torch.Tensor, wanted: torch.Tensor) -> float:
    actual = actual.float()
    wanted = wanted.float()
    return float(
        (actual - wanted).square().sum()
        / wanted.square().sum().clamp_min(1e-24)
    )


def _source_qkv(
    source,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length = hidden.shape[:2]
    projected = source.q_proj(hidden).view(
        batch, length, source.config.num_attention_heads, 2 * source.head_dim
    )
    query, gate = projected.chunk(2, dim=-1)
    key = source.k_proj(hidden).view(
        batch, length, source.config.num_key_value_heads, source.head_dim
    )
    value = source.v_proj(hidden).view(
        batch, length, source.config.num_key_value_heads, source.head_dim
    )
    query = source.q_norm(query).transpose(1, 2)
    key = source.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return query, key, value, gate.reshape(batch, length, -1)


@torch.no_grad()
def exact_gqa_attention(
    source,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return source causal Softmax value outputs as [B,T,H,D]."""

    query, key, value, _ = _source_qkv(source, hidden, position_embeddings)
    groups = source.config.num_attention_heads // source.config.num_key_value_heads
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.einsum(
        "bhtd,bhsd->bhts", query.float(), key.float()
    ) * source.scaling
    length = hidden.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
    scores = scores.masked_fill(~causal.view(1, 1, length, length), -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    heads = torch.einsum("bhts,bhsd->bhtd", probability, value.float())
    return heads.to(value.dtype).transpose(1, 2).contiguous()


@torch.no_grad()
def exact_gqa_teacher_targets(
    source,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    return exact_gqa_attention(source, hidden, position_embeddings)


@torch.no_grad()
def _source_tmix_output(
    source,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    heads = exact_gqa_attention(source, hidden, position_embeddings)
    projected = source.q_proj(hidden).view(
        hidden.shape[0],
        hidden.shape[1],
        source.config.num_attention_heads,
        2 * source.head_dim,
    )
    _, gate = projected.chunk(2, dim=-1)
    mixed = heads.reshape(*hidden.shape) * torch.sigmoid(
        gate.reshape(*hidden.shape)
    )
    return source.o_proj(mixed)


@torch.no_grad()
def initialize_gqa_layer(
    source,
    target,
    init_hidden: torch.Tensor,
    init_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    validation_hidden: torch.Tensor | None = None,
    validation_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Install the source shell and verify exact-prefix identity.

    Capacity, feature dimension, and readout form are fixed by the target
    checkpoint schema. No validation-driven architecture selection happens
    here.
    """

    if (
        target.num_heads != 8
        or target.num_kv_heads != 2
        or target.head_size != 256
        or target.feature_projection_dim != 64
        or target.feature_output_dim != 128
        or target.states_per_head != 2
        or target.sidecar_capacity != 128
        or (target.sink_slots, target.recent_slots, target.heavy_slots)
        != (8, 64, 56)
    ):
        raise ValueError(
            "GQA initialization requires fixed [QH8,KVH2,D256,F64x2,S2,"
            "sidecar=8+64+56] geometry"
        )
    target.load_source_attention(source)

    # A prefix no longer than the sidecar is an exact-attention identity. Use
    # one row to keep initializer memory independent of DDP world size.
    prefix_length = min(128, init_hidden.shape[1])
    probe_hidden = init_hidden[:1, :prefix_length]
    probe_embeddings = tuple(
        value[:1, :prefix_length] for value in init_position_embeddings
    )
    source_heads = exact_gqa_attention(source, probe_hidden, probe_embeddings)
    target_heads, _ = target.attention_heads_reference(
        probe_hidden,
        torch.arange(prefix_length, device=probe_hidden.device).view(1, -1),
    )
    attention_nmse = _nmse(target_heads.transpose(1, 2), source_heads)
    source_output = _source_tmix_output(source, probe_hidden, probe_embeddings)
    target_output = target.reference_forward(probe_hidden)
    tmix_nmse = _nmse(target_output, source_output)
    if (
        not math.isfinite(attention_nmse)
        or not math.isfinite(tmix_nmse)
        or attention_nmse > REFERENCE_NMSE_GATE
        or tmix_nmse > REFERENCE_NMSE_GATE
    ):
        raise RuntimeError(
            "bounded Hedgehog exact-prefix identity failed: "
            f"attention_nmse={attention_nmse:.8g}, tmix_nmse={tmix_nmse:.8g}"
        )

    sidecar_bytes = (
        target.sidecar_capacity
        * target.num_kv_heads
        * target.head_size
        * 2
        * 2
    )
    recurrent_bytes = (
        target.kernel_heads * target.head_size * target.head_size * 2
    )
    return {
        "hedgehog_exact_prefix_attention_nmse": attention_nmse,
        "hedgehog_exact_prefix_tmix_nmse": tmix_nmse,
        "gqa_feature_projection_dim": float(target.feature_projection_dim),
        "gqa_feature_output_dim": float(target.feature_output_dim),
        "gqa_sidecar_slots": float(target.sidecar_capacity),
        "gqa_sidecar_kv_bytes_fp16": float(sidecar_bytes),
        "gqa_recurrent_bytes_fp16": float(recurrent_bytes),
        "gqa_total_fixed_payload_bytes_fp16": float(
            sidecar_bytes + recurrent_bytes
        ),
    }


__all__ = [
    "ORACLE_BLOCK_NMSE_GATE",
    "exact_gqa_attention",
    "exact_gqa_teacher_targets",
    "initialize_gqa_layer",
]
