"""Pure RWKV feature-state initialization and frozen GQA teacher extraction."""

from __future__ import annotations

import math

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def _nmse(actual: torch.Tensor, wanted: torch.Tensor) -> float:
    actual = actual.float()
    wanted = wanted.float()
    return float((actual - wanted).square().sum() / wanted.square().sum().clamp_min(1e-24))


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
    scores = torch.einsum("bhtd,bhsd->bhts", query.float(), key.float()) * source.scaling
    length = hidden.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
    scores = scores.masked_fill(~causal.view(1, 1, length, length), -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    heads = torch.einsum("bhts,bhsd->bhtd", probability, value.float())
    return heads.to(value.dtype).transpose(1, 2).contiguous()


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
    mixed = heads.reshape(*hidden.shape) * torch.sigmoid(gate.reshape(*hidden.shape))
    return source.o_proj(mixed)


@torch.no_grad()
def initialize_gqa_layer(
    source,
    target,
    init_hidden: torch.Tensor,
    init_position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Install source projections and report the pure feature-state baseline."""

    if (
        target.num_heads != 8
        or target.num_kv_heads != 2
        or target.head_size != 256
        or target.feature_projection_dim != 64
        or target.feature_output_dim != 128
        or target.states_per_head != 2
    ):
        raise ValueError("GQA initialization requires fixed [QH8,KVH2,D256,F64x2,S2] geometry")
    target.load_source_attention(source)
    probe_length = min(128, init_hidden.shape[1])
    probe_hidden = init_hidden[:1, :probe_length]
    probe_embeddings = tuple(value[:1, :probe_length] for value in init_position_embeddings)
    source_output = _source_tmix_output(source, probe_hidden, probe_embeddings)
    target_output = target.reference_forward(probe_hidden)
    tmix_nmse = _nmse(target_output, source_output)
    if not math.isfinite(tmix_nmse):
        raise RuntimeError(f"pure RWKV GQA initialization produced non-finite NMSE {tmix_nmse}")
    recurrent_bytes = target.kernel_heads * target.head_size * target.head_size * 2
    return {
        "rwkv_feature_state_init_tmix_nmse": tmix_nmse,
        "gqa_feature_projection_dim": float(target.feature_projection_dim),
        "gqa_feature_output_dim": float(target.feature_output_dim),
        "gqa_recurrent_bytes_fp16": float(recurrent_bytes),
    }


@torch.no_grad()
def evaluate_gqa_recall(target, distances=(128, 256, 512, 1024), *, use_flash=False):
    """Feature-space retrieval with the learned RWKV dynamics enabled.

    The deterministic logits use a right-inverse of each current Q/K feature
    matrix. This keeps the fixture in feature space when shell weights are
    fine-tuned. Independent seeded hidden vectors drive decay, erase and value
    residuals; the readout includes learned normalization and the RKV shortcut.
    This is a controlled kernel probe, not a source-shell or language benchmark.
    """
    weight = target.feature_q_weight
    batch, heads, kv_heads, width = 2, target.num_heads, target.num_kv_heads, target.head_size
    codes = torch.tensor([[3, 11], [19, 27]], device=weight.device)
    metrics = {}
    for distance in distances:
        length = distance + 16
        positions = torch.arange(length, device=weight.device)
        distractors = (
            codes[..., None] + 1 + positions.remainder(31)
        ) % target.feature_projection_dim
        distractors[:, :, 15] = codes
        query_logits = torch.zeros(
            batch, heads, length, target.feature_projection_dim, device=weight.device
        )
        query_codes = codes[:, :, None, None].expand(batch, kv_heads, length, 1)
        query_codes = query_codes.repeat_interleave(heads // kv_heads, dim=1)
        query_logits.scatter_(-1, query_codes, 10.0)
        key_distractors = distractors.repeat_interleave(heads // kv_heads, dim=1)
        key_logits = torch.zeros(
            batch, heads, length, target.feature_projection_dim, device=weight.device
        )
        key_logits.scatter_(-1, key_distractors[..., None], 10.0)

        def right_inverse(logits, matrix):
            logits_batch, _, logits_length, _ = logits.shape
            outputs = []
            for head in range(matrix.shape[0]):
                flat = logits[:, head].reshape(-1, logits.shape[-1]).T
                # ``pinv`` gives the minimum-norm right-inverse for the
                # rectangular [F,D] system and remains deterministic across
                # CPU BLAS implementations (``lstsq`` may return a different
                # null-space solution for the same rank-deficient matrix).
                inverse = torch.linalg.pinv(matrix[head].float().T, rcond=1e-5)
                solution = (inverse @ flat).T
                outputs.append(solution.reshape(logits_batch, logits_length, width))
            return torch.stack(outputs, dim=1).to(weight.dtype)

        query = right_inverse(query_logits, target.feature_q_weight)
        key = right_inverse(key_logits, target.feature_k_weight)
        values = torch.zeros(batch, heads, length, width, device=weight.device, dtype=weight.dtype)
        values[..., 1] = 1
        values[:, :, 15, 1] = 0
        values[:, :, 15, 0] = 1
        # ``key`` is constructed per repeated query head so every learned
        # feature matrix is exercised directly.  Runtime GQA repeats the
        # source K/V before applying the per-head feature map; doing that
        # explicit repeat here would make a right-inverse impossible once K
        # feature matrices have diverged during distillation.
        feature_query = target._feature(query, target.feature_q_weight)
        feature_key = target._feature(key, target.feature_k_weight)
        control_hidden = torch.randn(
            batch,
            length,
            target.config.hidden_size,
            generator=torch.Generator().manual_seed(4096 + distance),
        ).to(device=weight.device, dtype=weight.dtype)
        delta = target._shift_delta(control_hidden)
        controls = target._dynamics(control_hidden, delta)
        source_values = values
        values, _ = target._mix_values(
            control_hidden, delta, values, torch.zeros_like(control_hidden)
        )
        recurrence = target._state_training if use_flash else target._state_reference
        numerator, denominator = recurrence(feature_query, feature_key, values, controls)
        recalled = target._heads(
            numerator, denominator, feature_query, feature_key, values, controls
        )[:, :, -1]
        metrics[f"recall_{distance}_hit_at_1"] = float((recalled.argmax(-1) == 0).float().mean())
        # The frozen source teacher is represented by the synthetic logits
        # themselves, so its hit remains an oracle even when the student
        # feature matrices are deliberately collapsed in the negative test.
        teacher_scores = torch.einsum(
            "bhf,bhtf->bht", query_logits[:, :, -1].float(), key_logits.float()
        )
        exact = torch.einsum("bht,bhtd->bhd", teacher_scores.softmax(-1), source_values.float())
        metrics[f"teacher_recall_{distance}_hit_at_1"] = float(
            (exact.argmax(-1) == 0).float().mean()
        )
    return metrics


__all__ = [
    "evaluate_gqa_recall",
    "exact_gqa_attention",
    "initialize_gqa_layer",
]
