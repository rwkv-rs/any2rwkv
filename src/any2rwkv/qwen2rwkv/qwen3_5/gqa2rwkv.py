"""Pure RWKV feature-state initialization and frozen GQA teacher extraction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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


def _gqa_source_components(
    source,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch, length = hidden.shape[:2]
    projected = source.q_proj(hidden).view(
        batch, length, source.config.num_attention_heads, 2 * source.head_dim
    )
    query_raw, gate = projected.chunk(2, dim=-1)
    key_raw = source.k_proj(hidden).view(
        batch, length, source.config.num_key_value_heads, source.head_dim
    )
    value = source.v_proj(hidden).view(
        batch, length, source.config.num_key_value_heads, source.head_dim
    )
    query_raw = query_raw.transpose(1, 2).float()
    key_raw = key_raw.transpose(1, 2).float()
    value = value.transpose(1, 2).float()
    groups = source.config.num_attention_heads // source.config.num_key_value_heads
    query_norm = source.q_norm(query_raw.transpose(1, 2)).transpose(1, 2).float()
    key_norm = source.k_norm(key_raw.transpose(1, 2)).transpose(1, 2).float()
    query_norm_gain = torch.rsqrt(query_raw.square().mean(-1) + source.q_norm.eps)
    key_norm_gain = torch.rsqrt(key_raw.square().mean(-1) + source.k_norm.eps)
    query_scaled = query_raw * (1.0 + source.q_norm.weight.float()).view(1, 1, 1, -1)
    key_scaled = key_raw * (1.0 + source.k_norm.weight.float()).view(1, 1, 1, -1)
    cos, sin = position_embeddings
    query_norm_rope, key_norm_rope = apply_rotary_pos_emb(query_norm, key_norm, cos, sin)
    query_scaled_rope, key_scaled_rope = apply_rotary_pos_emb(query_scaled, key_scaled, cos, sin)
    return {
        "q_norm_rope": query_norm_rope.float(),
        "k_norm_rope": key_norm_rope.repeat_interleave(groups, dim=1).float(),
        "q_norm_gain": query_norm_gain.float(),
        "k_norm_gain": key_norm_gain.repeat_interleave(groups, dim=1).float(),
        "q_norm_scale": query_norm_gain.reciprocal().float(),
        "k_norm_scale": key_norm_gain.reciprocal().repeat_interleave(groups, dim=1).float(),
        "q_norm": query_norm.float(),
        "k_norm": key_norm.repeat_interleave(groups, dim=1).float(),
        "q_scaled_rope": query_scaled_rope.float(),
        "k_scaled_rope": key_scaled_rope.repeat_interleave(groups, dim=1).float(),
        "value": value.repeat_interleave(groups, dim=1).float(),
        "gate": gate.reshape(batch, length, -1).float(),
        "query_raw": query_raw.float(),
        "key_raw": key_raw.repeat_interleave(groups, dim=1).float(),
    }


def _gqa_identity_feature_weight(heads: int, width: int, feature_dim: int, device) -> torch.Tensor:
    weight = torch.zeros(heads, width, feature_dim, device=device, dtype=torch.float32)
    indices = torch.arange(feature_dim, device=device)
    weight[:, indices, indices] = 1
    return weight


def _gqa_feature(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    projected = torch.einsum("bhtd,hdf->bhtf", value.float(), weight.float())
    scale = projected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((projected.relu(), (-projected).relu()), dim=-1) / scale + 1e-4


def _gqa_normal_fit(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    design = design.float()
    target = target.float()
    gram = design.transpose(0, 1) @ design
    scale = gram.diagonal().mean().clamp_min(1.0)
    gram = gram + torch.eye(gram.shape[0], device=gram.device) * (scale * 1e-6)
    return torch.linalg.solve(gram, design.transpose(0, 1) @ target)


def _gqa_fit_linear_feature(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    coefficients = []
    for head in range(value.shape[1]):
        design = value[:, head].transpose(0, 1).reshape(-1, value.shape[-1])
        wanted = target[:, head].transpose(0, 1).reshape(-1, target.shape[-1])
        coefficients.append(_gqa_normal_fit(design, wanted))
    return torch.stack(coefficients, dim=0)


def _gqa_linear_feature(value: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hdf->bhtf", value.float(), coefficient.float())


def _gqa_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_logit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, features = query.shape
    width = value.shape[-1]
    retention = math.exp(-math.exp(-0.5) * torch.sigmoid(torch.tensor(decay_logit)).item())
    state = torch.zeros(batch, heads, features, width, dtype=torch.float32, device=query.device)
    normalizer = torch.zeros(batch, heads, features, dtype=torch.float32, device=query.device)
    numerator, denominator = [], []
    for token in range(length):
        key_t = key[:, :, token].float()
        value_t = value[:, :, token].float()
        state = state * retention + torch.einsum("bhf,bhw->bhfw", key_t, value_t)
        normalizer = normalizer * retention + key_t
        query_t = query[:, :, token].float()
        numerator.append(torch.einsum("bhf,bhfw->bhw", query_t, state))
        denominator.append((normalizer * query_t).sum(-1))
    return torch.stack(numerator, dim=2), torch.stack(denominator, dim=2)


def _gqa_safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    sign = torch.where(denominator >= 0, 1.0, -1.0)
    divisor = sign * denominator.abs().clamp_min(1e-12)
    return numerator / divisor.unsqueeze(-1)


def _gqa_group_norm(
    numerator: torch.Tensor,
    heads: int,
    head_size: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, _, length, width = numerator.shape
    flat = numerator.transpose(1, 2).reshape(batch * length, heads * width).float()
    if weight is None:
        weight = torch.ones(heads * width, device=numerator.device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros_like(weight)
    normalized = F.group_norm(
        flat,
        heads * 2,
        weight=weight.float(),
        bias=bias.float(),
        eps=64e-5,
    )
    return normalized.view(batch, length, heads, width).transpose(1, 2)


def _gqa_output(
    source,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    gate: torch.Tensor,
    *,
    use_denominator: bool,
    group_weight: torch.Tensor | None = None,
    group_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if use_denominator:
        heads = _gqa_safe_divide(numerator, denominator)
    else:
        heads = _gqa_group_norm(
            numerator,
            source.config.num_attention_heads,
            source.head_dim,
            group_weight,
            group_bias,
        )
    mixed = heads.transpose(1, 2).reshape_as(gate) * gate.sigmoid()
    return source.o_proj(mixed).float()


def _gqa_layer_output(source_layer, hidden: torch.Tensor, tmix: torch.Tensor) -> torch.Tensor:
    residual = hidden.float() + tmix.float()
    return (
        residual + source_layer.mlp(source_layer.post_attention_layernorm(residual)).float()
    ).float()


def _gqa_metric(source_layer, hidden, source_tmix, candidate_tmix):
    return {
        "tmix_output_nmse": _nmse(candidate_tmix, source_tmix),
        "layer_output_nmse": _nmse(
            _gqa_layer_output(source_layer, hidden, candidate_tmix),
            _gqa_layer_output(source_layer, hidden, source_tmix),
        ),
    }


def _gqa_fit_channel_affine(actual: torch.Tensor, target: torch.Tensor):
    actual = actual.transpose(1, 2).reshape(-1, actual.shape[1] * actual.shape[-1]).float()
    target = target.transpose(1, 2).reshape(-1, target.shape[1] * target.shape[-1]).float()
    actual_mean = actual.mean(0)
    target_mean = target.mean(0)
    centered_actual = actual - actual_mean
    centered_target = target - target_mean
    weight = (centered_actual * centered_target).sum(0) / centered_actual.square().sum(0).clamp_min(
        1e-12
    )
    bias = target_mean - weight * actual_mean
    fitted = actual * weight + bias
    return weight, bias, _nmse(fitted, target)


def _gqa_geometric(value: torch.Tensor, dim=(0, 2)) -> torch.Tensor:
    return torch.exp(value.float().clamp_min(1e-30).log().mean(dim=dim))


def _gqa_teacher_attention(
    source,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    *,
    decay: float | None = None,
):
    length = query.shape[2]
    scores = torch.einsum("bhtd,bhsd->bhts", query.float(), key.float()) * source.scaling
    causal = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
    scores = scores.masked_fill(~causal.view(1, 1, length, length), -torch.inf)
    probability = scores.softmax(-1)
    if decay is not None:
        positions = torch.arange(length, device=query.device)
        delta = (positions[:, None] - positions[None, :]).clamp_min(0)
        probability = probability * (decay**delta).view(1, 1, length, length)
    heads = torch.einsum("bhts,bhsd->bhtd", probability, value.float())
    mixed = heads.transpose(1, 2).reshape_as(gate) * gate.float().sigmoid()
    return source.o_proj(mixed).float(), heads


def _gqa_geometry_check(query, key, value, decay_logit: float) -> float:
    batch, heads, length, features = query.shape
    width = value.shape[-1]
    half = width // 2
    retention = math.exp(-math.exp(-0.5) * torch.sigmoid(torch.tensor(decay_logit)).item())
    full_state = torch.zeros(
        batch, heads, features, width, dtype=torch.float32, device=query.device
    )
    left_state = torch.zeros(batch, heads, features, half, dtype=torch.float32, device=query.device)
    right_state = torch.zeros_like(left_state)
    full_outputs, split_outputs = [], []
    for token in range(length):
        key_t = key[:, :, token].float()
        value_t = value[:, :, token].float()
        full_state = full_state * retention + torch.einsum("bhf,bhw->bhfw", key_t, value_t)
        left_state = left_state * retention + torch.einsum(
            "bhf,bhw->bhfw", key_t, value_t[..., :half]
        )
        right_state = right_state * retention + torch.einsum(
            "bhf,bhw->bhfw", key_t, value_t[..., half:]
        )
        query_t = query[:, :, token].float()
        full_outputs.append(torch.einsum("bhf,bhfw->bhw", query_t, full_state))
        split_outputs.append(
            torch.cat(
                (
                    torch.einsum("bhf,bhfw->bhw", query_t, left_state),
                    torch.einsum("bhf,bhfw->bhw", query_t, right_state),
                ),
                dim=-1,
            )
        )
    return _nmse(torch.stack(split_outputs, dim=2), torch.stack(full_outputs, dim=2))


@torch.no_grad()
def audit_gqa_layer(
    source,
    source_layer,
    init_hidden: torch.Tensor,
    validation_hidden: torch.Tensor,
    init_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    validation_position_embeddings: tuple[torch.Tensor, torch.Tensor],
):
    """Stage -1 audit with teacher-side ablations and student deltas from E0."""
    if (
        source.config.num_attention_heads != 8
        or source.config.num_key_value_heads != 2
        or source.head_dim != 256
    ):
        raise ValueError("GQA audit requires source geometry [QH8, KVH2, D256]")
    identity_weight = _gqa_identity_feature_weight(
        source.config.num_attention_heads, source.head_dim, 64, init_hidden.device
    )
    raw_inputs = {
        "initialization": init_hidden.float(),
        "validation": validation_hidden.float(),
    }
    split_inputs = {
        split: source_layer.input_layernorm(hidden).float() for split, hidden in raw_inputs.items()
    }
    position_embeddings = {
        "initialization": init_position_embeddings,
        "validation": validation_position_embeddings,
    }
    parts = {
        split: _gqa_source_components(source, hidden, position_embeddings[split])
        for split, hidden in split_inputs.items()
    }
    teacher = {}
    teacher_heads = {}
    for split in split_inputs:
        output, heads = _gqa_teacher_attention(
            source,
            parts[split]["q_norm_rope"],
            parts[split]["k_norm_rope"],
            parts[split]["value"],
            parts[split]["gate"],
        )
        teacher[split] = output
        teacher_heads[split] = heads

    feature_q_init = _gqa_feature(parts["initialization"]["q_norm_rope"], identity_weight)
    feature_k_init = _gqa_feature(parts["initialization"]["k_norm_rope"], identity_weight)
    linear_q = _gqa_fit_linear_feature(parts["initialization"]["q_norm_rope"], feature_q_init)
    linear_k = _gqa_fit_linear_feature(parts["initialization"]["k_norm_rope"], feature_k_init)
    decay_initial = -30.0
    decay_canonical = -12.0
    retention_canonical = math.exp(
        -math.exp(-0.5) * torch.sigmoid(torch.tensor(decay_canonical)).item()
    )
    q_norm_constant = _gqa_geometric(parts["initialization"]["q_norm_scale"])
    k_norm_constant = _gqa_geometric(parts["initialization"]["k_norm_scale"])
    q_gamma = 1.0 + source.q_norm.weight.float()
    k_gamma = 1.0 + source.k_norm.weight.float()
    result = {
        "layer": int(source.layer_idx),
        "source_head_size": int(source.head_dim),
        "canonical_head_size": 128,
        "kernel_heads": 16,
        "feature_weight_source": "identity_first64_no_v3_checkpoint",
        "components": {},
        "fit": {
            "linear_feature_q_fit_nmse": _nmse(
                _gqa_linear_feature(parts["initialization"]["q_norm_rope"], linear_q),
                feature_q_init,
            ),
            "linear_feature_k_fit_nmse": _nmse(
                _gqa_linear_feature(parts["initialization"]["k_norm_rope"], linear_k),
                feature_k_init,
            ),
            "e2_q_norm_geometric_constant": list(q_norm_constant.tolist()),
            "e2_k_norm_geometric_constant": list(k_norm_constant.tolist()),
        },
    }

    def student_features(split, linear=False):
        current = parts[split]
        if linear:
            return (
                _gqa_linear_feature(current["q_norm_rope"], linear_q),
                _gqa_linear_feature(current["k_norm_rope"], linear_k),
            )
        return (
            _gqa_feature(current["q_norm_rope"], identity_weight),
            _gqa_feature(current["k_norm_rope"], identity_weight),
        )

    def add_e0_delta(metrics, baseline):
        return {
            split: {
                **value,
                "tmix_output_nmse_delta_from_E0": value["tmix_output_nmse"]
                - baseline[split]["tmix_output_nmse"],
                "layer_output_nmse_delta_from_E0": value["layer_output_nmse"]
                - baseline[split]["layer_output_nmse"],
            }
            for split, value in metrics.items()
        }

    e0_metrics = {}
    e0_numerators = {}
    for split in split_inputs:
        query, key = student_features(split)
        numerator, denominator = _gqa_recurrent(query, key, parts[split]["value"], decay_initial)
        e0_numerators[split] = numerator
        candidate = _gqa_output(
            source, numerator, denominator, parts[split]["gate"], use_denominator=True
        )
        e0_metrics[split] = _gqa_metric(source_layer, raw_inputs[split], teacher[split], candidate)
    result["components"]["E0_untrained_student_baseline"] = {
        "metrics": e0_metrics,
        "student": (
            "identity feature, decay_logit=-30, positive denominator, no component replacement"
        ),
    }

    e1_metrics = {}
    e2_metrics = {}
    for split in split_inputs:
        current = parts[split]
        e1_output, _ = _gqa_teacher_attention(
            source,
            current["q_norm"],
            current["k_norm"],
            current["value"],
            current["gate"],
        )
        e1_metrics[split] = _gqa_metric(source_layer, raw_inputs[split], teacher[split], e1_output)
        q_without_norm = current["query_raw"] * q_gamma.view(1, 1, 1, -1)
        k_without_norm = current["key_raw"] * k_gamma.view(1, 1, 1, -1)
        q_without_norm = q_without_norm / q_norm_constant.view(1, -1, 1, 1)
        k_without_norm = k_without_norm / k_norm_constant.view(1, -1, 1, 1)
        cos, sin = position_embeddings[split]
        q_without_norm, k_without_norm = apply_rotary_pos_emb(
            q_without_norm, k_without_norm, cos, sin
        )
        e2_output, _ = _gqa_teacher_attention(
            source,
            q_without_norm,
            k_without_norm,
            current["value"],
            current["gate"],
        )
        e2_metrics[split] = _gqa_metric(source_layer, raw_inputs[split], teacher[split], e2_output)
    result["components"]["E1_teacher_rope_removed"] = {
        "metrics": e1_metrics,
        "side": "source_softmax_teacher",
    }
    result["components"]["E2_teacher_qk_norm_geomean"] = {
        "metrics": e2_metrics,
        "side": "source_softmax_teacher",
        "q_gamma_absorbed": True,
        "k_gamma_absorbed": True,
    }

    e3_metrics = {}
    for split in split_inputs:
        query, key = student_features(split, linear=True)
        numerator, denominator = _gqa_recurrent(query, key, parts[split]["value"], decay_initial)
        candidate = _gqa_output(
            source, numerator, denominator, parts[split]["gate"], use_denominator=True
        )
        e3_metrics[split] = _gqa_metric(source_layer, raw_inputs[split], teacher[split], candidate)
    result["components"]["E3_linear_feature"] = {
        "metrics": add_e0_delta(e3_metrics, e0_metrics),
        "side": "untrained_student",
    }

    e4_base_metrics = {}
    for split in split_inputs:
        candidate = _gqa_output(
            source,
            e0_numerators[split],
            torch.zeros(e0_numerators[split].shape[:3], device=e0_numerators[split].device),
            parts[split]["gate"],
            use_denominator=False,
        )
        e4_base_metrics[split] = _gqa_metric(
            source_layer, raw_inputs[split], teacher[split], candidate
        )

    affine_weight, affine_bias, affine_fit_nmse = _gqa_fit_channel_affine(
        _gqa_group_norm(
            e0_numerators["initialization"],
            source.config.num_attention_heads,
            source.head_dim,
        ),
        teacher_heads["initialization"],
    )
    e4_ls_metrics = {}
    for split in split_inputs:
        candidate = _gqa_output(
            source,
            e0_numerators[split],
            torch.zeros(e0_numerators[split].shape[:3], device=e0_numerators[split].device),
            parts[split]["gate"],
            use_denominator=False,
            group_weight=affine_weight,
            group_bias=affine_bias,
        )
        e4_ls_metrics[split] = _gqa_metric(
            source_layer, raw_inputs[split], teacher[split], candidate
        )
    result["components"]["E4_denominator_removed_groupnorm"] = {
        "metrics": add_e0_delta(e4_base_metrics, e0_metrics),
        "side": "untrained_student",
    }
    result["components"]["E4_denominator_removed_groupnorm_ls_affine"] = {
        "metrics": add_e0_delta(e4_ls_metrics, e0_metrics),
        "side": "untrained_student",
        "affine_fit_target": "source_softmax_attention_heads_before_gate_and_o_proj",
        "affine_fit_nmse": affine_fit_nmse,
    }

    e5_metrics = {}
    for split in split_inputs:
        current = parts[split]
        e5_output, _ = _gqa_teacher_attention(
            source,
            current["q_norm_rope"],
            current["k_norm_rope"],
            current["value"],
            current["gate"],
            decay=retention_canonical,
        )
        e5_metrics[split] = _gqa_metric(source_layer, raw_inputs[split], teacher[split], e5_output)
    result["components"]["E5_teacher_canonical_decay"] = {
        "metrics": e5_metrics,
        "side": "source_softmax_teacher",
        "decay": retention_canonical,
        "decay_application": "softmax_probability_times_d_delta_t_without_renormalization",
        "one_minus_decay_T": {
            "512": 1 - retention_canonical**512,
            "4096": 1 - retention_canonical**4096,
        },
    }

    geometry = {}
    for split in split_inputs:
        query, key = student_features(split)
        geometry[split] = {
            "geometry_nmse": _gqa_geometry_check(query, key, parts[split]["value"], decay_initial)
        }
    result["components"]["E6_value_half_geometry_self_check"] = {
        "metrics": geometry,
        "expected": "zero up to FP32 roundoff",
    }
    return result


__all__ = [
    "audit_gqa_layer",
    "evaluate_gqa_recall",
    "exact_gqa_attention",
    "initialize_gqa_layer",
]
