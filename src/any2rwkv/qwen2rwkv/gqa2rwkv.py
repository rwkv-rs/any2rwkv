"""Compile Qwen GQA into a fixed two-state-per-query-head RWKV-7 TMix."""

from __future__ import annotations

import math

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from .gdn2rwkv import (
    W_SCALE,
    _apply_groupnorm_affine,
    _control_signal,
    _low_rank_target,
    _mixed,
    _native_decay_inverse,
    _native_forward,
    _native_raw_heads,
    _nmse,
    _restore,
    _snapshot,
    _state_is_finite,
    fit_groupnorm_affine,
    fit_native_gate,
    fit_time_mix,
    refit_native_output,
)

CLOSURE_CANDIDATES = (0.0, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def _previous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), 1)


def _trajectory_diagnostics(
    query: torch.Tensor,
    centered_key: torch.Tensor,
    value_innovation: torch.Tensor,
    mean_value: torch.Tensor,
    exact_attention: torch.Tensor,
    closure: float,
) -> tuple[float, float]:
    """Separate finite-state Softmax error from clamp-w native error."""
    affine_error = query.new_zeros((), dtype=torch.float64)
    affine_energy = query.new_zeros((), dtype=torch.float64)
    native_error = query.new_zeros((), dtype=torch.float64)
    batch, length = query.shape[:2]
    native_floor = math.exp(W_SCALE)
    matrix = query.new_zeros(batch, 8, 256, 256, dtype=torch.float32)
    native_matrix = torch.zeros_like(matrix)
    native_mean = query.new_zeros(batch, 8, 256, dtype=torch.float32)
    for token in range(length):
        rho = 1.0 / (token + 1)
        requested_decay = 1.0 - rho
        realized_decay = max(requested_decay, native_floor)
        key = centered_key[:, token].float()
        innovation = value_innovation[:, token].float()
        norm = key.square().sum(-1, keepdim=True).sqrt()
        direction = key / norm.clamp_min(1e-6)

        memory = torch.einsum("bhij,bhj->bhi", matrix, key)
        matrix = (
            matrix * requested_decay
            - closure * rho * torch.einsum("bhi,bhj->bhij", memory, key)
            + rho * torch.einsum("bhi,bhj->bhij", innovation, key)
        )
        affine = mean_value[:, token].float() + torch.einsum(
            "bhij,bhj->bhi", matrix, query[:, token].float()
        )

        native_memory = torch.einsum("bhij,bhj->bhi", native_matrix, direction)
        matrix_erase = (closure * rho * norm.square()).clamp(1e-6, 1 - 1e-6)
        native_matrix = (
            native_matrix * realized_decay
            - torch.einsum(
                "bhi,bhj->bhij", native_memory, direction * matrix_erase
            )
            + torch.einsum(
                "bhi,bhj->bhij", rho * norm * innovation, direction
            )
        )
        mean_erase = realized_decay - requested_decay
        previous_mean = mean_value[:, token - 1].float() if token else 0
        native_mean = (
            native_mean * (realized_decay - mean_erase)
            + rho * (value_innovation[:, token].float() + previous_mean)
        )
        native = native_mean + torch.einsum(
            "bhij,bhj->bhi", native_matrix, query[:, token].float()
        )
        wanted = exact_attention[:, :, token].float()
        affine_error += (affine - wanted).double().square().sum()
        affine_energy += wanted.double().square().sum()
        native_error += (native - affine).double().square().sum()
    return float(affine_error / affine_energy.clamp_min(1e-24)), float(
        native_error / affine_energy.clamp_min(1e-24)
    )


def _empirical_kernel_tail(scores: torch.Tensor, rank: int = 257) -> float:
    """Return a bounded empirical exponential-kernel tail diagnostic."""
    length = min(scores.shape[-1], 512)
    if length <= rank:
        return 0.0
    sampled = scores[: min(scores.shape[0], 2), :, :length, :length].float()
    sampled = sampled - sampled.amax(dim=(-2, -1), keepdim=True)
    singular_values = torch.linalg.svdvals(sampled.exp())
    energy = singular_values.square()
    return float(energy[..., rank:].sum() / energy.sum().clamp_min(1e-24))


def _gqa_trace(
    source,
    x: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor | float]:
    batch, length, channels = x.shape
    q, gate_pre = torch.chunk(source.q_proj(x).view(batch, length, 8, 512), 2, -1)
    q = source.q_norm(q).transpose(1, 2)
    k = source.k_norm(source.k_proj(x).view(batch, length, 2, 256)).transpose(1, 2)
    v = source.v_proj(x).view(batch, length, 2, 256).transpose(1, 2)
    q_pre_rope = q
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k = repeat_kv(k, 4)
    v = repeat_kv(v, 4)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / 256**0.5
    causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
    probabilities = scores.masked_fill(causal, -torch.inf).softmax(-1)
    exact_attention = torch.matmul(probabilities, v.float())
    gate = torch.sigmoid(gate_pre.reshape(batch, length, channels).float())
    source_pre_output = exact_attention.transpose(1, 2).reshape(batch, length, channels)
    reference_output = source.o_proj(
        (source_pre_output * gate).to(source.o_proj.weight.dtype)
    ).float()
    additive_causal = torch.full(
        (length, length),
        torch.finfo(x.dtype).min,
        dtype=x.dtype,
        device=x.device,
    ).triu(1)
    source_output = source(
        x,
        position_embeddings=position_embeddings,
        attention_mask=additive_causal.view(1, 1, length, length),
        past_key_values=None,
    )[0].float()

    scale = 256**-0.25
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

    q_pre_target = q_pre_rope.float().transpose(1, 2) * scale
    centered_heads = centered_direction.transpose(1, 2)
    centered_pre_rope, _ = apply_rotary_pos_emb(
        centered_heads,
        centered_heads,
        position_embeddings[0],
        -position_embeddings[1],
    )
    centered_pre_rope = centered_pre_rope.transpose(1, 2)
    q_target = torch.stack((q_pre_target, dc), dim=3).reshape(batch, length, 4096)
    k_target = torch.stack((centered_pre_rope, dc), dim=3).reshape(
        batch, length, 4096
    )
    matrix_write = rho * centered_norm * value_innovation
    mean_write = rho * source_value
    value_target = torch.stack((matrix_write, mean_write), dim=3).reshape(
        batch, length, 4096
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "query": q.float().transpose(1, 2) * scale,
        "q_target": q_target,
        "k_target": k_target,
        "value_target": value_target,
        "source_value": source_value,
        "centered_key": centered_key,
        "centered_norm": centered_norm,
        "value_innovation": value_innovation,
        "mean_value": mean_value,
        "rho": rho,
        "gate": gate,
        "exact_attention": exact_attention,
        "source_output": source_output,
        "reference_output": reference_output,
        "reference_source_output_nmse": _nmse(reference_output, source_output),
        "mean_exact_hazard": float(probabilities.diagonal(dim1=-2, dim2=-1).mean()),
        "empirical_kernel_tail_ratio": _empirical_kernel_tail(scores),
        "matrix_state_key_rms": float(centered_key.square().mean().sqrt()),
        "matrix_state_write_rms": float(matrix_write.square().mean().sqrt()),
        "mean_state_write_rms": float(mean_write.square().mean().sqrt()),
    }


def _fit_linear_projection(target, name, module, x, wanted):
    previous = _previous(x)
    ratio, fitted = fit_time_mix(x, previous, wanted)
    getattr(target, f"x_{name}").copy_(ratio.view(1, 1, -1))
    module.weight.copy_(fitted.T.to(module.weight))
    actual = _mixed(x, previous, ratio) @ module.weight.float().T
    return ratio, fitted, _nmse(actual, wanted)


def _selection_projection_nmse(x, wanted, ratio, projection):
    actual = _mixed(x, _previous(x), ratio) @ projection
    return _nmse(actual, wanted)


def _compile_gqa_candidate(
    source,
    target,
    x: torch.Tensor,
    trace,
    selection_hidden: torch.Tensor,
    selection_trace,
    closure: float,
) -> dict[str, float]:
    channels = x.shape[-1]
    fitted_projections = {}
    projection_metrics = {}
    for name, wanted, module in (
        ("r", trace["q_target"], target.receptance),
        ("k", trace["k_target"], target.key),
    ):
        ratio, fitted, calibration_nmse = _fit_linear_projection(
            target, name, module, x, wanted
        )
        fitted_projections[name] = (ratio, fitted)
        projection_metrics[f"{name}_projection_calibration_nmse"] = calibration_nmse

    ratio, fitted = fit_time_mix(x, _previous(x), trace["value_target"])
    target.x_v.copy_(ratio.view(1, 1, channels))
    target.value_base.weight.copy_(
        torch.eye(channels, dtype=target.value_base.weight.dtype, device=x.device)
    )
    target.value_expand.weight.copy_(fitted.T.to(target.value_expand.weight))
    value_actual = _mixed(x, _previous(x), ratio) @ target.value_expand.weight.float().T
    projection_metrics["v_projection_calibration_nmse"] = _nmse(
        value_actual, trace["value_target"]
    )
    fitted_projections["v"] = (ratio, fitted)

    rho = trace["rho"]
    batch, length = x.shape[:2]
    requested_decay = (1 - rho).expand(batch, length, 8, 256)
    log_decay = requested_decay.clamp_min(1e-6).log()
    realized_log_decay = log_decay.clamp_min(W_SCALE + 1e-6)
    realized_decay = realized_log_decay.exp()
    matrix_erase = closure * rho * trace["centered_norm"].square()
    mean_erase = (realized_decay - requested_decay).clamp_min(1e-6)
    erase = torch.stack((matrix_erase.expand_as(requested_decay), mean_erase), dim=3)
    decay_target = _native_decay_inverse(
        torch.stack((log_decay, log_decay), dim=3).reshape(batch, length, 4096)
    )
    erase_target = torch.logit(
        erase.clamp(1e-6, 1 - 1e-6).reshape(batch, length, 4096)
    )
    selection_rho = selection_trace["rho"]
    selection_batch, selection_length = selection_hidden.shape[:2]
    selection_requested_decay = (1 - selection_rho).expand(
        selection_batch, selection_length, 8, 256
    )
    selection_log_decay = selection_requested_decay.clamp_min(1e-6).log()
    selection_realized_decay = selection_log_decay.clamp_min(W_SCALE + 1e-6).exp()
    selection_matrix_erase = (
        closure * selection_rho * selection_trace["centered_norm"].square()
    )
    selection_mean_erase = (
        selection_realized_decay - selection_requested_decay
    ).clamp_min(1e-6)
    selection_erase = torch.stack(
        (
            selection_matrix_erase.expand_as(selection_requested_decay),
            selection_mean_erase,
        ),
        dim=3,
    )
    selection_decay_target = _native_decay_inverse(
        torch.stack((selection_log_decay, selection_log_decay), dim=3).reshape(
            selection_batch, selection_length, 4096
        )
    )
    selection_erase_target = torch.logit(
        selection_erase.clamp(1e-6, 1 - 1e-6).reshape(
            selection_batch, selection_length, 4096
        )
    )
    for stem, wanted in (("w", decay_target), ("a", erase_target)):
        ratio, _ = fit_time_mix(x, _previous(x), wanted)
        ratio_parameter = getattr(target, f"x_{stem}")
        ratio_parameter.copy_(ratio.view(1, 1, -1).to(ratio_parameter))
        mixed = _mixed(x, _previous(x), ratio)
        bias, first, second = _low_rank_target(
            mixed, wanted, 128, nonlinear=stem == "w"
        )
        getattr(target, f"{stem}0").copy_(bias.view(1, 1, -1).to(ratio_parameter))
        getattr(target, f"{stem}1").copy_(first.to(ratio_parameter))
        getattr(target, f"{stem}2").copy_(second.to(ratio_parameter))
        label = "decay" if stem == "w" else "erase"
        selection_wanted = (
            selection_decay_target if stem == "w" else selection_erase_target
        )
        projection_metrics[f"{label}_projection_calibration_nmse"] = _nmse(
            _control_signal(target, stem, x), wanted
        )
        projection_metrics[f"{label}_projection_selection_nmse"] = _nmse(
            _control_signal(target, stem, selection_hidden), selection_wanted
        )

    gate_nmse, gate_ratio, _ = fit_native_gate(
        target, x, _previous(x), trace["gate"]
    )
    target.k_k.fill_(1)
    target.k_a.zero_()
    target.r_k.zero_()
    target.value_residual_scale.zero_()
    groupnorm_nmse = fit_groupnorm_affine(
        target,
        _native_raw_heads(target, x),
        trace["exact_attention"].transpose(1, 2),
    )
    projection_metrics["groupnorm_affine_calibration_nmse"] = groupnorm_nmse
    projection_metrics["groupnorm_affine_selection_nmse"] = _nmse(
        _apply_groupnorm_affine(
            target, _native_raw_heads(target, selection_hidden)
        ),
        selection_trace["exact_attention"].transpose(1, 2),
    )
    target.output.weight.copy_(source.o_proj.weight.to(target.output.weight))
    boundary_metrics = refit_native_output(
        target,
        x,
        trace["source_output"],
        source.o_proj.weight,
        selection_hidden,
        selection_trace["source_output"],
    )

    selection_targets = {
        "r": selection_trace["q_target"],
        "k": selection_trace["k_target"],
        "v": selection_trace["value_target"],
    }
    for name, wanted in selection_targets.items():
        ratio, projection = fitted_projections[name]
        projection_metrics[f"{name}_projection_selection_nmse"] = (
            _selection_projection_nmse(selection_hidden, wanted, ratio, projection)
        )
    projection_metrics["gate_projection_calibration_nmse"] = gate_nmse
    selection_gate = (
        torch.sigmoid(
            _mixed(
                selection_hidden,
                _previous(selection_hidden),
                gate_ratio,
            )
            @ target.g1.float()
        )
        @ target.g2.float()
    )
    projection_metrics["gate_projection_selection_nmse"] = _nmse(
        selection_gate, selection_trace["gate"]
    )

    affine_nmse, native_nmse = _trajectory_diagnostics(
        trace["query"],
        trace["centered_key"],
        trace["value_innovation"],
        trace["mean_value"],
        trace["exact_attention"],
        closure,
    )
    selection_affine_nmse, selection_native_nmse = _trajectory_diagnostics(
        selection_trace["query"],
        selection_trace["centered_key"],
        selection_trace["value_innovation"],
        selection_trace["mean_value"],
        selection_trace["exact_attention"],
        closure,
    )
    return {
        "reference_affine_attention_calibration_nmse": affine_nmse,
        "reference_affine_attention_selection_nmse": selection_affine_nmse,
        "initialized_native_clamp_delta_calibration_nmse": native_nmse,
        "initialized_native_clamp_delta_selection_nmse": selection_native_nmse,
        "initialized_closure": closure,
        **{f"fitted_{name}": value for name, value in projection_metrics.items()},
        **{f"fitted_{name}": value for name, value in boundary_metrics.items()},
    }


@torch.no_grad()
def initialize_gqa_layer(
    source,
    target,
    normalized_hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    selection_hidden: torch.Tensor | None = None,
    selection_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Select a complete canonical two-state D256 initialization on development."""
    if selection_hidden is None or selection_position_embeddings is None:
        raise ValueError("GQA compilation requires an independent development split")
    trace = _gqa_trace(source, normalized_hidden, position_embeddings)
    selection_trace = _gqa_trace(
        source, selection_hidden, selection_position_embeddings
    )
    initial = _snapshot(target)
    candidates = []
    for closure in CLOSURE_CANDIDATES:
        _restore(target, initial)
        try:
            metrics = _compile_gqa_candidate(
                source,
                target,
                normalized_hidden,
                trace,
                selection_hidden,
                selection_trace,
                closure,
            )
            selection_nmse = _nmse(
                _native_forward(target, selection_hidden),
                selection_trace["source_output"],
            )
            candidate_state = _snapshot(target)
        except Exception:
            _restore(target, initial)
            raise
        candidates.append((selection_nmse, closure, candidate_state, metrics))
    finite = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate[0]) and _state_is_finite(candidate[2])
    ]
    if not finite:
        _restore(target, initial)
        raise FloatingPointError("all GQA closure candidates were non-finite")
    selection_nmse, closure, selected_state, selected_metrics = min(
        finite, key=lambda candidate: candidate[0]
    )
    _restore(target, selected_state)
    requested_decay = 1 - trace["rho"]
    return {
        "reference_mean_exact_hazard": trace["mean_exact_hazard"],
        "reference_source_trace_output_nmse": trace[
            "reference_source_output_nmse"
        ],
        "reference_empirical_kernel_tail_ratio": trace["empirical_kernel_tail_ratio"],
        "reference_selection_empirical_kernel_tail_ratio": selection_trace[
            "empirical_kernel_tail_ratio"
        ],
        "initialized_selected_closure": closure,
        "fitted_compiled_selection_mixer_nmse": selection_nmse,
        "initialized_matrix_state_key_rms": trace["matrix_state_key_rms"],
        "initialized_matrix_state_write_rms": trace["matrix_state_write_rms"],
        "initialized_mean_state_write_rms": trace["mean_state_write_rms"],
        "initialized_requested_decay_below_native_floor_fraction": float(
            (requested_decay.clamp_min(1e-6).log() < W_SCALE).float().mean()
        ),
        "reference_trace_tokens": float(
            normalized_hidden.shape[0] * normalized_hidden.shape[1]
        ),
        **{
            "fitted_closure_"
            f"{str(value).replace('.', '_')}_selection_mixer_nmse": metric
            for metric, value, _, _ in candidates
        },
        **selected_metrics,
    }


__all__ = ["initialize_gqa_layer"]
