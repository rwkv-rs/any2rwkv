"""Compile Qwen GQA into a fixed two-state-per-query-head RWKV-7 TMix."""

from __future__ import annotations

import math

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from .transformers.modeling_qwen2rwkv import W_SCALE

RIDGE_SCALE = 1e-4
BOUNDARY_ROUNDS = 4
CG_STEPS = 24
CLOSURE_CANDIDATES = (0.0, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def _previous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), 1)


def _nmse(actual: torch.Tensor, wanted: torch.Tensor) -> float:
    actual = actual.float()
    wanted = wanted.float()
    value = (actual - wanted).square().mean() / (wanted.square().mean() + 1e-12)
    return float(value.detach())


def _ridge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1]).float()
    y = y.reshape(-1, y.shape[-1]).float()
    gram = x.T @ x
    regularization = RIDGE_SCALE * gram.trace() / x.shape[-1]
    return torch.linalg.solve(
        gram + regularization * torch.eye(gram.shape[0], device=x.device),
        x.T @ y,
    )


def _truncated_map(full: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    left, singular, right = torch.linalg.svd(full.float(), full_matrices=False)
    root = singular[:rank].sqrt()
    return left[:, :rank] * root, root[:, None] * right[:rank]


def _mixed(current: torch.Tensor, previous: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    return current.float() + (previous.float() - current.float()) * ratio


def fit_time_mix(
    current: torch.Tensor,
    previous: torch.Tensor,
    target: torch.Tensor,
    rounds: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = current.reshape(-1, current.shape[-1]).float()
    previous = previous.reshape_as(current).float()
    target = target.reshape(-1, target.shape[-1]).float()
    delta = previous - current
    ratio = current.new_full((current.shape[-1],), 0.5)
    projection = None
    for _ in range(rounds):
        mixed = current + delta * ratio
        projection = _ridge(mixed, target)
        error = mixed @ projection - target
        gradient = (delta * (error @ projection.T)).sum(0)
        diagonal = delta.square().sum(0) * projection.square().sum(1) + 1e-12
        ratio = (ratio - gradient / diagonal).clamp_(0, 1)
    if projection is None:
        raise RuntimeError("time-mix fitting did not produce a projection")
    return ratio, projection


def _native_decay_inverse(log_decay: torch.Tensor) -> torch.Tensor:
    retention_log = log_decay.clamp(min=W_SCALE + 1e-6, max=-1e-6)
    return torch.logit((retention_log / W_SCALE).clamp(1e-6, 1 - 1e-6))


def _low_rank_target(x: torch.Tensor, y: torch.Tensor, rank: int, *, nonlinear: bool):
    bias = y.reshape(-1, y.shape[-1]).float().mean(0)
    full = _ridge(x, y - bias)
    first, second = _truncated_map(full, rank)
    if nonlinear:
        features = torch.tanh(x.reshape(-1, x.shape[-1]).float() @ first)
        second = _ridge(features, y.reshape(-1, y.shape[-1]).float() - bias)
    return bias, first, second


def _control_signal(target, stem: str, hidden: torch.Tensor) -> torch.Tensor:
    mixed = _mixed(
        hidden,
        _previous(hidden),
        getattr(target, f"x_{stem}").float().reshape(-1),
    )
    features = mixed @ getattr(target, f"{stem}1").float()
    if stem == "w":
        features = torch.tanh(features)
    return getattr(target, f"{stem}0").float() + (features @ getattr(target, f"{stem}2").float())


def _snapshot(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _restore(module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(state)


def _state_is_finite(state: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all() for value in state.values())


def _native_forward(target, hidden: torch.Tensor) -> torch.Tensor:
    return target(
        hidden,
        None,
        None,
        torch.ones_like(hidden[..., 0]),
    )[0].float()


def _native_raw_heads(target, hidden: torch.Tensor) -> torch.Tensor:
    _, raw, _, _, _, _, _ = target._training_components(hidden, None)
    if target.states_per_head == 2:
        raw = target._pair_sum(raw.reshape(-1, 16, 256)).view(
            hidden.shape[0], hidden.shape[1], 8, 256
        )
    else:
        raw = raw.view(hidden.shape[0], hidden.shape[1], 16, 128)
    return raw.float()


def _apply_groupnorm_affine(target, raw: torch.Tensor) -> torch.Tensor:
    standardized = (raw.float() - raw.float().mean(-1, keepdim=True)) * torch.rsqrt(
        raw.float().var(-1, keepdim=True, unbiased=False) + target.ln_x.eps
    )
    heads, dim = raw.shape[-2:]
    weight = target.ln_x.weight.float().view(heads, dim)
    bias = target.ln_x.bias.float().view(heads, dim)
    return standardized * weight + bias


def fit_groupnorm_affine(target, raw: torch.Tensor, wanted: torch.Tensor) -> float:
    standardized = (raw.float() - raw.float().mean(-1, keepdim=True)) * torch.rsqrt(
        raw.float().var(-1, keepdim=True, unbiased=False) + target.ln_x.eps
    )
    flattened = standardized.reshape(-1, standardized.shape[-2], standardized.shape[-1])
    wanted = wanted.float().reshape_as(flattened)
    feature_mean = flattened.mean(0)
    wanted_mean = wanted.mean(0)
    centered_feature = flattened - feature_mean
    centered_wanted = wanted - wanted_mean
    feature_energy = centered_feature.square().sum(0)
    regularization = RIDGE_SCALE * feature_energy.mean().clamp_min(1e-12)
    weight = (centered_feature * centered_wanted).sum(0) / (feature_energy + regularization)
    bias = wanted_mean - feature_mean * weight
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        return math.inf
    target.ln_x.weight.copy_(weight.reshape(-1).to(target.ln_x.weight))
    target.ln_x.bias.copy_(bias.reshape(-1).to(target.ln_x.bias))
    return _nmse(_apply_groupnorm_affine(target, raw).reshape_as(wanted), wanted)


def _capture_native_input(target, hidden: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def capture(_module, arguments):
        captured.append(arguments[0].detach())

    hook = target.output.register_forward_pre_hook(capture)
    try:
        _native_forward(target, hidden)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one native output activation, got {len(captured)}")
    return captured[0]


def _fit_output_weight(
    target,
    hidden: torch.Tensor,
    wanted: torch.Tensor,
    prior: torch.Tensor,
) -> float:
    native_input = _capture_native_input(target, hidden)
    residual = wanted.float() - native_input.float() @ prior.float().T
    fitted = prior.float() + _ridge(native_input, residual).T
    if not torch.isfinite(fitted).all():
        return math.inf
    target.output.weight.copy_(fitted.to(target.output.weight))
    return _nmse(target.output(native_input), wanted)


def _r_k_features(target, hidden: torch.Tensor):
    _, _, read, key, value, gate, _ = target._training_components(hidden, None)
    states = target.kernel_heads
    dim = target.head_size
    p = (read.float() * key.float()).reshape(-1, states, dim)
    value = value.float().reshape(-1, states, dim)
    if target.states_per_head == 1:
        expanded_gate = gate.float().reshape(-1, states, dim)
    else:
        expanded_gate = (
            gate.float().reshape(-1, 8, dim)[:, :, None, :].expand(-1, -1, 2, -1)
        ).reshape(-1, states, dim)
    return p, value * expanded_gate


def _r_k_jvp(target, p, q, vector: torch.Tensor) -> torch.Tensor:
    scalars = (p * vector[None]).sum(-1, keepdim=True)
    contribution = q * scalars
    if target.states_per_head == 2:
        contribution = contribution.view(-1, 8, 2, target.head_size).sum(2)
    pre_output = contribution.reshape(-1, target.config.hidden_size)
    return pre_output @ target.output.weight.float().T


def _r_k_vjp(target, p, q, output_gradient: torch.Tensor) -> torch.Tensor:
    pre_gradient = output_gradient.float() @ target.output.weight.float()
    if target.states_per_head == 1:
        pre_gradient = pre_gradient.view(-1, target.kernel_heads, target.head_size)
    else:
        pre_gradient = (
            pre_gradient.view(-1, 8, target.head_size)[:, :, None, :]
            .expand(-1, -1, 2, -1)
            .reshape(-1, target.kernel_heads, target.head_size)
        )
    scalar_gradient = (pre_gradient * q).sum(-1, keepdim=True)
    return (p * scalar_gradient).sum(0)


def _r_k_diagonal(target, p, q) -> torch.Tensor:
    output_weight = target.output.weight.float()
    energies = []
    for state in range(target.kernel_heads):
        output_head = state if target.states_per_head == 1 else state // 2
        start = output_head * target.head_size
        stop = start + target.head_size
        gram = output_weight[:, start:stop].T @ output_weight[:, start:stop]
        q_state = q[:, state]
        energies.append(torch.einsum("nd,de,ne->n", q_state, gram, q_state))
    energy = torch.stack(energies, 1)
    return (p.square() * energy[..., None]).sum(0)


def _fit_r_k(target, hidden: torch.Tensor, wanted: torch.Tensor) -> float:
    p, q = _r_k_features(target, hidden)
    target.r_k.zero_()
    baseline = _native_forward(target, hidden).reshape(-1, target.config.hidden_size)
    residual = wanted.float().reshape_as(baseline) - baseline
    right = _r_k_vjp(target, p, q, residual)
    diagonal = _r_k_diagonal(target, p, q)
    regularization = RIDGE_SCALE * diagonal.mean().clamp_min(1e-12)
    preconditioner = (diagonal + regularization).clamp_min(1e-12)

    def normal(vector: torch.Tensor) -> torch.Tensor:
        projected = _r_k_jvp(target, p, q, vector)
        return _r_k_vjp(target, p, q, projected) + regularization * vector

    solution = torch.zeros_like(right)
    conjugate_residual = right - normal(solution)
    preconditioned = conjugate_residual / preconditioner
    direction = preconditioned.clone()
    rz = (conjugate_residual * preconditioned).sum()
    initial = conjugate_residual.square().sum().sqrt().clamp_min(1e-24)
    for _ in range(CG_STEPS):
        projected = normal(direction)
        denominator = (direction * projected).sum()
        if not torch.isfinite(denominator) or denominator.abs() <= 1e-24:
            break
        step = rz / denominator
        solution = solution + step * direction
        conjugate_residual = conjugate_residual - step * projected
        if conjugate_residual.square().sum().sqrt() <= 1e-5 * initial:
            break
        preconditioned = conjugate_residual / preconditioner
        next_rz = (conjugate_residual * preconditioned).sum()
        direction = preconditioned + (next_rz / rz.clamp_min(1e-24)) * direction
        rz = next_rz
    if not torch.isfinite(solution).all():
        return math.inf
    target.r_k.copy_(solution.to(target.r_k))
    return _nmse(_native_forward(target, hidden), wanted)


def refit_native_output(
    target,
    normalized_hidden: torch.Tensor,
    source_output: torch.Tensor,
    prior_weight: torch.Tensor | None = None,
    selection_hidden: torch.Tensor | None = None,
    selection_source_output: torch.Tensor | None = None,
) -> dict[str, float]:
    if (selection_hidden is None) != (selection_source_output is None):
        raise ValueError("selection hidden and source output must be provided together")
    snapshot = _snapshot(target)
    prior_calibration_nmse = _nmse(_native_forward(target, normalized_hidden), source_output)
    prior_selection_nmse = math.nan
    if selection_hidden is not None:
        prior_selection_nmse = _nmse(
            _native_forward(target, selection_hidden), selection_source_output
        )
    prior = (
        target.output.weight.detach().float().clone()
        if prior_weight is None
        else prior_weight.float().clone()
    )
    previous_calibration = math.inf
    rounds = 0
    for round_index in range(BOUNDARY_ROUNDS):
        rounds = round_index + 1
        output_nmse = _fit_output_weight(target, normalized_hidden, source_output, prior)
        if not math.isfinite(output_nmse):
            break
        r_k_nmse = _fit_r_k(target, normalized_hidden, source_output)
        if not math.isfinite(r_k_nmse):
            break
        prior = target.output.weight.detach().float().clone()
        calibration_nmse = _fit_output_weight(target, normalized_hidden, source_output, prior)
        if not math.isfinite(calibration_nmse):
            break
        relative = (previous_calibration - calibration_nmse) / max(abs(previous_calibration), 1e-12)
        previous_calibration = calibration_nmse
        if relative >= 0 and relative < 1e-4:
            break

    candidate_calibration_nmse = _nmse(_native_forward(target, normalized_hidden), source_output)
    candidate_selection_nmse = math.nan
    selected = math.isfinite(candidate_calibration_nmse)
    if selection_hidden is not None:
        candidate_selection_nmse = _nmse(
            _native_forward(target, selection_hidden), selection_source_output
        )
        selected = (
            selected
            and math.isfinite(candidate_selection_nmse)
            and (candidate_selection_nmse < prior_selection_nmse)
        )
    if not selected:
        _restore(target, snapshot)
    installed_calibration_nmse = _nmse(_native_forward(target, normalized_hidden), source_output)
    installed_selection_nmse = math.nan
    if selection_hidden is not None:
        installed_selection_nmse = _nmse(
            _native_forward(target, selection_hidden), selection_source_output
        )
    return {
        "native_boundary_calibration_nmse": installed_calibration_nmse,
        "native_boundary_calibration_prior_nmse": prior_calibration_nmse,
        "native_boundary_calibration_candidate_nmse": candidate_calibration_nmse,
        "native_boundary_calibration_installed_nmse": installed_calibration_nmse,
        "native_boundary_selection_prior_nmse": prior_selection_nmse,
        "native_boundary_selection_candidate_nmse": candidate_selection_nmse,
        "native_boundary_selection_installed_nmse": installed_selection_nmse,
        "native_boundary_candidate_selected": float(selected),
        "native_boundary_rounds": float(rounds),
    }


def fit_native_gate(
    target,
    current: torch.Tensor,
    previous: torch.Tensor,
    wanted: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    ratio, full = fit_time_mix(current, previous, wanted)
    first, _ = _truncated_map(full, target.config.gate_low_rank_dim)
    target.x_g.copy_(ratio.view(1, 1, -1).to(target.x_g))
    target.g1.copy_(first.to(target.g1))
    mixed = _mixed(current, previous, ratio)
    features = torch.sigmoid(mixed @ target.g1.float())
    target.g2.copy_(_ridge(features, wanted).to(target.g2))
    actual = features @ target.g2.float()
    return _nmse(actual, wanted), ratio, target.g2.float()


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
            - torch.einsum("bhi,bhj->bhij", native_memory, direction * matrix_erase)
            + torch.einsum("bhi,bhj->bhij", rho * norm * innovation, direction)
        )
        mean_erase = realized_decay - requested_decay
        previous_mean = mean_value[:, token - 1].float() if token else 0
        native_mean = native_mean * (realized_decay - mean_erase) + rho * (
            value_innovation[:, token].float() + previous_mean
        )
        native = native_mean + torch.einsum("bhij,bhj->bhi", native_matrix, query[:, token].float())
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
    k_target = torch.stack((centered_pre_rope, dc), dim=3).reshape(batch, length, 4096)
    matrix_write = rho * centered_norm * value_innovation
    mean_write = rho * source_value
    value_target = torch.stack((matrix_write, mean_write), dim=3).reshape(batch, length, 4096)
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
        ratio, fitted, calibration_nmse = _fit_linear_projection(target, name, module, x, wanted)
        fitted_projections[name] = (ratio, fitted)
        projection_metrics[f"{name}_projection_calibration_nmse"] = calibration_nmse

    ratio, fitted = fit_time_mix(x, _previous(x), trace["value_target"])
    target.x_v.copy_(ratio.view(1, 1, channels))
    target.value_base.weight.copy_(
        torch.eye(channels, dtype=target.value_base.weight.dtype, device=x.device)
    )
    target.value_expand.weight.copy_(fitted.T.to(target.value_expand.weight))
    value_actual = _mixed(x, _previous(x), ratio) @ target.value_expand.weight.float().T
    projection_metrics["v_projection_calibration_nmse"] = _nmse(value_actual, trace["value_target"])
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
    erase_target = torch.logit(erase.clamp(1e-6, 1 - 1e-6).reshape(batch, length, 4096))
    selection_rho = selection_trace["rho"]
    selection_batch, selection_length = selection_hidden.shape[:2]
    selection_requested_decay = (1 - selection_rho).expand(
        selection_batch, selection_length, 8, 256
    )
    selection_log_decay = selection_requested_decay.clamp_min(1e-6).log()
    selection_realized_decay = selection_log_decay.clamp_min(W_SCALE + 1e-6).exp()
    selection_matrix_erase = closure * selection_rho * selection_trace["centered_norm"].square()
    selection_mean_erase = (selection_realized_decay - selection_requested_decay).clamp_min(1e-6)
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
        selection_erase.clamp(1e-6, 1 - 1e-6).reshape(selection_batch, selection_length, 4096)
    )
    for stem, wanted in (("w", decay_target), ("a", erase_target)):
        ratio, _ = fit_time_mix(x, _previous(x), wanted)
        ratio_parameter = getattr(target, f"x_{stem}")
        ratio_parameter.copy_(ratio.view(1, 1, -1).to(ratio_parameter))
        mixed = _mixed(x, _previous(x), ratio)
        bias, first, second = _low_rank_target(mixed, wanted, 128, nonlinear=stem == "w")
        getattr(target, f"{stem}0").copy_(bias.view(1, 1, -1).to(ratio_parameter))
        getattr(target, f"{stem}1").copy_(first.to(ratio_parameter))
        getattr(target, f"{stem}2").copy_(second.to(ratio_parameter))
        label = "decay" if stem == "w" else "erase"
        selection_wanted = selection_decay_target if stem == "w" else selection_erase_target
        projection_metrics[f"{label}_projection_calibration_nmse"] = _nmse(
            _control_signal(target, stem, x), wanted
        )
        projection_metrics[f"{label}_projection_selection_nmse"] = _nmse(
            _control_signal(target, stem, selection_hidden), selection_wanted
        )

    gate_nmse, gate_ratio, _ = fit_native_gate(target, x, _previous(x), trace["gate"])
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
        _apply_groupnorm_affine(target, _native_raw_heads(target, selection_hidden)),
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
        projection_metrics[f"{name}_projection_selection_nmse"] = _selection_projection_nmse(
            selection_hidden, wanted, ratio, projection
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
    selection_trace = _gqa_trace(source, selection_hidden, selection_position_embeddings)
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
        "reference_source_trace_output_nmse": trace["reference_source_output_nmse"],
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
        "reference_trace_tokens": float(normalized_hidden.shape[0] * normalized_hidden.shape[1]),
        **{
            f"fitted_closure_{str(value).replace('.', '_')}_selection_mixer_nmse": metric
            for metric, value, _, _ in candidates
        },
        **selected_metrics,
    }


__all__ = ["initialize_gqa_layer"]
