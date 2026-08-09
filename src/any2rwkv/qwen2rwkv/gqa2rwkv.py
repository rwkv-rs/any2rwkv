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
    init_hidden: torch.Tensor,
    source_output: torch.Tensor,
    prior_weight: torch.Tensor | None = None,
    validation_hidden: torch.Tensor | None = None,
    validation_source_output: torch.Tensor | None = None,
) -> dict[str, float]:
    if (validation_hidden is None) != (validation_source_output is None):
        raise ValueError("validation hidden and source output must be provided together")
    snapshot = _snapshot(target)
    prior_init_nmse = _nmse(_native_forward(target, init_hidden), source_output)
    prior_validation_nmse = math.nan
    if validation_hidden is not None:
        prior_validation_nmse = _nmse(
            _native_forward(target, validation_hidden), validation_source_output
        )
    prior = (
        target.output.weight.detach().float().clone()
        if prior_weight is None
        else prior_weight.float().clone()
    )
    previous_init = math.inf
    rounds = 0
    for round_index in range(BOUNDARY_ROUNDS):
        rounds = round_index + 1
        output_nmse = _fit_output_weight(target, init_hidden, source_output, prior)
        if not math.isfinite(output_nmse):
            break
        r_k_nmse = _fit_r_k(target, init_hidden, source_output)
        if not math.isfinite(r_k_nmse):
            break
        prior = target.output.weight.detach().float().clone()
        init_nmse = _fit_output_weight(target, init_hidden, source_output, prior)
        if not math.isfinite(init_nmse):
            break
        relative = (previous_init - init_nmse) / max(abs(previous_init), 1e-12)
        previous_init = init_nmse
        if relative >= 0 and relative < 1e-4:
            break

    candidate_init_nmse = _nmse(_native_forward(target, init_hidden), source_output)
    candidate_validation_nmse = math.nan
    accepted = math.isfinite(candidate_init_nmse)
    if validation_hidden is not None:
        candidate_validation_nmse = _nmse(
            _native_forward(target, validation_hidden), validation_source_output
        )
        accepted = (
            accepted
            and math.isfinite(candidate_validation_nmse)
            and (candidate_validation_nmse < prior_validation_nmse)
        )
    if not accepted:
        _restore(target, snapshot)
    installed_init_nmse = _nmse(_native_forward(target, init_hidden), source_output)
    installed_validation_nmse = math.nan
    if validation_hidden is not None:
        installed_validation_nmse = _nmse(
            _native_forward(target, validation_hidden), validation_source_output
        )
    return {
        "native_boundary_init_nmse": installed_init_nmse,
        "native_boundary_init_prior_nmse": prior_init_nmse,
        "native_boundary_init_candidate_nmse": candidate_init_nmse,
        "native_boundary_init_installed_nmse": installed_init_nmse,
        "native_boundary_validation_prior_nmse": prior_validation_nmse,
        "native_boundary_validation_candidate_nmse": candidate_validation_nmse,
        "native_boundary_validation_installed_nmse": installed_validation_nmse,
        "native_boundary_candidate_accepted": float(accepted),
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


def _validation_projection_nmse(x, wanted, ratio, projection):
    actual = _mixed(x, _previous(x), ratio) @ projection
    return _nmse(actual, wanted)


def _compile_gqa_candidate(
    source,
    target,
    init_hidden: torch.Tensor,
    init_trace,
    validation_hidden: torch.Tensor,
    validation_trace,
    closure: float,
) -> dict[str, float]:
    channels = init_hidden.shape[-1]
    fitted_projections = {}
    projection_metrics = {}
    for name, wanted, module in (
        ("r", init_trace["q_target"], target.receptance),
        ("k", init_trace["k_target"], target.key),
    ):
        ratio, fitted, init_nmse = _fit_linear_projection(target, name, module, init_hidden, wanted)
        fitted_projections[name] = (ratio, fitted)
        projection_metrics[f"{name}_projection_init_nmse"] = init_nmse

    ratio, fitted = fit_time_mix(init_hidden, _previous(init_hidden), init_trace["value_target"])
    target.x_v.copy_(ratio.view(1, 1, channels))
    target.value_base.weight.copy_(
        torch.eye(channels, dtype=target.value_base.weight.dtype, device=init_hidden.device)
    )
    target.value_expand.weight.copy_(fitted.T.to(target.value_expand.weight))
    value_actual = (
        _mixed(init_hidden, _previous(init_hidden), ratio) @ target.value_expand.weight.float().T
    )
    projection_metrics["v_projection_init_nmse"] = _nmse(value_actual, init_trace["value_target"])
    fitted_projections["v"] = (ratio, fitted)

    rho = init_trace["rho"]
    batch, length = init_hidden.shape[:2]
    requested_decay = (1 - rho).expand(batch, length, 8, 256)
    log_decay = requested_decay.clamp_min(1e-6).log()
    realized_log_decay = log_decay.clamp_min(W_SCALE + 1e-6)
    realized_decay = realized_log_decay.exp()
    matrix_erase = closure * rho * init_trace["centered_norm"].square()
    mean_erase = (realized_decay - requested_decay).clamp_min(1e-6)
    erase = torch.stack((matrix_erase.expand_as(requested_decay), mean_erase), dim=3)
    decay_target = _native_decay_inverse(
        torch.stack((log_decay, log_decay), dim=3).reshape(batch, length, 4096)
    )
    erase_target = torch.logit(erase.clamp(1e-6, 1 - 1e-6).reshape(batch, length, 4096))
    validation_rho = validation_trace["rho"]
    validation_batch, validation_length = validation_hidden.shape[:2]
    validation_requested_decay = (1 - validation_rho).expand(
        validation_batch, validation_length, 8, 256
    )
    validation_log_decay = validation_requested_decay.clamp_min(1e-6).log()
    validation_realized_decay = validation_log_decay.clamp_min(W_SCALE + 1e-6).exp()
    validation_matrix_erase = closure * validation_rho * validation_trace["centered_norm"].square()
    validation_mean_erase = (validation_realized_decay - validation_requested_decay).clamp_min(1e-6)
    validation_erase = torch.stack(
        (
            validation_matrix_erase.expand_as(validation_requested_decay),
            validation_mean_erase,
        ),
        dim=3,
    )
    validation_decay_target = _native_decay_inverse(
        torch.stack((validation_log_decay, validation_log_decay), dim=3).reshape(
            validation_batch, validation_length, 4096
        )
    )
    validation_erase_target = torch.logit(
        validation_erase.clamp(1e-6, 1 - 1e-6).reshape(validation_batch, validation_length, 4096)
    )
    for stem, wanted in (("w", decay_target), ("a", erase_target)):
        ratio, _ = fit_time_mix(init_hidden, _previous(init_hidden), wanted)
        ratio_parameter = getattr(target, f"x_{stem}")
        ratio_parameter.copy_(ratio.view(1, 1, -1).to(ratio_parameter))
        mixed = _mixed(init_hidden, _previous(init_hidden), ratio)
        bias, first, second = _low_rank_target(mixed, wanted, 128, nonlinear=stem == "w")
        getattr(target, f"{stem}0").copy_(bias.view(1, 1, -1).to(ratio_parameter))
        getattr(target, f"{stem}1").copy_(first.to(ratio_parameter))
        getattr(target, f"{stem}2").copy_(second.to(ratio_parameter))
        label = "decay" if stem == "w" else "erase"
        validation_wanted = validation_decay_target if stem == "w" else validation_erase_target
        projection_metrics[f"{label}_projection_init_nmse"] = _nmse(
            _control_signal(target, stem, init_hidden), wanted
        )
        projection_metrics[f"{label}_projection_validation_nmse"] = _nmse(
            _control_signal(target, stem, validation_hidden), validation_wanted
        )

    gate_nmse, gate_ratio, _ = fit_native_gate(
        target, init_hidden, _previous(init_hidden), init_trace["gate"]
    )
    target.k_k.fill_(1)
    target.k_a.zero_()
    target.r_k.zero_()
    target.value_residual_scale.zero_()
    groupnorm_nmse = fit_groupnorm_affine(
        target,
        _native_raw_heads(target, init_hidden),
        init_trace["exact_attention"].transpose(1, 2),
    )
    projection_metrics["groupnorm_affine_init_nmse"] = groupnorm_nmse
    projection_metrics["groupnorm_affine_validation_nmse"] = _nmse(
        _apply_groupnorm_affine(target, _native_raw_heads(target, validation_hidden)),
        validation_trace["exact_attention"].transpose(1, 2),
    )
    target.output.weight.copy_(source.o_proj.weight.to(target.output.weight))
    boundary_metrics = refit_native_output(
        target,
        init_hidden,
        init_trace["source_output"],
        source.o_proj.weight,
        validation_hidden,
        validation_trace["source_output"],
    )

    validation_targets = {
        "r": validation_trace["q_target"],
        "k": validation_trace["k_target"],
        "v": validation_trace["value_target"],
    }
    for name, wanted in validation_targets.items():
        ratio, projection = fitted_projections[name]
        projection_metrics[f"{name}_projection_validation_nmse"] = _validation_projection_nmse(
            validation_hidden, wanted, ratio, projection
        )
    projection_metrics["gate_projection_init_nmse"] = gate_nmse
    validation_gate = (
        torch.sigmoid(
            _mixed(
                validation_hidden,
                _previous(validation_hidden),
                gate_ratio,
            )
            @ target.g1.float()
        )
        @ target.g2.float()
    )
    projection_metrics["gate_projection_validation_nmse"] = _nmse(
        validation_gate, validation_trace["gate"]
    )

    init_affine_nmse, init_native_nmse = _trajectory_diagnostics(
        init_trace["query"],
        init_trace["centered_key"],
        init_trace["value_innovation"],
        init_trace["mean_value"],
        init_trace["exact_attention"],
        closure,
    )
    validation_affine_nmse, validation_native_nmse = _trajectory_diagnostics(
        validation_trace["query"],
        validation_trace["centered_key"],
        validation_trace["value_innovation"],
        validation_trace["mean_value"],
        validation_trace["exact_attention"],
        closure,
    )
    return {
        "reference_affine_attention_init_nmse": init_affine_nmse,
        "reference_affine_attention_validation_nmse": validation_affine_nmse,
        "initialized_native_clamp_delta_init_nmse": init_native_nmse,
        "initialized_native_clamp_delta_validation_nmse": validation_native_nmse,
        "initialized_closure": closure,
        **{f"fitted_{name}": value for name, value in projection_metrics.items()},
        **{f"fitted_{name}": value for name, value in boundary_metrics.items()},
    }


@torch.no_grad()
def initialize_gqa_layer(
    source,
    target,
    init_hidden: torch.Tensor,
    init_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    validation_hidden: torch.Tensor | None = None,
    validation_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Fit GQA candidates on init data and choose one on validation data."""
    if validation_hidden is None or validation_position_embeddings is None:
        raise ValueError("GQA compilation requires independent validation data")
    init_trace = _gqa_trace(source, init_hidden, init_position_embeddings)
    validation_trace = _gqa_trace(source, validation_hidden, validation_position_embeddings)
    initial = _snapshot(target)
    candidates = []
    for closure in CLOSURE_CANDIDATES:
        _restore(target, initial)
        try:
            metrics = _compile_gqa_candidate(
                source,
                target,
                init_hidden,
                init_trace,
                validation_hidden,
                validation_trace,
                closure,
            )
            validation_nmse = _nmse(
                _native_forward(target, validation_hidden),
                validation_trace["source_output"],
            )
            candidate_state = _snapshot(target)
        except Exception:
            _restore(target, initial)
            raise
        candidates.append((validation_nmse, closure, candidate_state, metrics))
    finite = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate[0]) and _state_is_finite(candidate[2])
    ]
    if not finite:
        _restore(target, initial)
        raise FloatingPointError("all GQA closure candidates were non-finite")
    validation_nmse, _, best_state, best_metrics = min(finite, key=lambda candidate: candidate[0])
    _restore(target, best_state)
    requested_decay = 1 - init_trace["rho"]
    return {
        "reference_init_mean_exact_hazard": init_trace["mean_exact_hazard"],
        "reference_init_source_trace_output_nmse": init_trace["reference_source_output_nmse"],
        "reference_init_empirical_kernel_tail_ratio": init_trace["empirical_kernel_tail_ratio"],
        "reference_validation_empirical_kernel_tail_ratio": validation_trace[
            "empirical_kernel_tail_ratio"
        ],
        "fitted_compiled_validation_tmix_output_nmse": validation_nmse,
        "init_matrix_state_key_rms": init_trace["matrix_state_key_rms"],
        "init_matrix_state_write_rms": init_trace["matrix_state_write_rms"],
        "init_mean_state_write_rms": init_trace["mean_state_write_rms"],
        "init_requested_decay_below_native_floor_fraction": float(
            (requested_decay.clamp_min(1e-6).log() < W_SCALE).float().mean()
        ),
        "reference_init_trace_tokens": float(init_hidden.shape[0] * init_hidden.shape[1]),
        **{
            f"fitted_closure_{str(value).replace('.', '_')}_validation_tmix_output_nmse": metric
            for metric, value, _, _ in candidates
        },
        **best_metrics,
    }


__all__ = ["initialize_gqa_layer"]
