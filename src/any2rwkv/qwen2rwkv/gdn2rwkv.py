"""Compile complete Qwen3.5 GDN mixers into canonical RWKV-7 TMix."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

RIDGE_SCALE = 1e-4
W_SCALE = -torch.exp(torch.tensor(-0.5)).item()
NATIVE_DECAY_FLOOR = math.exp(W_SCALE)
BOUNDARY_ROUNDS = 4
CG_STEPS = 24


def _nmse(actual: torch.Tensor, wanted: torch.Tensor) -> float:
    actual = actual.float()
    wanted = wanted.float()
    return float((actual - wanted).square().mean() / (wanted.square().mean() + 1e-12))


def ridge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the bias-free ridge map ``x @ W ~= y`` with the fixed recipe lambda."""
    x = x.reshape(-1, x.shape[-1]).float()
    y = y.reshape(-1, y.shape[-1]).float()
    gram = x.T @ x
    lam = RIDGE_SCALE * gram.trace() / x.shape[-1]
    return torch.linalg.solve(
        gram + lam * torch.eye(gram.shape[0], device=x.device), x.T @ y
    )


def truncated_map(full: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(full.float(), full_matrices=False)
    root = s[:rank].sqrt()
    return u[:, :rank] * root, root[:, None] * vh[:rank]


def _mixed(current: torch.Tensor, previous: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    return current.float() + (previous.float() - current.float()) * ratio


def fit_time_mix(
    current: torch.Tensor,
    previous: torch.Tensor,
    target: torch.Tensor,
    rounds: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed diagonal-ALS rounds for RWKV's channelwise one-token shift."""
    current = current.reshape(-1, current.shape[-1]).float()
    previous = previous.reshape_as(current).float()
    target = target.reshape(-1, target.shape[-1]).float()
    delta = previous - current
    ratio = current.new_full((current.shape[-1],), 0.5)
    projection = None
    for _ in range(rounds):
        mixed = current + delta * ratio
        projection = ridge(mixed, target)
        error = mixed @ projection - target
        gradient = (delta * (error @ projection.T)).sum(0)
        diagonal = delta.square().sum(0) * projection.square().sum(1) + 1e-12
        ratio = (ratio - gradient / diagonal).clamp_(0, 1)
    if projection is None:
        raise RuntimeError("time-mix fitting did not produce a projection")
    return ratio, projection


def _previous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def _native_decay_inverse(log_decay: torch.Tensor) -> torch.Tensor:
    retention_log = log_decay.clamp(min=W_SCALE + 1e-6, max=-1e-6)
    return torch.logit((retention_log / W_SCALE).clamp(1e-6, 1 - 1e-6))


def _low_rank_target(
    x: torch.Tensor, y: torch.Tensor, rank: int, *, nonlinear: bool
):
    bias = y.reshape(-1, y.shape[-1]).float().mean(0)
    full = ridge(x, y - bias)
    first, second = truncated_map(full, rank)
    if nonlinear:
        features = torch.tanh(x.reshape(-1, x.shape[-1]).float() @ first)
        second = ridge(features, y.reshape(-1, y.shape[-1]).float() - bias)
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
    return getattr(target, f"{stem}0").float() + (
        features @ getattr(target, f"{stem}2").float()
    )


def _rwkv_trace(r, k, v, erase, log_decay):
    batch, length, heads, dim = r.shape
    state = r.new_zeros(batch, heads, dim, dim, dtype=torch.float32)
    outputs = []
    for token in range(length):
        kt = k[:, token].float()
        memory = torch.einsum("bhij,bhj->bhi", state, kt)
        state = (
            state * log_decay[:, token].float().exp()[..., None, None]
            - erase[:, token].float()[..., None, None]
            * torch.einsum("bhi,bhj->bhij", memory, kt)
        )
        state = state + torch.einsum("bhi,bhj->bhij", v[:, token].float(), kt)
        outputs.append(torch.einsum("bhij,bhj->bhi", state, r[:, token].float()))
    return torch.stack(outputs, 1)


def _snapshot(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _restore(module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(state)


def _state_is_finite(state: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all() for value in state.values())


def _native_forward(target, hidden: torch.Tensor) -> torch.Tensor:
    v_first = None if target.layer_idx == 0 else torch.zeros_like(hidden)
    return target(
        hidden,
        v_first,
        None,
        torch.ones_like(hidden[..., 0]),
    )[0].float()


def _native_raw_heads(target, hidden: torch.Tensor) -> torch.Tensor:
    v_first = None if target.layer_idx == 0 else torch.zeros_like(hidden)
    _, raw, _, _, _, _, _ = target._training_components(hidden, v_first)
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


def fit_groupnorm_affine(
    target, raw: torch.Tensor, wanted: torch.Tensor
) -> float:
    """Fit per-channel affine terms after the canonical per-token/head GroupNorm."""
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
    lam = RIDGE_SCALE * feature_energy.mean().clamp_min(1e-12)
    weight = (centered_feature * centered_wanted).sum(0) / (feature_energy + lam)
    bias = wanted_mean - feature_mean * weight
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        return math.inf
    target.ln_x.weight.copy_(weight.reshape(-1).to(target.ln_x.weight))
    target.ln_x.bias.copy_(bias.reshape(-1).to(target.ln_x.bias))
    fitted = _apply_groupnorm_affine(target, raw).reshape_as(wanted)
    return _nmse(fitted, wanted)


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
    fitted = prior.float() + ridge(native_input, residual).T
    if not torch.isfinite(fitted).all():
        return math.inf
    target.output.weight.copy_(fitted.to(target.output.weight))
    return _nmse(target.output(native_input), wanted)


def _r_k_features(target, hidden: torch.Tensor):
    v_first = None if target.layer_idx == 0 else torch.zeros_like(hidden)
    _, _, r, k, v, gate, _ = target._training_components(hidden, v_first)
    states = target.kernel_heads
    dim = target.head_size
    p = (r.float() * k.float()).reshape(-1, states, dim)
    v = v.float().reshape(-1, states, dim)
    if target.states_per_head == 1:
        expanded_gate = gate.float().reshape(-1, states, dim)
    else:
        expanded_gate = (
            gate.float().reshape(-1, 8, dim)[:, :, None, :].expand(-1, -1, 2, -1)
        ).reshape(-1, states, dim)
    return p, v * expanded_gate


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
    lam = RIDGE_SCALE * diagonal.mean().clamp_min(1e-12)
    preconditioner = (diagonal + lam).clamp_min(1e-12)

    def normal(vector: torch.Tensor) -> torch.Tensor:
        projected = _r_k_jvp(target, p, q, vector)
        return _r_k_vjp(target, p, q, projected) + lam * vector

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
    """Alternately fit canonical post-GroupNorm ``r_k`` and output projection."""
    if (selection_hidden is None) != (selection_source_output is None):
        raise ValueError("selection hidden and source output must be provided together")
    snapshot = _snapshot(target)
    prior_calibration_nmse = _nmse(
        _native_forward(target, normalized_hidden), source_output
    )
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
        calibration_nmse = _fit_output_weight(
            target, normalized_hidden, source_output, prior
        )
        if not math.isfinite(calibration_nmse):
            break
        relative = (previous_calibration - calibration_nmse) / max(
            abs(previous_calibration), 1e-12
        )
        previous_calibration = calibration_nmse
        if relative >= 0 and relative < 1e-4:
            break

    candidate_calibration_nmse = _nmse(
        _native_forward(target, normalized_hidden), source_output
    )
    candidate_selection_nmse = math.nan
    selected = math.isfinite(candidate_calibration_nmse)
    if selection_hidden is not None:
        candidate_selection_nmse = _nmse(
            _native_forward(target, selection_hidden), selection_source_output
        )
        selected = selected and math.isfinite(candidate_selection_nmse) and (
            candidate_selection_nmse < prior_selection_nmse
        )
    if not selected:
        _restore(target, snapshot)
    installed_calibration_nmse = _nmse(
        _native_forward(target, normalized_hidden), source_output
    )
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
    """Fit the actual ``sigmoid(x_g @ g1) @ g2`` gate parameterization."""
    ratio, full = fit_time_mix(current, previous, wanted)
    first, _ = truncated_map(full, target.config.gate_low_rank_dim)
    target.x_g.copy_(ratio.view(1, 1, -1).to(target.x_g))
    target.g1.copy_(first.to(target.g1))
    mixed = _mixed(current, previous, ratio)
    features = torch.sigmoid(mixed @ target.g1.float())
    target.g2.copy_(ridge(features, wanted).to(target.g2))
    actual = features @ target.g2.float()
    return _nmse(actual, wanted), ratio, target.g2.float()


def _gdn_trace(source, x: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, length, _ = x.shape
    qkv = source.in_proj_qkv(x).transpose(1, 2)
    qkv = causal_conv1d_fn(
        qkv,
        source.conv1d.weight.squeeze(1),
        source.conv1d.bias,
        activation=source.activation,
    )[:, :, :length].transpose(1, 2)
    q, k, v = torch.split(qkv, (source.key_dim, source.key_dim, source.value_dim), -1)
    q = q.view(batch, length, 16, 128)
    k = k.view(batch, length, 16, 128)
    q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).float()
    k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).float()
    v = v.view(batch, length, 16, 128)
    beta = source.in_proj_b(x).sigmoid().float()
    log_decay = -source.A_log.float().exp() * F.softplus(
        source.in_proj_a(x).float() + source.dt_bias.float()
    )
    retention = log_decay.exp()
    source_erase = beta * retention
    write = beta[..., None] * v
    read = q / 128**0.5
    oracle_raw = _rwkv_trace(read, k, write, source_erase, log_decay)
    captured: dict[str, torch.Tensor] = {}

    def capture_source_boundary(_module, arguments, output):
        captured["raw"] = arguments[0].detach().reshape(batch, length, 16, 128)
        captured["gate_pre"] = arguments[1].detach().reshape(
            batch, length, 16, 128
        )
        captured["pre_output"] = output.detach().reshape(batch, length, 2048)

    hook = source.norm.register_forward_hook(capture_source_boundary)
    try:
        source_output = source(x).float()
    finally:
        hook.remove()
    if set(captured) != {"raw", "gate_pre", "pre_output"}:
        raise RuntimeError("failed to capture the complete source GDN output boundary")
    raw = captured["raw"].float()
    gate_pre = captured["gate_pre"]
    gate = F.silu(gate_pre.float())
    trace_pre_output = source.norm(
        oracle_raw.reshape(-1, 128).to(captured["raw"]),
        gate_pre.reshape(-1, 128),
    ).reshape(batch, length, 2048)
    trace_output = source.out_proj(trace_pre_output).float()
    return {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "log_decay": log_decay,
        "retention": retention,
        "source_erase": source_erase,
        "write": write,
        "read": read,
        "raw": raw,
        "oracle_raw": oracle_raw,
        "gate": gate,
        "source_pre_output": captured["pre_output"].float(),
        "trace_pre_output": trace_pre_output.float(),
        "source_output": source_output,
        "trace_output": trace_output,
    }


def _householder_basis(
    source, trace
) -> tuple[torch.Tensor, float, tuple[float, ...], int]:
    raw = trace["raw"].reshape(-1, 16, 128).float()
    gate = trace["gate"].reshape(-1, 16, 128).float()
    sample_count = min(raw.shape[0], 2048)
    if sample_count < raw.shape[0]:
        indices = torch.linspace(
            0, raw.shape[0] - 1, sample_count, device=raw.device
        ).long()
        raw = raw.index_select(0, indices)
        gate = gate.index_select(0, indices)
    gamma = source.norm.weight.float()
    output = source.out_proj.weight.float()
    identity = torch.eye(128, device=raw.device)
    dc = torch.ones(128, device=raw.device) / 128**0.5
    bases = []
    head_tail_ratios = []
    tail = raw.new_zeros(())
    total = raw.new_zeros(())
    for head in range(16):
        start = head * 128
        stop = start + 128
        output_gram = output[:, start:stop].T @ output[:, start:stop]
        gram = raw.new_zeros(128, 128)
        for offset in range(0, raw.shape[0], 128):
            z = raw[offset : offset + 128, head]
            diagonal = gate[offset : offset + 128, head] * gamma
            scale = torch.rsqrt(z.square().mean(-1) + source.layer_norm_epsilon)
            radial = scale.pow(3) / 128
            weighted = (
                diagonal[:, :, None]
                * output_gram[None]
                * diagonal[:, None, :]
            )
            weighted_z = torch.einsum("nij,nj->ni", weighted, z)
            z_weighted_z = (z * weighted_z).sum(-1)
            jacobian_gram = scale[:, None, None].square() * weighted
            jacobian_gram -= (
                scale * radial
            )[:, None, None] * (
                weighted_z[:, :, None] * z[:, None, :]
                + z[:, :, None] * weighted_z[:, None, :]
            )
            jacobian_gram += (
                radial.square() * z_weighted_z
            )[:, None, None] * (z[:, :, None] * z[:, None, :])
            gram += jacobian_gram.sum(0)
        gram = (gram + gram.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        weakest = eigenvectors[:, 0]
        if weakest @ dc < 0:
            weakest = -weakest
        difference = weakest - dc
        if difference.norm() <= 1e-6:
            basis = identity
        else:
            direction = difference / difference.norm()
            basis = identity - 2 * torch.outer(direction, direction)
        bases.append(basis)
        tail += eigenvalues[0].clamp_min(0)
        total += eigenvalues.clamp_min(0).sum()
        head_tail_ratios.append(
            float(
                eigenvalues[0].clamp_min(0)
                / eigenvalues.clamp_min(0).sum().clamp_min(1e-24)
            )
        )
    return (
        torch.stack(bases),
        float(tail / total.clamp_min(1e-24)),
        tuple(head_tail_ratios),
        sample_count,
    )


def _decay_metrics(source, trace) -> dict[str, float]:
    retention = trace["retention"]
    shortfall = (NATIVE_DECAY_FLOOR - retention).clamp_min(0)
    operator = 127**0.5 * shortfall
    key_norm_sq = trace["k"].square().sum(-1).clamp_min(1e-12)
    realized = retention.clamp_min(NATIVE_DECAY_FLOOR)
    corrected_erase = trace["source_erase"] + (realized - retention) / key_norm_sq
    floor_raw = _rwkv_trace(
        trace["read"],
        trace["k"],
        trace["write"],
        corrected_erase,
        realized.log(),
    )
    floor_normed = floor_raw * torch.rsqrt(
        floor_raw.square().mean(-1, keepdim=True) + source.layer_norm_epsilon
    )
    floor_normed = floor_normed * source.norm.weight.float()
    floor_output = source.out_proj(
        (floor_normed * trace["gate"]).flatten(2).to(source.out_proj.weight.dtype)
    ).float()
    return {
        "decay_operator_floor_bound_mean": float(operator.mean()),
        "decay_operator_floor_bound_rms": float(operator.square().mean().sqrt()),
        "decay_operator_floor_bound_max": float(operator.max()),
        "decay_below_native_floor_fraction": float((shortfall > 0).float().mean()),
        "decay_observable_oracle_nmse": _nmse(floor_output, trace["source_output"]),
    }


def _fit_projection(
    target,
    name: str,
    module,
    current: torch.Tensor,
    previous: torch.Tensor,
    wanted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    ratio, fitted = fit_time_mix(current, previous, wanted)
    getattr(target, f"x_{name}").copy_(ratio.view(1, 1, -1))
    module.weight.copy_(fitted.T.to(module.weight))
    actual = _mixed(current, previous, ratio) @ module.weight.float().T
    return ratio, fitted, _nmse(actual, wanted)


def _selection_projection_nmse(
    current: torch.Tensor,
    wanted: torch.Tensor,
    ratio: torch.Tensor,
    projection: torch.Tensor,
) -> float:
    actual = _mixed(current, _previous(current), ratio) @ projection
    return _nmse(actual, wanted)


def _write_gauge(trace, mode: str) -> torch.Tensor:
    """Choose a positive per-token/head factorization gauge without changing ``v kᵀ``."""
    shape = (*trace["beta"].shape, 1)
    if mode == "unity":
        return torch.ones(shape, dtype=torch.float32, device=trace["beta"].device)
    if mode != "balanced":
        raise ValueError(f"unknown GDN write gauge: {mode}")
    key_norm = trace["k"].float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    write_norm = trace["write"].float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (write_norm / key_norm).sqrt().clamp(0.25, 4.0)


def _compile_gdn_candidate(
    source,
    target,
    x: torch.Tensor,
    trace,
    selection_hidden: torch.Tensor,
    selection_trace,
    basis: torch.Tensor,
    gauge_mode: str,
) -> dict[str, float]:
    previous = _previous(x)
    gauge = _write_gauge(trace, gauge_mode)
    selection_gauge = _write_gauge(selection_trace, gauge_mode)
    gauged_key = trace["k"] * gauge
    rotated_write = (
        torch.einsum("hij,bthj->bthi", basis, trace["write"]) / gauge
    )
    projection_metrics = {}
    fitted_projections = {}
    for name, wanted, module in (
        ("r", trace["read"].flatten(2), target.receptance),
        ("k", gauged_key.flatten(2), target.key),
        ("v", rotated_write.flatten(2), target.value_base),
    ):
        ratio, fitted, calibration_nmse = _fit_projection(
            target, name, module, x, previous, wanted
        )
        fitted_projections[name] = (ratio, fitted)
        projection_metrics[f"{name}_projection_calibration_nmse"] = calibration_nmse
    target.value_expand.weight.copy_(
        torch.eye(2048, device=x.device, dtype=target.value_expand.weight.dtype)
    )
    target.k_k.fill_(1)
    target.k_a.zero_()
    target.r_k.zero_()

    key_norm_sq = trace["k"].square().sum(-1)
    native_erase = trace["beta"] * trace["retention"] * key_norm_sq
    decay_target = _native_decay_inverse(
        trace["log_decay"][..., None].expand(-1, -1, -1, 128).flatten(2)
    )
    erase_target = torch.logit(
        native_erase.clamp(1e-6, 1 - 1e-6)[..., None]
        .expand(-1, -1, -1, 128)
        .flatten(2)
    )
    selection_key_norm_sq = selection_trace["k"].square().sum(-1)
    selection_native_erase = (
        selection_trace["beta"]
        * selection_trace["retention"]
        * selection_key_norm_sq
    )
    selection_decay_target = _native_decay_inverse(
        selection_trace["log_decay"][..., None]
        .expand(-1, -1, -1, 128)
        .flatten(2)
    )
    selection_erase_target = torch.logit(
        selection_native_erase.clamp(1e-6, 1 - 1e-6)[..., None]
        .expand(-1, -1, -1, 128)
        .flatten(2)
    )
    for stem, wanted in (("w", decay_target), ("a", erase_target)):
        ratio, _ = fit_time_mix(x, previous, wanted)
        ratio_parameter = getattr(target, f"x_{stem}")
        ratio_parameter.copy_(ratio.view(1, 1, -1).to(ratio_parameter))
        mixed = _mixed(x, previous, ratio)
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

    gate_calibration_nmse, gate_ratio, _ = fit_native_gate(
        target, x, previous, trace["gate"].flatten(2)
    )
    source_normed = trace["raw"] * torch.rsqrt(
        trace["raw"].square().mean(-1, keepdim=True)
        + source.layer_norm_epsilon
    )
    source_normed = source_normed * source.norm.weight.float()
    wanted_groupnorm = torch.einsum("hij,bthj->bthi", basis, source_normed)
    groupnorm_calibration_nmse = fit_groupnorm_affine(
        target, _native_raw_heads(target, x), wanted_groupnorm
    )
    target.value_residual_scale.zero_()
    transformed_prior = source.out_proj.weight.float().clone()
    for head in range(16):
        start = head * 128
        stop = start + 128
        transformed_prior[:, start:stop] = (
            source.out_proj.weight.float()[:, start:stop] @ basis[head].T
        )
    target.output.weight.copy_(transformed_prior.to(target.output.weight))
    boundary_metrics = refit_native_output(
        target,
        x,
        trace["source_output"],
        transformed_prior,
        selection_hidden,
        selection_trace["source_output"],
    )

    selection_targets = {
        "r": selection_trace["read"].flatten(2),
        "k": (selection_trace["k"] * selection_gauge).flatten(2),
        "v": torch.einsum(
            "hij,bthj->bthi", basis, selection_trace["write"]
        ).div(selection_gauge).flatten(2),
    }
    for name, wanted in selection_targets.items():
        ratio, projection = fitted_projections[name]
        projection_metrics[f"{name}_projection_selection_nmse"] = (
            _selection_projection_nmse(selection_hidden, wanted, ratio, projection)
        )
    projection_metrics["gate_projection_calibration_nmse"] = gate_calibration_nmse
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
        selection_gate, selection_trace["gate"].flatten(2)
    )
    selection_source_normed = selection_trace["raw"] * torch.rsqrt(
        selection_trace["raw"].square().mean(-1, keepdim=True)
        + source.layer_norm_epsilon
    )
    selection_source_normed = selection_source_normed * source.norm.weight.float()
    selection_wanted_groupnorm = torch.einsum(
        "hij,bthj->bthi", basis, selection_source_normed
    )
    projection_metrics["groupnorm_affine_calibration_nmse"] = (
        groupnorm_calibration_nmse
    )
    projection_metrics["groupnorm_affine_selection_nmse"] = _nmse(
        _apply_groupnorm_affine(
            target, _native_raw_heads(target, selection_hidden)
        ),
        selection_wanted_groupnorm,
    )
    projection_metrics["write_gauge_calibration_mean"] = float(gauge.mean())
    projection_metrics["write_gauge_calibration_min"] = float(gauge.min())
    projection_metrics["write_gauge_calibration_max"] = float(gauge.max())
    return {**projection_metrics, **boundary_metrics}


@torch.no_grad()
def initialize_gdn_layer(
    source,
    target,
    normalized_hidden: torch.Tensor,
    selection_hidden: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compile one complete Qwen3.5 GDN mixer into one canonical D128 TMix."""
    if selection_hidden is None:
        raise ValueError("GDN compilation requires an independent development split")
    trace = _gdn_trace(source, normalized_hidden)
    selection_trace = _gdn_trace(source, selection_hidden)
    source_trace_nmse = _nmse(trace["trace_output"], trace["source_output"])
    source_pre_output_nmse = _nmse(
        trace["trace_pre_output"], trace["source_pre_output"]
    )
    source_raw_nmse = _nmse(trace["oracle_raw"], trace["raw"])
    basis, groupnorm_bound, groupnorm_head_bounds, groupnorm_tokens = (
        _householder_basis(source, trace)
    )
    identity = torch.eye(128, device=normalized_hidden.device).expand(16, -1, -1)
    initial = _snapshot(target)
    candidates = []
    for label, candidate_basis in (
        ("identity", identity),
        ("observability", basis),
    ):
        for gauge_mode in ("unity", "balanced"):
            _restore(target, initial)
            try:
                candidate_metrics = _compile_gdn_candidate(
                    source,
                    target,
                    normalized_hidden,
                    trace,
                    selection_hidden,
                    selection_trace,
                    candidate_basis,
                    gauge_mode,
                )
                selection_nmse = _nmse(
                    _native_forward(target, selection_hidden),
                    selection_trace["source_output"],
                )
                candidate_state = _snapshot(target)
            except Exception:
                _restore(target, initial)
                raise
            candidates.append(
                (
                    selection_nmse,
                    label,
                    gauge_mode,
                    candidate_state,
                    candidate_metrics,
                )
            )
    finite = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate[0]) and _state_is_finite(candidate[3])
    ]
    if not finite:
        _restore(target, initial)
        raise FloatingPointError("all GDN native compilation candidates were non-finite")
    selection_nmse, selected_label, selected_gauge, selected_state, selected_metrics = min(
        finite, key=lambda candidate: candidate[0]
    )
    _restore(target, selected_state)
    temporal_residual = max(
        selected_metrics["r_projection_selection_nmse"],
        selected_metrics["k_projection_selection_nmse"],
        selected_metrics["v_projection_selection_nmse"],
    )
    decay_metrics = _decay_metrics(source, trace)
    structural_diagnostic_max = max(
        decay_metrics["decay_observable_oracle_nmse"],
        groupnorm_bound,
        temporal_residual,
    )
    return {
        "source_trace_output_nmse": source_trace_nmse,
        "source_trace_pre_output_nmse": source_pre_output_nmse,
        "source_recurrence_raw_nmse": source_raw_nmse,
        "groupnorm_rank_tail_bound": groupnorm_bound,
        "groupnorm_observability_tokens": float(groupnorm_tokens),
        **{
            f"groupnorm_rank_tail_bound_head_{head:02d}": value
            for head, value in enumerate(groupnorm_head_bounds)
        },
        "temporal_conditional_residual": temporal_residual,
        "observable_structural_diagnostic_max": structural_diagnostic_max,
        "identity_basis_selection_mixer_nmse": min(
            candidate[0] for candidate in candidates if candidate[1] == "identity"
        ),
        "observability_basis_selection_mixer_nmse": min(
            candidate[0]
            for candidate in candidates
            if candidate[1] == "observability"
        ),
        "observability_basis_selected": float(selected_label == "observability"),
        "balanced_write_gauge_selected": float(selected_gauge == "balanced"),
        "compiled_selection_mixer_nmse": selection_nmse,
        "trace_tokens": float(normalized_hidden.shape[0] * normalized_hidden.shape[1]),
        **decay_metrics,
        **selected_metrics,
    }


__all__ = [
    "W_SCALE",
    "_control_signal",
    "fit_native_gate",
    "fit_groupnorm_affine",
    "fit_time_mix",
    "initialize_gdn_layer",
    "refit_native_output",
    "ridge",
    "truncated_map",
]
