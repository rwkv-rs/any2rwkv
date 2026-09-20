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
        f"{prefix}_clamp_w_tmix_output_nmse": _nmse(
            trace["clamped_output"], trace["source_output"]
        ),
    }


@torch.no_grad()
def initialize_gdn_layer(
    source,
    target,
    init_hidden: torch.Tensor,
    validation_hidden: torch.Tensor | None = None,
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

    init_trace = _gdn_trace(source, init_hidden)
    metrics = {
        "source_shell_parameter_tensors": float(len(source_state)),
        "source_shell_strict_copy": 1.0,
        "init_trace_tokens": float(init_hidden.shape[0] * init_hidden.shape[1]),
        **_trace_metrics("init", init_trace),
    }
    if validation_hidden is not None:
        validation_trace = _gdn_trace(source, validation_hidden)
        metrics.update(
            {
                "validation_trace_tokens": float(
                    validation_hidden.shape[0] * validation_hidden.shape[1]
                ),
                **_trace_metrics("validation", validation_trace),
            }
        )
    return metrics


def _gdn_front(source, hidden: torch.Tensor, *, taps=None, bias=None, activate=True):
    """Return the source QKV frontend, with optional audit-only substitutions."""
    projected = F.linear(hidden, source.in_proj_qkv.weight, source.in_proj_qkv.bias)
    weight = source.conv1d.weight.squeeze(1)
    if taps is not None:
        weight = weight.clone()
        disabled = [index for index in range(weight.shape[-1]) if index not in taps]
        if disabled:
            weight[:, disabled] = 0
    if bias is None and source.conv1d.bias is not None:
        bias = source.conv1d.bias
    pre_activation = causal_conv1d_fn(
        projected.transpose(1, 2), weight, bias, activation=None
    ).transpose(1, 2)
    if activate:
        return F.silu(pre_activation), pre_activation
    return pre_activation, pre_activation


def _gdn_components(
    source, hidden: torch.Tensor, frontend: torch.Tensor
) -> dict[str, torch.Tensor]:
    batch, length, _ = hidden.shape
    query, key, value = torch.split(
        frontend, (source.key_dim, source.key_dim, source.value_dim), dim=-1
    )
    query = query.view(batch, length, source.num_k_heads, source.head_k_dim)
    key = key.view(batch, length, source.num_k_heads, source.head_k_dim)
    value = value.view(batch, length, source.num_v_heads, source.head_v_dim)
    q_norm = query * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6)
    k_norm = key * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(source.in_proj_b(hidden).float())
    log_decay = -source.A_log.float().exp() * F.softplus(
        source.in_proj_a(hidden).float() + source.dt_bias.float()
    )
    z = source.in_proj_z(hidden).view(batch, length, source.num_v_heads, source.head_v_dim)
    return {
        "query_raw": query.float(),
        "key_raw": key.float(),
        "value": value.float(),
        "query_norm": q_norm.float(),
        "key_norm": k_norm.float(),
        "beta": beta,
        "log_decay": log_decay,
        "z": z.float(),
    }


def _rwkv_trace_separate(
    read: torch.Tensor,
    erase_key: torch.Tensor,
    erase: torch.Tensor,
    write_key: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
) -> torch.Tensor:
    """Reference DPLR trace with separate erase and write key directions."""
    batch, length, heads, dim = read.shape
    state = read.new_zeros(batch, heads, dim, dim, dtype=torch.float32)
    outputs = []
    for token in range(length):
        direction = erase_key[:, token].float()
        memory = torch.einsum("bhk,bhkv->bhv", direction, state)
        state = (
            state * log_decay[:, token].float().exp()[..., None, None]
            - erase[:, token].float()[..., None, None]
            * torch.einsum("bhk,bhv->bhkv", direction, memory)
            + torch.einsum("bhk,bhv->bhkv", write_key[:, token].float(), write[:, token].float())
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", read[:, token].float(), state))
    return torch.stack(outputs, 1)


def _gdn_boundary(source, raw: torch.Tensor, gate: torch.Tensor, *, group=False, gate_value=None):
    batch, length = raw.shape[:2]
    heads = source.num_v_heads
    dim = source.head_v_dim
    flat = raw.reshape(-1, heads * dim).float()
    if group:
        weight = source.norm.weight.float().repeat(heads)
        normalized = F.group_norm(
            flat,
            heads,
            weight=weight,
            bias=torch.zeros_like(weight),
            eps=64e-5,
        )
        normalized = normalized.view(batch, length, heads, dim)
    else:
        normalized = source.norm(raw.reshape(-1, dim).float(), gate.reshape(-1, dim).float()).view(
            batch, length, heads, dim
        )
        if gate_value is not None:
            rms = raw.float() * torch.rsqrt(
                raw.float().square().mean(-1, keepdim=True) + source.norm.variance_epsilon
            )
            normalized = rms * source.norm.weight.float().view(1, 1, 1, dim) * gate_value.float()
    if group and gate_value is None:
        normalized = normalized * F.silu(gate.float())
    elif group and gate_value is not None:
        normalized = normalized * gate_value.float()
    return source.out_proj(normalized.reshape(batch, length, -1)).float()


def _gdn_apply_shift(hidden: torch.Tensor, weight: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    previous = torch.cat((torch.zeros_like(hidden[:, :1]), hidden[:, :-1]), dim=1)
    shifted = hidden + (previous - hidden) * mu.view(1, 1, -1)
    return F.linear(shifted, weight).float()


def _gdn_normal_fit(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Small-ridge normal-equation least squares, returning output-by-input weights."""
    design = design.float()
    target = target.float()
    gram = design.transpose(0, 1) @ design
    scale = gram.diagonal().mean().clamp_min(1.0)
    gram = gram + torch.eye(gram.shape[0], device=gram.device) * (scale * 1e-6)
    return torch.linalg.solve(gram, design.transpose(0, 1) @ target).transpose(0, 1)


def _gdn_closed_shift(source, segment: slice) -> tuple[torch.Tensor, torch.Tensor]:
    weight = source.in_proj_qkv.weight[segment].float()
    conv_weight = source.conv1d.weight.squeeze(1)[segment].float()
    c0, c1 = conv_weight[:, 0], conv_weight[:, 1]
    output_weight = (c0 + c1).unsqueeze(1) * weight
    numerator = ((c0 + c1) * c1).unsqueeze(1) * weight.square()
    denominator = (c0 + c1).square().unsqueeze(1) * weight.square()
    mu = numerator.sum(0) / denominator.sum(0).clamp_min(1e-12)
    mu = torch.nan_to_num(mu)
    return output_weight, mu


def _gdn_fit_shift(
    hidden: torch.Tensor,
    target: torch.Tensor,
    closed_weight: torch.Tensor,
    closed_mu: torch.Tensor,
    *,
    direction: bool,
    steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.cat((torch.zeros_like(hidden[:, :1]), hidden[:, :-1]), dim=1)
    delta = (previous - hidden).reshape(-1, hidden.shape[-1]).float()
    inputs = hidden.reshape(-1, hidden.shape[-1]).float()
    target = target.float()
    if direction:
        target = target * torch.rsqrt(target.square().sum(-1, keepdim=True) + 1e-6)
    target = target.reshape(-1, target.shape[-1])
    mu = closed_mu.float().clone()
    weight = closed_weight.float().clone()
    for _ in range(steps):
        features = inputs + delta * mu
        weight = _gdn_normal_fit(features, target)
        output = features @ weight.transpose(0, 1)
        residual = target - output
        output_gram = weight.transpose(0, 1) @ weight
        delta_gram = delta.transpose(0, 1) @ delta
        gram = output_gram * delta_gram
        scale = gram.diagonal().mean().clamp_min(1.0)
        gram = gram + torch.eye(gram.shape[0], device=gram.device) * (scale * 1e-6)
        rhs = (delta * (residual @ weight)).sum(0)
        mu = torch.linalg.solve(gram, rhs)
    return weight, mu


def _gdn_apply_front_params(hidden, params, segments):
    pieces = []
    for name, segment in segments:
        pieces.append(_gdn_apply_shift(hidden, params[name]["weight"], params[name]["mu"]))
    return torch.cat(pieces, dim=-1)


def _gdn_geometric(value: torch.Tensor, dim=(0, 1)) -> torch.Tensor:
    return torch.exp(value.float().clamp_min(1e-30).log().mean(dim=dim))


def _gdn_layer_output(source_layer, hidden: torch.Tensor, tmix: torch.Tensor) -> torch.Tensor:
    residual = hidden.float() + tmix.float()
    return (
        residual + source_layer.mlp(source_layer.post_attention_layernorm(residual)).float()
    ).float()


def _gdn_metric(source_layer, hidden, source_tmix, candidate_tmix) -> dict[str, float]:
    source_layer_output = _gdn_layer_output(source_layer, hidden, source_tmix)
    candidate_layer_output = _gdn_layer_output(source_layer, hidden, candidate_tmix)
    return {
        "tmix_output_nmse": _nmse(candidate_tmix, source_tmix),
        "layer_output_nmse": _nmse(candidate_layer_output, source_layer_output),
    }


def _gdn_fit_erase(source, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    beta_logits = source.in_proj_b(hidden).float()
    decay_logits = source.in_proj_a(hidden).float()
    beta = beta_logits.sigmoid()
    log_decay = -source.A_log.float().exp() * F.softplus(decay_logits + source.dt_bias.float())
    target = torch.logit((beta * log_decay.exp()).clamp(1e-6, 1 - 1e-6))
    design = torch.cat((torch.ones_like(beta_logits[..., :1]), beta_logits, decay_logits), dim=-1)
    coefficient = _gdn_normal_fit(
        design.reshape(-1, design.shape[-1]), target.reshape(-1, target.shape[-1])
    ).transpose(0, 1)
    return {
        "a0": coefficient[0],
        "a2": coefficient[1:],
        "target_logit": target,
    }


def _gdn_apply_erase(source, hidden: torch.Tensor, fitted: dict[str, torch.Tensor]):
    beta_logits = source.in_proj_b(hidden).float()
    decay_logits = source.in_proj_a(hidden).float()
    design = torch.cat((torch.ones_like(beta_logits[..., :1]), beta_logits, decay_logits), dim=-1)
    return (design @ torch.cat((fitted["a0"].unsqueeze(0), fitted["a2"]), dim=0)).sigmoid()


def _gdn_fit_decay(source, hidden: torch.Tensor, alpha_grid: tuple[float, ...]):
    ua = source.in_proj_a(hidden).float()
    source_log_decay = -source.A_log.float().exp() * F.softplus(ua + source.dt_bias.float())
    ratio = (source_log_decay / W_SCALE).clamp(CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON)
    target = torch.logit(ratio)
    alpha = torch.tensor(alpha_grid, device=hidden.device, dtype=torch.float32)
    design = torch.cat((torch.ones_like(ua[..., None]), torch.tanh(ua[..., None] * alpha)), dim=-1)
    coefficients = []
    for head in range(ua.shape[-1]):
        coefficients.append(
            _gdn_normal_fit(
                design[..., head, :].reshape(-1, design.shape[-1]), target[..., head].reshape(-1, 1)
            )[0]
        )
    coefficients = torch.stack(coefficients, dim=1)
    return {
        "w0": coefficients[0],
        "w2": coefficients[1:],
        "alpha": alpha,
        "target": target,
    }


def _gdn_apply_decay(source, hidden: torch.Tensor, fitted: dict[str, torch.Tensor]):
    ua = source.in_proj_a(hidden).float()
    basis = torch.tanh(ua[..., None] * fitted["alpha"])
    fit = fitted["w0"] + torch.einsum("bthu,uh->bth", basis, fitted["w2"])
    return W_SCALE * fit.sigmoid(), fit


def _gdn_gate_scale(source, hidden: torch.Tensor) -> torch.Tensor:
    z = source.in_proj_z(hidden).float()
    sigmoid = z.sigmoid()
    return (z * sigmoid.square()).sum((0, 1)) / sigmoid.square().sum((0, 1)).clamp_min(1e-12)


def _gdn_fit_front_params(source, hidden: torch.Tensor, frontend: torch.Tensor):
    segments = (
        ("q", slice(0, source.key_dim)),
        ("k", slice(source.key_dim, 2 * source.key_dim)),
        ("v", slice(2 * source.key_dim, 2 * source.key_dim + source.value_dim)),
    )
    params = {}
    for name, segment in segments:
        closed_weight, closed_mu = _gdn_closed_shift(source, segment)
        target = frontend[..., segment]
        direction = name in ("q", "k")
        refined_weight, refined_mu = _gdn_fit_shift(
            hidden, target, closed_weight, closed_mu, direction=direction
        )
        params[name] = {
            "closed": {"weight": closed_weight, "mu": closed_mu},
            "refined": {"weight": refined_weight, "mu": refined_mu},
        }
    return params, segments


def _gdn_front_with_params(hidden, params, segments, mode, names=None):
    if names is None:
        names = {name for name, _ in segments}
    pieces = []
    for name, segment in segments:
        if name in names:
            piece = _gdn_apply_shift(hidden, params[name][mode]["weight"], params[name][mode]["mu"])
        else:
            piece = None
        pieces.append(piece)
    return pieces


@torch.no_grad()
def audit_gdn_layer(
    source, source_layer, init_hidden: torch.Tensor, validation_hidden: torch.Tensor
):
    """Stage -1 component audit for one source-shell GDN layer.

    Every candidate below changes only the named source component.  The returned
    records intentionally keep the initialization-fit parameters separate from
    validation measurements so that no validation activation is used for fitting.
    """
    alpha_grid = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    split_inputs = {"initialization": init_hidden.float(), "validation": validation_hidden.float()}
    normalized = {
        split: source_layer.input_layernorm(value).float() for split, value in split_inputs.items()
    }
    source_traces = {split: _gdn_trace(source, value) for split, value in normalized.items()}
    source_front = {
        split: _gdn_front(source, value)[0].float() for split, value in normalized.items()
    }
    source_parts = {
        split: _gdn_components(source, normalized[split], source_front[split])
        for split in split_inputs
    }

    fit_params, segment_defs = _gdn_fit_front_params(
        source, normalized["initialization"], source_front["initialization"]
    )
    erase_fit = _gdn_fit_erase(source, normalized["initialization"])
    decay_fit = _gdn_fit_decay(source, normalized["initialization"], alpha_grid)
    gate_scale = _gdn_gate_scale(source, normalized["initialization"])

    rho_initial = _gdn_geometric(
        torch.rsqrt(source_parts["initialization"]["query_raw"].square().sum(-1) + 1e-6)
        / math.sqrt(source.head_k_dim)
    )
    source_key = source_parts["initialization"]["key_raw"]
    source_a = (
        source_parts["initialization"]["beta"] * source_parts["initialization"]["log_decay"].exp()
    )
    source_key_norm = torch.sqrt(source_key.square().sum(-1) + 1e-6)
    ka_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    gamma_bars = {}
    gamma_log_variances = {}
    for ka in ka_grid:
        factor = 1 + (source_a - 1) * ka
        gamma = source_parts["initialization"]["beta"] / (source_key_norm * factor)
        gamma_bars[ka] = _gdn_geometric(gamma)
        gamma_log_variances[ka] = gamma.log().var(dim=(0, 1), unbiased=False)
    variance_grid = torch.stack([gamma_log_variances[ka] for ka in ka_grid])
    best_ka_heads = torch.tensor(
        [ka_grid[index] for index in variance_grid.argmin(0).tolist()],
        device=init_hidden.device,
        dtype=torch.float32,
    )
    selected_factor = 1 + (source_a - 1) * best_ka_heads.view(1, 1, -1)
    selected_gamma_bar = _gdn_geometric(
        source_parts["initialization"]["beta"] / (source_key_norm * selected_factor)
    )

    # The canonical all-at-once candidate recomputes q/k gain statistics after
    # the closed-form frontend replacement, rather than mixing two coordinate gauges.
    closed_front_initial = torch.cat(
        [
            _gdn_apply_shift(
                normalized["initialization"],
                fit_params[name]["closed"]["weight"],
                fit_params[name]["closed"]["mu"],
            )
            for name, _ in segment_defs
        ],
        dim=-1,
    )
    closed_parts_initial = _gdn_components(
        source, normalized["initialization"], closed_front_initial
    )
    canonical_rho = _gdn_geometric(
        torch.rsqrt(closed_parts_initial["query_raw"].square().sum(-1) + 1e-6)
        / math.sqrt(source.head_k_dim)
    )
    canonical_key_norm = torch.sqrt(closed_parts_initial["key_raw"].square().sum(-1) + 1e-6)

    def standard_raw(parts, *, read_gain=None, erase=None, log_decay=None):
        if read_gain is None:
            read = parts["query_norm"] / math.sqrt(source.head_k_dim)
        else:
            read = parts["query_raw"] * read_gain.view(1, 1, -1, 1)
        log_decay = parts["log_decay"] if log_decay is None else log_decay
        retention = log_decay.exp()
        erase = parts["beta"] * retention if erase is None else erase
        return _rwkv_trace(
            read,
            parts["key_norm"],
            parts["beta"][..., None] * parts["value"],
            erase,
            log_decay,
        )

    def write_gain_raw(parts, ka, gamma_bar):
        a = parts["beta"] * parts["log_decay"].exp()
        factor = 1 + (a - 1) * ka
        read = parts["query_norm"] / math.sqrt(source.head_k_dim)
        return _rwkv_trace_separate(
            read,
            parts["key_norm"],
            parts["beta"] * parts["log_decay"].exp(),
            parts["key_raw"] * factor[..., None],
            parts["value"] * gamma_bar.view(1, 1, -1, 1),
            parts["log_decay"],
        )

    def boundary(source_raw, parts, raw, *, group=False, fitted_gate=None):
        if fitted_gate is None:
            return _gdn_boundary(source, raw, parts["z"], group=group)
        gate = fitted_gate.view(1, 1, source.num_v_heads, source.head_v_dim) * parts["z"].sigmoid()
        return _gdn_boundary(source, raw, parts["z"], group=group, gate_value=gate)

    def all_metrics(split, candidate, *, reference=None):
        if reference is None:
            reference = source_traces[split]["source_output"]
        return _gdn_metric(source_layer, split_inputs[split], reference, candidate)

    def evaluate_front(
        split,
        frontend,
        *,
        read_gain=None,
        erase=None,
        log_decay=None,
        group=False,
        gate_fit=None,
        separate=None,
    ):
        parts = _gdn_components(source, normalized[split], frontend)
        if separate is None:
            raw = standard_raw(parts, read_gain=read_gain, erase=erase, log_decay=log_decay)
        else:
            raw = _rwkv_trace_separate(*separate(parts))
        candidate = boundary(
            source_traces[split]["raw"], parts, raw, group=group, fitted_gate=gate_fit
        )
        return all_metrics(split, candidate), parts, raw, candidate

    result = {
        "layer": int(source.layer_idx),
        "head_size": int(source.head_k_dim),
        "legacy_trace_metrics": {
            split: _trace_metrics(f"{split}", source_traces[split]) for split in split_inputs
        },
        "components": {},
        "fit": {
            "decay_alpha_grid": list(alpha_grid),
            "conv_bias_present": source.conv1d.bias is not None,
            "frontend": {},
        },
    }

    def put(name, metrics_by_split, **extra):
        result["components"][name] = {"metrics": metrics_by_split, **extra}

    for name, kwargs in (
        ("D1_conv_taps_2_3_zero", {"taps": (0, 1), "activate": True}),
        ("D2_silu_identity", {"taps": None, "activate": False}),
        (
            "D3_conv_bias_zero",
            {
                "taps": None,
                "activate": True,
                "bias": None
                if source.conv1d.bias is None
                else torch.zeros_like(source.conv1d.bias),
            },
        ),
    ):
        metrics = {}
        for split in split_inputs:
            frontend = _gdn_front(source, normalized[split], **kwargs)[0]
            metrics[split] = evaluate_front(split, frontend)[0]
        put(name, metrics)

    # D4: report each segment independently and all segments together, first with
    # the formula in §3.2 and then with two alternating normal-equation updates.
    d4 = {"closed_form": {}, "alternating_least_squares": {}}
    for mode in ("closed", "refined"):
        target_group = d4["closed_form"] if mode == "closed" else d4["alternating_least_squares"]
        for selected in ("q", "k", "v", "all"):
            metrics = {}
            fit_residual = {}
            for split in split_inputs:
                pieces = _gdn_front_with_params(
                    normalized[split],
                    fit_params,
                    segment_defs,
                    mode,
                    names=set(name for name, _ in segment_defs)
                    if selected == "all"
                    else {selected},
                )
                frontend = source_front[split].clone()
                for (name, segment), piece in zip(segment_defs, pieces, strict=True):
                    if piece is not None:
                        frontend[..., segment] = piece
                        target = source_front[split][..., segment]
                        fit_residual.setdefault(name, {})
                        fit_residual[name][split] = {
                            "frontend_output_nmse": _nmse(piece, target),
                            "frontend_direction_nmse": _nmse(
                                piece * torch.rsqrt(piece.square().sum(-1, keepdim=True) + 1e-6),
                                target * torch.rsqrt(target.square().sum(-1, keepdim=True) + 1e-6),
                            ),
                        }
                metrics[split] = evaluate_front(split, frontend)[0]
            target_group[selected] = {"metrics": metrics, "fit_residual": fit_residual}
    d4["metrics"] = d4["closed_form"]["all"]["metrics"]
    d4["alternating_metrics"] = d4["alternating_least_squares"]["all"]["metrics"]
    result["components"]["D4_input_side_2tap"] = d4
    for name in ("q", "k", "v"):
        result["fit"]["frontend"][name] = {
            "closed_mu_mean": float(fit_params[name]["closed"]["mu"].mean()),
            "refined_mu_mean": float(fit_params[name]["refined"]["mu"].mean()),
            "closed_refined_mu_nmse": _nmse(
                fit_params[name]["refined"]["mu"], fit_params[name]["closed"]["mu"]
            ),
        }

    rho_metrics = {}
    for split in split_inputs:
        rho_metrics[split] = evaluate_front(split, source_front[split], read_gain=rho_initial)[0]
    put("D5_q_l2norm_read_gain_geomean", rho_metrics, read_gain=list(rho_initial.tolist()))

    d6_metrics = {}
    for split in split_inputs:
        d6_metrics[split] = evaluate_front(
            split,
            source_front[split],
            separate=lambda parts: (
                parts["query_norm"] / math.sqrt(source.head_k_dim),
                parts["key_norm"],
                parts["beta"] * parts["log_decay"].exp(),
                parts["key_raw"],
                parts["value"] * gamma_bars[0.0].view(1, 1, -1, 1),
                parts["log_decay"],
            ),
        )[0]
    put(
        "D6_write_gain_ka_0",
        d6_metrics,
        k_a=0.0,
        gamma_geometric_mean=list(gamma_bars[0.0].tolist()),
    )

    grid_metrics = {}
    for ka in ka_grid:
        grid_metrics[str(ka)] = {}
        for split in split_inputs:
            gamma_bar = gamma_bars[ka]
            grid_metrics[str(ka)][split] = evaluate_front(
                split,
                source_front[split],
                separate=lambda parts, ka=ka, gamma_bar=gamma_bar: (
                    parts["query_norm"] / math.sqrt(source.head_k_dim),
                    parts["key_norm"],
                    parts["beta"] * parts["log_decay"].exp(),
                    parts["key_raw"]
                    * (1 + (parts["beta"] * parts["log_decay"].exp() - 1) * ka)[..., None],
                    parts["value"] * gamma_bar.view(1, 1, -1, 1),
                    parts["log_decay"],
                ),
            )[0]
    selected_metrics = {}
    for split in split_inputs:
        selected_metrics[split] = evaluate_front(
            split,
            source_front[split],
            separate=lambda parts: (
                parts["query_norm"] / math.sqrt(source.head_k_dim),
                parts["key_norm"],
                parts["beta"] * parts["log_decay"].exp(),
                parts["key_raw"]
                * (
                    1
                    + (parts["beta"] * parts["log_decay"].exp() - 1) * best_ka_heads.view(1, 1, -1)
                )[..., None],
                parts["value"] * selected_gamma_bar.view(1, 1, -1, 1),
                parts["log_decay"],
            ),
        )[0]
    best_ka = min(
        ka_grid, key=lambda ka: grid_metrics[str(ka)]["initialization"]["tmix_output_nmse"]
    )
    put(
        "D7_write_gain_best_ka_grid",
        selected_metrics,
        k_a=list(best_ka_heads.tolist()),
        gamma_geometric_mean=list(selected_gamma_bar.tolist()),
        selection="per_head_min_variance_log_gamma",
        best_scalar_output_k_a=best_ka,
        grid=grid_metrics,
    )

    erase_metrics = {}
    erase_residual = {}
    for split in split_inputs:
        fitted_erase = _gdn_apply_erase(source, normalized[split], erase_fit)
        target_erase = source_parts[split]["beta"] * source_parts[split]["log_decay"].exp()
        erase_residual[split] = {
            "logit_nmse": _nmse(
                torch.logit(fitted_erase.clamp(1e-6, 1 - 1e-6)),
                torch.logit(target_erase.clamp(1e-6, 1 - 1e-6)),
            ),
            "coefficient_nmse": _nmse(fitted_erase, target_erase),
        }
        erase_metrics[split] = evaluate_front(split, source_front[split], erase=fitted_erase)[0]
    put("D8_erase_logistic_linear_fit", erase_metrics, regression_residual=erase_residual)

    clamp_metrics = {}
    for split in split_inputs:
        parts = source_parts[split]
        ratio = parts["log_decay"] / W_SCALE
        projected = ratio.clamp(CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON)
        projected_log_decay = W_SCALE * projected
        clamp_metrics[split] = evaluate_front(
            split,
            source_front[split],
            erase=parts["beta"] * projected_log_decay.exp(),
            log_decay=projected_log_decay,
        )[0]
        clamp_metrics[split]["clamp_w_outside_fraction"] = float(
            ((ratio < CLAMP_W_EPSILON) | (ratio > 1 - CLAMP_W_EPSILON)).float().mean()
        )
        clamp_metrics[split]["clamp_w_tmix_output_nmse"] = clamp_metrics[split]["tmix_output_nmse"]
    put("D9a_clamp_w", clamp_metrics)

    decay_metrics = {}
    decay_residual = {}
    for split in split_inputs:
        fitted_log_decay, fitted_logit = _gdn_apply_decay(source, normalized[split], decay_fit)
        target_logit = (
            decay_fit["target"]
            if split == "initialization"
            else torch.logit(
                (source_parts[split]["log_decay"] / W_SCALE).clamp(
                    CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON
                )
            )
        )
        decay_residual[split] = {
            "logit_nmse": _nmse(fitted_logit, target_logit),
            "log_decay_nmse": _nmse(fitted_log_decay, source_parts[split]["log_decay"]),
        }
        decay_metrics[split] = evaluate_front(
            split,
            source_front[split],
            erase=source_parts[split]["beta"] * fitted_log_decay.exp(),
            log_decay=fitted_log_decay,
        )[0]
    put(
        "D9b_decay_tanh_fit",
        decay_metrics,
        fit_residual=decay_residual,
        alpha_grid=list(alpha_grid),
    )

    norm_metrics = {}
    gate_metrics = {}
    for split in split_inputs:
        parts = source_parts[split]
        source_raw = source_traces[split]["oracle_raw"]
        norm_metrics[split] = _gdn_metric(
            source_layer,
            split_inputs[split],
            source_traces[split]["source_output"],
            _gdn_boundary(source, source_raw, parts["z"], group=True),
        )
        fitted_gate = (
            gate_scale.view(1, 1, source.num_v_heads, source.head_v_dim) * parts["z"].sigmoid()
        )
        gate_metrics[split] = _gdn_metric(
            source_layer,
            split_inputs[split],
            source_traces[split]["source_output"],
            _gdn_boundary(source, source_raw, parts["z"], gate_value=fitted_gate),
        )
    put("D10_rmsnorm_to_groupnorm", norm_metrics)
    put(
        "D11_silu_gate_to_scale_sigmoid",
        gate_metrics,
        scale_fit_nmse=float(
            _nmse(
                gate_scale.view(1, 1, source.num_v_heads, source.head_v_dim)
                * source_parts["initialization"]["z"].sigmoid(),
                source_parts["initialization"]["z"] * source_parts["initialization"]["z"].sigmoid(),
            )
        ),
    )

    # D12: all closed-form canonical substitutions at once.  This is deliberately
    # a zero-training student: the only fitted quantities are the §3.2 closed
    # regressions used as initialization, and no validation tensor enters them.
    canonical_ka = best_ka_heads
    canonical_a_init = _gdn_apply_erase(source, normalized["initialization"], erase_fit)
    canonical_gamma = _gdn_geometric(
        source_parts["initialization"]["beta"]
        / (canonical_key_norm * (1 + (canonical_a_init - 1) * canonical_ka.view(1, 1, -1)))
    )
    d12_metrics = {}
    for split in split_inputs:
        hidden = normalized[split]
        frontend = torch.cat(
            [
                _gdn_apply_shift(
                    hidden, fit_params[name]["closed"]["weight"], fit_params[name]["closed"]["mu"]
                )
                for name, _ in segment_defs
            ],
            dim=-1,
        )
        parts = _gdn_components(source, hidden, frontend)
        fitted_erase = _gdn_apply_erase(source, hidden, erase_fit)
        fitted_log_decay, _ = _gdn_apply_decay(source, hidden, decay_fit)
        factor = 1 + (fitted_erase - 1) * canonical_ka.view(1, 1, -1)
        raw = _rwkv_trace_separate(
            parts["query_raw"] * canonical_rho.view(1, 1, -1, 1),
            parts["key_norm"],
            fitted_erase,
            parts["key_raw"] * factor[..., None],
            parts["value"] * canonical_gamma.view(1, 1, -1, 1),
            fitted_log_decay,
        )
        fitted_gate = (
            gate_scale.view(1, 1, source.num_v_heads, source.head_v_dim) * parts["z"].sigmoid()
        )
        candidate = _gdn_boundary(source, raw, parts["z"], group=True, gate_value=fitted_gate)
        d12_metrics[split] = all_metrics(split, candidate)
    put(
        "D12_all_D1_D11_canonical_zero_training",
        d12_metrics,
        selected_k_a=list(canonical_ka.tolist()),
        canonical_decay_floor=W_SCALE,
    )
    return result


__all__ = ["audit_gdn_layer", "initialize_gdn_layer"]
