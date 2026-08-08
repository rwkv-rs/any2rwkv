"""The single Gated-Delta-Net to RWKV-7 initializer used by Any2RWKV."""

from __future__ import annotations

import torch
import torch.nn.functional as F

RIDGE_SCALE = 1e-4
W_SCALE = -torch.exp(torch.tensor(-0.5)).item()


def ridge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the bias-free ridge map ``x @ W ~= y`` with the fixed recipe lambda."""
    x = x.reshape(-1, x.shape[-1]).float()
    y = y.reshape(-1, y.shape[-1]).float()
    gram = x.T @ x
    lam = RIDGE_SCALE * gram.trace() / x.shape[-1]
    return torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], device=x.device), x.T @ y)


def truncated_map(full: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(full.float(), full_matrices=False)
    root = s[:rank].sqrt()
    return u[:, :rank] * root, root[:, None] * vh[:rank]


def fit_time_mix(
    current: torch.Tensor,
    previous: torch.Tensor,
    target: torch.Tensor,
    rounds: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three fixed diagonal-ALS rounds for RWKV's channelwise token shift."""
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


def fit_native_gate(
    target,
    current: torch.Tensor,
    previous: torch.Tensor,
    wanted: torch.Tensor,
) -> None:
    """Fit the actual ``sigmoid(x_g @ g1) @ g2`` gate parameterization."""
    ratio, full = fit_time_mix(current, previous, wanted)
    first, second = truncated_map(full, target.config.gate_low_rank_dim)
    target.x_g.copy_(ratio.view(1, 1, -1).to(target.x_g))
    target.g1.copy_(first.to(target.g1))
    mixed = current.float() + (previous.float() - current.float()) * ratio
    features = torch.sigmoid(mixed @ target.g1.float())
    target.g2.copy_(ridge(features, wanted).to(target.g2))


def refit_native_output(
    target,
    normalized_hidden: torch.Tensor,
    source_output: torch.Tensor,
) -> float:
    """Fit output projection from the real FlashRWKV2 pre-output activation."""
    captured: list[torch.Tensor] = []

    def capture(_module, arguments):
        captured.append(arguments[0].detach())

    hook = target.output.register_forward_pre_hook(capture)
    try:
        v_first = None if target.layer_idx == 0 else torch.zeros_like(normalized_hidden)
        target(normalized_hidden, v_first, None, torch.ones_like(normalized_hidden[..., 0]))
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one native output activation, got {len(captured)}")
    target.output.weight.copy_(
        ridge(captured[0], source_output.float()).T.to(target.output.weight)
    )
    fitted = target.output(captured[0]).float()
    return float(
        (fitted - source_output.float()).square().mean()
        / (source_output.float().square().mean() + 1e-12)
    )


@torch.no_grad()
def initialize_gdn_layer(source, target, normalized_hidden: torch.Tensor) -> dict[str, float]:
    """Compile one Qwen3.5 GDN activation trace into one native D128 TMix."""
    x = normalized_hidden
    batch, length, _ = x.shape
    qkv = source.in_proj_qkv(x).transpose(1, 2)
    qkv = F.conv1d(
        qkv.float(),
        source.conv1d.weight.float(),
        padding=source.conv_kernel_size - 1,
        groups=source.conv_dim,
    )[:, :, :length]
    qkv = F.silu(qkv).transpose(1, 2).to(x.dtype)
    q, k, v = torch.split(qkv, (source.key_dim, source.key_dim, source.value_dim), -1)
    q = q.view(batch, length, 16, 128).float()
    k = k.view(batch, length, 16, 128).float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    v = v.view(batch, length, 16, 128).float()
    beta = source.in_proj_b(x).sigmoid().float()
    log_decay = -source.A_log.float().exp() * F.softplus(
        source.in_proj_a(x).float() + source.dt_bias.float()
    )
    erase = (beta * log_decay.exp()).clamp(1e-6, 1 - 1e-6)
    write = beta[..., None] * v
    read = q / 128**0.5
    previous = _previous(x)

    for name, wanted in (("r", q.flatten(2)), ("k", k.flatten(2)), ("v", write.flatten(2))):
        ratio, fitted = fit_time_mix(x, previous, wanted)
        getattr(target, f"x_{name}").copy_(ratio.view(1, 1, -1))
        module = {"r": target.receptance, "k": target.key, "v": target.value_base}[name]
        module.weight.copy_(fitted.T.to(module.weight))
    target.value_expand.weight.copy_(
        torch.eye(2048, device=x.device, dtype=target.value_expand.weight.dtype)
    )
    target.k_k.fill_(1)
    target.k_a.zero_()
    target.r_k.zero_()

    decay_target = _native_decay_inverse(log_decay[..., None].expand(-1, -1, -1, 128).flatten(2))
    a_target = torch.logit(erase)[..., None].expand(-1, -1, -1, 128).flatten(2)
    for stem, wanted, rank in (("w", decay_target, 128), ("a", a_target, 128)):
        ratio, _ = fit_time_mix(x, previous, wanted)
        ratio_parameter = target.x_w if stem == "w" else target.x_a
        ratio_parameter.copy_(ratio.view(1, 1, -1).to(ratio_parameter))
        mixed = x + (previous - x) * ratio
        bias, first, second = _low_rank_target(
            mixed, wanted, rank, nonlinear=stem == "w"
        )
        getattr(target, f"{stem}0").copy_(bias.view(1, 1, -1).to(ratio_parameter))
        getattr(target, f"{stem}1").copy_(first.to(ratio_parameter))
        getattr(target, f"{stem}2").copy_(second.to(ratio_parameter))

    source_gate = F.silu(source.in_proj_z(x).float())
    fit_native_gate(target, x, previous, source_gate)

    raw = _rwkv_trace(read, k, write, erase, log_decay)
    normed = F.group_norm(
        raw.reshape(batch * length, -1),
        16,
        source.norm.weight.float().repeat(16),
        None,
        source.layer_norm_epsilon,
    ).view(batch, length, -1)
    pre_output = normed * source_gate
    source_output = source(x).float()
    target.ln_x.weight.copy_(source.norm.weight.repeat(16).to(target.ln_x.weight))
    target.ln_x.bias.zero_()
    target.value_residual_scale.zero_()
    analytic_output = source.out_proj(pre_output.to(source.out_proj.weight.dtype)).float()
    analytic_output_nmse = float(
        (analytic_output - source_output).square().mean()
        / (source_output.square().mean() + 1e-12)
    )
    native_output_fit_nmse = refit_native_output(target, x, source_output)
    return {
        "decay_clipped_fraction": float((log_decay < W_SCALE).float().mean()),
        "analytic_pre_output_nmse": analytic_output_nmse,
        "native_output_fit_nmse": native_output_fit_nmse,
        "trace_tokens": float(batch * length),
    }


__all__ = [
    "fit_native_gate",
    "initialize_gdn_layer",
    "refit_native_output",
    "ridge",
    "truncated_map",
]
