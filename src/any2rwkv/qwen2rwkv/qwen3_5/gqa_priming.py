"""Signed, single-state RWKV7 priming of Qwen3.5 GQA projections.

This opt-in path is independent of the positive-feature/scaffold runtime.
It does not preserve the source softmax attention function at initialization.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

PRIMED_GQA_SCHEMA = "qwen35_primed_gqa_d256_v1"
W_SCALE = -math.exp(-0.5)


@dataclass
class PrimedState:
    """Per-mixer cache; matrix axes are [batch, head, key, value]."""

    matrix: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    handle: object | None = None
    length: int = 0


def reference_wkv(read, decay, key, value, left, right, initial=None):
    """Explicit FP32 DPLR oracle, including gradients, for small tests only."""
    batch, length, heads, width = read.shape
    state = (
        torch.zeros(batch, heads, width, width, device=read.device, dtype=torch.float32)
        if initial is None
        else initial.float()
    )
    outputs = []
    with torch.autocast(read.device.type, enabled=False):
        for index in range(length):
            previous_projection = torch.einsum("bhkv,bhk->bhv", state, left[:, index].float())
            retention = (W_SCALE * decay[:, index].float().sigmoid()).exp()
            state = (
                retention.unsqueeze(-1) * state
                + right[:, index].float().unsqueeze(-1) * previous_projection.unsqueeze(-2)
                + key[:, index].float().unsqueeze(-1) * value[:, index].float().unsqueeze(-2)
            )
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, read[:, index].float()))
    return torch.stack(outputs, 1).to(value.dtype), state


def run_wkv(inputs, backend, cache=None):
    read, value = inputs[0], inputs[3]
    batch, length, heads, width = read.shape
    if cache is not None and torch.is_grad_enabled():
        raise ValueError("cached inference requires torch.no_grad()")
    if cache is not None and cache.matrix is not None:
        if cache.matrix.shape != (batch, heads, width, width) or cache.matrix.device != read.device:
            raise ValueError("cache batch, head geometry or device changed; create a new cache")
    if backend == "reference":
        output, state = reference_wkv(*inputs, initial=None if cache is None else cache.matrix)
        if cache is not None:
            cache.matrix = state.detach()
        return output
    if backend != "flash":
        raise ValueError(f"unsupported backend {backend!r}")
    if not read.is_cuda or read.dtype != torch.bfloat16 or width not in (128, 256):
        raise ValueError("Flash priming requires CUDA BF16 and D128 or D256")
    import flashrwkv2 as flash

    if flash.__version__ != "0.1.0a17":
        raise RuntimeError("this priming provider was validated with FlashRWKV2 0.1.0a17")
    if cache is None:
        if length % 16:
            raise ValueError("Flash training/full-sequence scoring requires T divisible by 16")
        return flash.pretrain_tmix_wkv7_recurrent_bf16(
            *(tensor.flatten(2).contiguous() for tensor in inputs), head_size=width
        ).view_as(value)
    if cache.matrix is None:
        cache.matrix = torch.zeros(
            batch, heads, width, width, device=read.device, dtype=torch.float32
        )
        cache.handle = flash.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(cache.matrix)
    offsets = torch.arange(0, (batch + 1) * length, length, device=read.device, dtype=torch.int32)
    indices = torch.arange(batch, device=read.device, dtype=torch.int32)
    return flash.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *(tensor.reshape(batch * length, heads, width).contiguous() for tensor in inputs),
        state=cache.handle,
        cu_seqlens=offsets,
        state_indices=indices,
        max_seqlen=length,
    ).view_as(value)


class AdaptiveHeads(nn.Module):
    def __init__(self, kv_heads, heads, width, rank, enabled=True):
        super().__init__()
        self.heads, self.width, self.enabled = heads, width, enabled
        repeat = torch.zeros(heads, kv_heads)
        repeat[torch.arange(heads), torch.arange(heads) // (heads // kv_heads)] = 1
        self.repeat = nn.Parameter(repeat, requires_grad=enabled)
        self.down = nn.Linear(kv_heads * width, rank, bias=False)
        self.up = nn.Linear(rank, heads * width, bias=False)
        nn.init.zeros_(self.up.weight)
        if not enabled:
            self.requires_grad_(False)

    def forward(self, hidden):
        repeated = torch.einsum("...kd,hk->...hd", hidden, self.repeat)
        if not self.enabled:
            return repeated
        residual = self.up(functional.silu(self.down(hidden.flatten(-2))))
        return repeated + residual.view(*hidden.shape[:-2], self.heads, self.width)


class LowRankControl(nn.Module):
    def __init__(self, channels, width, rank, initial=0.0):
        super().__init__()
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.bias = nn.Parameter(torch.full((width,), initial))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.bias + self.up(torch.tanh(self.down(hidden)))


def rotary(query, key, positions, rope):
    width = int(query.shape[-1] * rope.get("partial_rotary_factor", 1.0))
    inverse = rope["rope_theta"] ** (
        -torch.arange(0, width, 2, dtype=torch.float32, device=query.device) / width
    )
    angles = positions.float()[..., None] * inverse
    angles = torch.cat((angles, angles), -1).unsqueeze(-2)

    def rotate(tensor):
        rotating, suffix = tensor[..., :width], tensor[..., width:]
        first, second = rotating.chunk(2, -1)
        rotated = rotating * angles.cos() + torch.cat((-second, first), -1) * angles.sin()
        return torch.cat((rotated, suffix), -1)

    return rotate(query), rotate(key)


class PrimedTimeMix(nn.Module):
    """Transfer Q/K/V/O, QK RMSNorm, partial RoPE and the per-head output gate.

    Parameters stay FP32. CUDA autocast applies to large projections; controls,
    AGQA, norms and readout arithmetic are FP32, WKV IO is BF16 on Flash.
    State-dict names match the measured AGQA v3 mixer, without experiment paths.
    """

    def __init__(self, source, config, rank=32, agqa=True, backend="flash"):
        super().__init__()
        self.heads, self.kv_heads, self.width = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        channels = config.hidden_size
        self.rope = deepcopy(config.rope_parameters)
        rotary_width = self.width * self.rope.get("partial_rotary_factor", 1.0)
        if (
            rank <= 0
            or self.kv_heads <= 0
            or self.heads % self.kv_heads
            or channels != self.heads * self.width
            or self.width < 2
            or self.rope.get("rope_type", "default") != "default"
            or not math.isfinite(self.rope.get("rope_theta", 0))
            or self.rope.get("rope_theta", 0) <= 0
            or not float(rotary_width).is_integer()
            or not 0 < rotary_width <= self.width
            or int(rotary_width) % 2
            or config.attention_dropout != 0
            or backend not in {"flash", "reference"}
        ):
            raise ValueError("unsupported Qwen GQA geometry, RoPE, dropout, rank or backend")
        self.backend, self.rank, self.agqa = backend, rank, agqa
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            setattr(self, name, deepcopy(getattr(source, name)).float())
        expected = {
            "q_proj": (2 * channels, channels),
            "k_proj": (self.kv_heads * self.width, channels),
            "v_proj": (self.kv_heads * self.width, channels),
            "o_proj": (channels, channels),
        }
        if any(
            tuple(getattr(self, name).weight.shape) != shape for name, shape in expected.items()
        ):
            raise ValueError("source projections lack Qwen's per-head interleaved Q/gate layout")
        self.key_heads = AdaptiveHeads(self.kv_heads, self.heads, self.width, rank, agqa)
        self.value_heads = AdaptiveHeads(self.kv_heads, self.heads, self.width, rank, agqa)
        self.shifts = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(channels)) for name in "rwkvag"}
        )
        self.decay = LowRankControl(channels, channels, rank, initial=-6.0)
        self.erase = LowRankControl(channels, channels, rank, initial=-2.0)
        self.value_mix = LowRankControl(channels, channels, rank)
        self.k_k = nn.Parameter(torch.ones(self.heads, self.width))
        self.k_a = nn.Parameter(torch.ones(self.heads, self.width))
        self.r_k = nn.Parameter(torch.zeros(self.heads, self.width))
        self.ln_x = nn.GroupNorm(self.heads, channels, eps=64e-5)
        self.to(device=source.q_proj.weight.device, dtype=torch.float32)

    def set_trainable(self, enabled=True):
        self.requires_grad_(enabled)
        if not self.agqa:
            self.key_heads.requires_grad_(False)
            self.value_heads.requires_grad_(False)

    def forward(self, hidden, positions, first=None, cache=None):
        if hidden.ndim != 3 or not hidden.shape[1] or positions.shape != hidden.shape[:2]:
            raise ValueError("expected nonempty [B,T,C] hidden and [B,T] text positions")
        if cache is not None:
            if torch.is_grad_enabled():
                raise ValueError("cached inference requires torch.no_grad()")
            expected = torch.arange(
                cache.length, cache.length + hidden.shape[1], device=hidden.device
            )
            if not torch.equal(positions, expected.expand(hidden.shape[0], -1)):
                raise ValueError("cache positions must continue the same unpadded text sequence")
            if cache.previous is not None and (
                cache.previous.shape != hidden[:, :1].shape
                or cache.previous.device != hidden.device
            ):
                raise ValueError("token-shift cache geometry or device changed")
        with torch.autocast(hidden.device.type, enabled=False):
            previous = torch.zeros_like(hidden[:, :1]).float()
            if cache is not None and cache.previous is not None:
                previous = cache.previous
            delta = torch.cat((previous, hidden[:, :-1].float()), 1) - hidden.float()
            shifted = {name: hidden.float() + shift * delta for name, shift in self.shifts.items()}
        shape = (*hidden.shape[:2], self.heads, self.width)
        packed = self.q_proj(shifted["r"]).view(*shape[:-1], 2 * self.width)
        key = self.k_proj(shifted["k"]).view(*shape[:2], self.kv_heads, self.width)
        value = self.v_proj(shifted["v"]).view_as(key)
        gate_weight = self.q_proj.weight.view(self.heads, 2, self.width, -1)[:, 1].flatten(0, 1)
        gate_bias = (
            None
            if self.q_proj.bias is None
            else self.q_proj.bias.view(self.heads, 2, self.width)[:, 1].flatten()
        )
        gate = functional.linear(shifted["g"], gate_weight, gate_bias).view(shape)
        with torch.autocast(hidden.device.type, enabled=False):
            query = self.q_norm(packed[..., : self.width].float())
            key = self.key_heads(self.k_norm(key.float()))
            value = self.value_heads(value.float())
            query, key = rotary(query, key, positions, self.rope)
            read, key = query / math.sqrt(self.width), key / math.sqrt(self.width)
            if first is None:
                first = value.flatten(2)
            if first.shape != hidden.shape:
                raise ValueError("first GQA values must match the current [B,T,C] chunk")
            value = value + self.value_mix(shifted["v"]).tanh().view(shape) * (
                first.view(shape) - value
            )
            decay = self.decay(shifted["w"]).view(shape)
            erase = self.erase(shifted["a"]).sigmoid().view(shape)
            direction = functional.normalize(key * self.k_k, dim=-1)
            write_key = key * (1 + (erase - 1) * self.k_a)
            inputs = (read, decay, write_key, value, -direction, direction * erase)
            if self.backend == "flash":
                inputs = tuple(tensor.to(torch.bfloat16) for tensor in inputs)
            output = run_wkv(inputs, self.backend, cache)
            normalized = functional.group_norm(
                output.float().flatten(0, 1).flatten(1),
                self.heads,
                self.ln_x.weight,
                self.ln_x.bias,
                self.ln_x.eps,
            ).view(shape)
            shortcut = (read * write_key * self.r_k).sum(-1, keepdim=True)
            mixed = (normalized + shortcut * value) * gate.float().sigmoid()
        result = self.o_proj(mixed.flatten(2))
        if cache is not None:
            cache.previous = hidden[:, -1:].detach().float().clone()
            cache.length += hidden.shape[1]
        return result, first


class PrimedDecoderLayer(nn.Module):
    """Keep the source FFN and both external norms; replace only its mixer."""

    def __init__(self, source, mixer):
        super().__init__()
        self.mixer = mixer
        self.input_layernorm = source.input_layernorm
        self.post_attention_layernorm = source.post_attention_layernorm
        self.mlp = source.mlp

    def forward(self, hidden, positions, first=None):
        mixed, first = self.mixer(self.input_layernorm(hidden), positions, first)
        residual = hidden.float() + mixed.float()
        output = residual + self.mlp(self.post_attention_layernorm(residual)).float()
        return output, mixed, first


class FrozenGDN(nn.Module):
    """Adapter for an already-converted any2rwkv source-shell GDN, not a converter."""

    def __init__(self, shell, backend="flash"):
        super().__init__()
        if not all(
            callable(getattr(shell, name, None))
            for name in ("_source_activations", "_wkv_inputs", "_source_boundary")
        ):
            raise ValueError("retained GDN must provide the any2rwkv source-shell interface")
        self.shell = deepcopy(shell).eval().requires_grad_(False)
        self.backend = backend

    def forward(self, hidden, positions, first=None):
        query, key, value, beta, decay, gate = self.shell._source_activations(hidden, None, None)
        inputs = self.shell._wkv_inputs(query, key, value, beta, decay, training=False)
        raw = run_wkv(tuple(tensor.view_as(value) for tensor in inputs), self.backend)
        return self.shell._source_boundary(raw, gate), first


def source_outputs(layer, rotary_embedding, hidden):
    """Original attention and complete block on the same student-prefix input."""
    normalized = layer.input_layernorm(hidden)
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
    mask = torch.full(
        (hidden.shape[1], hidden.shape[1]),
        torch.finfo(hidden.dtype).min,
        device=hidden.device,
        dtype=hidden.dtype,
    ).triu(1)[None, None]
    mixed = layer.self_attn(
        normalized,
        position_embeddings=rotary_embedding(normalized, positions),
        attention_mask=mask,
        past_key_values=None,
    )[0]
    residual = hidden + mixed
    return residual + layer.mlp(layer.post_attention_layernorm(residual)), mixed


class PrimedQwen(nn.Module):
    """Full-sequence model with frozen unselected blocks and FFNs.

    Retained mixers are explicit adapters, e.g. FrozenGDN, keyed by GDN index.
    Forward returns final normalized hidden states, not an AutoModel export.
    """

    def __init__(
        self, source, selected=None, rank=32, agqa=True, backend="flash", retained_mixers=None
    ):
        super().__init__()
        text = source.model
        available = {
            index for index, kind in enumerate(text.config.layer_types) if kind == "full_attention"
        }
        selected = sorted(available) if selected is None else list(selected)
        retained_mixers = retained_mixers or {}
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= available:
            raise ValueError("select unique source GQA layer indices")
        if any(
            index not in range(len(text.layers)) or index in available for index in retained_mixers
        ):
            raise ValueError("retained mixers must only replace GDN positions")
        self.selected = sorted(selected)
        self.config = deepcopy(text.config)
        self.rank, self.agqa, self.backend = rank, agqa, backend
        self.embed_tokens, self.norm, self.lm_head = text.embed_tokens, text.norm, source.lm_head
        self.rotary_emb = text.rotary_emb
        blocks = []
        for index, layer in enumerate(text.layers):
            if index in self.selected:
                mixer = PrimedTimeMix(layer.self_attn, self.config, rank, agqa, backend)
            elif index in retained_mixers:
                mixer = deepcopy(retained_mixers[index]).eval().requires_grad_(False)
            else:
                blocks.append(layer)
                continue
            blocks.append(PrimedDecoderLayer(layer, mixer))
        self.layers = nn.ModuleList(blocks)
        self.eval().requires_grad_(False)

    def step_layer(self, index, hidden, first=None):
        layer = self.layers[index]
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
        if isinstance(layer, PrimedDecoderLayer):
            return layer(hidden, positions, first)
        if self.config.layer_types[index] == "full_attention":
            output, mixed = source_outputs(layer, self.rotary_emb, hidden)
            return output, mixed, first
        output = layer(
            hidden,
            position_embeddings=None,
            position_ids=positions,
            attention_mask=None,
            past_key_values=None,
        )
        return output, None, first

    def forward(self, input_ids):
        hidden, first = self.embed_tokens(input_ids).float(), None
        for index in range(len(self.layers)):
            hidden, _, first = self.step_layer(index, hidden, first)
        return self.norm(hidden)
