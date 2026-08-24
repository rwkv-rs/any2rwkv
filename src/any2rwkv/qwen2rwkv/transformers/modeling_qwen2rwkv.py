"""Qwen3.5 text blocks with hybrid GDN/WKV and converted GQA mixers."""

from __future__ import annotations

import importlib
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionLayer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    causal_conv1d_fn,
    causal_conv1d_update,
)

GDN = "linear_attention"
GQA = "full_attention"
_SOURCE_TYPES = [GDN, GDN, GDN, GQA] * 6
GDN_MODE = "source_shell_wkv7"
GDN_CHECKPOINT_SCHEMA = "source_gdn_state_dict_v1"
# These three aliases are imported by the existing source-model builder.  They
# are deliberately not part of the persisted Qwen2RWKV config; ``to_dict``
# removes the compatibility-only constructor fields below.
GQA_NUM_EXPERTS = 1
GQA_STATES_PER_EXPERT = 2
GQA_ROUTER_LOW_RANK_DIM = 0
GQA_FEATURE_PROJECTION_DIM = 64
GQA_FEATURE_OUTPUT_DIM = 128
GQA_STATES_PER_QUERY_HEAD = 2
GQA_SIDECAR_CAPACITY = 128
GQA_SINK_SLOTS = 8
GQA_RECENT_SLOTS = 64
GQA_HEAVY_SLOTS = 56
GQA_READOUT_MODE = "hedgehog_h2o_shared_norm"
GQA_CHECKPOINT_SCHEMA = "gqa_hedgehog_h2o_d256x2_v1"
GQA_TAIL_DECAY_LOGITS = -30.0
CLAMP_W_EPSILON = 1e-4
W_SCALE = -math.exp(-0.5)


class _FP32RotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """Keep RoPE frequencies in FP32 across model-wide dtype conversions."""

    def _apply(self, fn, recurse: bool = True):
        names = ("inv_freq", "original_inv_freq")
        protected = {name: self._buffers[name] for name in names}
        for name in names:
            self._buffers[name] = None
        try:
            result = super()._apply(fn, recurse=recurse)
            device = fn(protected["inv_freq"].new_empty(0)).device
        except Exception:
            for name, value in protected.items():
                self._buffers[name] = value
            raise
        for name, value in protected.items():
            self._buffers[name] = value.to(device=device, dtype=torch.float32)
        return result


def _orthogonal_(parameter: torch.Tensor, gain: float) -> None:
    value = torch.empty_like(parameter, dtype=torch.float32)
    nn.init.orthogonal_(value, gain=gain)
    parameter.copy_(value)


class Qwen2RWKVConfig(Qwen3_5TextConfig):
    """The one supported Qwen3.5-2B text geometry."""

    model_type = "qwen2rwkv"
    source_layer_types: list[str] | None = None
    decay_low_rank_dim: int = 128
    a_low_rank_dim: int = 128
    v_low_rank_dim: int = 64
    gate_low_rank_dim: int = 224
    group_norm_epsilon: float = 1e-5
    value_residual: bool = True
    gdn_mode: str = GDN_MODE
    gdn_checkpoint_schema: str = GDN_CHECKPOINT_SCHEMA
    # Deprecated builder-only inputs.  They are accepted so the existing
    # source shell can construct this config without editing a fifth file.
    gqa_num_experts: int = GQA_NUM_EXPERTS
    gqa_states_per_expert: int = GQA_STATES_PER_EXPERT
    gqa_router_low_rank_dim: int = GQA_ROUTER_LOW_RANK_DIM
    gqa_feature_projection_dim: int = GQA_FEATURE_PROJECTION_DIM
    gqa_feature_output_dim: int = GQA_FEATURE_OUTPUT_DIM
    gqa_states_per_query_head: int = GQA_STATES_PER_QUERY_HEAD
    gqa_sidecar_capacity: int = GQA_SIDECAR_CAPACITY
    gqa_sink_slots: int = GQA_SINK_SLOTS
    gqa_recent_slots: int = GQA_RECENT_SLOTS
    gqa_heavy_slots: int = GQA_HEAVY_SLOTS
    gqa_readout_mode: str = GQA_READOUT_MODE
    gqa_checkpoint_schema: str = GQA_CHECKPOINT_SCHEMA

    def __post_init__(self, **kwargs):
        if self.source_layer_types is None:
            self.source_layer_types = list(self.layer_types or _SOURCE_TYPES)
        self.layer_types = list(self.source_layer_types)
        if self.gdn_mode != GDN_MODE:
            raise ValueError(f"unsupported GDN mode {self.gdn_mode!r}; expected {GDN_MODE!r}")
        if self.gdn_checkpoint_schema != GDN_CHECKPOINT_SCHEMA:
            raise ValueError(
                "unsupported GDN checkpoint schema "
                f"{self.gdn_checkpoint_schema!r}; expected {GDN_CHECKPOINT_SCHEMA!r}"
            )
        if self.gqa_readout_mode != GQA_READOUT_MODE:
            raise ValueError(
                "unsupported GQA readout mode "
                f"{self.gqa_readout_mode!r}; expected {GQA_READOUT_MODE!r}"
            )
        if self.gqa_checkpoint_schema != GQA_CHECKPOINT_SCHEMA:
            raise ValueError(
                "unsupported GQA checkpoint schema "
                f"{self.gqa_checkpoint_schema!r}; expected {GQA_CHECKPOINT_SCHEMA!r}"
            )
        geometry = (
            self.gqa_feature_projection_dim,
            self.gqa_feature_output_dim,
            self.gqa_states_per_query_head,
            self.gqa_sidecar_capacity,
            self.gqa_sink_slots,
            self.gqa_recent_slots,
            self.gqa_heavy_slots,
        )
        expected = (
            GQA_FEATURE_PROJECTION_DIM,
            GQA_FEATURE_OUTPUT_DIM,
            GQA_STATES_PER_QUERY_HEAD,
            GQA_SIDECAR_CAPACITY,
            GQA_SINK_SLOTS,
            GQA_RECENT_SLOTS,
            GQA_HEAVY_SLOTS,
        )
        if geometry != expected or sum(geometry[-3:]) != geometry[3]:
            raise ValueError(
                f"unsupported bounded Hedgehog GQA geometry {geometry}; expected {expected}"
            )
        super().__post_init__(**kwargs)

    def to_dict(self):
        values = super().to_dict()
        for legacy in (
            "gqa_num_experts",
            "gqa_states_per_expert",
            "gqa_router_low_rank_dim",
        ):
            values.pop(legacy, None)
        return values

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        source_types = config_dict.get("source_layer_types") or config_dict.get("layer_types", ())
        if config_dict.get("model_type") == cls.model_type and GDN in source_types:
            missing = {
                "gdn_mode",
                "gdn_checkpoint_schema",
            }.difference(config_dict)
            if missing:
                raise ValueError(
                    "legacy canonical-RWKV GDN artifact is incompatible with the "
                    f"source-shell WKV runtime; missing config keys {sorted(missing)}"
                )
        if config_dict.get("model_type") == cls.model_type and GQA in source_types:
            missing = {
                "gqa_feature_projection_dim",
                "gqa_feature_output_dim",
                "gqa_states_per_query_head",
                "gqa_sidecar_capacity",
                "gqa_sink_slots",
                "gqa_recent_slots",
                "gqa_heavy_slots",
                "gqa_readout_mode",
                "gqa_checkpoint_schema",
            }.difference(config_dict)
            if missing:
                raise ValueError(
                    "legacy GQA artifact is incompatible with the bounded Hedgehog/H2O "
                    f"runtime; missing config keys {sorted(missing)}"
                )
        return super().from_dict(config_dict, **kwargs)

    def geometry(self, layer_idx: int) -> tuple[int, int, int]:
        if self.source_layer_types[layer_idx] == GDN:
            return 16, 128, 1
        return (
            self.num_attention_heads * self.gqa_states_per_query_head,
            self.head_dim,
            self.gqa_states_per_query_head,
        )


def _flash(mode: str, tensor: torch.Tensor, *, gdn: bool = False):
    try:
        module = importlib.import_module("flashrwkv2")
    except ImportError as error:
        raise RuntimeError(f"{mode} requires the pinned FlashRWKV2 provider") from error
    if gdn and mode == "training":
        required = ("pretrain_recurrent_bf16",)
    elif gdn:
        required = (
            "prepare_recurrent_metadata",
            "infer_recurrent_fp16_forward_varlen",
        )
    elif mode == "training":
        required = ("pretrain_recurrent_bf16",)
    else:
        required = (
            "prepare_recurrent_metadata",
            "infer_recurrent_fp16_forward_varlen",
        )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"FlashRWKV2 is missing required public operators: {missing}")
    if not tensor.is_cuda:
        raise RuntimeError(f"{mode} requires CUDA tensors; got {tensor.device}")
    return module


class Qwen2RWKVCacheLayer(LinearAttentionLayer, CacheLayerMixin):
    is_sliding = False

    def __init__(self):
        CacheLayerMixin.__init__(self)
        LinearAttentionLayer.__init__(self, number_of_states=1)
        self.cumulative_length = 0
        self.gqa_keys: torch.Tensor | None = None
        self.gqa_values: torch.Tensor | None = None
        self.gqa_scores: torch.Tensor | None = None
        self.gqa_positions: torch.Tensor | None = None

    def update(self, key_states, value_states, *args, **kwargs):
        raise RuntimeError("Qwen2RWKV updates recurrent state through FlashRWKV2")

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    @property
    def batch_size(self) -> int:
        if self.is_conv_states_initialized[0]:
            return self.conv_states[0].shape[0]
        if self.is_recurrent_states_initialized[0]:
            return self.recurrent_states[0].shape[0]
        if self.gqa_keys is not None:
            return self.gqa_keys.shape[0]
        return -1

    def mark_updated(self, length: int) -> None:
        self.cumulative_length += length
        self.has_previous_state[0] = True

    def initialize_gqa_sidecar(
        self,
        batch: int,
        kv_heads: int,
        capacity: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (batch, kv_heads, capacity, head_dim)
        if self.gqa_keys is None:
            self.gqa_keys = torch.zeros(shape, dtype=dtype, device=device)
            self.gqa_values = torch.zeros_like(self.gqa_keys)
            self.gqa_scores = torch.zeros(batch, capacity, dtype=torch.float32, device=device)
            self.gqa_positions = torch.full((batch, capacity), -1, dtype=torch.int32, device=device)
        elif (
            self.gqa_keys.shape != shape
            or self.gqa_keys.dtype != dtype
            or self.gqa_keys.device != device
            or self.gqa_values is None
            or self.gqa_scores is None
            or self.gqa_positions is None
            or self.gqa_values.shape != shape
            or self.gqa_values.dtype != dtype
            or self.gqa_values.device != device
            or self.gqa_scores.shape != (batch, capacity)
            or self.gqa_scores.device != device
            or self.gqa_positions.shape != (batch, capacity)
            or self.gqa_positions.device != device
        ):
            raise RuntimeError(
                "GQA sidecar shape/dtype changed after initialization: "
                f"have={tuple(self.gqa_keys.shape)}/{self.gqa_keys.dtype}, "
                f"wanted={shape}/{dtype}"
            )

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0
        for value in (self.gqa_keys, self.gqa_values, self.gqa_scores):
            if value is not None:
                value.zero_()
        if self.gqa_positions is not None:
            self.gqa_positions.fill_(-1)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.is_conv_states_initialized[0]:
            self.conv_states[0] = self.conv_states[0].repeat_interleave(repeats, 0)
        if self.is_recurrent_states_initialized[0]:
            self.recurrent_states[0] = self.recurrent_states[0].repeat_interleave(repeats, 0)
        for name in ("gqa_keys", "gqa_values", "gqa_scores", "gqa_positions"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.repeat_interleave(repeats, 0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.is_conv_states_initialized[0]:
            self.conv_states[0] = self.conv_states[0].index_select(0, indices.to(self.device))
        if self.is_recurrent_states_initialized[0]:
            self.recurrent_states[0] = self.recurrent_states[0].index_select(
                0, indices.to(self.device)
            )
        for name in ("gqa_keys", "gqa_values", "gqa_scores", "gqa_positions"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, indices.to(value.device)))


class Qwen2RWKVCache(Cache):
    """Per-layer Conv4-or-shift state, WKV state, elapsed state, and GQA ``v_first``."""

    def __init__(self, config: Qwen2RWKVConfig):
        super().__init__(layers=[Qwen2RWKVCacheLayer() for _ in range(config.num_hidden_layers)])
        self.config = config
        self.elapsed: list[torch.Tensor | None] = [None] * config.num_hidden_layers
        self.v_first: torch.Tensor | None = None
        self._metadata_key = None
        self._metadata = None

    def recurrent_metadata(self, flash, batch: int, length: int, device: torch.device):
        key = (batch, length, device.type, device.index)
        if key != self._metadata_key:
            offsets = torch.arange(
                0, (batch + 1) * length, length, dtype=torch.int32, device=device
            )
            indices = torch.arange(batch, dtype=torch.int32, device=device)
            ticket = flash.prepare_recurrent_metadata(
                offsets,
                indices,
                total_tokens=batch * length,
                state_pool_size=batch,
                max_seqlen=length,
            )
            self._metadata_key = key
            self._metadata = offsets, indices, ticket
        return self._metadata

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        self.elapsed = [
            None if x is None else x.repeat_interleave(repeats, 0) for x in self.elapsed
        ]
        if self.v_first is not None:
            self.v_first = self.v_first.repeat_interleave(repeats, 0)
        self._metadata_key = None
        self._metadata = None

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        self.elapsed = [
            None if x is None else x.index_select(0, indices.to(x.device)) for x in self.elapsed
        ]
        if self.v_first is not None:
            self.v_first = self.v_first.index_select(0, indices.to(self.v_first.device))
        self._metadata_key = None
        self._metadata = None

    def reset(self) -> None:
        super().reset()
        for value in self.elapsed:
            if value is not None:
                value.zero_()
        self.v_first = None
        self._metadata_key = None
        self._metadata = None


def _cache_states(cache: Qwen2RWKVCache, layer_idx: int, x: torch.Tensor, heads: int, dim: int):
    batch = x.shape[0]
    layer = cache.layers[layer_idx]
    if not layer.is_conv_states_initialized[0]:
        layer.lazy_initialization(
            conv_states=x.new_zeros(batch, x.shape[-1], 1), state_idx=0, conv_kernel_size=1
        )
    if not layer.is_recurrent_states_initialized[0]:
        layer.lazy_initialization(
            recurrent_states=torch.zeros(
                batch, heads, dim, dim, dtype=torch.float16, device=x.device
            ),
            state_idx=0,
        )
    if cache.elapsed[layer_idx] is None:
        cache.elapsed[layer_idx] = torch.zeros(batch, dtype=torch.int32, device=x.device)
    return layer.conv_states[0].squeeze(-1), layer.recurrent_states[0], cache.elapsed[layer_idx]


def _gdn_cache_states(cache: Qwen2RWKVCache, layer_idx: int, x: torch.Tensor, heads: int, dim: int):
    batch = x.shape[0]
    layer = cache.layers[layer_idx]
    if not layer.is_recurrent_states_initialized[0]:
        layer.lazy_initialization(
            recurrent_states=torch.zeros(
                batch, heads, dim, dim, dtype=torch.float16, device=x.device
            ),
            state_idx=0,
        )
    if cache.elapsed[layer_idx] is None:
        cache.elapsed[layer_idx] = torch.zeros(batch, dtype=torch.int32, device=x.device)
    return layer.recurrent_states[0], cache.elapsed[layer_idx]


def _clamp_w_logits(log_decay: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
    ratio = log_decay.float() / W_SCALE
    projected = ratio.clamp(CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON)
    if straight_through:
        projected = ratio + (projected - ratio).detach()
    return torch.logit(projected)


class Qwen2RWKVGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Source Qwen3.5 GDN shell whose matrix recurrence is executed by RWKV-7 WKV."""

    def _source_activations(
        self,
        hidden_states: torch.Tensor,
        cache: Qwen2RWKVCache | None,
        attention_mask: torch.Tensor | None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch, length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        use_precomputed = cache is not None and cache.has_previous_state(self.layer_idx)
        if use_precomputed and length == 1 and not cache.layers[self.layer_idx].record_past:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                cache.layers[self.layer_idx].conv_states[0],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache is not None:
                mixed_qkv = cache.update_conv_state(
                    mixed_qkv,
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )
            if cache is not None:
                mixed_qkv = mixed_qkv[:, :, -length:]
        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.view(batch, length, self.num_k_heads, self.head_k_dim)
        key = key.view(batch, length, self.num_k_heads, self.head_k_dim)
        value = value.view(batch, length, self.num_v_heads, self.head_v_dim)
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        log_decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.float()
        )
        z = self.in_proj_z(hidden_states).view(batch, length, self.num_v_heads, self.head_v_dim)
        return query, key, value, beta, log_decay, z

    def _wkv_inputs(self, query, key, value, beta, log_decay, *, training: bool):
        read = query / math.sqrt(self.head_k_dim)
        write = beta[..., None] * value
        decay_logits = _clamp_w_logits(log_decay, straight_through=training)
        realized_log_decay = W_SCALE * decay_logits.sigmoid()
        retention = realized_log_decay.exp()
        erase = -(beta.float() * retention)[..., None] * key.float()
        decay = decay_logits[..., None].expand_as(key)
        dtype = value.dtype
        return tuple(
            tensor.flatten(2).to(dtype).contiguous()
            for tensor in (read, decay, key, write, key, erase)
        )

    def _source_boundary(self, raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch, length = raw.shape[:2]
        mixed = self.norm(
            raw.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        ).reshape(batch, length, self.value_dim)
        return self.out_proj(mixed)

    def _training_forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None):
        flash = _flash("training", x, gdn=True)
        if x.dtype != torch.bfloat16 or x.shape[1] % 16:
            raise RuntimeError(
                "training requires contiguous BF16 [B,T,2048] with T divisible by 16"
            )
        query, key, value, beta, log_decay, z = self._source_activations(x, None, attention_mask)
        r, w, k, v, a, b = self._wkv_inputs(query, key, value, beta, log_decay, training=True)
        raw = flash.pretrain_recurrent_bf16(r, w, k, v, a, b, head_size=self.head_v_dim)
        return self._source_boundary(raw.view_as(value), z)

    def _inference_forward(
        self,
        x: torch.Tensor,
        cache: Qwen2RWKVCache,
        attention_mask: torch.Tensor | None,
    ):
        flash = _flash("inference", x, gdn=True)
        if x.dtype != torch.float16:
            raise RuntimeError("inference requires a float16 checkpoint")
        batch, length, _ = x.shape
        query, key, value, beta, log_decay, z = self._source_activations(x, cache, attention_mask)
        r, w, k, v, a, b = self._wkv_inputs(query, key, value, beta, log_decay, training=False)
        state, elapsed = _gdn_cache_states(
            cache, self.layer_idx, x, self.num_v_heads, self.head_v_dim
        )
        offsets, indices, ticket = cache.recurrent_metadata(flash, batch, length, x.device)
        raw = flash.infer_recurrent_fp16_forward_varlen(
            r.view(-1, self.num_v_heads, self.head_v_dim),
            w.view(-1, self.num_v_heads, self.head_v_dim),
            k.view(-1, self.num_v_heads, self.head_v_dim),
            v.view(-1, self.num_v_heads, self.head_v_dim),
            a.view(-1, self.num_v_heads, self.head_v_dim),
            b.view(-1, self.num_v_heads, self.head_v_dim),
            state_pool=state,
            elapsed_state_pool=elapsed,
            cu_seqlens=offsets,
            state_indices=indices,
            max_seqlen=length,
            validated_metadata=ticket,
        ).view_as(value)
        return self._source_boundary(raw, z)

    def forward(self, x, v_first=None, past_key_values=None, attention_mask=None):
        if self.training:
            return self._training_forward(x, attention_mask), v_first
        if not isinstance(past_key_values, Qwen2RWKVCache):
            raise TypeError("inference requires Qwen2RWKVCache")
        return self._inference_forward(x, past_key_values, attention_mask), v_first


def _repeat_gqa(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    return tensor.repeat_interleave(groups, dim=1)


def _slot_values(tensor: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch, :, slots.to(torch.long), :]


def _admit_sidecar_token(
    positions: torch.Tensor,
    scores: torch.Tensor,
    position: int,
    *,
    sink_slots: int,
    recent_slots: int,
    heavy_slots: int,
    keys: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Admit one token and return the token evicted into the linear tail."""

    if position < 0:
        raise ValueError(f"sidecar position must be non-negative; got {position}")
    capacity = sink_slots + recent_slots + heavy_slots
    if positions.shape != scores.shape or positions.shape[1] != capacity:
        raise ValueError("invalid bounded sidecar metadata geometry")
    provided = (keys, values, current_key, current_value)
    if any(value is None for value in provided) and not all(value is None for value in provided):
        raise ValueError("sidecar K/V arguments must be provided together")
    batch = positions.shape[0]
    batch_index = torch.arange(batch, device=positions.device)
    evicted_position = torch.full((batch,), -1, dtype=torch.int32, device=positions.device)
    evicted_key = None if keys is None else torch.zeros_like(current_key)
    evicted_value = None if values is None else torch.zeros_like(current_value)

    if position < sink_slots:
        current_slot = position
    elif position < sink_slots + recent_slots:
        current_slot = sink_slots + position - sink_slots
    else:
        current_slot = sink_slots + ((position - sink_slots) % recent_slots)
        candidate_position = positions[:, current_slot].clone()
        candidate_score = scores[:, current_slot].clone()
        candidate_key = None if keys is None else keys[:, :, current_slot, :].clone()
        candidate_value = None if values is None else values[:, :, current_slot, :].clone()
        heavy_start = sink_slots + recent_slots
        if position < capacity:
            heavy_slot = heavy_start + position - sink_slots - recent_slots
            positions[:, heavy_slot] = candidate_position
            scores[:, heavy_slot] = candidate_score
            if keys is not None:
                keys[:, :, heavy_slot, :] = candidate_key
                values[:, :, heavy_slot, :] = candidate_value
        else:
            heavy_scores = scores[:, heavy_start:]
            heavy_score = heavy_scores.min(dim=1).values
            tied_for_minimum = heavy_scores == heavy_score[:, None]
            heavy_positions = positions[:, heavy_start:]
            oldest_tied_position = torch.where(
                tied_for_minimum,
                heavy_positions,
                torch.full_like(heavy_positions, torch.iinfo(torch.int32).max),
            )
            heavy_relative = oldest_tied_position.argmin(dim=1)
            heavy_slot = heavy_start + heavy_relative
            selected_heavy_position = positions[batch_index, heavy_slot]
            # Resolve equal scores by original position, evicting the older
            # non-sink token regardless of its physical heavy slot.
            keep_candidate = (candidate_score > heavy_score) | (
                (candidate_score == heavy_score) & (candidate_position > selected_heavy_position)
            )
            evicted_slot = torch.where(
                keep_candidate,
                heavy_slot,
                torch.full_like(heavy_slot, current_slot),
            )
            evicted_position = positions[batch_index, evicted_slot].clone()
            if keys is not None:
                evicted_key = _slot_values(keys, evicted_slot).clone()
                evicted_value = _slot_values(values, evicted_slot).clone()
            old_heavy_position = positions[batch_index, heavy_slot].clone()
            old_heavy_score = scores[batch_index, heavy_slot].clone()
            positions[batch_index, heavy_slot] = torch.where(
                keep_candidate, candidate_position, old_heavy_position
            )
            scores[batch_index, heavy_slot] = torch.where(
                keep_candidate, candidate_score, old_heavy_score
            )
            if keys is not None:
                old_heavy_key = _slot_values(keys, heavy_slot).clone()
                old_heavy_value = _slot_values(values, heavy_slot).clone()
                keys[batch_index, :, heavy_slot, :] = torch.where(
                    keep_candidate[:, None, None], candidate_key, old_heavy_key
                )
                values[batch_index, :, heavy_slot, :] = torch.where(
                    keep_candidate[:, None, None], candidate_value, old_heavy_value
                )

    positions[:, current_slot] = position
    scores[:, current_slot] = 0
    if keys is not None:
        keys[:, :, current_slot, :] = current_key
        values[:, :, current_slot, :] = current_value
    return evicted_position, evicted_key, evicted_value


def _require_positive_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise FloatingPointError(f"{name} must be finite and strictly positive")


def _validate_sidecar_plan(
    snapshots: torch.Tensor,
    evicted: torch.Tensor,
) -> None:
    batch, length, capacity = snapshots.shape
    if evicted.shape != (batch, length):
        raise RuntimeError("sidecar eviction stream has incompatible geometry")
    valid = snapshots >= 0
    expected_exact = torch.arange(
        1, length + 1, dtype=torch.int32, device=snapshots.device
    ).clamp_max(capacity)
    exact_count = valid.sum(-1).to(torch.int32)
    token_index = torch.arange(length, device=snapshots.device).view(1, -1, 1)
    bounded_positions = (~valid) | ((snapshots >= 0) & (snapshots <= token_index))
    sorted_positions = snapshots.masked_fill(~valid, torch.iinfo(torch.int32).max).sort(-1).values
    duplicate = (sorted_positions[..., 1:] == sorted_positions[..., :-1]) & (
        sorted_positions[..., 1:] != torch.iinfo(torch.int32).max
    )
    expected_evictions = (
        torch.arange(1, length + 1, dtype=torch.int32, device=snapshots.device) - capacity
    ).clamp_min(0)
    eviction_count = (evicted >= 0).cumsum(-1).to(torch.int32)
    if (
        not torch.equal(exact_count, expected_exact.view(1, -1).expand_as(exact_count))
        or not bounded_positions.all()
        or duplicate.any()
        or not torch.equal(
            eviction_count,
            expected_evictions.view(1, -1).expand_as(eviction_count),
        )
    ):
        raise RuntimeError("sidecar lost, duplicated, or exceeded a token partition")
    if length > capacity:
        tail = evicted[:, capacity:]
        sorted_tail = tail.sort(-1).values
        if (sorted_tail[:, 1:] == sorted_tail[:, :-1]).any():
            raise RuntimeError("linear tail received the same original position twice")
        final_exact = snapshots[:, -1]
        if (tail[:, :, None] == final_exact[:, None, :]).any():
            raise RuntimeError("a token appears in both exact sidecar and linear tail")


def _validate_live_sidecar(
    positions: torch.Tensor,
    scores: torch.Tensor,
    latest_position: int,
) -> None:
    valid = positions >= 0
    sorted_positions = positions.masked_fill(~valid, torch.iinfo(torch.int32).max).sort(-1).values
    duplicate = (sorted_positions[:, 1:] == sorted_positions[:, :-1]) & (
        sorted_positions[:, 1:] != torch.iinfo(torch.int32).max
    )
    expected_count = min(latest_position + 1, positions.shape[1])
    if (
        (positions[valid] > latest_position).any()
        or not (valid.sum(-1) == expected_count).all()
        or not (positions == latest_position).any(-1).all()
        or duplicate.any()
        or not torch.isfinite(scores).all()
        or (scores < 0).any()
    ):
        raise RuntimeError("live sidecar metadata is non-finite or inconsistent")


class Qwen2RWKVTimeMix(nn.Module):
    """Bounded Hedgehog/LoLCATs GQA shell over an additive RWKV-7 tail."""

    def __init__(self, config: Qwen2RWKVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_size = config.head_dim
        self.feature_projection_dim = config.gqa_feature_projection_dim
        self.feature_output_dim = config.gqa_feature_output_dim
        self.states_per_head = config.gqa_states_per_query_head
        self.kernel_heads = self.num_heads * self.states_per_head
        self.recurrent_width = self.kernel_heads * self.head_size
        self.scaling = self.head_size**-0.5
        self.sink_slots = config.gqa_sink_slots
        self.recent_slots = config.gqa_recent_slots
        self.heavy_slots = config.gqa_heavy_slots
        self.sidecar_capacity = config.gqa_sidecar_capacity

        channels = config.hidden_size
        self.q_proj = nn.Linear(
            channels, self.num_heads * self.head_size * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            channels, self.num_kv_heads * self.head_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            channels, self.num_kv_heads * self.head_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_size, channels, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_size, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_size, eps=config.rms_norm_eps)
        self.rotary_emb = _FP32RotaryEmbedding(config)
        self.feature_q_weight = nn.Parameter(
            torch.zeros(
                self.num_heads,
                self.head_size,
                self.feature_projection_dim,
            )
        )
        self.feature_k_weight = nn.Parameter(torch.zeros_like(self.feature_q_weight))
        self.beta_logit = nn.Parameter(torch.full((self.num_heads,), math.log(0.1 / 0.9)))
        self._last_attention_heads: torch.Tensor | None = None
        self._last_sidecar_metrics: dict[str, torch.Tensor] = {}

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # LoLCATs names this ``zero_init_``, but its no-skip implementation
            # uses ``eye_``.  Literal all-zero paired-softmax maps are a
            # symmetry point with zero Wq/Wk gradients.
            for weight in (self.feature_q_weight, self.feature_k_weight):
                weight.zero_()
                for head in weight:
                    nn.init.eye_(head)
            self.beta_logit.fill_(math.log(0.1 / 0.9))

    def load_source_attention(self, source: nn.Module) -> None:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
        self.reset_parameters()

    def attention_transfer_parameters(self) -> list[nn.Parameter]:
        return [self.feature_q_weight, self.feature_k_weight, self.beta_logit]

    def enable_lora(self, rank: int = 16, alpha: float = 32.0) -> None:
        if hasattr(self, "q_lora_a"):
            raise RuntimeError("GQA LoRA is already enabled")
        dtype = self.q_proj.weight.dtype
        device = self.q_proj.weight.device
        specs = {
            "q": (self.config.hidden_size, self.num_heads * self.head_size),
            "k": (self.config.hidden_size, self.num_kv_heads * self.head_size),
            "v": (self.config.hidden_size, self.num_kv_heads * self.head_size),
            "o": (self.num_heads * self.head_size, self.config.hidden_size),
        }
        self.lora_rank = rank
        self.lora_scale = float(alpha / rank)
        for name, (input_dim, output_dim) in specs.items():
            first = nn.Parameter(torch.empty(rank, input_dim, dtype=dtype, device=device))
            second = nn.Parameter(torch.zeros(output_dim, rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(first, a=math.sqrt(5))
            self.register_parameter(f"{name}_lora_a", first)
            self.register_parameter(f"{name}_lora_b", second)

    def lora_parameters(self) -> list[nn.Parameter]:
        if not hasattr(self, "q_lora_a"):
            return []
        return [
            getattr(self, f"{name}_lora_{part}")
            for name in ("q", "k", "v", "o")
            for part in ("a", "b")
        ]

    def _lora(self, x: torch.Tensor, name: str) -> torch.Tensor:
        first = getattr(self, f"{name}_lora_a")
        second = getattr(self, f"{name}_lora_b")
        return F.linear(F.linear(x, first), second) * self.lora_scale

    def merge_lora(self) -> None:
        if not hasattr(self, "q_lora_a"):
            return
        with torch.no_grad():
            deltas = {
                name: (
                    getattr(self, f"{name}_lora_b").float()
                    @ getattr(self, f"{name}_lora_a").float()
                    * self.lora_scale
                )
                for name in ("q", "k", "v", "o")
            }
            q_weight = self.q_proj.weight.view(
                self.num_heads, 2, self.head_size, self.config.hidden_size
            )
            q_weight[:, 0].add_(deltas["q"].view_as(q_weight[:, 0]).to(q_weight))
            self.k_proj.weight.add_(deltas["k"].to(self.k_proj.weight))
            self.v_proj.weight.add_(deltas["v"].to(self.v_proj.weight))
            self.o_proj.weight.add_(deltas["o"].to(self.o_proj.weight))
        for name in ("q", "k", "v", "o"):
            delattr(self, f"{name}_lora_a")
            delattr(self, f"{name}_lora_b")
        del self.lora_rank
        del self.lora_scale

    def drop_lora(self) -> None:
        if not hasattr(self, "q_lora_a"):
            return
        for name in ("q", "k", "v", "o"):
            delattr(self, f"{name}_lora_a")
            delattr(self, f"{name}_lora_b")
        del self.lora_rank
        del self.lora_scale

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length = x.shape[:2]
        projected = self.q_proj(x).view(batch, length, self.num_heads, 2 * self.head_size)
        query, gate = projected.chunk(2, dim=-1)
        key = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_size)
        value = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_size)
        if hasattr(self, "q_lora_a"):
            query = query + self._lora(x, "q").view_as(query)
            key = key + self._lora(x, "k").view_as(key)
            value = value + self._lora(x, "v").view_as(value)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = self.rotary_emb(x, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value, gate.reshape(batch, length, -1)

    def _feature(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("bhtd,hdf->bhtf", value.float(), weight.float())
        feature = torch.cat(
            (torch.softmax(projected, dim=-1), torch.softmax(-projected, dim=-1)),
            dim=-1,
        )
        return feature.to(value.dtype)

    def _features(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_heads = _repeat_gqa(key, self.num_kv_groups)
        return (
            self._feature(query, self.feature_q_weight),
            self._feature(key_heads, self.feature_k_weight),
        )

    @torch.no_grad()
    def _sidecar_plan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        feature_query: torch.Tensor,
        feature_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, length, _ = query.shape
        positions = torch.full(
            (batch, self.sidecar_capacity),
            -1,
            dtype=torch.int32,
            device=query.device,
        )
        scores = torch.zeros(batch, self.sidecar_capacity, dtype=torch.float32, device=query.device)
        snapshots = torch.empty(
            batch,
            length,
            self.sidecar_capacity,
            dtype=torch.int32,
            device=query.device,
        )
        evicted = torch.full((batch, length), -1, dtype=torch.int32, device=query.device)
        tail_feature_sum = torch.zeros(
            batch,
            self.num_heads,
            self.feature_output_dim,
            dtype=torch.float32,
            device=query.device,
        )
        key_heads = _repeat_gqa(key, self.num_kv_groups)
        beta = torch.sigmoid(self.beta_logit.float()).view(1, self.num_heads)
        alpha = 1 - beta
        denominators = []
        for token in range(length):
            evicted_position, _, _ = _admit_sidecar_token(
                positions,
                scores,
                token,
                sink_slots=self.sink_slots,
                recent_slots=self.recent_slots,
                heavy_slots=self.heavy_slots,
            )
            evicted[:, token] = evicted_position
            valid_eviction = evicted_position >= 0
            gather = evicted_position.clamp_min(0).to(torch.long)
            gather = gather[:, None, None, None].expand(
                -1, self.num_heads, 1, self.feature_output_dim
            )
            evicted_feature = torch.gather(feature_key, 2, gather).squeeze(2).float()
            tail_feature_sum.add_(evicted_feature * valid_eviction[:, None, None])
            snapshots[:, token] = positions
            slot_index = positions.clamp_min(0).to(torch.long)
            slot_index = slot_index[:, None, :, None].expand(-1, self.num_heads, -1, self.head_size)
            exact_key = torch.gather(key_heads, 2, slot_index)
            logits = (
                torch.einsum("bhd,bhsd->bhs", query[:, :, token].float(), exact_key.float())
                * self.scaling
            )
            valid = positions[:, None] >= 0
            logits = logits.masked_fill(~valid, -torch.inf)
            maximum = logits.amax(dim=-1, keepdim=True)
            exact_kernel = torch.exp(logits - maximum).masked_fill(~valid, 0)
            exact_denominator = exact_kernel.sum(-1)
            linear_denominator = (feature_query[:, :, token].float() * tail_feature_sum).sum(-1)
            raw_denominator = beta * exact_denominator + alpha * linear_denominator
            denominators.append(raw_denominator)
            denominator = raw_denominator.clamp_min(1e-12)
            exact_probability = beta[..., None] * exact_kernel / denominator[..., None]
            scores.add_(exact_probability.mean(1))
        _validate_sidecar_plan(snapshots, evicted)
        _require_positive_finite(
            "bounded Hedgehog planning shared denominator",
            torch.stack(denominators, dim=-1),
        )
        if not torch.isfinite(scores).all() or (scores < 0).any():
            raise FloatingPointError("H2O cumulative scores must be finite and non-negative")
        return snapshots, evicted

    def _eviction_stream(
        self,
        feature_key: torch.Tensor,
        value_heads: torch.Tensor,
        evicted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, length, _ = feature_key.shape
        valid = evicted >= 0
        feature_index = evicted.clamp_min(0).to(torch.long)
        feature_index = feature_index[:, None, :, None].expand(
            -1, self.num_heads, -1, self.feature_output_dim
        )
        value_index = evicted.clamp_min(0).to(torch.long)
        value_index = value_index[:, None, :, None].expand(-1, self.num_heads, -1, self.head_size)
        write_key = torch.gather(feature_key, 2, feature_index)
        write_value = torch.gather(value_heads, 2, value_index)
        write_key = write_key * valid[:, None, :, None]
        write_value = write_value * valid[:, None, :, None]
        return write_key, write_value, valid

    def _tail_reference(
        self,
        feature_query: torch.Tensor,
        feature_key: torch.Tensor,
        value_heads: torch.Tensor,
        evicted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_key, write_value, valid = self._eviction_stream(feature_key, value_heads, evicted)
        batch, heads, length, _ = write_key.shape
        numerator_state = torch.zeros(
            batch,
            heads,
            self.head_size,
            self.feature_output_dim,
            dtype=torch.float32,
            device=write_key.device,
        )
        denominator_state = torch.zeros(
            batch,
            heads,
            self.feature_output_dim,
            dtype=torch.float32,
            device=write_key.device,
        )
        numerator = torch.empty(
            batch,
            heads,
            length,
            self.head_size,
            dtype=torch.float32,
            device=write_key.device,
        )
        denominator = torch.empty(
            batch, heads, length, dtype=torch.float32, device=write_key.device
        )
        for token in range(length):
            key_token = write_key[:, :, token].float()
            value_token = write_value[:, :, token].float()
            numerator_update = value_token.unsqueeze(-1) * key_token.unsqueeze(-2)
            denominator_update = key_token * valid[:, token, None, None]
            if torch.is_grad_enabled():
                # A differentiable probe cannot mutate a state version saved
                # by autograd for an earlier read.
                numerator_state = numerator_state + numerator_update
                denominator_state = denominator_state + denominator_update
            else:
                numerator_state.add_(numerator_update)
                denominator_state.add_(denominator_update)
            read = feature_query[:, :, token].float()
            numerator[:, :, token] = torch.einsum("bhdf,bhf->bhd", numerator_state, read)
            denominator[:, :, token] = torch.einsum("bhf,bhf->bh", denominator_state, read)
        return numerator, denominator

    def _tail_training(
        self,
        x: torch.Tensor,
        feature_query: torch.Tensor,
        feature_key: torch.Tensor,
        value_heads: torch.Tensor,
        evicted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flash = _flash("training", x)
        if x.dtype != torch.bfloat16 or x.shape[1] % 16:
            raise RuntimeError(
                "GQA training requires contiguous BF16 [B,T,2048] with T divisible by 16"
            )
        write_key, write_value, valid = self._eviction_stream(feature_key, value_heads, evicted)
        query_padded = F.pad(feature_query, (0, self.head_size - self.feature_output_dim)).permute(
            0, 2, 1, 3
        )
        key_padded = F.pad(write_key, (0, self.head_size - self.feature_output_dim)).permute(
            0, 2, 1, 3
        )
        value_write = write_value.permute(0, 2, 1, 3)
        denominator_write = torch.ones_like(value_write) * valid[:, :, None, None]
        read = torch.stack((query_padded, query_padded), dim=3)
        key = torch.stack((key_padded, key_padded), dim=3)
        value = torch.stack((value_write, denominator_write), dim=3)
        shape = (x.shape[0], x.shape[1], self.recurrent_width)
        read = read.reshape(shape).contiguous()
        key = key.reshape(shape).contiguous()
        value = value.reshape(shape).contiguous()
        decay = torch.full_like(read, GQA_TAIL_DECAY_LOGITS)
        erase = torch.zeros_like(read)
        raw = flash.pretrain_recurrent_bf16(
            read, decay, key, value, erase, erase, head_size=self.head_size
        ).view(x.shape[0], x.shape[1], self.num_heads, 2, self.head_size)
        numerator = raw[..., 0, :].permute(0, 2, 1, 3).float()
        denominator = raw[..., 1, :].float().mean(-1).permute(0, 2, 1)
        return numerator, denominator

    def _combine(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        snapshots: torch.Tensor,
        tail_numerator: torch.Tensor,
        tail_denominator: torch.Tensor,
    ) -> torch.Tensor:
        key_heads = _repeat_gqa(key, self.num_kv_groups)
        value_heads = _repeat_gqa(value, self.num_kv_groups)
        beta = torch.sigmoid(self.beta_logit.float()).view(1, self.num_heads)
        alpha = 1 - beta
        outputs = []
        exact_mass = query.new_zeros((), dtype=torch.float32)
        linear_mass = query.new_zeros((), dtype=torch.float32)
        sink_mass = query.new_zeros((), dtype=torch.float32)
        recent_mass = query.new_zeros((), dtype=torch.float32)
        heavy_mass = query.new_zeros((), dtype=torch.float32)
        denominators = []
        for token in range(query.shape[2]):
            positions = snapshots[:, token]
            index = positions.clamp_min(0).to(torch.long)
            gather = index[:, None, :, None].expand(-1, self.num_heads, -1, self.head_size)
            exact_key = torch.gather(key_heads, 2, gather)
            exact_value = torch.gather(value_heads, 2, gather)
            logits = (
                torch.einsum("bhd,bhsd->bhs", query[:, :, token].float(), exact_key.float())
                * self.scaling
            )
            valid = positions[:, None] >= 0
            logits = logits.masked_fill(~valid, -torch.inf)
            maximum = logits.amax(-1, keepdim=True)
            kernel = torch.exp(logits - maximum).masked_fill(~valid, 0)
            exact_denominator = kernel.sum(-1)
            exact_numerator = torch.einsum("bhs,bhsd->bhd", kernel, exact_value.float())
            raw_denominator = (
                beta * exact_denominator + alpha * tail_denominator[:, :, token].float()
            )
            denominators.append(raw_denominator)
            denominator = raw_denominator.clamp_min(1e-12)
            output = (
                beta[..., None] * exact_numerator
                + alpha[..., None] * tail_numerator[:, :, token].float()
            ) / denominator[..., None]
            outputs.append(output)
            exact_mass = exact_mass + (beta * exact_denominator / denominator).mean()
            linear_mass = (
                linear_mass + (alpha * tail_denominator[:, :, token].float() / denominator).mean()
            )
            exact_probability = beta[..., None] * kernel / denominator[..., None]
            sink_mass = sink_mass + exact_probability[..., : self.sink_slots].sum(-1).mean()
            recent_mass = (
                recent_mass
                + exact_probability[..., self.sink_slots : self.sink_slots + self.recent_slots]
                .sum(-1)
                .mean()
            )
            heavy_mass = (
                heavy_mass
                + exact_probability[..., self.sink_slots + self.recent_slots :].sum(-1).mean()
            )
        _require_positive_finite(
            "bounded Hedgehog shared denominator",
            torch.stack(denominators, dim=-1),
        )
        length = query.shape[2]
        final_valid = snapshots[:, -1] >= 0
        self._last_sidecar_metrics = {
            "exact_mass": (exact_mass / length).detach(),
            "linear_mass": (linear_mass / length).detach(),
            "sink_mass": (sink_mass / length).detach(),
            "recent_mass": (recent_mass / length).detach(),
            "heavy_mass": (heavy_mass / length).detach(),
            "sink_count": final_valid[:, : self.sink_slots].sum(-1).float().mean(),
            "recent_count": final_valid[:, self.sink_slots : self.sink_slots + self.recent_slots]
            .sum(-1)
            .float()
            .mean(),
            "heavy_count": final_valid[:, self.sink_slots + self.recent_slots :]
            .sum(-1)
            .float()
            .mean(),
        }
        return torch.stack(outputs, dim=2).to(value.dtype)

    def attention_heads_reference(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).view(1, -1)
            position_ids = position_ids.expand(x.shape[0], -1)
        query, key, value, gate = self._project_qkv(x, position_ids)
        feature_query, feature_key = self._features(query, key)
        snapshots, evicted = self._sidecar_plan(query, key, feature_query, feature_key)
        value_heads = _repeat_gqa(value, self.num_kv_groups)
        numerator, denominator = self._tail_reference(
            feature_query, feature_key, value_heads, evicted
        )
        return (
            self._combine(query, key, value, snapshots, numerator, denominator),
            gate,
        )

    def reference_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        heads, gate = self.attention_heads_reference(x, position_ids)
        mixed = heads.transpose(1, 2).reshape(*x.shape)
        mixed = mixed * torch.sigmoid(gate)
        if hasattr(self, "o_lora_a"):
            return self.o_proj(mixed) + self._lora(mixed, "o")
        return self.o_proj(mixed)

    def _training_forward(
        self, x: torch.Tensor, v_first: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
        positions = positions.expand(x.shape[0], -1)
        query, key, value, gate = self._project_qkv(x, positions)
        feature_query, feature_key = self._features(query, key)
        snapshots, evicted = self._sidecar_plan(query, key, feature_query, feature_key)
        value_heads = _repeat_gqa(value, self.num_kv_groups)
        numerator, denominator = self._tail_training(
            x, feature_query, feature_key, value_heads, evicted
        )
        heads = self._combine(query, key, value, snapshots, numerator, denominator)
        self._last_attention_heads = heads
        mixed = heads.transpose(1, 2).reshape(*x.shape)
        mixed = mixed * torch.sigmoid(gate)
        output = self.o_proj(mixed)
        if hasattr(self, "o_lora_a"):
            output = output + self._lora(mixed, "o")
        return output, v_first

    def _inference_forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        cache: Qwen2RWKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flash = _flash("inference", x)
        if x.dtype != torch.float16:
            raise RuntimeError("GQA inference requires a float16 checkpoint")
        layer = cache.layers[self.layer_idx]
        layer.initialize_gqa_sidecar(
            x.shape[0],
            self.num_kv_heads,
            self.sidecar_capacity,
            self.head_size,
            dtype=x.dtype,
            device=x.device,
        )
        state, elapsed = _gdn_cache_states(
            cache, self.layer_idx, x, self.kernel_heads, self.head_size
        )
        if not torch.equal(elapsed, elapsed[:1].expand_as(elapsed)):
            raise RuntimeError("bounded GQA cache requires equal sequence lengths in a batch")
        start = int(elapsed[0].item())
        positions = start + torch.arange(x.shape[1], device=x.device)
        position_ids = positions.view(1, -1).expand(x.shape[0], -1)
        query, key, value, gate = self._project_qkv(x, position_ids)
        feature_query, _ = self._features(query, key)
        beta = torch.sigmoid(self.beta_logit.float()).view(1, self.num_heads)
        alpha = 1 - beta
        outputs = []
        exact_mass = x.new_zeros((), dtype=torch.float32)
        linear_mass = x.new_zeros((), dtype=torch.float32)
        sink_mass = x.new_zeros((), dtype=torch.float32)
        recent_mass = x.new_zeros((), dtype=torch.float32)
        heavy_mass = x.new_zeros((), dtype=torch.float32)
        denominators = []
        for offset in range(x.shape[1]):
            position = start + offset
            evicted_position, evicted_key, evicted_value = _admit_sidecar_token(
                layer.gqa_positions,
                layer.gqa_scores,
                position,
                sink_slots=self.sink_slots,
                recent_slots=self.recent_slots,
                heavy_slots=self.heavy_slots,
                keys=layer.gqa_keys,
                values=layer.gqa_values,
                current_key=key[:, :, offset],
                current_value=value[:, :, offset],
            )
            valid_eviction = evicted_position >= 0
            evicted_key_heads = _repeat_gqa(evicted_key.unsqueeze(2), self.num_kv_groups)
            write_key = self._feature(evicted_key_heads, self.feature_k_weight).squeeze(2)
            write_key = write_key * valid_eviction[:, None, None]
            write_key = F.pad(write_key, (0, self.head_size - self.feature_output_dim))
            read = F.pad(
                feature_query[:, :, offset],
                (0, self.head_size - self.feature_output_dim),
            )
            value_heads = _repeat_gqa(evicted_value.unsqueeze(2), self.num_kv_groups).squeeze(2)
            value_heads = value_heads * valid_eviction[:, None, None]
            denominator_value = torch.ones_like(value_heads) * valid_eviction[:, None, None]
            read = (
                torch.stack((read, read), dim=2)
                .reshape(x.shape[0], self.kernel_heads, self.head_size)
                .contiguous()
            )
            write_key = torch.stack((write_key, write_key), dim=2).reshape_as(read).contiguous()
            write_value = (
                torch.stack((value_heads, denominator_value), dim=2).reshape_as(read).contiguous()
            )
            decay = torch.full_like(read, GQA_TAIL_DECAY_LOGITS)
            erase = torch.zeros_like(read)
            offsets, indices, ticket = cache.recurrent_metadata(flash, x.shape[0], 1, x.device)
            raw = flash.infer_recurrent_fp16_forward_varlen(
                read,
                decay,
                write_key,
                write_value,
                erase,
                erase,
                state_pool=state,
                elapsed_state_pool=elapsed,
                cu_seqlens=offsets,
                state_indices=indices,
                max_seqlen=1,
                validated_metadata=ticket,
            ).view(x.shape[0], self.num_heads, 2, self.head_size)
            tail_numerator = raw[:, :, 0].float()
            tail_denominator = raw[:, :, 1].float().mean(-1)
            slot_index = layer.gqa_positions.clamp_min(0).to(torch.long)
            gather = slot_index[:, None, :, None].expand(-1, self.num_kv_heads, -1, self.head_size)
            exact_key = torch.gather(layer.gqa_keys, 2, gather)
            exact_value = torch.gather(layer.gqa_values, 2, gather)
            exact_key = _repeat_gqa(exact_key, self.num_kv_groups)
            exact_value = _repeat_gqa(exact_value, self.num_kv_groups)
            logits = (
                torch.einsum(
                    "bhd,bhsd->bhs",
                    query[:, :, offset].float(),
                    exact_key.float(),
                )
                * self.scaling
            )
            valid = layer.gqa_positions[:, None] >= 0
            logits = logits.masked_fill(~valid, -torch.inf)
            maximum = logits.amax(-1, keepdim=True)
            kernel = torch.exp(logits - maximum).masked_fill(~valid, 0)
            exact_denominator = kernel.sum(-1)
            exact_numerator = torch.einsum("bhs,bhsd->bhd", kernel, exact_value.float())
            raw_denominator = beta * exact_denominator + alpha * tail_denominator
            denominators.append(raw_denominator)
            denominator = raw_denominator.clamp_min(1e-12)
            output = (
                beta[..., None] * exact_numerator + alpha[..., None] * tail_numerator
            ) / denominator[..., None]
            outputs.append(output.to(value.dtype))
            exact_probability = beta[..., None] * kernel / denominator[..., None]
            layer.gqa_scores.add_(exact_probability.detach().mean(1))
            exact_mass = exact_mass + exact_probability.sum(-1).mean()
            linear_mass = linear_mass + (alpha * tail_denominator / denominator).mean()
            sink_mass = sink_mass + exact_probability[..., : self.sink_slots].sum(-1).mean()
            recent_mass = (
                recent_mass
                + exact_probability[..., self.sink_slots : self.sink_slots + self.recent_slots]
                .sum(-1)
                .mean()
            )
            heavy_mass = (
                heavy_mass
                + exact_probability[..., self.sink_slots + self.recent_slots :].sum(-1).mean()
            )
        _require_positive_finite(
            "bounded Hedgehog inference shared denominator",
            torch.stack(denominators, dim=-1),
        )
        _validate_live_sidecar(
            layer.gqa_positions,
            layer.gqa_scores,
            start + x.shape[1] - 1,
        )
        final_valid = layer.gqa_positions >= 0
        length = x.shape[1]
        self._last_sidecar_metrics = {
            "exact_mass": (exact_mass / length).detach(),
            "linear_mass": (linear_mass / length).detach(),
            "sink_mass": (sink_mass / length).detach(),
            "recent_mass": (recent_mass / length).detach(),
            "heavy_mass": (heavy_mass / length).detach(),
            "sink_count": final_valid[:, : self.sink_slots].sum(-1).float().mean(),
            "recent_count": final_valid[:, self.sink_slots : self.sink_slots + self.recent_slots]
            .sum(-1)
            .float()
            .mean(),
            "heavy_count": final_valid[:, self.sink_slots + self.recent_slots :]
            .sum(-1)
            .float()
            .mean(),
        }
        heads = torch.stack(outputs, dim=2)
        self._last_attention_heads = heads
        mixed = heads.transpose(1, 2).reshape(*x.shape)
        mixed = mixed * torch.sigmoid(gate)
        return self.o_proj(mixed), v_first

    def forward(self, x, v_first=None, past_key_values=None, attention_mask=None):
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("packed Qwen2RWKV batches cannot contain padding")
        if self.training:
            return self._training_forward(x, v_first)
        if not isinstance(past_key_values, Qwen2RWKVCache):
            raise TypeError("inference requires Qwen2RWKVCache")
        return self._inference_forward(x, v_first, past_key_values)


class Qwen2RWKVDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2RWKVConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if config.source_layer_types[layer_idx] == GDN:
            self.tmix = Qwen2RWKVGatedDeltaNet(config, layer_idx)
        else:
            self.tmix = Qwen2RWKVTimeMix(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, v_first=None, past_key_values=None, attention_mask=None):
        residual = hidden_states
        mixed, v_first = self.tmix(
            self.input_layernorm(hidden_states), v_first, past_key_values, attention_mask
        )
        hidden_states = residual + mixed
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if past_key_values is not None:
            past_key_values.layers[self.layer_idx].mark_updated(hidden_states.shape[1])
        return hidden_states, v_first


class Qwen2RWKVPreTrainedModel(PreTrainedModel):
    config_class = Qwen2RWKVConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2RWKVDecoderLayer"]
    _is_stateful = True
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, Qwen2RWKVTimeMix):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class Qwen2RWKVModel(Qwen2RWKVPreTrainedModel):
    def __init__(self, config: Qwen2RWKVConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Qwen2RWKVDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids and inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, Qwen2RWKVCache):
            raise TypeError("past_key_values must be Qwen2RWKVCache")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        cache = None if self.training else (past_key_values or Qwen2RWKVCache(self.config))
        v_first = None
        for layer in self.layers:
            hidden, v_first = layer(hidden, v_first, cache, attention_mask)
        hidden = self.norm(hidden)
        if cache is not None:
            cache.v_first = v_first
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=cache if use_cache else None
        )
        return result if return_dict else (hidden, result.past_key_values)


class Qwen2RWKVForCausalLM(Qwen2RWKVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen2RWKVConfig):
        super().__init__(config)
        self.model = Qwen2RWKVModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, past_key_values=None, use_cache=None, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": self.config.use_cache if use_cache is None else use_cache,
            "logits_to_keep": kwargs.get("logits_to_keep", 1),
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids, attention_mask, past_key_values, inputs_embeds, use_cache, True, **kwargs
        )
        hidden = outputs.last_hidden_state
        index = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep
            else slice(None)
        )
        logits = self.lm_head(hidden[:, index])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values
        )
        return (
            result
            if return_dict
            else tuple(x for x in (loss, logits, outputs.past_key_values) if x is not None)
        )


__all__ = ["Qwen2RWKVCache", "Qwen2RWKVConfig", "Qwen2RWKVForCausalLM"]
