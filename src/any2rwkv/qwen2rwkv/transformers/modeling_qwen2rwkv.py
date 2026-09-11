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
GQA_FEATURE_PROJECTION_DIM = 64
GQA_FEATURE_OUTPUT_DIM = 128
GQA_STATES_PER_QUERY_HEAD = 2
GQA_READOUT_MODE = "rwkv_feature_state"
GQA_CHECKPOINT_SCHEMA = "gqa_rwkv_feature_state_d256x2_v2"
GQA_DECAY_LOGITS = -30.0
# FlashRWKV2 keeps recurrent state in the operator dtype.  A shared scale is
# applied to numerator and denominator writes so the ratio is unchanged while
# long all-token prefixes stay inside BF16/FP16 range.
GQA_STATE_SCALE = 1.0 / 256.0
CLAMP_W_EPSILON = 1e-4
W_SCALE = -math.exp(-0.5)


class _FP32RotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """Keep RoPE frequencies in FP32 across model-wide dtype conversions."""

    def _build_inv_freq(self, device: torch.device) -> torch.Tensor:
        base = self.config.rope_parameters["rope_theta"]
        partial_rotary_factor = self.config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        return (
            1.0
            / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        )

    def _apply(self, fn, recurse: bool = True):
        probe = torch.empty(0, device=self.inv_freq.device, dtype=torch.float32)
        device = fn(probe).device
        result = super()._apply(fn, recurse=recurse)
        value = self._build_inv_freq(device)
        self._buffers["inv_freq"] = value
        self._buffers["original_inv_freq"] = value.clone()
        return result

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``from_pretrained(low_cpu_mem_usage=True)`` may materialize the
        # non-persistent inherited buffers through ``to_empty``.  Build the
        # frequencies from the immutable config on every call so RoPE never
        # depends on that transient buffer contents.
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self._build_inv_freq(x.device)
        inv_freq_expanded = (
            inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2RWKVConfig(Qwen3_5TextConfig):
    """The one supported Qwen3.5-2B text geometry."""

    model_type = "qwen2rwkv"
    source_layer_types: list[str] | None = None
    gdn_mode: str = GDN_MODE
    gdn_checkpoint_schema: str = GDN_CHECKPOINT_SCHEMA
    gqa_feature_projection_dim: int = GQA_FEATURE_PROJECTION_DIM
    gqa_feature_output_dim: int = GQA_FEATURE_OUTPUT_DIM
    gqa_states_per_query_head: int = GQA_STATES_PER_QUERY_HEAD
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
        )
        expected = (
            GQA_FEATURE_PROJECTION_DIM,
            GQA_FEATURE_OUTPUT_DIM,
            GQA_STATES_PER_QUERY_HEAD,
        )
        if geometry != expected:
            raise ValueError(
                f"unsupported RWKV feature-state GQA geometry {geometry}; expected {expected}"
            )
        super().__post_init__(**kwargs)

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
            legacy_fields = {
                "gqa_num_experts",
                "gqa_states_per_expert",
                "gqa_router_low_rank_dim",
                "gqa_sidecar_capacity",
                "gqa_sink_slots",
                "gqa_recent_slots",
                "gqa_heavy_slots",
                "beta",
                "beta_logit",
                "gqa_beta",
                "gqa_beta_logit",
            }.intersection(config_dict)
            if legacy_fields:
                raise ValueError(
                    f"bounded-Hedgehog GQA artifact contains removed fields {sorted(legacy_fields)}"
                )
            if config_dict.get("gqa_readout_mode") != GQA_READOUT_MODE:
                raise ValueError(
                    "bounded-Hedgehog GQA artifact is incompatible with the pure RWKV "
                    f"feature-state runtime; expected readout mode {GQA_READOUT_MODE!r}"
                )
            if config_dict.get("gqa_checkpoint_schema") != GQA_CHECKPOINT_SCHEMA:
                raise ValueError(
                    "bounded-Hedgehog GQA artifact is incompatible with the pure RWKV "
                    f"feature-state checkpoint schema {GQA_CHECKPOINT_SCHEMA!r}"
                )
            missing = {
                "gqa_feature_projection_dim",
                "gqa_feature_output_dim",
                "gqa_states_per_query_head",
                "gqa_readout_mode",
                "gqa_checkpoint_schema",
            }.difference(config_dict)
            if missing:
                raise ValueError(
                    "legacy GQA artifact is incompatible with the pure RWKV feature-state "
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
        return -1

    def mark_updated(self, length: int) -> None:
        self.cumulative_length += length
        self.has_previous_state[0] = True

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.is_conv_states_initialized[0]:
            self.conv_states[0] = self.conv_states[0].repeat_interleave(repeats, 0)
        if self.is_recurrent_states_initialized[0]:
            self.recurrent_states[0] = self.recurrent_states[0].repeat_interleave(repeats, 0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.is_conv_states_initialized[0]:
            self.conv_states[0] = self.conv_states[0].index_select(0, indices.to(self.device))
        if self.is_recurrent_states_initialized[0]:
            self.recurrent_states[0] = self.recurrent_states[0].index_select(
                0, indices.to(self.device)
            )


class Qwen2RWKVCache(Cache):
    """Per-layer convolution/recurrent state and elapsed cursors."""

    def __init__(self, config: Qwen2RWKVConfig):
        super().__init__(layers=[Qwen2RWKVCacheLayer() for _ in range(config.num_hidden_layers)])
        self.config = config
        self.elapsed: list[torch.Tensor | None] = [None] * config.num_hidden_layers
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

    def mark_updated(self, layer_idx: int, length: int) -> None:
        self.layers[layer_idx].mark_updated(length)
        elapsed = self.elapsed[layer_idx]
        if elapsed is None:
            raise RuntimeError("recurrent operator did not initialize elapsed state")
        elapsed.add_(length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        self.elapsed = [
            None if x is None else x.repeat_interleave(repeats, 0) for x in self.elapsed
        ]
        self._metadata_key = None
        self._metadata = None

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        self.elapsed = [
            None if x is None else x.index_select(0, indices.to(x.device)) for x in self.elapsed
        ]
        self._metadata_key = None
        self._metadata = None

    def reset(self) -> None:
        super().reset()
        for value in self.elapsed:
            if value is not None:
                value.zero_()
        self._metadata_key = None
        self._metadata = None


def _recurrent_cache_states(
    cache: Qwen2RWKVCache, layer_idx: int, x: torch.Tensor, heads: int, dim: int
):
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
        state, elapsed = _recurrent_cache_states(
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


class Qwen2RWKVTimeMix(nn.Module):
    """Pure positive-feature RWKV state used for converted GQA layers."""

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
            torch.zeros(self.num_heads, self.head_size, self.feature_projection_dim)
        )
        self.feature_k_weight = nn.Parameter(torch.zeros_like(self.feature_q_weight))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for weight in (self.feature_q_weight, self.feature_k_weight):
                weight.zero_()
                for head in weight:
                    nn.init.eye_(head)

    def load_source_attention(self, source: nn.Module) -> None:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
        self.reset_parameters()

    def attention_transfer_parameters(self) -> list[nn.Parameter]:
        return [self.feature_q_weight, self.feature_k_weight]

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
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = self.rotary_emb(x, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value, gate.reshape(batch, length, -1)

    def _feature(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("bhtd,hdf->bhtf", value.float(), weight.float())
        scale = projected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        positive = projected.relu()
        negative = (-projected).relu()
        positive = positive / scale
        negative = negative / scale
        feature = torch.cat((positive, negative), dim=-1) + 1e-4
        return feature.to(value.dtype)

    def _features(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_heads = _repeat_gqa(key, self.num_kv_groups)
        return (
            self._feature(query, self.feature_q_weight),
            self._feature(key_heads, self.feature_k_weight),
        )

    def _state_reference(
        self,
        feature_query: torch.Tensor,
        feature_key: torch.Tensor,
        value_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, length, _ = feature_query.shape
        numerator_state = torch.zeros(
            batch,
            heads,
            self.head_size,
            self.feature_output_dim,
            dtype=torch.float32,
            device=feature_query.device,
        )
        denominator_state = torch.zeros(
            batch,
            heads,
            self.feature_output_dim,
            dtype=torch.float32,
            device=feature_query.device,
        )
        numerator = torch.empty(
            batch, heads, length, self.head_size, dtype=torch.float32, device=feature_query.device
        )
        denominator = torch.empty(
            batch, heads, length, dtype=torch.float32, device=feature_query.device
        )
        for token in range(length):
            key_token = feature_key[:, :, token].float()
            value_token = value_heads[:, :, token].float()
            numerator_state = numerator_state + value_token.unsqueeze(-1) * key_token.unsqueeze(-2)
            denominator_state = denominator_state + key_token
            read = feature_query[:, :, token].float()
            numerator[:, :, token] = torch.einsum("bhdf,bhf->bhd", numerator_state, read)
            denominator[:, :, token] = torch.einsum("bhf,bhf->bh", denominator_state, read)
        return numerator, denominator

    def _state_training(
        self,
        feature_query: torch.Tensor,
        feature_key: torch.Tensor,
        value_heads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flash = _flash("training", feature_query)
        batch, _, length, _ = feature_query.shape
        if feature_query.dtype != torch.bfloat16 or length % 16:
            raise RuntimeError(
                "GQA training requires contiguous BF16 [B,T,2048] with T divisible by 16"
            )
        query_padded = F.pad(feature_query, (0, self.head_size - self.feature_output_dim))
        key_padded = F.pad(feature_key, (0, self.head_size - self.feature_output_dim))
        value_write = value_heads * GQA_STATE_SCALE
        denominator_write = torch.ones_like(value_write) * GQA_STATE_SCALE
        read = torch.stack((query_padded, query_padded), dim=3)
        key = torch.stack((key_padded, key_padded), dim=3)
        value = torch.stack((value_write, denominator_write), dim=3)
        shape = (batch, length, self.recurrent_width)
        read = read.permute(0, 2, 1, 3, 4).reshape(shape).contiguous()
        key = key.permute(0, 2, 1, 3, 4).reshape(shape).contiguous()
        value = value.permute(0, 2, 1, 3, 4).reshape(shape).contiguous()
        decay = torch.full_like(read, GQA_DECAY_LOGITS)
        erase = torch.zeros_like(read)
        raw = flash.pretrain_recurrent_bf16(
            read, decay, key, value, erase, erase, head_size=self.head_size
        ).view(batch, length, self.num_heads, 2, self.head_size)
        numerator = raw[..., 0, :].permute(0, 2, 1, 3).float() / GQA_STATE_SCALE
        denominator = raw[..., 1, :].float().mean(-1).permute(0, 2, 1) / GQA_STATE_SCALE
        return numerator, denominator

    def _readout(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        heads = numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
        mixed = heads.transpose(1, 2).reshape(gate.shape[0], gate.shape[1], -1)
        mixed = (mixed * torch.sigmoid(gate).float()).to(self.o_proj.weight.dtype)
        return self.o_proj(mixed)

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
        value_heads = _repeat_gqa(value, self.num_kv_groups)
        numerator, denominator = self._state_reference(feature_query, feature_key, value_heads)
        heads = numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
        return heads, gate

    def reference_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        heads, gate = self.attention_heads_reference(x, position_ids)
        mixed = heads.transpose(1, 2).reshape(*x.shape)
        mixed = (mixed * torch.sigmoid(gate).float()).to(self.o_proj.weight.dtype)
        return self.o_proj(mixed)

    def _training_forward(
        self, x: torch.Tensor, v_first: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
        positions = positions.expand(x.shape[0], -1)
        query, key, value, gate = self._project_qkv(x, positions)
        feature_query, feature_key = self._features(query, key)
        value_heads = _repeat_gqa(value, self.num_kv_groups)
        numerator, denominator = self._state_training(feature_query, feature_key, value_heads)
        return self._readout(numerator, denominator, gate), v_first

    def _inference_forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        cache: Qwen2RWKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flash = _flash("inference", x)
        if x.dtype != torch.float16:
            raise RuntimeError("GQA inference requires a float16 checkpoint")
        batch, length, _ = x.shape
        state, elapsed = _recurrent_cache_states(
            cache, self.layer_idx, x, self.kernel_heads, self.head_size
        )
        if not torch.equal(elapsed, elapsed[:1].expand_as(elapsed)):
            raise RuntimeError("RWKV cache requires equal sequence lengths in a batch")
        start = int(elapsed[0].item())
        positions = start + torch.arange(length, device=x.device)
        position_ids = positions.view(1, -1).expand(batch, -1)
        query, key, value, gate = self._project_qkv(x, position_ids)
        feature_query, feature_key = self._features(query, key)
        value_heads = _repeat_gqa(value, self.num_kv_groups)
        query_padded = F.pad(feature_query, (0, self.head_size - self.feature_output_dim))
        key_padded = F.pad(feature_key, (0, self.head_size - self.feature_output_dim))
        value_heads = value_heads * GQA_STATE_SCALE
        denominator_value = torch.ones_like(value_heads) * GQA_STATE_SCALE
        read = torch.stack((query_padded, query_padded), dim=3).permute(0, 2, 1, 3, 4)
        write_key = torch.stack((key_padded, key_padded), dim=3).permute(0, 2, 1, 3, 4)
        write_value = torch.stack((value_heads, denominator_value), dim=3).permute(0, 2, 1, 3, 4)
        read = read.reshape(batch * length, self.kernel_heads, self.head_size).contiguous()
        write_key = write_key.reshape_as(read).contiguous()
        write_value = write_value.reshape_as(read).contiguous()
        decay = torch.full_like(read, GQA_DECAY_LOGITS)
        erase = torch.zeros_like(read)
        offsets, indices, ticket = cache.recurrent_metadata(flash, batch, length, x.device)
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
            max_seqlen=length,
            validated_metadata=ticket,
        ).view(batch, length, self.num_heads, 2, self.head_size)
        numerator = raw[..., 0, :].permute(0, 2, 1, 3).float()
        denominator = raw[..., 1, :].float().mean(-1).permute(0, 2, 1)
        output = self._readout(numerator, denominator, gate)
        return output, v_first

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
            past_key_values.mark_updated(self.layer_idx, hidden_states.shape[1])
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
