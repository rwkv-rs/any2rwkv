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
from transformers.models.rwkv.configuration_rwkv import RwkvConfig
from transformers.models.rwkv.modeling_rwkv import RwkvTimeMix

GDN = "linear_attention"
GQA = "full_attention"
_SOURCE_TYPES = [GDN, GDN, GDN, GQA] * 6
GDN_MODE = "source_shell_wkv7"
GDN_CHECKPOINT_SCHEMA = "source_gdn_state_dict_v1"
GQA_FEATURE_PROJECTION_DIM = 64
GQA_FEATURE_OUTPUT_DIM = 128
GQA_STATES_PER_QUERY_HEAD = 2
GQA_READOUT_MODE = "rwkv_feature_state"
GQA_CHECKPOINT_SCHEMA = "gqa_rwkv_tmix_d256x2_v3"
GQA_DECAY_LOGITS = -30.0
# FlashRWKV2 keeps recurrent state in the operator dtype.  A shared scale is
# applied to numerator and denominator writes so the ratio is unchanged while
# long all-token prefixes stay inside BF16/FP16 range.
GQA_STATE_SCALE = 1.0 / 256.0
CLAMP_W_EPSILON = 1e-4
W_SCALE = -math.exp(-0.5)
CANONICAL_HEAD_SIZE = 128
CANONICAL_HEADS = 16
CANONICAL_DECAY_LOW_RANK_DIM = 128
CANONICAL_A_LOW_RANK_DIM = 32
CANONICAL_V_LOW_RANK_DIM = 32
CANONICAL_GATE_LOW_RANK_DIM = 2048


def _canonical_reference_spec(hidden_size: int = 2048) -> dict[str, tuple[int, ...]]:
    """Build the upstream parameter contract with the deliberate r_k reshape exception."""
    reference_config = RwkvConfig(
        hidden_size=hidden_size,
        head_size=64,
        num_hidden_layers=24,
        vocab_size=248320,
        decay_low_rank_dim=CANONICAL_DECAY_LOW_RANK_DIM,
        a_low_rank_dim=CANONICAL_A_LOW_RANK_DIM,
        v_low_rank_dim=CANONICAL_V_LOW_RANK_DIM,
        gate_low_rank_dim=CANONICAL_GATE_LOW_RANK_DIM,
    )
    reference = RwkvTimeMix(reference_config, 0)
    return {name: tuple(value.shape) for name, value in reference.state_dict().items()}


class _FP32RotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """Keep RoPE frequencies in FP32 across model-wide dtype conversions."""

    def _build_inv_freq(self, device: torch.device) -> torch.Tensor:
        base = self.config.rope_parameters["rope_theta"]
        partial_rotary_factor = self.config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

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
    tmix_schema: str = "source_shell_v1"
    scaffold_stage: list[int] | None = None
    decay_low_rank_dim: int = CANONICAL_DECAY_LOW_RANK_DIM
    a_low_rank_dim: int = CANONICAL_A_LOW_RANK_DIM
    v_low_rank_dim: int = CANONICAL_V_LOW_RANK_DIM
    gate_low_rank_dim: int = CANONICAL_GATE_LOW_RANK_DIM

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
        if self.scaffold_stage is None:
            self.scaffold_stage = [0] * self.num_hidden_layers

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
                    "incompatible GQA checkpoint schema "
                    f"{config_dict.get('gqa_checkpoint_schema')!r}; "
                    f"expected {GQA_CHECKPOINT_SCHEMA!r}"
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
    cache: Qwen2RWKVCache,
    layer_idx: int,
    x: torch.Tensor,
    heads: int,
    dim: int,
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
        return self._source_boundary(raw.view_as(value), z), value.flatten(2)

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
        return self._source_boundary(raw, z), value.flatten(2)

    def forward(self, x, v_first=None, past_key_values=None, attention_mask=None):
        if self.training:
            output, value = self._training_forward(x, attention_mask)
        else:
            if not isinstance(past_key_values, Qwen2RWKVCache):
                raise TypeError("inference requires Qwen2RWKVCache")
            output, value = self._inference_forward(x, past_key_values, attention_mask)
        return output, value if v_first is None else v_first


class ScaffoldTimeMix(nn.Module):
    """S1/T1 homotopy wrapper around the canonical RWKV-7 TMix.

    The source module is deliberately retained as a child module during the
    scaffold stages.  At the two endpoints the wrapper uses one complete
    reference path: this makes the lambda-zero identity and the lambda-one
    export independently testable before any distillation is run.
    """

    def __init__(
        self,
        config: Qwen2RWKVConfig,
        layer_idx: int,
        *,
        origin: str,
        source: nn.Module,
        teacher: nn.Module | None = None,
    ):
        super().__init__()
        if origin not in (GDN, GQA):
            raise ValueError(f"unknown scaffold origin {origin!r}")
        if config.hidden_size != 2048:
            raise ValueError("ScaffoldTimeMix currently supports hidden_size=2048 only")
        self.config = config
        self.layer_idx = layer_idx
        self.origin = origin
        self.heads = CANONICAL_HEADS
        self.head_size = CANONICAL_HEAD_SIZE
        self.channels = config.hidden_size
        self.source = source
        # The frozen teacher is a construction-time aid and is intentionally
        # not part of the scaffold checkpoint state dict.
        self.__dict__["_teacher"] = teacher
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.zeros(1, 1, self.channels)))
        self.w0 = nn.Parameter(torch.zeros(1, 1, self.channels))
        self.w1 = nn.Parameter(torch.zeros(self.channels, CANONICAL_DECAY_LOW_RANK_DIM))
        self.w2 = nn.Parameter(torch.zeros(CANONICAL_DECAY_LOW_RANK_DIM, self.channels))
        self.a0 = nn.Parameter(torch.zeros(1, 1, self.channels))
        self.a1 = nn.Parameter(torch.zeros(self.channels, CANONICAL_A_LOW_RANK_DIM))
        self.a2 = nn.Parameter(torch.zeros(CANONICAL_A_LOW_RANK_DIM, self.channels))
        self.v0 = nn.Parameter(torch.zeros(1, 1, self.channels))
        self.v1 = nn.Parameter(torch.zeros(self.channels, CANONICAL_V_LOW_RANK_DIM))
        self.v2 = nn.Parameter(torch.zeros(CANONICAL_V_LOW_RANK_DIM, self.channels))
        self.g1 = nn.Parameter(torch.zeros(self.channels, CANONICAL_GATE_LOW_RANK_DIM))
        self.g2 = nn.Parameter(torch.zeros(CANONICAL_GATE_LOW_RANK_DIM, self.channels))
        self.k_k = nn.Parameter(torch.ones(1, 1, self.channels))
        self.k_a = nn.Parameter(torch.zeros(1, 1, self.channels))
        self.r_k = nn.Parameter(torch.zeros(self.heads, self.head_size))
        self.receptance = nn.Linear(self.channels, self.channels, bias=False)
        self.key = nn.Linear(self.channels, self.channels, bias=False)
        self.value = nn.Linear(self.channels, self.channels, bias=False)
        self.output = nn.Linear(self.channels, self.channels, bias=False)
        self.ln_x = nn.GroupNorm(self.heads, self.channels, eps=64e-5)

        if origin == GDN:
            lambda_names = (
                "lambda_front",
                "lambda_rho",
                "lambda_gamma",
                "lambda_erase",
                "lambda_decay",
                "lambda_norm",
                "lambda_gate",
            )
        else:
            lambda_names = ("lambda_den", "lambda_rope", "lambda_qk", "lambda_phi")
        for name in lambda_names:
            self.register_buffer(name, torch.zeros(()), persistent=False)
        self._assert_canonical_contract()

    @staticmethod
    def canonical_state_spec() -> dict[str, tuple[int, ...]]:
        return _canonical_reference_spec()

    def _assert_canonical_contract(self) -> None:
        expected = self.canonical_state_spec()
        actual = {
            name: tuple(value.shape)
            for name, value in self.state_dict().items()
            if not name.startswith("source.")
        }
        # source.* is the only non-canonical state prefix; the nonpersistent
        # lambda buffers do not enter state_dict().
        missing = set(expected).difference(actual)
        if missing:
            raise RuntimeError(f"scaffold canonical parameters are missing {sorted(missing)}")
        for name, shape in expected.items():
            if name == "r_k":
                if self.r_k.numel() != math.prod(shape):
                    raise RuntimeError("canonical r_k has the wrong number of elements")
            elif actual[name] != shape:
                raise RuntimeError(
                    f"canonical shape mismatch for {name}: {actual[name]} != {shape}"
                )

    def set_lambdas(self, **values: float) -> None:
        for name, value in values.items():
            if not name.startswith("lambda_") or not hasattr(self, name):
                raise ValueError(f"unknown scaffold lambda {name!r}")
            tensor = getattr(self, name)
            tensor.fill_(float(value))

    def lambda_values(self) -> dict[str, float]:
        return {
            name: float(value) for name, value in self.named_buffers() if name.startswith("lambda_")
        }

    def _all_lambdas(self, value: float) -> bool:
        return all(abs(current - value) <= 1e-7 for current in self.lambda_values().values())

    @staticmethod
    def _shift(x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        return x + parameter.to(x.dtype) * (previous - x)

    def _source_gdn_reference(self, x: torch.Tensor):
        source = self.source
        if x.is_cuda and x.dtype == torch.bfloat16 and hasattr(source, "_training_forward"):
            was_training = source.training
            source.train()
            try:
                output, value = source._training_forward(
                    x, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                )
            finally:
                source.train(was_training)
            return output.float(), value.float()
        batch, length, _ = x.shape
        mixed = causal_conv1d_fn(
            source.in_proj_qkv(x).transpose(1, 2),
            source.conv1d.weight.squeeze(1),
            source.conv1d.bias,
            activation=source.activation,
        )[:, :, :length].transpose(1, 2)
        query, key, value = torch.split(
            mixed, (source.key_dim, source.key_dim, source.value_dim), dim=-1
        )
        query = query.view(batch, length, source.num_k_heads, source.head_k_dim)
        key = key.view(batch, length, source.num_k_heads, source.head_k_dim)
        value = value.view(batch, length, source.num_v_heads, source.head_v_dim)
        query = query * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)
        if source.num_v_heads != source.num_k_heads:
            repeats = source.num_v_heads // source.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        beta = torch.sigmoid(source.in_proj_b(x)).float()
        log_decay = -source.A_log.float().exp() * F.softplus(
            source.in_proj_a(x).float() + source.dt_bias.float()
        )
        ratio = (log_decay / W_SCALE).clamp(CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON)
        log_decay = W_SCALE * ratio
        state = torch.zeros(
            batch,
            source.num_v_heads,
            source.head_v_dim,
            source.head_v_dim,
            dtype=torch.float32,
            device=x.device,
        )
        outputs = []
        for token in range(length):
            direction = key[:, token].float()
            memory = torch.einsum("bhk,bhkv->bhv", direction, state)
            state = (
                state * log_decay[:, token].float().exp()[..., None, None]
                - (beta[:, token] * log_decay[:, token].float().exp())[..., None, None]
                * torch.einsum("bhk,bhv->bhkv", direction, memory)
                + torch.einsum(
                    "bhk,bhv->bhkv", direction, beta[:, token, :, None] * value[:, token].float()
                )
            )
            outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, token].float(), state))
        raw = torch.stack(outputs, dim=1).to(x.dtype)
        gate = source.in_proj_z(x).view(batch, length, source.num_v_heads, source.head_v_dim)
        boundary = source.norm(
            raw.reshape(-1, source.head_v_dim), gate.reshape(-1, source.head_v_dim)
        ).view(batch, length, source.value_dim)
        return source.out_proj(boundary).float(), value.reshape(batch, length, -1).float()

    def _source_forward(
        self, x: torch.Tensor, v_first=None, position_ids=None, past_key_values=None
    ):
        if (
            past_key_values is not None
            and not isinstance(past_key_values, dict)
            and hasattr(self.source, "forward")
        ):
            attention_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            return self.source(
                x,
                v_first=v_first,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
        if self.origin == GDN:
            return self._source_gdn_reference(x)
        source = self.source
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and hasattr(source, "_training_forward")
        ):
            was_training = source.training
            source.train()
            try:
                return source._training_forward(x, v_first)
            finally:
                source.train(was_training)
        if hasattr(source, "reference_forward"):
            if position_ids is None:
                position_ids = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
            return source.reference_forward(x, position_ids, v_first), None
        raise RuntimeError("GQA scaffold source must provide reference_forward")

    def _canonical_inputs(self, x: torch.Tensor, v_first=None, previous=None):
        if previous is None:
            shifts = {
                name: self._shift(x, getattr(self, f"x_{name}"))
                for name in ("r", "w", "k", "v", "a", "g")
            }
        else:
            shifts = {
                name: x + getattr(self, f"x_{name}").to(x.dtype) * (previous - x)
                for name in ("r", "w", "k", "v", "a", "g")
            }
        read = F.linear(shifts["r"], self.receptance.weight)
        key = F.linear(shifts["k"], self.key.weight)
        value = F.linear(shifts["v"], self.value.weight)
        if self.layer_idx == 0:
            first = value
        else:
            first = value if v_first is None else v_first
            residual = torch.sigmoid(self.v0 + torch.tanh(shifts["v"] @ self.v1) @ self.v2)
            value = value + residual * (first - value)
        decay_state = self.w0 + torch.tanh(shifts["w"] @ self.w1) @ self.w2
        log_decay = W_SCALE * torch.sigmoid(decay_state)
        learning_rate = torch.sigmoid(self.a0 + torch.tanh(shifts["a"] @ self.a1) @ self.a2)
        direction = F.normalize(
            key.view(*key.shape[:2], self.heads, self.head_size).float()
            * self.k_k.view(1, 1, self.heads, self.head_size).float(),
            dim=-1,
        )
        write_key = key.view(*key.shape[:2], self.heads, self.head_size) * (
            1
            + (learning_rate.view(*learning_rate.shape[:2], self.heads, self.head_size) - 1)
            * self.k_a.view(1, 1, self.heads, self.head_size)
        )
        return (
            read.view(*read.shape[:2], self.heads, self.head_size),
            direction,
            learning_rate.view(*learning_rate.shape[:2], self.heads, self.head_size),
            write_key,
            value.view(*value.shape[:2], self.heads, self.head_size),
            log_decay.view(*log_decay.shape[:2], self.heads, self.head_size),
            shifts["g"],
            first,
        )

    def _canonical_scan(self, inputs):
        read, direction, learning_rate, write_key, value, log_decay, _gate, first = inputs
        batch, length, heads, width = read.shape
        if read.is_cuda and read.dtype == torch.bfloat16:
            try:
                flash = _flash("training", read)
                decay_logits = torch.logit(
                    (log_decay.float() / W_SCALE).clamp(CLAMP_W_EPSILON, 1 - CLAMP_W_EPSILON)
                ).to(read.dtype)
                raw = flash.pretrain_recurrent_bf16(
                    read.reshape(batch, length, -1).contiguous(),
                    decay_logits.reshape(batch, length, -1).contiguous(),
                    write_key.reshape(batch, length, -1).contiguous(),
                    value.reshape(batch, length, -1).contiguous(),
                    (-direction).to(read.dtype).reshape(batch, length, -1).contiguous(),
                    (direction * learning_rate)
                    .to(read.dtype)
                    .reshape(batch, length, -1)
                    .contiguous(),
                    head_size=self.head_size,
                )
                return raw.view(batch, length, heads, width).float(), first
            except (RuntimeError, AttributeError):
                pass
        state = torch.zeros(batch, heads, width, width, dtype=torch.float32, device=read.device)
        raw = []
        for token in range(length):
            direction_t = direction[:, token].float()
            memory = torch.einsum("bhk,bhkv->bhv", direction_t, state)
            state = (
                state * log_decay[:, token].float().exp().unsqueeze(-1)
                - torch.einsum(
                    "bhk,bhv->bhkv",
                    direction_t * learning_rate[:, token].float(),
                    memory,
                )
                + torch.einsum(
                    "bhk,bhv->bhkv", write_key[:, token].float(), value[:, token].float()
                )
            )
            raw.append(torch.einsum("bhk,bhkv->bhv", read[:, token].float(), state))
        return torch.stack(raw, dim=1), first

    def _canonical_readout(self, x: torch.Tensor, raw: torch.Tensor, inputs):
        read, _direction, _learning_rate, write_key, value, _log_decay, gate_input, _ = inputs
        flat = raw.reshape(x.shape[0] * x.shape[1], self.channels)
        normalized = self.ln_x(flat.to(self.ln_x.weight.dtype)).float().view_as(raw)
        shortcut = (
            read.float()
            * write_key.float()
            * self.r_k.view(1, 1, self.heads, self.head_size).float()
        ).sum(-1, keepdim=True) * value.float()
        heads = normalized + shortcut
        gate = torch.sigmoid(gate_input.float() @ self.g1.float()) @ self.g2.float()
        mixed = heads.reshape(x.shape[0], x.shape[1], self.channels) * gate
        return F.linear(mixed, self.output.weight.float()).to(x.dtype)

    def _canonical_step(self, x: torch.Tensor, v_first, cache: dict):
        inputs = self._canonical_inputs(x, v_first, cache.get("previous"))
        read, direction, learning_rate, write_key, value, log_decay, _gate, first = inputs
        state = cache.get(
            "state",
            torch.zeros(
                x.shape[0], self.heads, self.head_size, self.head_size,
                dtype=torch.float32, device=x.device
            ),
        )
        direction_t = direction[:, 0].float()
        memory = torch.einsum("bhk,bhkv->bhv", direction_t, state)
        state = (
            state * log_decay[:, 0].float().exp().unsqueeze(-1)
            - torch.einsum(
                "bhk,bhv->bhkv", direction_t * learning_rate[:, 0].float(), memory
            )
            + torch.einsum(
                "bhk,bhv->bhkv", write_key[:, 0].float(), value[:, 0].float()
            )
        )
        raw = torch.einsum("bhk,bhkv->bhv", read[:, 0].float(), state).unsqueeze(1)
        cache["state"] = state
        cache["previous"] = x[:, -1:].detach()
        cache["first"] = first.detach()
        return self._canonical_readout(x, raw, inputs), first

    def _canonical_forward(self, x: torch.Tensor, v_first=None, past_key_values=None):
        if isinstance(past_key_values, dict) and x.shape[1] == 1:
            return self._canonical_step(x, v_first, past_key_values)
        inputs = self._canonical_inputs(x, v_first)
        raw, first = self._canonical_scan(inputs)
        return self._canonical_readout(x, raw, inputs), first

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None = None,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
    ):
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("ScaffoldTimeMix requires an all-ones attention mask")
        if self._all_lambdas(0.0):
            return self._source_forward(
                hidden_states, v_first, position_ids, past_key_values
            )
        canonical, first = self._canonical_forward(
            hidden_states, v_first, past_key_values
        )
        if self._all_lambdas(1.0):
            return canonical, first
        source, source_first = self._source_forward(
            hidden_states, v_first, position_ids, past_key_values
        )
        values = list(self.lambda_values().values())
        blend = sum(values) / len(values)
        output = source.float() * (1 - blend) + canonical.float() * blend
        if first is None:
            first = source_first
        return output.to(hidden_states.dtype), first

    def load_canonical_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.canonical_state_spec())
        if set(state) != expected:
            raise ValueError(f"canonical initialization keys differ: {set(state) ^ expected}")
        with torch.no_grad():
            for name, value in state.items():
                target = self.r_k if name == "r_k" else self.get_parameter(name)
                target.copy_(value.to(device=target.device, dtype=target.dtype).reshape_as(target))

    def to_canonical(self) -> dict[str, torch.Tensor]:
        if not self._all_lambdas(1.0):
            raise RuntimeError(f"cannot export scaffold with lambdas {self.lambda_values()}")
        expected = self.canonical_state_spec()
        state = {}
        for name in expected:
            value = self.r_k if name == "r_k" else self.get_parameter(name)
            state[name] = value.detach().clone()
            if name != "r_k" and tuple(state[name].shape) != expected[name]:
                raise RuntimeError(f"canonical export shape mismatch for {name}")
        if set(state) != set(expected) or state["r_k"].numel() != math.prod(expected["r_k"]):
            raise RuntimeError("canonical export key/shape contract failed")
        return state


def _repeat_gqa(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    return tensor.repeat_interleave(groups, dim=1)


class Qwen2RWKVTimeMix(nn.Module):
    """Source GQA shell with function-preserving, learnable RWKV-7 dynamics.

    The numerator uses diagonal-plus-rank-one updates. The positive denominator
    shares the diagonal decay, but does not undergo signed state corrections.
    Source projections, RoPE and output gate remain the transfer boundary.
    """

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
        feature_channels = self.num_heads * self.feature_output_dim
        rank = 32
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
            torch.empty(self.num_heads, self.head_size, self.feature_projection_dim)
        )
        self.feature_k_weight = nn.Parameter(torch.empty_like(self.feature_q_weight))
        for name in ("r", "w", "k", "v", "a", "g"):
            setattr(self, f"x_{name}", nn.Parameter(torch.zeros(channels)))
        for name, width in (("w", feature_channels), ("a", feature_channels), ("v", channels)):
            setattr(self, f"{name}0", nn.Parameter(torch.zeros(width)))
            setattr(self, f"{name}1", nn.Parameter(torch.empty(channels, rank)))
            setattr(self, f"{name}2", nn.Parameter(torch.zeros(rank, width)))
        self.k_k = nn.Parameter(torch.ones(self.num_heads, self.feature_output_dim))
        self.k_a = nn.Parameter(torch.zeros_like(self.k_k))
        self.r_k = nn.Parameter(torch.zeros_like(self.k_k))
        self.norm_mix = nn.Parameter(torch.zeros(self.num_heads))
        self.ln_x = nn.GroupNorm(self.num_heads, channels, eps=64e-5)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for weight in (self.feature_q_weight, self.feature_k_weight):
            for head in weight:
                nn.init.eye_(head)
        for name in ("r", "w", "k", "v", "a", "g"):
            getattr(self, f"x_{name}").zero_()
        for name in ("w", "a", "v"):
            getattr(self, f"{name}0").zero_()
            nn.init.normal_(getattr(self, f"{name}1"), std=self.config.hidden_size**-0.5)
            getattr(self, f"{name}2").zero_()
        self.k_k.fill_(1)
        self.k_a.zero_()
        self.r_k.zero_()
        self.norm_mix.zero_()
        self.ln_x.weight.fill_(1)
        self.ln_x.bias.zero_()

    def load_source_attention(self, source: nn.Module) -> None:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
        self.reset_parameters()

    def attention_transfer_parameters(self) -> list[nn.Parameter]:
        return [self.feature_q_weight, self.feature_k_weight]

    @staticmethod
    def _shift_delta(x: torch.Tensor, cache_layer=None) -> torch.Tensor:
        previous = torch.zeros_like(x[:, :1])
        if cache_layer is not None and cache_layer.is_conv_states_initialized[0]:
            previous = cache_layer.conv_states[0].transpose(1, 2)
        delta = torch.cat((previous, x[:, :-1]), dim=1) - x
        if cache_layer is not None:
            last = x[:, -1:].transpose(1, 2).contiguous()
            if not cache_layer.is_conv_states_initialized[0]:
                cache_layer.lazy_initialization(conv_states=last, state_idx=0)
            cache_layer.conv_states[0].copy_(last)
        return delta

    def _project_qkv(self, x, position_ids, delta=None):
        if delta is None:
            delta = self._shift_delta(x)
        batch, length, _ = x.shape
        projected = self.q_proj(x + self.x_r * delta).view(
            batch, length, self.num_heads, 2 * self.head_size
        )
        query, gate = projected.chunk(2, dim=-1)
        # The joint source projection remains intact at initialization. The gate
        # gets its own token shift through an initially zero input correction.
        gate_weight = self.q_proj.weight.view(self.num_heads, 2, self.head_size, -1)[:, 1].reshape(
            -1, x.shape[-1]
        )
        gate = gate.reshape(batch, length, -1) + F.linear(
            (self.x_g - self.x_r) * delta, gate_weight
        )
        key = self.k_proj(x + self.x_k * delta).view(
            batch, length, self.num_kv_heads, self.head_size
        )
        value = self.v_proj(x + self.x_v * delta).view(
            batch, length, self.num_kv_heads, self.head_size
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        cos, sin = self.rotary_emb(x, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value.transpose(1, 2), gate

    def _feature(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("bhtd,hdf->bhtf", value.float(), weight.float())
        scale = projected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        feature = torch.cat((projected.relu(), (-projected).relu()), dim=-1) / scale + 1e-4
        return feature.to(value.dtype)

    def _features(self, query, key):
        return (
            self._feature(query, self.feature_q_weight),
            self._feature(_repeat_gqa(key, self.num_kv_groups), self.feature_k_weight),
        )

    def _dynamics(self, x, delta):
        # Train the bounded retention fraction, not saturated decay logits. The
        # small positive floor maps zero exactly to the previous -30 logits and
        # keeps the inverse-logit chain rule finite in FP32 at that boundary.
        fraction = (
            self.w0.float()
            + torch.tanh((x + self.x_w * delta).float() @ self.w1.float()) @ self.w2.float()
        )
        fraction = fraction.clamp(0, 1 - CLAMP_W_EPSILON) + math.exp(GQA_DECAY_LOGITS)
        decay = torch.logit(fraction).to(x.dtype)
        erase = (
            self.a0.float()
            + torch.tanh((x + self.x_a * delta).float() @ self.a1.float()) @ self.a2.float()
        ).clamp(0, 1)
        shape = (*x.shape[:2], self.num_heads, self.feature_output_dim)
        return decay.view(shape).transpose(1, 2), erase.to(x.dtype).view(shape).transpose(1, 2)

    def _inputs(self, x, positions, v_first=None, delta=None):
        if delta is None:
            delta = self._shift_delta(x)
        query, key, value, gate = self._project_qkv(x, positions, delta)
        feature_query, feature_key = self._features(query, key)
        value = _repeat_gqa(value, self.num_kv_groups)
        value, first = self._mix_values(x, delta, value, v_first)
        return feature_query, feature_key, value, gate, self._dynamics(x, delta), first

    def _mix_values(self, x, delta, value, v_first=None):
        first = value.transpose(1, 2).reshape_as(x) if v_first is None else v_first
        blend = torch.tanh(
            self.v0.float()
            + torch.sigmoid((x + self.x_v * delta).float() @ self.v1.float()) @ self.v2.float()
        )
        first_heads = first.view(*x.shape[:2], self.num_heads, self.head_size).transpose(1, 2)
        value = value + blend.view(*x.shape[:2], self.num_heads, self.head_size).transpose(1, 2).to(
            value.dtype
        ) * (first_heads - value)
        return value, first

    def _state_components(self, feature_key, controls=None):
        if controls is None:
            decay = torch.full_like(feature_key, GQA_DECAY_LOGITS)
            erase = torch.zeros_like(feature_key)
        else:
            decay, erase = controls
        key = feature_key * (1 + (erase - 1) * self.k_a[None, :, None, :])
        direction = F.normalize(
            feature_key.float() * self.k_k[None, :, None, :].float(), dim=-1
        ).to(feature_key.dtype)
        return decay, key, -direction, direction * erase

    def _state_reference(self, feature_query, feature_key, value_heads, controls=None):
        batch, heads, length, features = feature_query.shape
        decay, key, a, b = self._state_components(feature_key, controls)
        state = torch.zeros(
            batch, heads, self.head_size, features, dtype=torch.float32, device=feature_query.device
        )
        normalizer = torch.zeros(
            batch, heads, features, dtype=torch.float32, device=feature_query.device
        )
        numerator, denominator = [], []
        for token in range(length):
            retention = (W_SCALE * decay[:, :, token].float().sigmoid()).exp()
            correction = torch.einsum("bhdf,bhf->bhd", state, a[:, :, token].float())
            state = (
                state * retention.unsqueeze(-2)
                + correction.unsqueeze(-1) * b[:, :, token].float().unsqueeze(-2)
                + value_heads[:, :, token].float().unsqueeze(-1)
                * key[:, :, token].float().unsqueeze(-2)
            )
            normalizer = normalizer * retention + feature_key[:, :, token].float()
            read = feature_query[:, :, token].float()
            numerator.append(torch.einsum("bhdf,bhf->bhd", state, read))
            denominator.append((normalizer * read).sum(-1))
        return torch.stack(numerator, dim=2), torch.stack(denominator, dim=2)

    def _wkv_inputs(self, feature_query, feature_key, value_heads, controls=None):
        decay, key, a, b = self._state_components(feature_key, controls)
        pad = self.head_size - self.feature_output_dim

        def paired(left, right):
            return torch.stack((left, right), dim=3).permute(0, 2, 1, 3, 4).contiguous()

        read = F.pad(feature_query, (0, pad))
        decay = F.pad(decay, (0, pad), value=GQA_DECAY_LOGITS)
        key = F.pad(key, (0, pad))
        denominator_key = F.pad(feature_key, (0, pad))
        a, b = F.pad(a, (0, pad)), F.pad(b, (0, pad))
        zeros = torch.zeros_like(a)
        return tuple(
            tensor.flatten(2)
            for tensor in (
                paired(read, read),
                paired(decay, decay),
                paired(key, denominator_key),
                paired(
                    value_heads * GQA_STATE_SCALE, torch.ones_like(value_heads) * GQA_STATE_SCALE
                ),
                paired(a, zeros),
                paired(b, zeros),
            )
        )

    def _state_training(self, feature_query, feature_key, value_heads, controls=None):
        flash = _flash("training", feature_query)
        batch, _, length, _ = feature_query.shape
        if feature_query.dtype != torch.bfloat16 or length % 16:
            raise RuntimeError("GQA training requires BF16 inputs with T divisible by 16")
        inputs = self._wkv_inputs(feature_query, feature_key, value_heads, controls)
        raw = flash.pretrain_recurrent_bf16(*inputs, head_size=self.head_size).view(
            batch, length, self.num_heads, 2, self.head_size
        )
        return (
            raw[..., 0, :].permute(0, 2, 1, 3).float() / GQA_STATE_SCALE,
            raw[..., 1, :].float().mean(-1).permute(0, 2, 1) / GQA_STATE_SCALE,
        )

    def _heads(
        self,
        numerator,
        denominator,
        feature_query=None,
        feature_key=None,
        value=None,
        controls=None,
    ):
        heads = numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
        batch, _, length, _ = heads.shape
        flat = heads.transpose(1, 2).reshape(batch * length, -1)
        normalized = F.group_norm(
            flat, self.num_heads, self.ln_x.weight.float(), self.ln_x.bias.float(), self.ln_x.eps
        )
        normalized = normalized.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)
        heads = heads + self.norm_mix.tanh()[None, :, None, None] * (normalized - heads)
        if feature_query is not None:
            _, key, _, _ = self._state_components(feature_key, controls)
            shortcut = (
                feature_query.float() * key.float() * self.r_k[None, :, None, :].float()
            ).sum(-1, keepdim=True)
            heads = heads + shortcut * value.float()
        return heads

    def _readout(
        self, numerator, denominator, gate, query=None, key=None, value=None, controls=None
    ):
        heads = self._heads(numerator, denominator, query, key, value, controls)
        mixed = heads.transpose(1, 2).reshape_as(gate)
        return self.o_proj((mixed * gate.float().sigmoid()).to(self.o_proj.weight.dtype))

    def attention_heads_reference(self, x, position_ids=None, v_first=None):
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        query, key, value, gate, controls, _ = self._inputs(x, position_ids, v_first)
        numerator, denominator = self._state_reference(query, key, value, controls)
        return self._heads(numerator, denominator, query, key, value, controls), gate

    def reference_forward(self, x, position_ids=None, v_first=None):
        heads, gate = self.attention_heads_reference(x, position_ids, v_first)
        mixed = heads.transpose(1, 2).reshape_as(gate) * gate.float().sigmoid()
        return self.o_proj(mixed.to(self.o_proj.weight.dtype))

    def _training_forward(self, x, v_first):
        positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        query, key, value, gate, controls, first = self._inputs(x, positions, v_first)
        numerator, denominator = self._state_training(query, key, value, controls)
        return self._readout(numerator, denominator, gate, query, key, value, controls), first

    def _inference_forward(self, x, v_first, cache):
        flash = _flash("inference", x)
        if x.dtype != torch.float16:
            raise RuntimeError("GQA inference requires a float16 checkpoint")
        batch, length, _ = x.shape
        state, elapsed = _recurrent_cache_states(
            cache, self.layer_idx, x, self.kernel_heads, self.head_size
        )
        if not torch.equal(elapsed, elapsed[:1].expand_as(elapsed)):
            raise RuntimeError("RWKV cache requires equal sequence lengths in a batch")
        positions = (int(elapsed[0].item()) + torch.arange(length, device=x.device)).expand(
            batch, -1
        )
        delta = self._shift_delta(x, cache.layers[self.layer_idx])
        query, key, value, gate, controls, first = self._inputs(x, positions, v_first, delta)
        inputs = tuple(
            tensor.view(batch * length, self.kernel_heads, self.head_size)
            for tensor in self._wkv_inputs(query, key, value, controls)
        )
        offsets, indices, ticket = cache.recurrent_metadata(flash, batch, length, x.device)
        raw = flash.infer_recurrent_fp16_forward_varlen(
            *inputs,
            state_pool=state,
            elapsed_state_pool=elapsed,
            cu_seqlens=offsets,
            state_indices=indices,
            max_seqlen=length,
            validated_metadata=ticket,
        ).view(batch, length, self.num_heads, 2, self.head_size)
        numerator = raw[..., 0, :].permute(0, 2, 1, 3).float() / GQA_STATE_SCALE
        denominator = raw[..., 1, :].float().mean(-1).permute(0, 2, 1) / GQA_STATE_SCALE
        return self._readout(numerator, denominator, gate, query, key, value, controls), first

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
