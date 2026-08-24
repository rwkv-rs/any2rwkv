"""Qwen3.5 text blocks with a validated GDN/WKV prefix.

The first unsupported GQA layer is represented by an explicit fail-closed
module. Rejected GQA approximations must not leak into checkpoint schemas,
cache state, training, or inference.
"""

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
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
)

GDN = "linear_attention"
GQA = "full_attention"
_SOURCE_TYPES = [GDN, GDN, GDN, GQA] * 6
GDN_MODE = "source_shell_wkv7"
GDN_CHECKPOINT_SCHEMA = "source_gdn_state_dict_v1"
CLAMP_W_EPSILON = 1e-4
W_SCALE = -math.exp(-0.5)


class Qwen2RWKVConfig(Qwen3_5TextConfig):
    """The one supported Qwen3.5-2B text geometry."""

    model_type = "qwen2rwkv"
    source_layer_types: list[str] | None = None
    gdn_mode: str = GDN_MODE
    gdn_checkpoint_schema: str = GDN_CHECKPOINT_SCHEMA

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
        return super().from_dict(config_dict, **kwargs)

    def geometry(self, layer_idx: int) -> tuple[int, int, int]:
        if self.source_layer_types[layer_idx] == GDN:
            return 16, 128, 1
        raise RuntimeError(
            "GQA -> RWKV geometry is unavailable because all attempted GQA "
            "approximations failed their strict acceptance gates"
        )


def _flash(mode: str, tensor: torch.Tensor):
    try:
        module = importlib.import_module("flashrwkv2")
    except ImportError as error:
        raise RuntimeError(f"{mode} requires the pinned FlashRWKV2 provider") from error
    if mode == "training":
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
    """Per-layer GDN Conv4 state, WKV state, and elapsed state."""

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
        flash = _flash("training", x)
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
        flash = _flash("inference", x)
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

    def forward(self, x, past_key_values=None, attention_mask=None):
        if self.training:
            return self._training_forward(x, attention_mask)
        if not isinstance(past_key_values, Qwen2RWKVCache):
            raise TypeError("inference requires Qwen2RWKVCache")
        return self._inference_forward(x, past_key_values, attention_mask)


class Qwen2RWKVUnimplementedGQA(nn.Module):
    """Fail closed at every rejected GQA -> RWKV product boundary."""

    _ERROR = (
        "GQA -> RWKV conversion is not implemented: bounded-hazard, "
        "dual-expert, and Hedgehog/H2O approximations were removed after "
        "failing their strict layer-output NMSE gates"
    )

    def __init__(self, config: Qwen2RWKVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(self, *args, **kwargs):
        raise RuntimeError(self._ERROR)


class Qwen2RWKVDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2RWKVConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if config.source_layer_types[layer_idx] == GDN:
            self.tmix = Qwen2RWKVGatedDeltaNet(config, layer_idx)
        else:
            self.tmix = Qwen2RWKVUnimplementedGQA(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None):
        residual = hidden_states
        mixed = self.tmix(self.input_layernorm(hidden_states), past_key_values, attention_mask)
        hidden_states = residual + mixed
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if past_key_values is not None:
            past_key_values.layers[self.layer_idx].mark_updated(hidden_states.shape[1])
        return hidden_states


class Qwen2RWKVPreTrainedModel(PreTrainedModel):
    config_class = Qwen2RWKVConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2RWKVDecoderLayer"]
    _is_stateful = True
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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
        for layer in self.layers:
            hidden = layer(hidden, cache, attention_mask)
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
