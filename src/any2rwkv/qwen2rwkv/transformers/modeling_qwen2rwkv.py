"""Qwen3.5 text blocks with every token mixer replaced by RWKV-7."""

from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionLayer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

GDN = "linear_attention"
GQA = "full_attention"
_SOURCE_TYPES = [GDN, GDN, GDN, GQA] * 6


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

    def __post_init__(self, **kwargs):
        if self.source_layer_types is None:
            self.source_layer_types = list(self.layer_types or _SOURCE_TYPES)
        self.layer_types = list(self.source_layer_types)
        super().__post_init__(**kwargs)

    def geometry(self, layer_idx: int) -> tuple[int, int, int]:
        if self.source_layer_types[layer_idx] == GDN:
            return 16, 128, 1
        return 16, 256, 2


def _flash(mode: str, tensor: torch.Tensor):
    try:
        module = importlib.import_module("flashrwkv2")
    except ImportError as error:
        raise RuntimeError(f"{mode} requires the pinned FlashRWKV2 provider") from error
    required = (
        (
            "pretrain_tmix_mix6_bf16",
            "pretrain_tmix_a_gate_bf16",
            "pretrain_tmix_kk_pre_bf16",
            "pretrain_recurrent_bf16",
            "pretrain_tmix_lnx_rkvres_xg_bf16",
        )
        if mode == "training"
        else (
            "prepare_recurrent_metadata",
            "infer_tmix_mix6_forward_varlen",
            "infer_tmix_linear_attention_c2c_forward_varlen",
            "infer_tmix_kk_a_gate_forward_varlen",
            "infer_recurrent_fp16_forward_varlen",
            "infer_tmix_lnx_rkvres_xg_forward_varlen",
        )
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
        return -1

    def mark_updated(self, length: int) -> None:
        self.cumulative_length += length
        self.has_previous_state[0] = True

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
    """Per-layer shift/WKV state plus the shared current-token ``v_first``."""

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

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        self.elapsed = [
            None if x is None else x.index_select(0, indices.to(x.device)) for x in self.elapsed
        ]
        if self.v_first is not None:
            self.v_first = self.v_first.index_select(0, indices.to(self.v_first.device))


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


class Qwen2RWKVTimeMix(nn.Module):
    """RWKV-7 TMix with per-layer D128/D256 recurrent geometry."""

    def __init__(self, config: Qwen2RWKVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kernel_heads, self.head_size, self.states_per_head = config.geometry(layer_idx)
        self.recurrent_width = self.kernel_heads * self.head_size
        channels = config.hidden_size
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, channels)))
        self.w0 = nn.Parameter(torch.empty(1, 1, self.recurrent_width))
        self.w1 = nn.Parameter(torch.empty(channels, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.empty(config.decay_low_rank_dim, self.recurrent_width))
        self.a0 = nn.Parameter(torch.empty(1, 1, self.recurrent_width))
        self.a1 = nn.Parameter(torch.empty(channels, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.empty(config.a_low_rank_dim, self.recurrent_width))
        self.v0 = nn.Parameter(torch.empty(1, 1, channels))
        self.v1 = nn.Parameter(torch.empty(channels, config.v_low_rank_dim))
        self.v2 = nn.Parameter(torch.empty(config.v_low_rank_dim, channels))
        self.value_residual_scale = nn.Parameter(torch.zeros(()))
        self.g1 = nn.Parameter(torch.empty(channels, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.empty(config.gate_low_rank_dim, channels))
        self.k_k = nn.Parameter(torch.empty(1, 1, self.recurrent_width))
        self.k_a = nn.Parameter(torch.empty(1, 1, self.recurrent_width))
        self.r_k = nn.Parameter(torch.empty(self.kernel_heads, self.head_size))
        self.receptance = nn.Linear(channels, self.recurrent_width, bias=False)
        self.key = nn.Linear(channels, self.recurrent_width, bias=False)
        self.value_base = nn.Linear(channels, channels, bias=False)
        self.value_expand = nn.Linear(channels, self.recurrent_width, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        norm_heads = 16 if self.states_per_head == 1 else 8
        self.ln_x = nn.GroupNorm(norm_heads, channels, eps=config.group_norm_epsilon)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        c = self.config.hidden_size
        depth = self.layer_idx / max(self.config.num_hidden_layers - 1, 1)
        reverse = 1.0 - self.layer_idx / self.config.num_hidden_layers
        ddd = torch.arange(c, dtype=torch.float32).view(1, 1, -1) / c
        with torch.no_grad():
            for name, power in {"r": 0.2, "w": 0.9, "k": 0.7, "v": 0.7, "a": 0.9, "g": 0.2}.items():
                getattr(self, f"x_{name}").copy_(1.0 - ddd.pow(power * reverse))
            axis = torch.arange(self.recurrent_width, dtype=torch.float32)
            linear = axis / max(self.recurrent_width - 1, 1) - 0.5
            within = axis.remainder(self.head_size)
            zigzag = (within - (self.head_size - 1) / 2) / ((self.head_size - 1) / 2)
            zigzag = zigzag * zigzag.abs()
            decay = -6 + 6 * (axis / max(self.recurrent_width - 1, 1)).pow(1 + depth**0.3)
            self.w0.copy_((decay + 0.5 + 2.5 * zigzag).view(1, 1, -1))
            self.a0.copy_((-0.19 + 0.3 * zigzag + 0.4 * linear).view(1, 1, -1))
            self.v0.copy_((0.73 - 0.4 * (torch.arange(c) / max(c - 1, 1) - 0.5)).view(1, 1, -1))
            self.k_k.copy_((0.71 - 0.1 * linear).view(1, 1, -1))
            self.k_a.fill_(1.02)
            self.r_k.fill_(-0.04)
            for name in ("w1", "a1", "v1", "g1"):
                getattr(self, name).zero_()
            for name in ("w2", "a2", "v2", "g2"):
                _orthogonal_(getattr(self, name), gain=0.1)
            for module, gain in ((self.receptance, 1.0), (self.key, 0.1), (self.value_base, 1.0)):
                _orthogonal_(module.weight, gain=gain)
            self.value_expand.weight.zero_()
            self.value_expand.weight[:c].copy_(torch.eye(c))
            self.output.weight.zero_()
            self.ln_x.weight.fill_(((self.layer_idx + 1) / self.config.num_hidden_layers) ** 0.7)
            self.ln_x.bias.zero_()
            self.value_residual_scale.zero_()

    def _value(self, xv: torch.Tensor, v_first: torch.Tensor | None):
        base = self.value_base(xv)
        if self.layer_idx == 0:
            return self.value_expand(base), base
        if v_first is None:
            raise ValueError("v_first is required after layer 0")
        gate = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        base = base + self.value_residual_scale * gate * (v_first - base)
        return self.value_expand(base), v_first

    def _pair_sum(self, raw, r, k, v):
        n = raw.shape[0]
        diag = (r * k * self.r_k).sum(-1, keepdim=True) * v
        return (raw + diag).view(n, 8, 2, 256).sum(2).reshape(n, 2048)

    def _training_forward(self, x: torch.Tensor, v_first: torch.Tensor | None):
        flash = _flash("training", x)
        if x.dtype != torch.bfloat16 or x.shape[1] % 16:
            raise RuntimeError(
                "training requires contiguous BF16 [B,T,2048] with T divisible by 16"
            )
        xr, xw, xk, xv, xa, xg = flash.pretrain_tmix_mix6_bf16(
            x.contiguous(),
            *(
                getattr(self, n).reshape(-1).contiguous()
                for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
            ),
        )
        r = self.receptance(xr)
        decay = -F.softplus(-(self.w0 + (torch.tanh(xw @ self.w1) @ self.w2))) - 0.5
        k = self.key(xk)
        v, v_first = self._value(xv, v_first)
        a = flash.pretrain_tmix_a_gate_bf16(
            self.a0.reshape(-1).contiguous(), ((xa @ self.a1) @ self.a2).contiguous()
        )
        g = (torch.sigmoid(xg @ self.g1) @ self.g2).contiguous()
        k, aa, bb = flash.pretrain_tmix_kk_pre_bf16(
            k.contiguous(),
            self.k_k.reshape(-1).contiguous(),
            a.contiguous(),
            self.k_a.reshape(-1).contiguous(),
            head_size=self.head_size,
        )
        raw = flash.pretrain_recurrent_bf16(
            r.contiguous(), decay.contiguous(), k, v.contiguous(), aa, bb, head_size=self.head_size
        )
        if self.states_per_head == 1:
            mixed = flash.pretrain_tmix_lnx_rkvres_xg_bf16(
                raw,
                r.contiguous(),
                k,
                v.contiguous(),
                self.r_k.contiguous(),
                self.ln_x.weight.contiguous(),
                self.ln_x.bias.contiguous(),
                g,
                head_size=128,
            )
        else:
            mixed = self._pair_sum(
                raw.reshape(-1, 16, 256),
                r.reshape(-1, 16, 256),
                k.reshape(-1, 16, 256),
                v.reshape(-1, 16, 256),
            ).view(x.shape[0], x.shape[1], 2048)
            zeros = torch.zeros(2048, dtype=mixed.dtype, device=mixed.device)
            mixed = flash.pretrain_tmix_lnx_rkvres_xg_bf16(
                mixed.contiguous(),
                mixed.contiguous(),
                mixed.contiguous(),
                mixed.contiguous(),
                zeros,
                self.ln_x.weight.contiguous(),
                self.ln_x.bias.contiguous(),
                g,
                head_size=256,
            )
        return self.output(mixed), v_first

    def _inference_forward(self, x: torch.Tensor, v_first, cache: Qwen2RWKVCache):
        flash = _flash("inference", x)
        if x.dtype != torch.float16:
            raise RuntimeError("inference requires a float16 checkpoint")
        b, t, c = x.shape
        shift, state, elapsed = _cache_states(
            cache, self.layer_idx, x, self.kernel_heads, self.head_size
        )
        offsets, indices, ticket = cache.recurrent_metadata(flash, b, t, x.device)
        packed = x.reshape(-1, c).contiguous()
        xr, xw, xk, xv, xa, xg = flash.infer_tmix_mix6_forward_varlen(
            packed,
            *(
                getattr(self, n).reshape(-1).contiguous()
                for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
            ),
            shift_state_pool=shift,
            cu_seqlens=offsets,
            state_indices=indices,
            max_seqlen=t,
            validated_metadata=ticket,
        )
        linear = flash.infer_tmix_linear_attention_c2c_forward_varlen
        r = linear(xr, self.receptance.weight.contiguous())
        k = linear(xk, self.key.weight.contiguous())
        base = linear(xv, self.value_base.weight.contiguous()).view(b, t, c)
        if self.layer_idx == 0:
            v_first = base
        else:
            gate_v = torch.sigmoid(self.v0 + (xv.view(b, t, c) @ self.v1) @ self.v2)
            base = base + self.value_residual_scale * gate_v * (v_first - base)
        v = linear(base.reshape(-1, c).contiguous(), self.value_expand.weight.contiguous())
        decay_delta = torch.tanh(xw @ self.w1) @ self.w2
        a_delta = (xa @ self.a1) @ self.a2
        g = (torch.sigmoid(xg @ self.g1) @ self.g2).contiguous()
        k, aa, bb = flash.infer_tmix_kk_a_gate_forward_varlen(
            k,
            self.k_k.reshape(-1).contiguous(),
            self.a0.reshape(-1).contiguous(),
            a_delta.contiguous(),
            self.k_a.reshape(-1).contiguous(),
            head_size=self.head_size,
            batch_size=b,
            max_seqlen=t,
        )
        raw = flash.infer_recurrent_fp16_forward_varlen(
            r.view(-1, self.kernel_heads, self.head_size).contiguous(),
            decay_delta.view(-1, self.kernel_heads, self.head_size).contiguous(),
            k.view(-1, self.kernel_heads, self.head_size).contiguous(),
            v.view(-1, self.kernel_heads, self.head_size).contiguous(),
            aa.view(-1, self.kernel_heads, self.head_size).contiguous(),
            bb.view(-1, self.kernel_heads, self.head_size).contiguous(),
            state_pool=state,
            elapsed_state_pool=elapsed,
            cu_seqlens=offsets,
            state_indices=indices,
            decay_bias=self.w0.view(self.kernel_heads, self.head_size).contiguous(),
            max_seqlen=t,
            validated_metadata=ticket,
        ).reshape(-1, self.recurrent_width)
        if self.states_per_head == 1:
            mixed = flash.infer_tmix_lnx_rkvres_xg_forward_varlen(
                raw,
                r,
                k,
                v,
                self.r_k.reshape(-1).contiguous(),
                self.ln_x.weight.contiguous(),
                self.ln_x.bias.contiguous(),
                g,
                head_size=128,
                batch_size=b,
                max_seqlen=t,
            )
        else:
            mixed = self._pair_sum(
                raw.view(-1, 16, 256), r.view(-1, 16, 256), k.view(-1, 16, 256), v.view(-1, 16, 256)
            )
            zeros = torch.zeros(2048, dtype=x.dtype, device=x.device)
            mixed = flash.infer_tmix_lnx_rkvres_xg_forward_varlen(
                mixed.contiguous(),
                mixed.contiguous(),
                mixed.contiguous(),
                mixed.contiguous(),
                zeros,
                self.ln_x.weight.contiguous(),
                self.ln_x.bias.contiguous(),
                g,
                head_size=256,
                batch_size=b,
                max_seqlen=t,
            )
        return linear(mixed, self.output.weight.contiguous()).view(b, t, c), v_first

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
