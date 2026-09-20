"""Construct the target and copy every non-TMix Qwen parameter."""

from __future__ import annotations

from copy import deepcopy

import torch

from ..gdn2rwkv import initialize_gdn_scaffold
from ..gqa2rwkv import initialize_gqa_layer, initialize_gqa_scaffold
from ..transformers.modeling_qwen2rwkv import (
    GDN,
    GDN_CHECKPOINT_SCHEMA,
    GDN_MODE,
    GQA,
    GQA_CHECKPOINT_SCHEMA,
    GQA_READOUT_MODE,
    Qwen2RWKVConfig,
    Qwen2RWKVForCausalLM,
    ScaffoldTimeMix,
)


def _config(source_config) -> Qwen2RWKVConfig:
    keys = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "rope_parameters",
        "attention_bias",
        "attention_dropout",
        "head_dim",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
    )
    values = {name: deepcopy(getattr(source_config, name)) for name in keys}
    values["source_layer_types"] = list(source_config.layer_types)
    values["layer_types"] = list(source_config.layer_types)
    values["gdn_mode"] = GDN_MODE
    values["gdn_checkpoint_schema"] = GDN_CHECKPOINT_SCHEMA
    values["gqa_readout_mode"] = GQA_READOUT_MODE
    values["gqa_checkpoint_schema"] = GQA_CHECKPOINT_SCHEMA
    values["tmix_schema"] = "source_shell_v1"
    values["scaffold_stage"] = [0] * source_config.num_hidden_layers
    return Qwen2RWKVConfig(**values)


def _validate(config) -> None:
    rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
    rotary_width = config.head_dim * rotary_factor
    rotary_dim = int(rotary_width)
    expected_prefix = (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    if (
        config.num_hidden_layers != 24
        or config.hidden_size != 2048
        or config.layer_types.count("linear_attention") != 18
        or config.layer_types.count("full_attention") != 6
        or config.linear_num_key_heads != 16
        or config.linear_num_value_heads != 16
        or config.linear_key_head_dim != 128
        or config.linear_value_head_dim != 128
        or config.linear_conv_kernel_dim != 4
        or config.num_attention_heads != 8
        or config.num_key_value_heads != 2
        or config.head_dim != 256
        or config.hidden_act != "silu"
        or config.attention_bias
        or tuple(config.layer_types[:4]) != expected_prefix
        or not float(rotary_width).is_integer()
        or rotary_dim <= 0
        or rotary_dim >= config.head_dim
        or rotary_dim % 2
    ):
        raise ValueError("source must be the supported Qwen3.5-2B text geometry")


def build_qwen2rwkv(source_outer, source_text) -> Qwen2RWKVForCausalLM:
    _validate(source_text.config)
    source_weight = source_text.embed_tokens.weight
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(source_weight.dtype)
        with torch.device(source_weight.device):
            target = Qwen2RWKVForCausalLM(_config(source_text.config))
    finally:
        torch.set_default_dtype(previous_dtype)
    target.model.embed_tokens.load_state_dict(source_text.embed_tokens.state_dict())
    target.model.norm.load_state_dict(source_text.norm.state_dict())
    for source_layer, target_layer in zip(source_text.layers, target.model.layers, strict=True):
        target_layer.input_layernorm.load_state_dict(source_layer.input_layernorm.state_dict())
        target_layer.post_attention_layernorm.load_state_dict(
            source_layer.post_attention_layernorm.state_dict()
        )
        target_layer.mlp.load_state_dict(source_layer.mlp.state_dict())
        if source_layer.block_type == "linear_attention":
            target_layer.tmix.load_state_dict(source_layer.linear_attn.state_dict(), strict=True)
    target.lm_head.load_state_dict(source_outer.lm_head.state_dict())
    target.tie_weights()
    return target


@torch.no_grad()
def initialize_scaffold_layers(target, source_text, activations):
    """Wrap the copied S0/T0 layers with the zero-training S1/T1 scaffolds."""
    if len(activations) != len(source_text.layers):
        raise ValueError("one source activation tensor is required per layer")
    for layer_idx, source_layer in enumerate(source_text.layers):
        target_layer = target.model.layers[layer_idx]
        hidden = activations[layer_idx].to(next(target_layer.parameters()).device).float()
        normalized = source_layer.input_layernorm(hidden).float()
        if source_layer.block_type == GDN:
            scaffold = ScaffoldTimeMix(
                target.config,
                layer_idx,
                origin=GDN,
                source=target_layer.tmix,
                teacher=source_layer.linear_attn,
            )
            initialize_gdn_scaffold(source_layer.linear_attn, scaffold, normalized[:8])
            scaffold.to(
                device=next(target_layer.tmix.parameters()).device,
                dtype=next(target_layer.tmix.parameters()).dtype,
            )
            target_layer.tmix = scaffold
            target.config.scaffold_stage[layer_idx] = 1
        else:
            positions = torch.arange(normalized.shape[1], device=normalized.device).expand(
                normalized.shape[0], -1
            )
            embeddings = source_text.rotary_emb(normalized, positions)
            init_embeddings = tuple(value[:8] for value in embeddings)
            initialize_gqa_layer(
                source_layer.self_attn, target_layer.tmix, normalized[:8], init_embeddings
            )
            scaffold = ScaffoldTimeMix(
                target.config,
                layer_idx,
                origin=GQA,
                source=target_layer.tmix,
                teacher=source_layer.self_attn,
            )
            initialize_gqa_scaffold(
                source_layer.self_attn, scaffold, normalized[:8], init_embeddings
            )
            scaffold.to(
                device=next(target_layer.tmix.parameters()).device,
                dtype=next(target_layer.tmix.parameters()).dtype,
            )
            target_layer.tmix = scaffold
            target.config.scaffold_stage[layer_idx] = 1
    target.config.tmix_schema = "scaffold_h128_s1_t1"
    return target


__all__ = ["build_qwen2rwkv", "initialize_scaffold_layers"]
