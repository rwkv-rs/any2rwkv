"""Construct the target and copy every non-TMix Qwen parameter."""

from __future__ import annotations

from copy import deepcopy

import torch

from ..transformers.modeling_qwen2rwkv import Qwen2RWKVConfig, Qwen2RWKVForCausalLM


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
    return Qwen2RWKVConfig(**values)


def _validate(config) -> None:
    if (
        config.num_hidden_layers != 24
        or config.hidden_size != 2048
        or config.layer_types.count("linear_attention") != 18
        or config.layer_types.count("full_attention") != 6
        or config.linear_num_key_heads != 16
        or config.linear_num_value_heads != 16
        or config.linear_key_head_dim != 128
        or config.linear_value_head_dim != 128
        or config.num_attention_heads != 8
        or config.num_key_value_heads != 2
        or config.head_dim != 256
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
    target.lm_head.load_state_dict(source_outer.lm_head.state_dict())
    target.tie_weights()
    return target


__all__ = ["build_qwen2rwkv"]
