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


@torch.no_grad()
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
        if source_layer.layer_type == "linear_attention":
            source_mixer = source_layer.linear_attn
            target_mixer = target_layer.tmix
            target_mixer.gdn_in_proj_qkv.load_state_dict(
                source_mixer.in_proj_qkv.state_dict()
            )
            target_mixer.gdn_conv1d.load_state_dict(source_mixer.conv1d.state_dict())
            target_mixer.gdn_in_proj_z.load_state_dict(source_mixer.in_proj_z.state_dict())
            target_mixer.gdn_in_proj_b.load_state_dict(source_mixer.in_proj_b.state_dict())
            target_mixer.gdn_in_proj_a.load_state_dict(source_mixer.in_proj_a.state_dict())
            target_mixer.gdn_dt_bias.copy_(source_mixer.dt_bias)
            target_mixer.gdn_A_log.copy_(source_mixer.A_log)
            target_mixer.gdn_norm_weight.copy_(source_mixer.norm.weight)
            target_mixer.gdn_out_proj.load_state_dict(source_mixer.out_proj.state_dict())
    target.lm_head.load_state_dict(source_outer.lm_head.state_dict())
    target.tie_weights()
    return target


__all__ = ["build_qwen2rwkv"]
