import pytest
import torch

import any2rwkv
from any2rwkv.qwen2rwkv.gqa2rwkv import initialize_gqa_layer
from any2rwkv.qwen2rwkv.transformers.modeling_qwen2rwkv import (
    Qwen2RWKVConfig,
    Qwen2RWKVForCausalLM,
)


def test_package_version() -> None:
    assert any2rwkv.__version__ == "0.1.0"


def test_rejected_gqa_conversion_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="GQA -> RWKV conversion is not implemented"):
        initialize_gqa_layer()


def test_rejected_gqa_model_forward_fails_closed() -> None:
    config = Qwen2RWKVConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        layer_types=["full_attention"],
        source_layer_types=["full_attention"],
    )
    model = Qwen2RWKVForCausalLM(config).train()
    with pytest.raises(RuntimeError, match="GQA -> RWKV conversion is not implemented"):
        model(input_ids=torch.ones(1, 2, dtype=torch.long))
