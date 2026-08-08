"""Mathematical initialization and greedy alignment into RWKV-7."""

from .qwen2rwkv.transformers.modeling_qwen2rwkv import (
    Qwen2RWKVConfig,
    Qwen2RWKVForCausalLM,
)


def convert_qwen3_5_2b(*args, **kwargs):
    from .qwen2rwkv.align.train import convert_qwen3_5_2b as convert

    return convert(*args, **kwargs)


__version__ = "0.1.0"
__all__ = ["Qwen2RWKVConfig", "Qwen2RWKVForCausalLM", "convert_qwen3_5_2b"]
