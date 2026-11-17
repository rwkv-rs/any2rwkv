"""Mathematical initialization and greedy alignment into RWKV-7."""


def __getattr__(name):
    if name in {"Qwen2RWKVConfig", "Qwen2RWKVForCausalLM"}:
        from .qwen2rwkv.qwen3_5.transformers import modeling_qwen2rwkv

        value = getattr(modeling_qwen2rwkv, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def convert_qwen3_5_2b(*args, **kwargs):
    from .qwen2rwkv.qwen3_5.align.train import convert_qwen3_5_2b as convert

    return convert(*args, **kwargs)


__version__ = "0.1.0"
__all__ = ["Qwen2RWKVConfig", "Qwen2RWKVForCausalLM", "convert_qwen3_5_2b"]
