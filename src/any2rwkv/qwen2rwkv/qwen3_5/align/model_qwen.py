"""Frozen Qwen3.5-2B text teacher loader."""

from transformers import AutoConfig, Qwen3_5ForCausalLM


def load_qwen_teacher(source: str, dtype, device):
    outer_config = AutoConfig.from_pretrained(source)
    teacher = Qwen3_5ForCausalLM.from_pretrained(
        source,
        config=outer_config.text_config,
        dtype=dtype,
        attn_implementation="eager",
        key_mapping={r"^model\.language_model\.": "model."},
    ).to(device)
    teacher.eval().requires_grad_(False)
    return teacher, teacher.model


__all__ = ["load_qwen_teacher"]
