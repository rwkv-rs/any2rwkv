"""The fixed Nemotron 1:1 packing recipe for Qwen2RWKV alignment."""

from __future__ import annotations

from itertools import cycle

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

AGENTIC_SPLITS = ("interactive_agent", "search", "tool_calling")


class PackedSequences(Dataset):
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        ids = self.input_ids[index]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _stream(repo: str, split: str, seed: int):
    return iter(
        load_dataset(repo, "default", split=split, streaming=True).shuffle(
            seed=seed, buffer_size=10_000
        )
    )


def _tokens(tokenizer, row) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": False}
    if row.get("tools"):
        kwargs["tools"] = row["tools"]
    encoded = tokenizer.apply_chat_template(row["messages"], **kwargs)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    return encoded


def build_packed_sequences(
    tokenizer,
    agentic: str = "nvidia/Nemotron-SFT-Agentic-v2",
    math: str = "nvidia/Nemotron-SFT-Math-v4",
    count: int = 4096,
    context_length: int = 512,
    seed: int = 42,
) -> PackedSequences:
    """Alternate Math/Agentic rows and emit exactly ``count`` dense blocks."""
    agentic_streams = [_stream(agentic, split, seed + i) for i, split in enumerate(AGENTIC_SPLITS)]
    agentic_order = cycle(range(len(agentic_streams)))
    math_stream = _stream(math, "train", seed)
    buffer: list[int] = []
    packed: list[list[int]] = []
    choose_math = True
    while len(packed) < count:
        if choose_math:
            row = next(math_stream)
        else:
            row = next(agentic_streams[next(agentic_order)])
        choose_math = not choose_math
        buffer.extend(_tokens(tokenizer, row))
        while len(buffer) >= context_length and len(packed) < count:
            packed.append(buffer[:context_length])
            del buffer[:context_length]
    return PackedSequences(torch.tensor(packed, dtype=torch.long))


__all__ = ["PackedSequences", "build_packed_sequences"]
