"""The fixed Nemotron 1:1 packing recipe for Qwen2RWKV alignment."""

from __future__ import annotations

import json
from itertools import cycle

import torch
from datasets import IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

AGENTIC_SPLITS = ("interactive_agent", "search", "tool_calling")
AGENTIC_PARQUET_REVISION = "4fb69cd40dbf36da60c73321e094e093946e60e9"


def _jsonl_rows(path: str):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield {"messages": row["messages"], "tools": row.get("tools")}


class PackedSequences(Dataset):
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        ids = self.input_ids[index]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _stream(repo: str, split: str, seed: int):
    if split in ("search", "tool_calling"):
        data_files = hf_hub_download(
            repo,
            f"data/{split}.parquet",
            repo_type="dataset",
            revision=AGENTIC_PARQUET_REVISION,
        )
        builder = "parquet"
        dataset = load_dataset(builder, data_files=data_files, split="train", streaming=True)
    else:
        data_files = hf_hub_download(repo, f"data/{split}.jsonl", repo_type="dataset")
        dataset = IterableDataset.from_generator(_jsonl_rows, gen_kwargs={"path": data_files})
    return iter(dataset.shuffle(seed=seed, buffer_size=128).repeat(None))


def _tokens(tokenizer, row) -> list[int]:
    messages = [
        json.loads(message) if isinstance(message, str) else message for message in row["messages"]
    ]
    for index, message in enumerate(messages):
        if message.get("tool_calls"):
            message = dict(message)
            calls = (
                json.loads(message["tool_calls"])
                if isinstance(message["tool_calls"], str)
                else message["tool_calls"]
            )
            message["tool_calls"] = []
            for value in calls:
                call = json.loads(value) if isinstance(value, str) else value
                function = call.get("function")
                if function and isinstance(function.get("arguments"), str):
                    call = dict(call)
                    call["function"] = dict(function)
                    call["function"]["arguments"] = json.loads(function["arguments"])
                elif isinstance(call.get("arguments"), str):
                    call = dict(call)
                    call["arguments"] = json.loads(call["arguments"])
                message["tool_calls"].append(call)
            messages[index] = message
    kwargs = {"tokenize": True, "add_generation_prompt": False}
    if row.get("tools"):
        tools = []
        for value in row["tools"]:
            tool = json.loads(value) if isinstance(value, str) else value
            function = tool.get("function")
            if function and isinstance(function.get("parameters"), str):
                tool = dict(tool)
                tool["function"] = dict(function)
                tool["function"]["parameters"] = json.loads(function["parameters"])
            tools.append(tool)
        kwargs["tools"] = tools
    encoded = tokenizer.apply_chat_template(messages, **kwargs)
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
    offset = 0
    packed: list[list[int]] = []
    choose_math = True
    while len(packed) < count:
        if choose_math:
            row = next(math_stream)
        else:
            row = next(agentic_streams[next(agentic_order)])
        choose_math = not choose_math
        buffer.extend(_tokens(tokenizer, row))
        while len(buffer) - offset >= context_length and len(packed) < count:
            packed.append(buffer[offset : offset + context_length])
            offset += context_length
        if offset >= len(buffer) // 2:
            buffer = buffer[offset:]
            offset = 0
    return PackedSequences(torch.tensor(packed, dtype=torch.long))


__all__ = ["PackedSequences", "build_packed_sequences"]
