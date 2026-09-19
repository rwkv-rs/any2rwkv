"""Current/next hidden states and the first layer's values for RWKV value residuals."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class LastLayerCache:
    def __init__(self, directory: str | Path, rank: int):
        self.directory = Path(directory)
        self.rank = rank
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, generation: str) -> Path:
        return self.directory / f"rank_{self.rank:02d}_{generation}.safetensors"

    def load(self, generation: str = "current") -> torch.Tensor:
        with safe_open(self.path(generation), framework="pt", device="cpu") as tensors:
            return tensors.get_tensor("hidden")

    def load_v_first(self, generation: str = "current") -> torch.Tensor | None:
        with safe_open(self.path(generation), framework="pt", device="cpu") as tensors:
            return tensors.get_tensor("v_first") if "v_first" in tensors.keys() else None

    def store(self, hidden: torch.Tensor, generation: str = "next", *, v_first=None) -> None:
        tensors = {"hidden": hidden}
        if v_first is not None:
            tensors["v_first"] = v_first
        save_file(
            {
                name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                for name, value in tensors.items()
            },
            self.path(generation).as_posix(),
        )

    def advance(self) -> None:
        current = self.path("current")
        next_path = self.path("next")
        current.unlink(missing_ok=True)
        next_path.replace(current)


__all__ = ["LastLayerCache"]
