"""Only the current and next rank-local BF16 hidden caches."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class LastLayerCache:
    def __init__(self, directory: str | Path, rank: int):
        self.directory = Path(directory)
        self.rank = rank
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, generation: str) -> Path:
        return self.directory / f"rank_{self.rank:02d}_{generation}.safetensors"

    def load(self, generation: str = "current") -> torch.Tensor:
        return load_file(self.path(generation).as_posix())["hidden"]

    def store(self, hidden: torch.Tensor, generation: str = "next") -> None:
        save_file(
            {"hidden": hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()},
            self.path(generation).as_posix(),
        )

    def advance(self) -> None:
        current = self.path("current")
        next_path = self.path("next")
        current.unlink(missing_ok=True)
        next_path.replace(current)


__all__ = ["LastLayerCache"]
