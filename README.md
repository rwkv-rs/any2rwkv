# Any2RWKV

Any2RWKV is a research library for converting attention and linear-attention
architectures to RWKV time-mixing layers. The project focuses on mathematically
grounded weight initialization followed by layer-wise distillation.

The conversion implementation is still under development. This repository
currently provides the Python package skeleton and development toolchain.

## Development

The project uses [uv](https://docs.astral.sh/uv/) and a `src` package layout.

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```
