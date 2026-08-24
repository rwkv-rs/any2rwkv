# Any2RWKV

Any2RWKV is a research implementation for converting source-model TMix layers
to RWKV. The current Qwen3.5-2B path supports only the first three GDN layers
(`0..2`): each layer keeps the complete source frontend, control projections,
`RMSNormGated`, and `out_proj`, while FlashRWKV2 executes its matrix-state
recurrence as RWKV-7 WKV.

The first GQA layer is deliberately unsupported. Bounded-hazard, dual-expert,
PISA/PWT-sidecar, and Hedgehog/H2O attempts all failed their strict feasibility
or layer-output gates. Their executable product paths have been removed, and
the layer-3 boundary fails before training or checkpoint creation. See
[`docs/gqa2rwkv.md`](docs/gqa2rwkv.md) for the measured rejection evidence.

The GDN prefix uses a strict source `state_dict` copy plus recurrence and
Clamp-W diagnostics, followed by complete decoder-layer alignment. The product
path requires the pinned `rwkv-rs/transformers-rwkv` revision and FlashRWKV2's
native D128 training operators. There is no Torch, FLA, or alternate recurrence
fallback.

## Usage

The project uses [uv](https://docs.astral.sh/uv/) and a `src` package layout.

```bash
uv sync
uv run python -m any2rwkv.qwen2rwkv.align.train \
  --source /home/caizus/Weights/Qwen/Qwen3.5-2B \
  --output /path/to/Qwen3.5-2B-RWKV-GDN-prefix \
  --through-layer 2 \
  --agentic nvidia/Nemotron-SFT-Agentic-v2 \
  --math nvidia/Nemotron-SFT-Math-v4
```

This command produces validated layer checkpoints and the post-layer cache for
the GDN prefix. It does not produce or advertise a complete converted model.
