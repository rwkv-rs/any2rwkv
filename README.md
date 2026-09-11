# Any2RWKV

Any2RWKV implements a research conversion of the Qwen3.5-2B text backbone to RWKV
recurrent token mixing. Each GDN
layer keeps its complete source frontend, control projections, `RMSNormGated`,
and `out_proj`; only its matrix-state recurrence is executed by RWKV-7 WKV. Each
GQA layer copies the source Q/K/V/O, norms, RoPE, and output gate into a pure
RWKV positive-feature state. Qwen embeddings, decoder RMSNorm, MLP, residual
structure, tokenizer, and tied LM head remain unchanged.

During layerwise distillation the frozen source GQA block is a teacher only. An
external gate mixes teacher and pure RWKV outputs through the fixed schedule
`0.9 → 0.75 → 0.5 → 0.25 → 0.1 → 0.0`; the loss always includes a four-times
weighted pure RWKV term. Distillation optimizes the pure RWKV TMix parameters
directly. Only a checkpoint that passes gate `0.0` acceptance is saved.
Runtime and final checkpoints contain no GQA teacher, sidecar,
eviction policy, beta, or temporary gate. With fixed inputs this loss equals
`[4 + (1 - g)^2] * NMSE(R, G)`: the gate schedule changes the loss scale but adds
no retrieval capacity. Validation Block NMSE tracks improvement from initialization;
there is no arbitrary `0.003` quality cutoff. Each final GQA layer must pass recall,
reference/operator parity, and fixed-state cache checks before saving.

The layer-3 gate=0 run reduced validation Block NMSE from `0.1416` to `0.03286`.
Runtime tests establish the feature-state implementation; model quality is assessed
separately after full conversion and generation.
See [the GQA migration analysis and acceptance evidence](docs/gqa2rwkv.md).

The product path requires the pinned `rwkv-rs/transformers-rwkv` revision and
FlashRWKV2's native D128/D256 training and FP16 inference kernels. There is no
Torch, FLA, or FP32-state recurrence fallback.

## Usage

The project uses [uv](https://docs.astral.sh/uv/) and a `src` package layout.

```bash
uv sync
uv run python -m any2rwkv.qwen2rwkv.align.train \
  --source /home/caizus/Weights/Qwen/Qwen3.5-2B \
  --output /path/to/Qwen3.5-2B-RWKV \
  --agentic nvidia/Nemotron-SFT-Agentic-v2 \
  --math nvidia/Nemotron-SFT-Math-v4
```
