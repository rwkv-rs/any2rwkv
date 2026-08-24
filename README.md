# Any2RWKV

Any2RWKV converts the Qwen3.5-2B text backbone to a hybrid RWKV model. Each GDN
layer keeps its complete source frontend, control projections,
`RMSNormGated`, and `out_proj`; only its matrix-state recurrence is executed by
RWKV-7 WKV. Each GQA layer is still converted to a canonical RWKV-7 TMix. Qwen
embeddings, decoder RMSNorm, MLP, residual structure, tokenizer, and tied LM
head remain unchanged.

GDN initialization is a strict source `state_dict` copy plus recurrence and
Clamp-W diagnostics. GQA keeps its mathematical compiler. Both paths then align
each complete decoder-layer output while the original Qwen model remains fixed;
the full-model path later adds logits-KL TMix fine-tuning. RWKV value residual
belongs only to GQA layers: it remains zero during initialization and layerwise
alignment, and opens for later GQA layers only during full-model fine-tuning.

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
