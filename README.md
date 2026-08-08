# Any2RWKV

Any2RWKV converts the Qwen3.5-2B text backbone's 18 GDN and 6 GQA token mixers
to RWKV-7 while retaining Qwen embeddings, RMSNorm, MLP, residual structure,
tokenizer and tied LM head. It uses one mathematical initializer per source
layer type, greedy block-output alignment, then a frozen-teacher logits-KL TMix
fine-tune. Value residual is present in every target layer, remains exactly zero
during initialization/layerwise alignment, and opens only in the final stage.

The product path requires the pinned `rwkv-rs/transformers-rwkv` revision and
FlashRWKV2's native D128/D256 training and FP16 inference kernels. There is no
Torch, FLA, or FP32-state recurrence fallback.

## Development

The project uses [uv](https://docs.astral.sh/uv/) and a `src` package layout.

```bash
uv sync
uv run python -m any2rwkv.qwen2rwkv.align.train \
  --source /home/caizus/Weights/Qwen/Qwen3.5-2B \
  --output /path/to/Qwen3.5-2B-RWKV \
  --agentic nvidia/Nemotron-SFT-Agentic-v2 \
  --math nvidia/Nemotron-SFT-Math-v4
```
