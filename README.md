# Any2RWKV

Any2RWKV implements a research conversion of the Qwen3.5-2B text backbone to RWKV
recurrent token mixing. Each GDN
layer keeps its complete source frontend, control projections, `RMSNormGated`,
and `out_proj`; only its matrix-state recurrence is executed by RWKV-7 WKV. Each
GQA layer copies the source Q/K/V/O, norms, RoPE, and output gate into a pure
RWKV positive-feature state, with trainable token shifts, decay, rank-one state
corrections, value residuals, normalization and the RKV readout term. New components
start with zero contribution or identity behavior, preserving transferred outputs.
The first layer's values are carried through both layerwise caches and model forward.
Qwen embeddings, decoder RMSNorm, MLP, residual
structure, tokenizer, and tied LM head remain unchanged.

During layerwise distillation the frozen source GQA block is a teacher only. An
external gate mixes teacher and pure RWKV outputs through the fixed schedule
`0.9 → 0.75 → 0.5 → 0.25 → 0.1 → 0.0`; the loss always includes a four-times
weighted pure RWKV term. Distillation optimizes the pure RWKV TMix parameters
directly. Each epoch selects for both validation Block NMSE and recall with learned
dynamics enabled. Only an accepted gate-zero candidate becomes a layer checkpoint.
Runtime and final checkpoints contain no GQA teacher, sidecar,
eviction policy, beta, or temporary gate. With fixed inputs this loss equals
`[4 + (1 - g)^2] * NMSE(R, G)`: the gate schedule changes the loss scale but adds
no retrieval capacity. Validation Block NMSE tracks improvement from initialization;
there is no arbitrary `0.003` quality cutoff. Each final GQA layer must pass recall,
reference/operator parity, and fixed-state cache checks before saving.

The historical additive layer-3 baseline has validation Block NMSE `0.03286`.
The current v3 schema expands its trainable RWKV dynamics. Controlled experiments
compare against continued training with the added components frozen. The synthetic
recall probe exercises learned forgetting, but does not establish language recall.
The selected conservative 12-epoch candidate reaches Block NMSE `0.02949` with
hit@1 `1.0` at all four tested distances. A longer candidate reached `0.02573` but
was rejected after its full-model math generation regressed. Model generation is
evaluated separately, including whether answers finish with an EOS token.
The directory-level warm-start path has now completed all 24 layers: six GQA layers
passed gate-zero recall, global KL fell from `0.219678` to `0.183399`, and the three
fixed generation prompts all finished with EOS. This is a smoke test rather than a
full language evaluation.
See [the GQA migration analysis and acceptance evidence](docs/gqa2rwkv.md).

The product path requires the pinned `rwkv-rs/transformers-rwkv` revision and
FlashRWKV2's native D128/D256 training and FP16 inference kernels. There is no
Torch, FLA, or FP32-state recurrence fallback.

## Usage

The project uses [uv](https://docs.astral.sh/uv/) and a `src` package layout.

```bash
uv sync
uv run python -m any2rwkv.qwen2rwkv.qwen3_5.align.train \
  --source /home/caizus/Weights/Qwen/Qwen3.5-2B \
  --output /path/to/Qwen3.5-2B-RWKV \
  --agentic nvidia/Nemotron-SFT-Agentic-v2 \
  --math nvidia/Nemotron-SFT-Math-v4
```
