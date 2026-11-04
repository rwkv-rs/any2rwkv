# GQA priming with adaptive heads and a signed RWKV7 state

This opt-in method replaces Qwen3.5 GQA with a source-initialized RWKV7 mixer.
It keeps Q/K/V/O projections, QK RMSNorm, text partial RoPE, the source output gate,
both external decoder norms, the FFN and residual ordering. Unlike the existing
positive-feature converter, it uses the full signed D256 representation and one
matrix state per query head, with no feature compression or denominator state.
Initialization is **not** functionally equivalent to softmax attention.

The design is inspired by [Priming](https://arxiv.org/abs/2605.08301), adapted to
RWKV7 dynamics and greedy layerwise alignment. It is not a reproduction of the
paper's end-to-end training budget or all of its architectures.

## Scope and entry points

- `qwen3_5/gqa_priming.py`: transfer, AGQA, token shifts, recurrence/readout,
  per-mixer recurrent cache, frozen-backbone full-sequence model.
- `qwen3_5/align/primed_gqa.py`: ascending GQA-only alignment, local block/mixer
  NMSE, strict delta checkpoints, optimizer resume and local-data CLI.
- `tests/test_gqa_priming.py`: CPU mathematical/integration tests and optional
  CUDA provider checks.

The reference production geometry is Qwen3.5-2B: hidden=2048, 8 query heads,
2 KV heads, D256, converted indices `3,7,11,15,19,23`. Small Qwen geometries are
supported for tests. Llama, multimodal position streams, non-default RoPE scaling,
masked/padded packed training and scaffold endpoints are outside this path.

The CLI leaves native GDN layers unchanged. To reproduce the already-converted
GDN parent experiment, explicitly supply those frozen mixers through the Python
API below. No GDN conversion or global training runs implicitly.

## Transfer and recurrence

Qwen packs Q and its output gate **within each head**. Reshape to `[B,T,H,2D]`
before splitting; reading the first half of the entire projection is incorrect.
The output gate has its own shifted input and uses the corresponding original
projection rows, including bias when present.

For K and V independently, adaptive head expansion is

```text
A(X) = R X + reshape(up(SiLU(down(flatten(X)))))
```

`R` starts as the exact grouped-query repeat matrix. Each adapter has rank32,
random down weights and zero up weights. Thus initialization exactly repeats KV,
but heads can subsequently learn different writes. `--no-agqa` freezes both the
repeat map and residual branch, even when the current mixer is unfrozen.

Six zero-initialized channelwise shifts apply to `r,w,k,v,a,g`:
`x_p(t) = x(t) + shift_p * (x(t-1) - x(t))`, with zero previous input at sequence
start. QK normalization precedes RoPE; K normalization also precedes AGQA.
Read and key vectors each receive `1/sqrt(D)` scaling. This is a parameterization
choice, not a softmax kernel identity.

With states indexed `[key,value]`, the update is

```text
read      = RoPE(QNorm(Q(x_r))) / sqrt(D)
key       = RoPE(AGQA_K(KNorm(K(x_k)))) / sqrt(D)
value     = AGQA_V(V(x_v))
decay     = decay_control(x_w)
erase     = sigmoid(erase_control(x_a))
direction = L2Normalize(key * k_k)
write_key = key * (1 + (erase - 1) * k_a)
value     = value + tanh(value_control(x_v)) * (first_gqa_value - value)
retention = exp(-exp(-0.5) * sigmoid(decay))

S_t = diag(retention) S_previous
      - (direction * erase) (direction^T S_previous)
      + write_key value^T
raw = read^T S_t
heads = GroupNorm_per_head(raw) + sum(read * write_key * r_k) * value
output = O(heads * sigmoid(source_gate(x_g)))
```

The rank-one term reads the **old**, not already decayed, state. The Flash six
inputs are `(read, decay, write_key, value, -direction, direction * erase)`.
Decay/erase biases start at -6/-2; controller up weights and `r_k` start at zero;
`k_k` and `k_a` start at one. GroupNorm operates on each token independently.

`first_gqa_value` is the first converted GQA's expanded V, not the first GDN's V
or a value from the previous token. Retained layers carry it through. It remains
in the gradient graph for callers doing end-to-end differentiation; the layerwise
trainer caches it without gradients after freezing the completed prefix.

At D256 this is 15,266,336 parameters and `8*256*256` state elements per mixer.
With FP32 state that is 2 MiB per sequence per GQA layer, or 12 MiB for six layers,
plus token-shift state. This does not include GDN state, weights or activations.

## Precision and backends

Masters/Adam, controls, AGQA, norms, RoPE and readout arithmetic are FP32. Large
projections use CUDA BF16 autocast; Flash WKV inputs/outputs are BF16 with FP32
internal state. Restoring FP32 masters prevents further master rounding, but cannot
recover information already absent from an input checkpoint.

`backend="reference"` is an explicit differentiable FP32 oracle for small CPU
tests, not an automatic runtime fallback. `backend="flash"` requires CUDA,
D128/D256 and the audited **FlashRWKV2 0.1.0a17** APIs. Full-sequence training/scoring
requires a context divisible by 16. Cached mixer inference supports shorter chunks
under `torch.no_grad()`, with equal unpadded batch lengths and continuous positions.

The repository's existing Flash git pin is for the original runtime. This PR
does not silently replace that dependency or claim compatibility between its old
FP16 APIs and the opt-in provider. Use a separate research environment with
PyTorch 2.13, the project's pinned Transformers revision, FlashRWKV2 0.1.0a17 and
its TileLang dependency. For an existing project environment dedicated to this
path, install the provider with `uv pip install --python <python> flashrwkv2==0.1.0a17`;
run that interpreter directly rather than allowing `uv run` to resync the old pin.
Standard Transformers 5.15 also supports the new CPU tests; the original scaffold
tests require the pinned `rwkv-rs/transformers-rwkv` package.

`PrimedQwen.forward()` returns full-sequence final-normalized hidden states;
`student.lm_head(hidden)` produces logits. Per-mixer cached inference is tested,
but this PR does not register an AutoModel checkpoint, whole-model generation
cache, vLLM backend or beam search adapter. Unknown caches are not accepted.

## Layerwise alignment

The active GQA is the **only trainable mixer**. Every FFN, external norm,
embedding, LM head, retained mixer and other GQA stays frozen. The teacher is the
original corresponding GQA block, evaluated on the current student-prefix input:

```text
loss = 0.5 * (block_NMSE + mixer_NMSE)
NMSE = sum((student - teacher)^2) / sum(teacher^2)
```

Both validation NMSE values must reach 0.0025, corresponding to relative RMS 5%.
After the first hit, explore 20 more epochs, with a hard maximum of 100 epochs per
layer. At the limit, restore the minimum-validation-loss weights and explicitly
report `target_met=false` when appropriate; do not silently weaken the target.
Epoch 0 is a candidate. These are local same-input comparisons, not accumulated
end-to-end errors or a percentage of the initialization loss.

Default optimizer: AdamW `(0.9,0.95)`, no decay, gradient clip 1; new parameters
start at 1e-4, copied Q/K/V/O and QK-norm parameters at 1e-5, cosine decay to 10%.
Training randomizes complete 20-row groups, preserving any caller-provided data
rotation within a group. No teacher-output blending, recall proxy gate or KL is
used. The existing converter's positive-feature recall gate is not applicable.

Input is a **local** `torch.save` file with two `LongTensor[N,T]` entries:
`training` and `validation`. Split documents before packing; use Qwen's original
tokenizer/chat template. Rows are fully valid tokens, without padding. Exact row
overlap/duplicates and invalid token IDs are rejected. Row checks do not prove
document/semantic deduplication; that remains the data preparer's responsibility.
No dataset or model download is performed by this entry.

```bash
PYTHONPATH=src /path/to/research-env/bin/python -u \
  -m any2rwkv.qwen2rwkv.qwen3_5.align.primed_gqa \
  --source /path/to/local/Qwen3.5-2B \
  --packed /path/to/disjoint_qwen_tokens.pt \
  --output /path/to/new/primed-gqa-run \
  --layers 3,7,11,15,19,23 --rank 32 \
  --epochs 100 --extra-epochs 20 --batch-size 16
```

This entry is single-process; select a GPU with `CUDA_VISIBLE_DEVICES` and do not
launch it with torchrun. It needs CPU RAM for the current/next hidden prefixes and
first-GQA values. Logs are written to both terminal and a unique `testlog/` file;
`ANY2RWKV_LOG_DIR` overrides that location.

Each epoch atomically saves current/best mixer states, Adam, scheduler, history
and CPU/CUDA RNG. Repeating the identical command resumes the last complete epoch;
unfinished work in that epoch is replayed. Source weights/config, retained GDNs,
packed splits, implementation and options bind the experiment identity. An OS
file lock prevents two writers. Completed layers are checked by SHA256 and loaded
without retraining; prefix activations are regenerated from their saved best.
The command never deletes or overwrites a different experiment.

## Using an existing converted GDN parent

The API accepts already-loaded source-shell GDNs through `FrozenGDN`; no server
paths, copied experiment scripts or checkpoints are bundled into the package:

```python
from any2rwkv.qwen2rwkv.qwen3_5.gqa_priming import FrozenGDN
from any2rwkv.qwen2rwkv.qwen3_5.align.primed_gqa import align, load_aligned

retained = {
    index: FrozenGDN(converted_parent.model.layers[index].tmix, backend="flash")
    for index, kind in enumerate(original.config.layer_types)
    if kind == "linear_attention"
}
student, reports = align(
    original, training_ids, validation_ids, output_directory,
    retained_mixers=retained,
)
restored = load_aligned(original, output_directory, retained_mixers=retained)
```

`original` is the unconverted Qwen text causal LM, frozen and on the selected
device; `converted_parent` is the separately verified source-shell parent. Its
GDNs must expose `_source_activations`, `_wkv_inputs`, `_source_boundary`; scaffold
and unrelated architectures are rejected. Source and retained modules must be
compatible with the chosen device and precision. GDN outputs carry the current
first-GQA values through unchanged. Their parameters are never optimized.

The output is a **GQA delta artifact**, with `experiment.json`, per-layer
`best.safetensors`/`complete.json` and resumable progress. Loading requires the exact
original source and, when used, retained GDN parent. It cannot be loaded as an old
`gqa_rwkv_tmix_d256x2_v3` positive-feature checkpoint or a self-contained HF model.

## Evidence and limits

The pre-port experiment used an already-distilled 18-GDN parent and six GQA layers,
each trained for 100 epochs on 1,996,800 training tokens. It used the earlier
any2rwkv source snapshot `60eb9b28bc3f1a8466be56e5b7c141c3b0ec7244` and the external
AGQA v3 runner. This PR ports that algorithm onto the current fork; the new entry
has not itself repeated the full 2B server experiment.

Historical Uncheatable scores (same 100-document, 512-target-token newline-reset
protocol; lower NLL is better):

| Variant | Overall | C++ | JavaScript | Python |
|---|---:|---:|---:|---:|
| Original Qwen3.5-2B Instruct | 1.981295 | 1.148222 | 1.042401 | 1.294794 |
| Older positive-feature conversion plus KL | 2.171631 | 1.526368 | 1.331658 | 1.518159 |
| Primed GQA on the frozen GDN parent | 2.087862 | 1.339159 | 1.194989 | 1.414823 |
| Later 24-TMix global MSE experiment | 2.074941 | 1.316400 | 1.166346 | 1.398982 |

These are historical products with different training budgets, GDN parents and
precision policies, **not a controlled GQA-only ablation**. The last row additionally
updates all 24 mixers and is not produced by this layerwise entry. None establishes
code pass@1, long-context accuracy, parity with the teacher or guaranteed gains.
The old positive-feature teacher score was 1.981193 rather than 1.981295 due to its
different scoring precision. The older score is a recorded report excerpt, not a
new evaluation of its weights.

Local checks cover explicit DPLR matrix/gradient parity, exact initial KV repeat,
source projection isolation, nontrivial chunk-cache parity and constant state,
first-GQA gradients through retained blocks, freezing, mixed native GDN/GQA,
strict save/reload, exact epoch resume, data rejection and output locking. The
ported FP32 mixer was also compared against the measured AGQA v3 implementation
with nonzero shifts/controllers/adapters: output, first values, input gradients
and parameter gradients matched exactly. CUDA tests are skipped without the
required GPU/provider; CPU tests do not certify CUDA or 2B quality.

```bash
PYTHONPATH=src /path/to/research-env/bin/python -m pytest tests/test_gqa_priming.py -q
```
