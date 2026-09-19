## Core Objectives

This project is the authoritative architecture conversion research implementation repository for the RWKV community.
Existing LLMs have gradually migrated from O(n^2) attention architectures to hybrid architectures with linear attention/sparse attention, such as Qwen3.5, Kimi-K3, Deepseek-v4, etc., thereby reducing inference costs by reducing KV Cache usage, but they still contain O(n^2) attention.
RWKV is an RNN with great LLM performance and parallelizable like a Transformer. It's combining the best of RNN and transformer - great performance, linear time, constant space (no kv-cache), fast training, infinite ctxlen, and free text. Better than Gated-Delta-Net(GDN) and Kimi-Delta-Attention(KDA).
Through certain mathematical analysis methods, we can find weight transfer methods from attention architectures GQA/MLA, linear attention architectures GDN/KDA, and sparse attention architectures to RWKV-Tmix, so that the same input can produce the same output. Therefore, any Tmix layer in models such as Qwen3.5, Kimi-K3, Deepseek-v4, etc. can be converted to RWKV-Tmix while retaining their Cmix (MoE and other architectures), thereby drastically reducing inference cost with almost no retraining and allowing a larger BatchSize under the same VRAM. (any linear-attention to rwkv is very easy)
The weights obtained through mathematical transfer will serve as initialization weights for greedy layerwise distillation alignment, thereby enabling fast convergence during distillation alignment and reducing training time. Therefore, it is not completely training-free.

## Authoritative RWKV7 Implementations

(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- authoritative low-level inference engine implementation repository (CUDA, for PRO6000, no scheduling, no varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- authoritative pretraining implementation repository (CUDA, for H100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- mathematical principles of Diagonal Plus Low Rank (DPLR): parallel computation of explicit transition matrices
(6) https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv -- authoritative transformers adaptation repository (with rust tokenizer 10x faster than python implementation, and FlashRWKV2 RapidSampling 5x faster than FlashInfer)

## Weight Transfer Resources

If the layerwise NMSE is too high and the transfer effect is poor, please first read and understand their transfer ideas:
(1) https://spaces.ac.cn/archives/11823
(2) https://www.haoyizhu.site/blog/sparse-linear-attention/
Then check whether the code implementation deviates from the theoretical analysis.

## Distillation Alignment Datasets

(1) https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2
(2) https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4
Download them under `~/Datasets/`.

## Directory Conventions

This project uses a `src` layout. Official Python packages are placed uniformly under `src/any2rwkv/`, and tests under `tests/`. The directory is organized by “source model family -> transfer mechanism/runtime boundary”, not split by temporary experiments, developers, or task names.

```text
src/any2rwkv/
├── __init__.py
├── py.typed
└── <model>2rwkv/
    ├── <any_tmix>2rwkv.py
    ├── transformers/
    │   └── modeling_<model>2rwkv.py
    └── align/
        ├── datasets.py
        ├── last_layer_cache.py
        ├── model_<model>.py
        ├── model_<model>2rwkv.py
        └── train.py
```
Adding any file requires waiting for user confirmation.
For solutions rejected after cross-validation between theoretical analysis and experimental results, they must be cleaned up promptly, and the reason must be stated in the GitHub commit.

## Env

Use uv to manage the local and remote dedicated environment `./.venv`. Using any other environment is strictly prohibited to avoid environment contamination issues.

## Machine for Testing and Benchmarking
```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/transformers-rwkv
```
use git to sync your changes instead of rsync.
