## 核心目标

本项目为 RWKV 社区权威架构转换研究实现仓库.
现有 LLM 已经逐步从O(n^2)的注意力架构迁移到带有线性注意力/稀疏注意力的混合架构, 如 Qwen3.5 Kimi-K3 Deepseek-v4 等, 从而通过减少 KV Cache 的使用来降低推理成本, 但其中仍然包含O(n^2)的注意力.
RWKV is an RNN with great LLM performance and parallelizable like a Transformer. It's combining the best of RNN and transformer - great performance, linear time, constant space (no kv-cache), fast training, infinite ctxlen, and free text. Better than Gated-Delta-Net(GDN) and Kimi-Delta-Attention(KDA).
通过一些数学的分析方法, 我们可以找到注意力架构 GQA/MLA 以及线性注意力架构 GDN/KDA 以及稀疏注意力架构 到 RWKV-Tmix 的权重迁移方法, 使得相同的输入可以得到相同的输出, 因此可以将 Qwen3.5 Kimi-K3 Deepseek-v4 等模型中任何 Tmix 层转换为 RWKV-Tmix, 但保留它们的 Cmix (Moe等架构), 从而在几乎无需重新训练的情况下使得推理成本骤降, 相同的显存下允许使用更大的 BatchSize. (any linear-attention to rwkv is very easy)
通过数学方法迁移得到的权重, 将作为 greedy layerwise 蒸馏对齐的初始化权重, 从而在蒸馏对齐过程中快速收敛, 减少训练时间, 因此并非完全无需训练.

## 权威 RWKV7 实现

(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- 权威底层推理引擎实现仓库 (cuda, for pro6000, 无调度, 无varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- 权威预训练实现仓库 (cuda, for h100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Diagonal Plus Low Rank(DPLR）的数学原理：显式转移矩阵的并行计算
(6) https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv -- 权威 transformers 适配仓库 (with rust tokenizer 10x faster than python implementation, and FlashRWKV2 RapidSampling 5x faster than FlashInfer)

## 权重迁移资料

如果 layerwise NMSE 过高, 迁移效果很差, 请先阅读并理解它们的迁移思想:
(1) https://spaces.ac.cn/archives/11823
(2) https://www.haoyizhu.site/blog/sparse-linear-attention/
然后检查代码实现是否与理论分析存在偏差.

## 蒸馏对齐数据集

(1) https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2
(2) https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4
下载在 `~/Datasets/` 下.


## 目录规范

本项目使用 `src` layout，正式 Python 包统一放在 `src/any2rwkv/`，测试统一放在
`tests/`。目录按“源模型族 -> 迁移机制/运行边界”组织，不按临时实验、开发者或
任务名称拆分。

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
新增任何文件, 都需要等待用户确认.
对于理论分析与实验结果互相校验而判废的方案, 需要及时清理, 并在 github 的提交中写明原因.

## Env

使用 uv 管理本机和远端专属环境 ./.venv, 严禁使用其它环境, 避免环境污染问题。

## Machine for Testing and Benchmarking
```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/transformers-rwkv
```
use git to sync your changes instead of rsync.
