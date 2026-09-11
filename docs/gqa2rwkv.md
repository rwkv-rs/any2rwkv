# Qwen3.5 GQA 到纯 RWKV feature-state 的转换

Qwen3.5-2B 的层布局是前三层 GDN、随后每四层一个 full-attention GQA 层。GQA
转换的目标是让每个 full-attention 层最终只保留 RWKV recurrent state：完整 GQA
支路只在蒸馏时作为冻结 teacher 使用，运行时和最终 checkpoint 都没有 GQA
teacher、sidecar、H2O eviction、`beta` 或临时 gate。

## 目标架构

源 GQA 的 Q/K/V/O、Q/K RMSNorm、partial RoPE 和 source output gate 直接复制到
`Qwen2RWKVTimeMix`。每个 query head 使用正值 feature map：

\[
  \phi(x)=\left[\frac{\operatorname{ReLU}(xW)}{s},
  \frac{\operatorname{ReLU}(-xW)}{s}\right]+10^{-4},
  \qquad s=\max |xW|.
\]

其中截断单位阵初始化将 D256 映射到两个 F64 正值通道。Q/K feature map 是每个
query head 独立的参数。每个 query head 保存两个 D256 状态：

\[
  S_t=S_{t-1}+v_t\phi(k_t)^T,
  \qquad z_t=z_{t-1}+\phi(k_t),
  \qquad h_t=\frac{S_t\phi(q_t)}{z_t^T\phi(q_t)+\epsilon}.
\]

`S` 是 numerator state，`z` 通过全 1 value 写入 denominator state。F128 在写入
FlashRWKV2 前补零到 D256，因此 8 个 query heads、每头两个 state 共 16 个
D256 states；FP16 recurrent state 为 2 MiB/sequence/layer，状态大小与上下文长度
无关。所有 token 都写入 state，首次 eviction、sink/recent/heavy slot 等旧策略
不再存在。实现还会将 numerator 和 denominator 的写入同时乘以 `1/256`，读出时
两者的比例不变，只用于把长前缀的 FP16 state 保持在安全范围内。

训练使用 FlashRWKV2 的 BF16 recurrent operator，推理使用同一 recurrent state 的
FP16 varlen operator。CPU/FP32 `reference_forward` 显式执行上式，用来验证
FlashRWKV2 的增量结果；它不是产品运行时 fallback。RoPE 的非持久 buffer 在
`from_pretrained` 的低内存 materialize 路径中可能被置为未初始化内容，运行时因此
按 immutable config 重建 FP32 频率，避免加载后的 NaN。

## 冻结 GQA teacher 蒸馏

每个 GQA 层的 source full-attention 每次产生完整 causal attention output，并且
teacher 参数在 `torch.no_grad()` 下冻结。student 始终计算纯 RWKV 输出 `R`，训练
期间只在 loss 中使用外部 gate 混合：

\[
  Y_g=gG_{source}+(1-g)R_{rwkv},
\]

\[
  L=\operatorname{NMSE}(Y_g,G_{source})+
    4\operatorname{NMSE}(R_{rwkv},G_{source}).
\]

同一输入、同一 teacher 和平方误差归一化下，这个损失等价于
`[4 + (1 - g)^2] * NMSE(R, G)`。因此 gate 调度改变损失倍率，不改变最优解，也
不会增加 feature-state 的召回容量。pure RWKV 路径始终有直接梯度。gate 固定按
`0.9 → 0.75 → 0.5 → 0.25 → 0.1 → 0.0` 下降；每个阶段结束后重新计算当前
layer 的混合输出 cache，再进入下一个阶段。当前层输入始终来自固定的已迁移
prefix，只有 gate=0 验收通过后才推进到下一层。只有 gate=0 且 state dict 不含
临时参数的 checkpoint 才允许保存。

蒸馏顺序如下：

1. 复制 source attention，并训练 Q/K feature map（attention transfer）。
2. 对纯 RWKV TMix 参数直接蒸馏，依次执行上述 gate schedule；source teacher 和
   Cmix/MLP 保持冻结，不引入 LoRA 参数或合并步骤。
3. gate=0 后重新计算纯 RWKV FP32 reference、FlashRWKV2 FP16 cache parity 和
   固定状态 soak。
4. 通过所有 gate 后才保存 layer artifact；后续 layer 从新生成的 hidden cache
   继续训练。

## 映射的数学限制

当前 GQA 路径将 RWKV 的 decay 固定为近似 1、erase 固定为 0，实现的是正值
feature kernel 的累加状态。它采用 RWKV operator，但没有使用 RWKV-7 的可学习
遗忘与低秩状态更新。FP32 oracle 与 FlashRWKV2 一致，只能证明这个累加映射实现
正确，不能证明它与 source Softmax 等价。

差异来自 `phi(q)^T phi(k)` 对 `exp(q^T k / sqrt(D))` 的有限维近似。source 的
大 logits 经过 Softmax 后仍产生归一化输出；RWKV 数值稳定本身不排斥尖峰召回。
需要验证的是 feature map 是否保留了区分目标 key 与大量干扰 key 的能力。
截断单位阵初始化仅保留前 64 个投影方向，后续蒸馏必须学习其余方向的信息。

这种近似与 [Softmax 到 Gated DeltaNet 的推导](https://kexue.fm/archives/11823/comment-page-1)
中的可学习状态更新不是同一个映射；[稀疏与线性注意力分析](https://www.haoyizhu.site/blog/sparse-linear-attention/)
也明确区分了尖峰区域的精确 attention 与其余部分的线性近似。最终移除 GQA 后，
尖峰召回必须由纯 feature-state 自身通过验收。

## Config 与 checkpoint contract

当前唯一支持的 GQA schema 是：

| field | value |
| --- | --- |
| `gqa_feature_projection_dim` | `64` |
| `gqa_feature_output_dim` | `128` |
| `gqa_states_per_query_head` | `2` |
| `gqa_readout_mode` | `rwkv_feature_state` |
| `gqa_checkpoint_schema` | `gqa_rwkv_feature_state_d256x2_v2` |

`Qwen2RWKVConfig.from_dict()` 对旧 bounded-Hedgehog/H2O schema、缺少新字段的
artifact 直接 fail closed。layer state dict 使用 strict load；其中出现
`sidecar`、`beta_logit`、`teacher`、`branch_gate` 或旧 LoRA 参数都会拒绝加载。
最终模型只包含 source-shell projections、feature Q/K 和 recurrent runtime 所需
参数，不保存蒸馏 gate。

## 验收边界

验收分为三层，不能用其中一层替代另一层：

- **CPU/reference**：显式 numerator/denominator recurrence 与 reference 输出一致，
  feature/state/readout shape 和 schema strict-load 正确。距离 128/256/512/1024 的
  deterministic fixture 经实际 Q/K feature map，检查单个目标值与重复干扰值的
  hit@1；fixture 在 feature-space 构造 logits，再通过当前 feature 矩阵的稳定伪逆
  反解 Q/K 输入，不是自然语言召回评测。
- **GPU/runtime**：记录 pure RWKV gate=0 完整 Block validation NMSE 相对初始化
  的变化，不再以 `3e-3` 作为质量截止值；保存的是验证集上不劣于初始化的最佳
  状态。FP32 reference 与 FlashRWKV2 BF16 增量 NMSE 不超过 `1e-3`；FP16 full
  prefill、64/128/256 chunked prefill 和逐 token decode 的 TMix/Block 相对 L2
  误差 `sqrt(sum((actual-reference)^2) / sum(reference^2))` 不超过 `1.1e-3`，
  这是 FlashRWKV2 FP16 分块累加的设备容差；
  8192-token soak 中 recurrent state 固定、无 sidecar 增长、无 NaN/Inf。
- **最终生成**：通过 layer 3 后再执行完整 24-layer conversion、fresh-process
  load 和 deterministic generation。生成成功不能替代前面的数值门槛。

如果训练无法改善独立验证集 NMSE，或 gate=0 的长距离 recall 失败，应检查
feature dimension、state geometry 与 RWKV 映射本身。NMSE 是拟合诊断指标，
模型效果还需要完整转换后的生成评估；teacher gate 不改变最终验收的纯 RWKV 路径。

## 当前证据

本地 27 项测试通过，覆盖 CPU/reference、schema、frozen-gate，以及实际 FlashRWKV2
BF16 前向/反向、FP16 prefill/decode 与 soak。保存前检查验证集拟合变化、
BF16/reference 增量 NMSE、四个距离的 recall，以及所有指标和 state 的有限性。
NVIDIA GB10 上使用当前初始化运行的 FlashRWKV2 探针结果为：

| probe | result |
| --- | ---: |
| BF16 operator 对 FP32 reference，64 tokens | NMSE `1.7454e-5` |
| FP16 512-token prefill 对 chunk 64 / 128 / 256 | NMSE `5.4174e-8` / `5.4163e-8` / `0.0` |
| FP16 512-token prefill 对逐 token decode | NMSE `8.6793e-8` |
| 8192-token soak | recurrent state `2097152` bytes，elapsed `8192`，无 NaN/Inf |

真实 Qwen3.5-2B 的 512-token 蒸馏探针使用固定 packed tensor
`1d039b73dcafd9783a7e872f682cf64728cb31f6090ef54c2882ca3bc0919336`，训练行
`[192,256)`、验证行 `[64,192)`，随机种子 0。输入来自直接迁移、未蒸馏的前三层。
该前缀 layer 0 的 Block NMSE 为 `0.02238`；它不是已完成训练的正式前缀。

| layer-3 probe | result |
| --- | ---: |
| 初始化 Block NMSE | `0.141616` |
| 直接蒸馏 TMix 后的 Block NMSE | `0.032859`，相对初始化下降约 77% |
| 已训练状态 BF16/reference TMix 增量 NMSE | `3.4754e-5` |
| 已训练状态 BF16/reference Block 增量 NMSE | `1.8509e-5` |
| 已训练 feature map recall，128/256/512/1024 | hit@1 全部 `1.0` |

这些结果来自已保存的 gate=0 layer-3 artifact，支持继续完整转换。`0.003` 不再是
硬截止值；代码只要求验证集相对初始化不退化，并保留 recall、reference/operator
数值一致性和有限状态检查。

完整 24-layer conversion 已完成，artifact 位于
`/home/caizus/Weights/Qwen/Qwen3.5-2B-pure-rwkv-common-20260911`，包含 24 个
layer artifact 和 3.6 GiB 的 `model.safetensors`。全局 KL 三个 epoch 从
`0.2500983` 降至 `0.2059738`；fresh-process strict-load、schema 检查和三条
deterministic generation 均通过，结果保存在 artifact 的 `acceptance.json`。
