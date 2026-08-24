# Qwen3.5 GQA 到有界 Hedgehog/RWKV7 的转换

## 1. 范围与唯一效果门槛

Qwen3.5-2B 的前三层是 GDN，随后是第一个 full-attention GQA 层：

| layer | source mixer |
| ---: | --- |
| 0 | GDN |
| 1 | GDN |
| 2 | GDN |
| 3 | GQA，8 个 query heads、2 个 KV heads、D256 |

layer 3 必须在已转换 layer 0–2 的 rank-local prefix cache 之后评估。
`--through-layer 3` 仍为包含 layer 3 的边界；本轮在 layer 3 停止，不自动进入
layer 4、global KL、generation 或全模型导出。

效果层面的唯一硬指标是 unified validation 的完整 layer-3 Block NMSE：

\[
  \operatorname{NMSE}_{Block,layer3}^{validation} \le 3\times10^{-3}.
\]

attention、TMix、exact/cache mass、FP32/native 增量及吞吐均为诊断指标。
FP32 reference 必须先达到 `1.5e-3`，再允许使用 BF16/FP16 的数值余量。

## 2. 已判废路线

PISA/PWT、bounded-hazard、exact-local-tangent 和 dual-expert 的可执行路径、
配置与 checkpoint contract 已删除，不提供隐式迁移或 fallback。仅保留以下负结果：

| 判废路线 | 最佳相关结果 | 判废原因 |
| --- | ---: | --- |
| dual exact-local-tangent，4 个 D256 states/head | validation attention `0.20121`；完整 Block `0.11361` | oracle 已发生结构性失败，并非增加训练 epoch 可以解决 |
| PISA/PWT，recent 256 + sink 8 | validation attention `0.00208100`；完整 Block `0.00158776` | 需要 264 个精确槽、528 KiB KV sidecar；只有 FP32 oracle，没有 native/runtime/BF16 证据；理论来源也不满足本轮顶会约束 |

这些值只是历史诊断，不对应当前受支持的 artifact。当前固定方案失败时不得重新启用
上述路线。

## 3. 研究依据

当前实现是固定的工程组合，不是架构网格搜索或逐篇大规模复现：

- [Hedgehog，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/ebba182cb97864368fdb6ae00773a5e4-Paper-Conference.pdf)：可学习正值特征映射；
- [LoLCATs，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/72163d1c3c1726f1c29157d06e9e93c1-Paper-Conference.pdf)：attention transfer 后接 low-rank correction，以及 exact-window 与 linear-tail 共享分母；
- [LoLCATs 官方实现](https://github.com/HazyResearch/lolcats/blob/main/src/model/linear_attention/linear_window_attention_tk_gen.py)：共享归一化的实际计算顺序；
- [BASED，ICML 2024](https://proceedings.mlr.press/v235/arora24a.html)：局部精确注意力和长程线性状态的互补结构；
- [StreamingLLM，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html)：永久保留初始 attention sink；
- [H2O，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract.html)：recent 与累计重要度 heavy hitter；
- [Transformers are RNNs，ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html)：加法线性注意力状态。

本轮不测试 Performer、Taylor、T2R、其他容量或随机种子。

## 4. 固定目标架构

### 4.1 精确 sidecar 分区

每条序列、每个被转换 GQA 层最多持有 128 个 token 槽：

- `0:8`：最初 8 个 sink token；
- `8:72`：64-token recent ring；
- `72:128`：56 个 H2O heavy hitters。

每个 token 必须且只能位于 exact sidecar 或 linear tail 之一。recent 溢出时，
最旧 recent token 与最低累计分的 heavy token 比较；同分时按原始 position 稳定
排序，淘汰更老的非 sink token。被淘汰 token 在当前 query 读取之前写入 tail。

heavy score 是 hybrid 最终归一化 exact probability 在 8 个 query heads 上的均值
累加。slot 决策和 score 更新均与反向传播分离；两个 KV heads 共用同一套 token
位置。

2 个 KV heads、D256 下，FP16/BF16 K+V payload 为：

\[
  128 \times 2 \times 256 \times 2 \times 2
  =262144\text{ bytes}=256\text{ KiB}.
\]

FP32 score、INT32 original position、固定槽类别和 elapsed cursor 单独报告，
不计入“256 KiB KV payload”。当前固定布局由 slot range 推导类别和 valid 状态，
不额外保存随上下文增长的索引结构。

### 4.2 Hedgehog linear tail

每个 query head 独立使用：

\[
 \phi(x)=
 [\operatorname{softmax}(xW),\operatorname{softmax}(-xW)].
\]

原始 head dimension 为 256，projection dimension 为 64，正值 feature dimension
为 128；进入 RWKV state 前补零到 D256。Q/K feature map 分别持有每-head 投影。

LoLCATs 配置中的 `zero_init` 不能解释成“把权重全部置零”。其官方
`FeatureMapMLP.zero_init_()` 在无 skip connection 时实际执行 `eye_`；因此本实现
把每个 `256×64` Q/K 投影初始化为截断单位阵。若使用全零矩阵，成对 softmax
映射会落入对称驻点，初始 `W_q/W_k` 梯度为零。

每个 query head 使用两个 D256 RWKV7 states：

1. numerator state：累计 `value ⊗ phi_k`；
2. denominator state：以全 1 value 累计 `1 ⊗ phi_k`，readout 后对 256 个 value coordinates 取均值。

8 个 query heads 共 16 个 D256 states，FP16 recurrent state 为 2 MiB/layer。
与 exact sidecar 合计约 2.25 MiB/sequence/layer，且不随上下文长度增长。原始
2-KV-head GQA 约消耗 2 KiB/token，固定实现的 cache 收支平衡点约为 1153
tokens；短上下文不宣称节省显存。

仅被淘汰 token 写入 tail。FlashRWKV2 固定接收 `decay_logits=-30`、`a=b=0`；
decay/erase 不训练。所有矩阵状态更新只调用 FlashRWKV2 公共 recurrent operator，
没有 Torch、FLA 或 local-kernel 产品 fallback。

### 4.3 共享分母 readout

当前 exact slots 为 \(E_t\) 时：

\[
 A_E=\exp(qK_E^\top/\sqrt{256}-m).
\]

linear tail 为：

\[
 N_L=S_L\phi(q),\qquad D_L=z_L^\top\phi(q).
\]

每个 query head 训练一个
\(\beta_h=\sigma(b_h)\)，初始值为 0.1，并令
\(\alpha_h=1-\beta_h\)：

\[
 o_t=
 \frac{\beta_h A_EV_E+\alpha_hN_L}
      {\beta_h\sum A_E+\alpha_hD_L+\epsilon}.
\]

source Q/K/V、q/k RMSNorm、partial RoPE、sigmoid gate、`o_proj`、residual 与
Cmix 保持兼容。sidecar 保存原始绝对位置上完成 partial RoPE 的 K，不做位置
重编号。

当 prefix 长度不超过 128 时，tail 为空，exact 分支必须与 source causal Softmax
一致且不依赖 beta。实现会 fail closed 检查 denominator 有限且严格为正、position
无重复、容量不溢出、每个 prefix token 不丢失且不重复计数。

## 5. Config 与 checkpoint contract

唯一受支持的 GQA 配置为：

| field | value |
| --- | --- |
| `gqa_feature_projection_dim` | 64 |
| `gqa_feature_output_dim` | 128 |
| `gqa_states_per_query_head` | 2 |
| `gqa_sidecar_capacity` | 128 |
| `gqa_sink_slots` | 8 |
| `gqa_recent_slots` | 64 |
| `gqa_heavy_slots` | 56 |
| `gqa_readout_mode` | `hedgehog_h2o_shared_norm` |
| `gqa_checkpoint_schema` | `gqa_hedgehog_h2o_d256x2_v1` |

未修改的 source-model builder 仍会在进程内传入旧 expert/router 构造参数；它们
只是 builder compatibility input，`Qwen2RWKVConfig.to_dict()` 会删除它们，因而
不会进入新 config 或 checkpoint contract。持久化 artifact 缺少上表任一字段时
直接拒绝加载；旧 dual/PWT state 也会因 strict `state_dict` 不匹配而失败，并提示
使用 fresh output directory。

## 6. 数据与两阶段训练

每个 rank 固定划分：

- rows `0:8`：initializer 与 exact-prefix preflight；
- rows `8:24`：唯一 unified validation；
- rows `24:`：optimizer training。

prefix-cache 模式只接受 SHA-256 为
`1d039b73dcafd9783a7e872f682cf64728cb31f6090ef54c2882ca3bc0919336`
的 immutable `packed_sequences.pt`。序列长度为 512，batch size 为 8/rank，
顺序固定，seed 为 0；不执行 feature/window/capacity/seed 组合搜索。

### Phase A：Attention Transfer

只训练 Hedgehog Q/K feature weights 和 8 个 `beta_logit`：

\[
 L_A=\operatorname{NMSE}(o_{hybrid}^{heads},o_{source}^{heads}).
\]

- AdamW，LR `1e-2`，betas `(0.9,0.99)`，weight decay 0；
- gradient clip 1；
- 5% warmup + cosine decay；
- 最多 16 epochs；
- validation 完整 Block NMSE 连续 4 epochs 没有降低即停止。

### Phase B：Low-rank Block Correction

冻结 feature maps、beta、sidecar policy 和 recurrence。在 Q/K/V/O 上训练
rank-16、alpha-32、dropout-0 LoRA。Qwen 合并 Q/gate projection 中只有每个
head 的 Q slice 接收 LoRA，sigmoid gate slice 保持 source 权重不变。

\[
 L_B=4\,\operatorname{NMSE}(Block_s,Block_t)
     +\operatorname{NMSE}(TMix_s,TMix_t).
\]

- AdamW，LR `3e-5`，betas `(0.9,0.99)`，weight decay 0.1；
- gradient clip 1；
- 5% warmup + cosine decay；
- 最多 48 epochs；
- validation 完整 Block NMSE 连续 8 epochs 没有降低即停止。

两个阶段都只按 unified-validation 完整 Block NMSE 选择最佳 checkpoint；
TMix/attention 不能覆盖 Block 排序。LoRA 在保存前合并回 Q/K/V/O，runtime 不
保留 adapter 分支。本 GQA 路径没有额外 96-epoch corrective phase。

## 7. Fail-closed 证据阶梯

layer-3 artifact 只有依次通过下列门槛后才会保存：

1. **Static/reference**
   - 每个 token 恰好属于 exact sidecar 或 tail；
   - sink/recent/heavy 不超过 8/64/56；
   - 第一次 eviction 恰好发生在 token index 128；
   - denominator 有限且严格为正；
   - `T<=128` 的 head、TMix 和完整 Block NMSE 均不超过 `1e-6`。
2. **FP32 reference effect**
   - unified-validation 完整 Block NMSE 不超过 `1.5e-3`。
3. **FlashRWKV2 BF16**
   - unified-validation 完整 Block NMSE 不超过 `3e-3`；
   - 相对 FP32 reference 的完整 Block 增量 NMSE 不超过 `1e-3`。
4. **FP16 inference/cache**
   - 子进程 fresh strict-load 新 config/schema 和已合并 state；
   - full prefill、64/128/256-token chunked prefill 与 T=1 decode 的完整 Block
     输出相对 full prefill NMSE 均不超过 `1e-3`；
   - `batch_repeat_interleave`、`batch_select_indices`、reset 后继续运行与基线
     NMSE 不超过 `1e-3`；
   - 512、4096、8192-token soak 中 exact slots 永远不超过 128，sidecar K/V
     固定为 256 KiB，recurrent state 固定为 2 MiB，elapsed 与 token count 一致，
     且没有 NaN/Inf 或随序列增长的持久 tensor/list/cache。

任一门槛失败都会恢复训练前 source-shell snapshot、不保存 layer artifact，并输出
最佳 validation Block/TMix、失败证据层和 sidecar mass 诊断。不会扩大容量、换
seed、加入专家或切回非顶会方案。

## 8. 当前证据与尚未完成的证据

当前本机 `.venv` 的 CPU/reference 临时 probe 已证明：

- 新 schema 可 round-trip，旧 GQA config 会 fail closed；
- 16-token exact-prefix attention NMSE 为 `6.129229085670472e-15`，完整 TMix
  NMSE 为 `3.100761292694487e-14`；
- 201-token policy probe 中 exact/tail 无重复、无丢失，最终分类数为
  sink/recent/heavy=`8/64/56`，第一次 tail eviction 是 position 8；
- LoLCATs 截断单位阵初始化在 129-token overflow 后对 `W_q`、`W_k`、beta 都
  产生有限非零梯度；
- 模拟公共 recurrent provider 的 D256×2 numerator/denominator 相对显式 tail
  和式 NMSE 分别约为 `3.17e-6` 与 `3.00e-6`；
- 模拟 provider 下整段 prefill 与逐 token decode NMSE 为 `0.0`；
- cache repeat/select/reset 与 fresh-process strict load 均通过；
- rank-16 LoRA merge 只产生浮点舍入量级差异，merge 后 state 中没有 LoRA tensor。

这些只属于 static/CPU/reference 证据，不能证明目标 GPU 上的 FlashRWKV2 BF16
正确性、FP16 prefill/decode parity、8192-token soak 或最终 layer-3 `3e-3` 效果门。
上述 GPU gate 已接入训练控制流，但必须等待经授权的 8×PRO6000 运行后才能填写
实测值。本轮没有 commit、push、远端同步、GPU 运行或 layer-3 artifact。
