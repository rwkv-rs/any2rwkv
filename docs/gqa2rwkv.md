# Qwen3.5 GQA 到 RWKV7：当前判废结论

## 1. 当前状态

Qwen3.5-2B 的前三层是 GDN，第 3 层是第一个 full-attention GQA 层：

| layer | source TMix |
| --- | --- |
| 0–2 | GDN |
| 3 | GQA，8 个 query heads、2 个 KV heads、head dim 256 |

当前仓库只保留 layer `0..2` 的 GDN source-shell/WKV 转换。GQA 尚无通过严格门槛的
转换方法，因此：

- `gqa2rwkv.initialize_gqa_layer()` 固定 fail closed；
- target model 中的 GQA layer 是无参数 fail-closed 边界；
- 不存在 GQA checkpoint schema、sidecar cache、训练 phase、LoRA correction 或推理
  runtime；
- `--through-layer` 只接受 `0..2`；
- 不生成 `layer_03.safetensors`，也不生成或宣称完整 Qwen3.5-2B 转换产物。

这不是“暂时关闭一个已完成实现”，而是避免已被理论分析和实验共同否定的方案继续
出现在产品路径中。

## 2. 判废证据

### 2.1 单中心 bounded-hazard / affine state

误差分解为：

```text
exact hazard ≈ 7.16e-14
bounded hazard ≈ 0.11054
affine state ≈ 0.19726
complete Block ≈ 0.02636
```

主要误差在 query-dependent sharp attention 被压入单中心、固定 affine state 时已经
产生，不是 FlashRWKV2、state layout、训练时长或 BF16 kernel 的问题。继续调 epoch、
candidate closure 或数值实现不能修复错误的函数类。

### 2.2 两个 local-affine experts / 4×D256

两个 expert 的负载并未坍缩，但 validation 仍为：

```text
attention NMSE = 0.20121071   (gate <= 0.011)
complete Block NMSE = 0.11360577   (gate <= 0.0015)
```

因此增加两个局部切点和四个 D256 state 仍不足。该方案不能作为 fallback，也不能以旧
checkpoint alias、schema upgrade 或隐藏兼容路径保留。

### 2.3 PISA/PWT fixed-capacity exact sidecar oracle

PISA/PWT 证明“sharp content 精确计算、tail 近似、共享 numerator/denominator”是有效诊断
方向，但固定 sidecar oracle 仍未给后续 runtime 和数值误差留下预算。最佳候选
`recent=256, sink=8` 在长度 512 上保留超过一半上下文，结果为：

```text
init-tune attention NMSE = 0.00265735
init-tune complete Block NMSE = 0.00152742
validation attention NMSE = 0.00208100
validation complete Block NMSE = 0.00158776
oracle complete Block gate <= 0.0015
```

attention 通过，但 init-tune 和 validation 的完整 Block 都未通过 oracle gate。不能因其
低于最终产品 gate `0.003` 就继续进入 static projection、native recurrence、BF16、cache
和训练；oracle gate 更严格，目的正是为这些后续误差保留预算。

### 2.4 Hedgehog/H2O bounded sidecar + learned linear tail

后续实验使用固定 `8 sink / 64 recent / 56 heavy` exact slots、共享 denominator、
Hedgehog/LoLCATs feature map、线性 tail 和 rank-16 Q/K/V/O correction。修复 RoPE 数值
边界后，exact-prefix preflight 通过，但真实 layer-3 训练仍未达到产品门槛：

```text
Phase A best complete Block NMSE = 0.0066076923
Phase B best complete Block NMSE = 0.0050153371
product complete Block gate <= 0.003
```

Phase B 在 epoch 34 因每卡 batch size 仅为 8、显存利用率过低而停止；没有保存正式
artifact。即使将“停止原因”和“质量结果”分开看，当前最好结果仍高于产品 gate，不能
作为已验收 runtime 合入主线。

## 3. 保留的数学与验收边界

后续重新研究 GQA 时可以复用以下边界，但不能复活上述实现：

1. 比较 converted layer-0–2 prefix 上完整 decoder-block output，而不是只看 attention
   或 TMix。
2. source/student 必须使用一致的 RoPE dtype、attention reference、output dtype 和数值
   边界；混用 FP32 student reference 与 BF16 source eager output 不构成有效 gate。
3. sharp exact branch 与 approximate tail 必须共享未归一化 numerator/denominator；两个
   已独立归一化的输出再混合不等价。
4. architecture/capacity 只能用 init split 选择；统一 validation 只用于 checkpoint
   选择和最终验收。
5. oracle、static compiler、native recurrence、BF16/FP16、prefill/decode cache 和完整
   Block 是独立证据层，前一层未通过时立即停止。
6. 不得降低阈值、改 seed、扩大隐含 KV 容量、保存失败 artifact 或静默回退。

新的 GQA 方法只有在这些边界下通过严格 oracle 和产品 gate 后，才应重新引入 config、
checkpoint schema、训练和 runtime 代码。
