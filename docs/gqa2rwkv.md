# Qwen3.5 GQA 到可学习 RWKV TimeMix 的迁移

GQA 只在蒸馏阶段作为冻结 teacher。运行时使用固定大小的 RWKV recurrent state，
不保存 attention KV、teacher、sidecar 或蒸馏门控。保留 source Q/K/V/O、Q/K RMSNorm、
partial RoPE 和输出 gate，新增 RWKV 组件以不改变已有迁移函数的方式初始化。

这里扩展的是 source shell 内的 RWKV-7 记忆更新与读出能力。正值 feature map、
分母状态和恒等初始化的归一化接入方式属于迁移结构，不能将它称为逐项等同于原生
RWKV-7 模型。数学迁移决定初始化，后续蒸馏可以学习新增的状态更新参数。

## 状态与读出

每个 query head 的 feature projection 为 D256→F64，正负两部分形成 F128：

\[
\phi(x)=\frac{[\operatorname{ReLU}(xW),\operatorname{ReLU}(-xW)]}
{\max |xW|}+10^{-4}.
\]

记 \(r_t=\phi(q_t)\)、\(f_t=\phi(k_t)\)，学习到的控制量为逐通道保留率
\(d_t\) 和擦写系数 \(\eta_t\)：

\[
u_t=\operatorname{normalize}(f_t\odot k_k),\qquad
\kappa_t=f_t\odot[1+(\eta_t-1)\odot k_a],
\]
\[
S_t=S_{t-1}\operatorname{diag}(d_t)
 -(S_{t-1}u_t)(u_t\odot\eta_t)^T+v'_t\kappa_t^T,
\qquad z_t=d_t\odot z_{t-1}+f_t.
\]

Numerator 使用 RWKV-7 的 diagonal-plus-rank-one 更新。Denominator 使用同样的
对角衰减和正值写入，不执行有符号的低秩修正，保证分母为正。开启低秩更新后，
分母是独立的归一化尺度，整个读出不再解释为概率权重之和为 1 的 attention。

\[
h_t^0=\frac{S_tr_t}{z_t^Tr_t},\qquad
h_t=h_t^0+\tanh(\gamma)\odot[\operatorname{GN}(h_t^0)-h_t^0]
+\left[\sum_f r_{t,f}\kappa_{t,f}(r_k)_f\right]v'_t.
\]

最后仍通过 source `sigmoid(gate)` 和 `o_proj`。六个 token-shift 系数分别控制
read、decay、key、value、erase 和 gate 输入。Value residual 使用整个模型首层
GDN 的原始 value，按 token 跨层传递；逐层蒸馏缓存同时保存对应的 `v_first`。

两个 F128 状态补零到 FlashRWKV2 的 D256。8 个 query heads 对应 16 个 kernel
heads，FP16 recurrent state 为 **2,097,152 bytes/sequence/layer**；另外保存
**4,096 bytes** 的上一 token hidden 和 **4 bytes** 的 elapsed cursor，共
**2,101,252 bytes**，与上下文长度无关。Numerator/denominator 写入共同乘以
`1/256`，读取恢复尺度。没有 eviction、sink/recent/heavy slots。

训练使用 BF16 `pretrain_recurrent_bf16`，推理使用 FP16
`infer_recurrent_fp16_forward_varlen`。FP32 reference 显式执行同一更新式，供数值
验证使用。RoPE 保留 source 定义，按 config 重建 FP32 频率；本次没有移除 RoPE。

## 保持迁移函数的初始化

| 部件 | 初始化 | 初始作用 |
| --- | --- | --- |
| 已迁移 Q/K/V/O、norm、feature weights | 原样保留 | 保留迁移结果 |
| `x_r/x_w/x_k/x_v/x_a/x_g` | 0 | token shift 为恒等 |
| `w0/w2` | 0 | 保留原来的 `-30` decay logits |
| `a0/a2` | 0 | 低秩状态修正为 0 |
| `v0/v2` | 0 | value residual 为 0 |
| `w1/a1/v1` | 非零随机初始化，std=`1/sqrt(hidden_size)` | 零输出矩阵具有有效梯度 |
| `k_k` | 1 | 使用未额外缩放的归一化 key 方向 |
| `k_a/r_k` | 0 | 额外 key 调制与 RKV readout 为 0 |
| `norm_mix` | 0 | 新增归一化不改变输出 |
| GroupNorm affine weight/bias | 1 / 0 | 在 `norm_mix` 学习后参与计算 |

Value residual 使用 `tanh(v0 + sigmoid(x_v @ v1) @ v2)`，以有限参数取得精确零贡献。
GroupNorm affine 的 1/0 本身不构成恒等，因此通过 `norm_mix` 的残差形式接入。
这两个选择都是为了保持迁移函数，参数化与原生 sigmoid gate 不完全相同。

Decay 直接优化保留率对应的有界量，再转换到算子 logits：

\[
p_t=\operatorname{clip}(w_0+\tanh(x_wW_1)W_2,0,1-10^{-4})+e^{-30},
\quad w_t=\operatorname{logit}(p_t),
\quad d_t=\exp(-e^{-1/2}\sigma(w_t)).
\]

初始化时算子仍接收 `-30`，但 FP32 inverse-logit 链式求导避免把可训练参数困在
饱和 logits 上。`w0/w2` 学习率为普通参数的 **0.02 倍**：保留率的单位变化会在
长序列中累积，不能直接沿用投影矩阵的更新尺度。对 `w0/a0` 的更新在边界处投影。

初始化验收比较完整输出，不只是参数值；还检查 BF16 算子的真实反向传播，以及
非零动态参数下的 prefill/chunk/decode 一致性。

## 蒸馏与 checkpoint 选择

\[
Y_g=g\,\operatorname{stopgrad}(G)+(1-g)R,\qquad
L=\operatorname{NMSE}(Y_g,G)+4\operatorname{NMSE}(R,G).
\]

Gate 固定为 `0.9 → 0.75 → 0.5 → 0.25 → 0.1 → 0.0`。固定输入下损失等价于
`[4+(1-g)^2] * NMSE(R,G)`，gate 不增加学生容量。每阶段重新计算输出 cache，
当前层的输入 prefix 和首层 value 保持固定。Teacher 和 source MLP 始终冻结。

新迁移先做 feature attention transfer，再蒸馏全部 TimeMix 参数。从已有 layer
checkpoint 初始化时跳过 attention transfer。每个 epoch 在同一验证集测量完整
Block NMSE，同时运行启用实际动态参数的 recall probe；只选择 recall 通过的
候选中的最佳 Block NMSE。没有任意的 `0.003` 截止值。

Gate=0 的 `gqa_candidate.safetensors` 是可供重放验收的研究候选。只有完整验收
通过才写入正式 `layer_03.safetensors`；正式 layer checkpoint 用原子替换保存。

## 复现 layer-3 对照

使用固定的 4096×512 packed tensor，SHA-256 为
`1d039b73dcafd9783a7e872f682cf64728cb31f6090ef54c2882ca3bc0919336`。
8 卡运行时，全局行 `[0,64)` 用于初始化，`[64,192)` 用于验证，`[192,4096)` 用于
训练。随机种子为 42。两组从相同 v2 layer-3 权重及前三层 GDN checkpoint 开始。

将基线的 `packed_sequences.pt` 和 `layer_00/01/02.safetensors` **复制**到新的
输出目录，然后执行：

```bash
uv run python -m any2rwkv.qwen2rwkv.align.train \
  --source /path/to/Qwen3.5-2B \
  --output /path/to/new-experiment \
  --through-layer 3 \
  --gqa-initial-checkpoint /path/to/v2-baseline/layer_03.safetensors \
  --gqa-epochs 2 \
  --gqa-learning-rate 3e-5
```

`--gqa-epochs 2` 是每个 gate 两个 epoch，总上限 12 个 epoch。加入
`--gqa-freeze-dynamics` 得到相同预算、只训练已迁移参数的对照。默认训练新增组件。
`--gqa-epochs 0` 可重放候选的验收。旧 hidden-only prefix cache 缺少 `v_first`，
需要从前三层 checkpoint 重建；标准 `--through-layer 3` 流程会自动重建。

完整模型复用 v2 的所有 layer checkpoint 时，将 `--gqa-initial-checkpoint` 指向
该目录。GDN layer 会严格加载并只重建后续 cache；每个 GQA layer 则只接收其
source Q/K/V/O、norm 和 feature 权重，新增 RWKV 参数仍按中性初始化展开。

## Schema 与验收边界

当前 schema 为 `gqa_rwkv_tmix_d256x2_v3`；feature geometry 仍为 D256→F64→F128，
每个 query head 两个 D256 state。运行时 strict-load 拒绝 v2 及 bounded-Hedgehog
artifact。只有显式的训练初始化入口接受完整 v2 layer 权重，并以中性初始化补齐
新增组件；最终 state dict 不含 teacher、sidecar 或蒸馏 gate。

- **函数与梯度**：扩展初始化与已有迁移输出一致；零贡献分支具有有效梯度入口；
  首层 value 在逐层缓存与模型 forward 中保持一致。
- **数值与 runtime**：BF16/reference 增量 NMSE ≤`1e-3`；FP16 整段、64/128/256
  分块及逐 token 输出相对 L2 误差 ≤`1e-3`；8192-token soak 无 NaN/Inf，状态固定。
- **受控 recall**：距离 128/256/512/1024 的 hit@1 ≥0.9。Fixture 在 feature-space
  构造 key/query，并使用当前 projection 的伪逆，因此不是语言 benchmark，也不
  经过 source Q/K/V/O。独立固定随机 hidden 驱动已学习的 decay、erase、value
  residual，读出包含归一化和 RKV shortcut。负例检查 feature collapse 和过强
  learned decay，防止在测试中绕过已学习的遗忘机制。
- **整模型生成**：必须单独验证，不能由 Block NMSE 或上述合成 recall 推断；本次
  24-layer run 已完成该检查。

## 实验证据

v2 加法状态基线的 layer-3 validation Block NMSE 为约 `0.032859`。相同初始权重
再训练 12 个 epoch、冻结新增组件的对照得到 **`0.0321935403`**。

第一轮新增全部组件、但 decay 使用普通学习率的实验得到 Block NMSE
`0.0261223067`，BF16/reference TMix 增量 NMSE `3.1484e-5`，最大 FP16 chunk/decode
相对 L2 误差 `7.0731e-4`。然而，启用已学习动态参数后的 recall 在距离 512、1024
只有 **0.75 / 0.5**，因此该候选不满足本次验收。它说明只检查加法 kernel 会漏掉
新学习到的遗忘。后续实验使用较小的 decay 更新尺度，并将 recall 纳入逐轮选择。

使用 decay 学习率缩放 `0.02` 后的 12-epoch 候选在四个距离均为
`recall_hit_at_1=1.0`，validation Block NMSE 为 **`0.0294897953`**，
BF16/reference 增量 NMSE 为 `1.7750e-5`，最大 FP16 分块/解码相对 L2 为
`7.005e-4`。这是当前选择的 layer-3 候选。将相同流程延长到每个 gate 八个
epoch 后，Block NMSE 降至 `0.0257349016`，但 1024 距离 recall 降为 `0.9375`，
且完整模型数学生成出现错误答案；因此该较低 NMSE 候选被拒绝，没有替换正式
研究候选。

在 v2 完整模型上做 v3 中性扩展并分别 graft 两个 layer-3 checkpoint 的生成
smoke test 中，保守候选的中文解释和二次方程回答均正常结束；二次方程输出
`x=2`、`x=3`。代码提示产生了可读的 Python 草稿，但仍把 Fibonacci 函数返回
为二元组，这个问题在中性扩展基线中同样存在，不能归因于 layer-3 改动。两种
候选均没有 teacher、sidecar 或临时 gate 参数。该 smoke test 只证明运行和回答
结束，不替代语言质量评测。

完整 24-layer v3 run 使用同一目录级 v2 warm-start，6 个 GQA layer 均完成 gate=0
验收，四距 recall 全部为 `1.0`。最终模型位于
`/home/caizus/Weights/Qwen/Qwen3.5-2B-rwkv-tmix-v3-full-20260911`，包含 24 个
纯 RWKV layer checkpoint 和约 3.77 GB 的 `model.safetensors`；global KL 三个
epoch 为 `0.219678 → 0.192705 → 0.183399`。三条生成分别为 73、394、440 token，
均以 EOS 结束；中文解释、二次方程答案（`x=2,x=3`）和 Fibonacci 列表均可读。
这是一次固定提示的 generation smoke test，不替代完整语言评测。

layer-3 保守候选的完整模型 graft 产物仍保留在
`/home/caizus/Weights/Qwen/Qwen3.5-2B-rwkv-tmix-generation-20260911/retention`；
其中 `experiment.json` 记录了 v2 全模型基线、layer-3 checkpoint SHA-256 和
中性 v3 扩展方式，可用于复核 layer-3 生成差异。

历史 v2 完整模型位于
`/home/caizus/Weights/Qwen/Qwen3.5-2B-pure-rwkv-common-20260911`，其三轮 global KL
从 `0.2500983` 降至 `0.2059738`，有独立 generation smoke test。该历史结果不构成
当前 v3 的完整模型验收。
