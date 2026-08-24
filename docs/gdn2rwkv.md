# Qwen3.5 GDN：保留原 GDN 外围结构，仅替换 WKV

当前 GDN 路径不是把整个 token mixer 编译成 canonical RWKV7 TMix。转换后模型保留
`Qwen3_5GatedDeltaNet` 的完整外围结构和原始参数，只把 matrix-state recurrence
交给 FlashRWKV2 的 RWKV7 WKV operator 执行。

实现位于：

- [`gdn2rwkv.py`](../src/any2rwkv/qwen2rwkv/gdn2rwkv.py)：严格复制检查、解析
  recurrence 自检和 Clamp-W 诊断；
- [`modeling_qwen2rwkv.py`](../src/any2rwkv/qwen2rwkv/transformers/modeling_qwen2rwkv.py)：
  hybrid GDN runtime、Conv/WKV cache 和 FlashRWKV2 调用；
- [`train.py`](../src/any2rwkv/qwen2rwkv/align/train.py)：逐层蒸馏、checkpoint schema
  检查以及统一验证集验收。

## 1. 保留的原 GDN 边界

每个转换后 GDN 都直接继承 `Qwen3_5GatedDeltaNet`，并严格复制原模型
`linear_attn.state_dict()`。以下模块和参数全部保留：

- `in_proj_qkv`；
- kernel size 为 4 的 depthwise causal `conv1d` 及 SiLU；
- Q/K L2Norm；
- `in_proj_a`、`in_proj_b`、`A_log`、`dt_bias`；
- `in_proj_z`；
- per-head D128 `RMSNormGated`；
- `out_proj`。

因此蒸馏前不存在 GDN token shift、静态 Q/K/V projection、low-rank control
gate、GroupNorm、basis/gauge、`r_k` 或重新拟合的 output/readout。构建转换后模型与
每层初始化函数都严格复制 `state_dict`；任一 tensor key 或值不一致都会直接失败。

输出模型的 config 写入：

```text
gdn_mode = "source_shell_wkv7"
gdn_checkpoint_schema = "source_gdn_state_dict_v1"
```

缺少这两个标识的旧 `qwen2rwkv` GDN config 不按新运行路径静默加载。逐层恢复时，
checkpoint key 集合也必须和当前 TMix 完全一致；旧 canonical-RWKV GDN checkpoint
会被明确拒绝，需要使用新的输出目录。

## 2. 原 GDN activation 到 RWKV7 WKV

原 GDN frontend 先执行

```text
hidden
  -> in_proj_qkv
  -> Conv4
  -> SiLU
  -> split(q, k, v)
  -> q/k L2Norm
```

其它原 GDN control signal 为

$$
\beta_t=\sigma(\operatorname{in\_proj\_b}(x_t)),
$$

$$
\ell_t
=
-\exp(A_{\log})
\operatorname{softplus}
\left(\operatorname{in\_proj\_a}(x_t)+dt_{\rm bias}\right),
\qquad d_t=\exp(\ell_t),
$$

以及

$$
z_t=\operatorname{in\_proj\_z}(x_t).
$$

对 D128 WKV kernel，代码逐项生成

$$
r_t=\frac{q_t}{\sqrt{128}},\qquad
k_t^{\rm wkv}=k_t,
$$

$$
v_t^{\rm wkv}=\beta_tv_t,\qquad
a_t^{\rm wkv}=k_t,
$$

$$
b_t^{\rm wkv}=-\beta_td_tk_t.
$$

在 kernel 的 `[key, value]` state 方向上，这组 activation 产生

$$
S_t
=d_tS_{t-1}
-\beta_td_tk_t(k_t^\top S_{t-1})
+\beta_tk_tv_t^\top,
$$

并以 $r_t^\top S_t$ 读出。它正是原 GDN 在 Q/K L2Norm 后的 delta-rule
recurrence。这里没有把 Conv4、原 GDN gate 或输出边界吸收到 WKV 权重中。

WKV raw read reshape 为 per-head D128 后，仍执行原 GDN 边界：

```text
raw WKV read -> RMSNormGated(raw, z) -> out_proj
```

它不经过 RWKV GroupNorm、low-rank gate、`r_k` 或转换后 TMix 的 output projection。

## 3. Clamp-W 投影与梯度

当前 FlashRWKV2 D128 kernel 使用

$$
\ell^{\rm native}=-e^{-1/2}\sigma(w).
$$

令

$$
u=\frac{\ell}{-e^{-1/2}}.
$$

运行路径将 $u$ 投影到

$$
\widehat u=\operatorname{clamp}(u,10^{-4},1-10^{-4}),
$$

再传入

$$
w=\operatorname{logit}(\widehat u).
$$

令

$$
\widehat\ell=-e^{-1/2}\widehat u,
\qquad \widehat d=\exp(\widehat\ell).
$$

运行路径对同一个 realized decay 一致地生成 `w` 和
$b_t^{\rm wkv}=-\beta_t\widehat d_tk_t$。因此域内
$\widehat d_t=d_t$ 时仍是上面的精确映射；域外则把完整 decay/erase 转移项投影到
Clamp-W 可达边界，而不是只投影各向同性 decay。

forward 始终使用投影后的值。训练时 clamp 使用 straight-through estimator：

```python
projected = ratio + (ratio.clamp(1e-4, 1 - 1e-4) - ratio).detach()
```

所以域外 token 的 forward 仍服从 Clamp-W 原生可达域，而 backward 对原 GDN
`A_log`、`in_proj_a` 和 `dt_bias` 保留有限梯度。训练入口继续对本层 trainable
parameters 使用现有 global gradient clipping。

Clamp-W 是当前 GDN 蒸馏前唯一保留的结构近似。初始化函数分别在初始化集和
验证集 activation 上报告：

- 原 GDN recurrence raw 自检 NMSE；
- 原 GDN `RMSNormGated/out_proj` 完整边界自检 NMSE；
- decay 域外 fraction；
- log-decay 投影 NMSE；
- 仅替换成投影后 decay 所产生的 raw 与 TMix 输出 NMSE。

这些数值只用于诊断，不参与权重拟合或 checkpoint 选择。

## 4. 训练、推理和 cache

训练路径要求 contiguous BF16 `[B,T,2048]` 且 `T` 能被 16 整除，调用
`pretrain_recurrent_bf16`。原 GDN 外围结构的全部 copied parameters 都参与当前层
`1e-5`、最多 48 epochs 的逐层蒸馏；“只替换 WKV”约束的是架构和蒸馏前权重
来源，不冻结原 GDN 外围结构。验证集负责选择最佳 checkpoint，现有 global
gradient clipping 保持不变。

推理路径使用 FP16 varlen recurrent operator。每个 GDN cache layer 保存：

- Conv4 的最近 4 个 projected-QKV frontend states；
- D128 WKV recurrent state；
- FP16 recurrent operator 的 elapsed state。

prefill 通过原 GDN causal Conv4 更新 conv cache；单 token decode 使用原 GDN
`causal_conv1d_update` 原地推进相同 cache。WKV state 和 elapsed state 由现有 varlen
operator 原地更新。GQA 层继续保存原有 one-token shift/WKV cache。

GDN 不生成、覆盖或消费 RWKV value-residual `v_first`。执行到第一个 GQA 层时，
该 GQA 层从自己的 `value_base` 建立本次 forward 的 `v_first`；后续 GQA 层沿用。

## 5. 数据、指标和验收边界

“蒸馏前”指严格复制原 GDN 参数，再执行 Clamp-W forward projection，不做参数拟合。
每张卡上的数据固定分为三部分：

- rows `0:8`：初始化集；GDN 只做蒸馏前诊断，GQA 用它计算解析初始化；
- rows `8:24`：统一验证集；选择 GQA 初始化候选和逐层最佳 checkpoint；
- rows `24:`：训练集；用于反向传播和参数更新。

当前 `--through-layer 3` 依次覆盖前三层 GDN 和第一层 GQA。验证集选择出的最佳
权重会在训练集和统一验证集上分别测量。统一验证集的整层输出 NMSE 必须不高于
`3e-3`；TMix 输出 NMSE 单独报告和优化，但不是独立硬门。

某层未达到严格目标时，当前进程仍会在内存中继续测到 `--through-layer` 指定的层，
以获得完整误差曲线；命令结束时仍返回失败。从第一个未通过层开始不保存正式
`layer_XX.safetensors`，后续结果只表示“基于未通过前缀的诊断”，不构成验收通过。

验收证据必须分层记录：

1. CPU/FP32 reference 只证明 state 方向、read scale、write、erase、原 GDN 输出边界
   与 Clamp-W 可见误差计算正确；
2. D128 BF16 FlashRWKV2 forward/backward 才能证明训练 kernel 路径有限；
3. FP16 prefill/decode cache-reuse 对比才能证明 Conv/WKV cache 路径一致；
4. 逐层统一验证集结果决定整层输出 NMSE 是否达到 `<=3e-3`，不再维护第二套 test
   指标。

任一层证据都不能替代其它层。静态检查或 CPU reference 不构成 GPU kernel 与完整
逐层验收。
