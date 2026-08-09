# Qwen3.5 GDN：保留 source shell，仅替换 WKV

当前 GDN 路径不是把整个 token mixer 编译成 canonical RWKV7 TMix。target 保留
`Qwen3_5GatedDeltaNet` 的完整外围结构和原始参数，只把 matrix-state recurrence
交给 FlashRWKV2 的 RWKV7 WKV operator 执行。

实现位于：

- [`gdn2rwkv.py`](../src/any2rwkv/qwen2rwkv/gdn2rwkv.py)：严格复制检查、解析
  recurrence 自检和 Clamp-W 诊断；
- [`modeling_qwen2rwkv.py`](../src/any2rwkv/qwen2rwkv/transformers/modeling_qwen2rwkv.py)：
  hybrid GDN runtime、Conv/WKV cache 和 FlashRWKV2 调用；
- [`train.py`](../src/any2rwkv/qwen2rwkv/align/train.py)：逐层蒸馏、checkpoint schema
  检查和 frozen-final 验收。

## 1. 保留的 source 边界

每个 GDN target 都直接继承 `Qwen3_5GatedDeltaNet`，并严格复制 source
`linear_attn.state_dict()`。以下模块和参数全部保留：

- `in_proj_qkv`；
- kernel size 为 4 的 depthwise causal `conv1d` 及 SiLU；
- Q/K L2Norm；
- `in_proj_a`、`in_proj_b`、`A_log`、`dt_bias`；
- `in_proj_z`；
- per-head D128 `RMSNormGated`；
- `out_proj`。

因此 zero-step 不存在 GDN token shift、静态 Q/K/V projection、low-rank control
gate、GroupNorm、basis/gauge、`r_k` 或重新拟合的 output/readout。构建 target 与
每层 initializer 都使用 strict `state_dict` copy；任一 tensor key 或值不一致都会
直接失败。

artifact config 写入：

```text
gdn_mode = "source_shell_wkv7"
gdn_checkpoint_schema = "source_gdn_state_dict_v1"
```

缺少这两个标识的旧 `qwen2rwkv` GDN config 不按新 runtime 静默加载。逐层恢复时，
checkpoint key 集合也必须和当前 TMix 完全一致；旧 canonical-RWKV GDN checkpoint
会被明确拒绝，需要使用新的输出目录。

## 2. source activation 到 RWKV7 WKV

source frontend 先执行

```text
hidden
  -> in_proj_qkv
  -> Conv4
  -> SiLU
  -> split(q, k, v)
  -> q/k L2Norm
```

其它 source control signal 为

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

并以 $r_t^\top S_t$ 读出。它正是 source GDN 在 Q/K L2Norm 后的 delta-rule
recurrence。这里没有把 Conv4、source gate 或输出边界吸收到 WKV 权重中。

WKV raw read reshape 为 per-head D128 后，仍执行 source 边界：

```text
raw WKV read -> RMSNormGated(raw, z) -> out_proj
```

它不经过 RWKV GroupNorm、low-rank gate、`r_k` 或 target output projection。

## 3. Clamp-W 投影与梯度

当前 FlashRWKV2 D128 kernel 使用

$$
\ell^{\rm native}=-e^{-1/2}\sigma(w).
$$

令

$$
u=\frac{\ell}{-e^{-1/2}}.
$$

runtime 将 $u$ 投影到

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

runtime 对同一个 realized decay 一致地生成 `w` 和
$b_t^{\rm wkv}=-\beta_t\widehat d_tk_t$。因此域内
$\widehat d_t=d_t$ 时仍是上面的精确映射；域外则把完整 decay/erase 转移项投影到
Clamp-W 可达边界，而不是只投影各向同性 decay。

forward 始终使用投影后的值。训练时 clamp 使用 straight-through estimator：

```python
projected = ratio + (ratio.clamp(1e-4, 1 - 1e-4) - ratio).detach()
```

所以域外 token 的 forward 仍服从 native Clamp-W 可达域，而 backward 对 source
`A_log`、`in_proj_a` 和 `dt_bias` 保留有限梯度。runner 继续对本层 trainable
parameters 使用现有 global gradient clipping。

Clamp-W 是当前 GDN zero-step 唯一保留的结构近似。initializer 分别在 calibration
和 development activation 上报告：

- source recurrence raw 自检 NMSE；
- source `RMSNormGated/out_proj` 完整边界自检 NMSE；
- decay 域外 fraction；
- log-decay 投影 NMSE；
- 仅替换成 projected decay 后的 raw 与 mixer NMSE。

这些数值只用于诊断，不参与权重拟合或 checkpoint 选择。

## 4. 训练、推理和 cache

训练路径要求 contiguous BF16 `[B,T,2048]` 且 `T` 能被 16 整除，调用
`pretrain_recurrent_bf16`。GDN source shell 的全部 copied parameters 都参与当前层
`1e-5`、最多 48 epochs 的逐层蒸馏；“只替换 WKV”约束的是架构和 zero-step 权重
来源，不冻结 source shell。development 负责选择 checkpoint，现有 global
gradient clipping 保持不变。

推理路径使用 FP16 varlen recurrent operator。每个 GDN cache layer 保存：

- Conv4 的最近 4 个 projected-QKV frontend states；
- D128 WKV recurrent state；
- FP16 recurrent operator 的 elapsed state。

prefill 通过 source causal Conv4 更新 conv cache；单 token decode 使用 source
`causal_conv1d_update` 原地推进相同 cache。WKV state 和 elapsed state 由现有 varlen
operator 原地更新。GQA 层继续保存原有 one-token shift/WKV cache。

GDN 不生成、覆盖或消费 RWKV value-residual `v_first`。执行到第一个 GQA 层时，
该 GQA 层从自己的 `value_base` 建立本次 forward 的 `v_first`；后续 GQA 层沿用。

## 5. 指标和验收边界

zero-step 指 strict source-shell copy 加 Clamp-W forward projection，没有 calibration
fit。随后逐层蒸馏仍使用固定 split：

- rows `0:8`：calibration，只供 initializer 诊断或 GQA initializer；
- rows `8:16`：development，选择逐层 checkpoint；
- rows `16:24`：frozen-final，只在方法和 checkpoint 固定后评估；
- rows `24:`：optimizer-train。

当前 `--through-layer 3` 依次覆盖前三层 GDN 和第一层 GQA。每层完整 Block NMSE
在 development 和 frozen-final 上都必须不高于 `1e-3`；mixer NMSE 单独报告和
优化，但不是独立硬门。

验收证据必须分层记录：

1. CPU/FP32 reference 只证明 state 方向、read scale、write、erase、source 输出边界
   与 Clamp-W 可见误差计算正确；
2. D128 BF16 FlashRWKV2 forward/backward 才能证明训练 kernel 路径有限；
3. FP16 prefill/decode cache-reuse 对比才能证明 Conv/WKV cache 路径一致；
4. layerwise development/frozen-final 结果才决定 `<=1e-3` Block NMSE 硬门。

任一层证据都不能替代其它层。静态检查或 CPU reference 不构成 GPU kernel 与完整
layerwise acceptance。
