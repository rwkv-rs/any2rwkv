# 从 GDN 到 RWKV7：先保持递推，再编译信号

参考：

- [将 Softmax Attention 线性化为 Gated DeltaNet](https://spaces.ac.cn/archives/11823)
- [Qwen3.5 GDN 与 RWKV7 逐项参考实现](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py)

Gated DeltaNet 与 RWKV7 的关系，比 GQA 与 RWKV7 更直接：两者的状态更新本来就属于同一个 DPLR 家族。因此迁移时最重要的不是重新发明一套记忆，而是先把 GDN 的真实 activation 编译成一个逐项相等的 RWKV oracle，再处理生成这些 activation 的参数化差异。

这里把“zero-step”定义为第一次 optimizer step 之前的初始化。它包括精确 oracle、闭式投影和一次 native free-running 校正；随后仍然进行逐层蒸馏。这样，蒸馏只需要修正 native decay、time-mix、低秩控制和归一化留下的误差，而不必重新学习状态递推。

## 1. GDN 递推可以逐项写成 RWKV7

以 `[value, key]` 为状态坐标，GDN 的单步更新为

$$
S_t
=
d_tS_{t-1}
-d_t\beta_t(S_{t-1}k_t)k_t^\top
+\beta_tv_tk_t^\top,
\qquad
y_t=S_t\frac{q_t}{\sqrt N}.
\tag{1}
$$

native RWKV7 的更新为

$$
\widehat S_t
=
\widehat S_{t-1}\operatorname{Diag}(\delta_t)
-(\widehat S_{t-1}n_t)(n_t\odot a_t)^\top
+u_t\kappa_t^\top,
\qquad
\widehat y_t=\widehat S_tr_t.
\tag{2}
$$

Qwen 的带 epsilon L2 normalization 并不保证归一化后的 Key 恰好是单位向量。记

$$
k_t=\frac{\bar k_t}{\sqrt{\|\bar k_t\|^2+\epsilon}},
\qquad
m_t=\|k_t\|.
$$

对任意正数 $\mu_t>0$，取

$$
\delta_t=d_t\mathbf 1,
\qquad
n_t=\frac{k_t}{m_t},
\qquad
a_t=d_t\beta_tm_t^2\mathbf 1,
\qquad
\kappa_t=\mu_tk_t,
\qquad
u_t=\frac{\beta_tv_t}{\mu_t},
\qquad
r_t=\frac{q_t}{\sqrt N}.
\tag{3}
$$

代入式 $(2)$，有

$$
-(S_{t-1}n_t)(n_t\odot a_t)^\top
=
-d_t\beta_t(S_{t-1}k_t)k_t^\top,
$$

以及

$$
u_t\kappa_t^\top=\beta_tv_tk_t^\top.
$$

因此，只要直接使用式 $(3)$ 中的 activation，状态和 readout 都与 GDN 逐项相等。$\mu_t$ 是 write 的尺度 gauge：它只在 Key 与 Value 之间搬运尺度，不改变 rank-one 写入。

这个恒等式给出了迁移的锚点。后续所有误差都应相对于这个 oracle 测量，而不是混入“GDN 与 RWKV7 的递推是否兼容”这一已经解决的问题。

## 2. native decay 应按可观测误差投影

式 $(3)$ 中的 DPLR 算子允许任意 $d_t\in(0,1]$，但 native RWKV7 的 decay link 为

$$
d_t^R
=
\exp[-c_w\sigma(z_{w,t})],
\qquad
c_w=e^{-1/2}.
\tag{4}
$$

所以它的可达域是

$$
d_t^R\in[\exp(-e^{-1/2}),1)
\approx[0.54524,1).
\tag{5}
$$

域内 target 可以先做 inverse link：

$$
z_{w,t}^{*}
=
\operatorname{logit}
\left(
\frac{-\log d_t}{c_w}
\right).
\tag{6}
$$

域外值则不能仅凭逐元素距离决定如何修正，因为 decay、erase 和 write 会共同影响最终读出。定义

$$
A_t^G=d_tI-d_t\beta_tk_tk_t^\top,
\qquad
B_t^G=\beta_tv_tk_t^\top,
\tag{7}
$$

以及 native 算子

$$
A_t^R
=
\operatorname{Diag}(\delta_t)
-n_t(n_t\odot a_t)^\top,
\qquad
B_t^R=u_t\kappa_t^\top.
\tag{8}
$$

由未来 read vectors 构造 Gram

$$
G_t
=
\sum_{\tau\ge t}
w_{t,\tau}r_\tau r_\tau^\top.
\tag{9}
$$

正确的 projection 是在 native 可达域内联合求解 decay、erase 和 write，使

$$
\sum_t
\operatorname{Tr}
\left[
(A_t^G-A_t^R)G_t(A_t^G-A_t^R)^\top
\right]
+
\lambda_B
\left\|
(B_t^G-B_t^R)G_t^{1/2}
\right\|_F^2
\tag{10}
$$

最小。把式 $(6)$ 的 clipped inverse link 当作初值，再用完整 recurrence rollout 修正式 $(10)$，就能把 native decay 的不可达部分优先放到真实 Query 看不见或不敏感的方向。

## 3. 先生成 activation oracle，再拟合有限维参数

式 $(3)$ 给出的信号随 token 变化，而 target 只能用 time-mix、静态投影和低秩 control subspace 生成它们。因而参数编译分成两层：

1. 从 source 真实前向中保存 $\bar q_t,\bar k_t,v_t,d_t,\beta_t$、gate、state read 和 mixer output；
2. 用式 $(3)$ 生成精确的 $r_t,\kappa_t,u_t,\delta_t,n_t,a_t$ activation oracle；
3. 再把 oracle 投影到 native 参数化。

write gauge $\mu_t$ 不应预先固定。对每个 head，在 calibration set 上交替求解

$$
\min_{\mu_t>0}
\mathcal E_{\kappa}(\mu_tk_t)
+
\mathcal E_u(\beta_tv_t/\mu_t),
\tag{11}
$$

其中两项分别是 native Key 与 Value signal compiler 的 output-weighted 回归残差。这样，动态 normalization 的尺度会被放到更容易拟合的一侧。

GDN 的四阶 causal convolution 与 RWKV7 的 one-step time-mix 不必逐项相等。对每类 oracle signal，先做

$$
\min_{B_0,B_1}
\sum_t
\left\|
s_t^*-(B_0x_t+B_1x_{t-1})
\right\|_{M_t}^2,
\tag{12}
$$

再把 $(B_0,B_1)$ 按 native 的共享 mixing 结构做 covariance-weighted rank-one projection，从而恢复 time-mix ratio 与静态 projection。decay、erase 和 gate 的 control subspace 则用 weighted reduced-rank regression 初始化；保留多少方向由 held-out full-mixer NMSE 决定，而不是只根据 driver 数量作维数推断。

## 4. 用完整轨迹联合校正 readout

teacher-forced 的单步 activation 拟合只能生成初值，最终校正必须在 free-running recurrence 上进行。令状态误差为 $E_t=\widehat S_t-S_t$，一阶传播满足

$$
E_t
\approx
E_{t-1}A_t^G
+S_{t-1}\Delta A_t
+\Delta B_t,
\qquad
\Delta y_t
=
E_tr_t
+S_t\Delta r_t.
\tag{13}
$$

将 source normalization、gate 和 `o_proj` 的局部可观测度量记为 $C_t$，闭式校正求解

$$
\min_{\Delta\theta}
\sum_t
\left\|
C_t
\left(
E_tr_t+S_t\Delta r_t
\right)
\right\|_2^2
+
\lambda\|\Delta\theta\|_2^2.
\tag{14}
$$

随后在真实 native rollout 上联合重算：

- `g_norm.weight/bias`；
- gate up projection；
- `o_proj`；
- `r_k`；
- read/write gauge 的静态近似。

RMSNorm 与 GroupNorm 的差异也在这个最终可见目标中处理。可以用 value-space 正交变换选择更有利的初始坐标，但模型选择必须看独立 held-out 文本上的 full-mixer output，而不能仅看归一化前的状态 NMSE。

## 5. 当前实现与指标边界

当前 initializer 位于
[`src/any2rwkv/qwen2rwkv/gdn2rwkv.py`](../src/any2rwkv/qwen2rwkv/gdn2rwkv.py)，
标准 RWKV7 TMix 位于
[`src/any2rwkv/qwen2rwkv/transformers/modeling_qwen2rwkv.py`](../src/any2rwkv/qwen2rwkv/transformers/modeling_qwen2rwkv.py)，
逐层运行和冻结 split 验收位于
[`src/any2rwkv/qwen2rwkv/align/train.py`](../src/any2rwkv/qwen2rwkv/align/train.py)。

从 Helicopter 搬来的旧文档曾引用本仓库不存在的 `evidence/` 和 `scripts/` probe，
并记录 GB10/FP32 的历史数字。它们不是当前 PRO6000/BF16 运行的验收证据，因此
不再作为本仓库的当前结论。

必须严格区分三类指标：

- canonical recurrence 自检只证明式 $(3)$ 的状态方向、尺度和 readout 正确；
- native zero-step mixer NMSE 还包含 signal compiler、decay 可达域、GroupNorm、
  gate 和 output projection 的误差；
- frozen-final block NMSE 才是本阶段硬验收，且 final rows 不参与拟合、选参或重试。

## 6. 完整迁移顺序

最终流程可以写成八步：

1. 在真实 layer input 上回放 GDN，保存归一化前后的 Q/K/V、$d$、$\beta$、gate、state read 和 mixer output。
2. 按式 $(3)$ 构造逐项相等的 RWKV activation oracle，并用它校验状态坐标与实现方向。
3. 交替选择 write gauge，按式 $(12)$ 初始化 time-mix 与静态 Q/K/V projection。
4. 对 decay、erase 和 gate 做 weighted reduced-rank regression；decay 先用式 $(6)$ 初始化。
5. 按式 $(9)$、式 $(10)$ 将域外 decay 与结构化残差投影到 native recurrence。
6. 运行 native free-running rollout，按式 $(14)$ 联合重算 norm、gate、`r_k` 与 `o_proj`。
7. 只用独立 held-out full-mixer NMSE、cosine 和长序列 drift 选择 zero-step checkpoint。
8. 冻结其余层，以真实 layer input 开始逐层蒸馏；完成当前层后再推进下一层。

GDN→RWKV7 的核心优势不是“无需训练”，而是状态递推已经有机器精度的解析锚点。zero-step 的任务因此非常明确：把 native 参数化留下的偏差压到尽可能小，再让逐层蒸馏修正最后的 signal compiler、decay 可达域和 readout 差异。
