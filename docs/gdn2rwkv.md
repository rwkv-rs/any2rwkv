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

这个可达域还给出一个严格的 operator 下界。若 $d_t<d_{\min}$，记
$\alpha_t=d_{\min}-d_t$。任意 native transition 都是

$$
A_t^R=\operatorname{Diag}(\delta_t)-uv^\top,
\qquad \delta_{t,j}\ge d_{\min}.
$$

而

$$
\operatorname{Diag}(\delta_t-d_t)+d_t\beta_tk_tk_t^\top\succeq\alpha_tI.
$$

根据 Eckart--Young 定理，一个 rank-one erase 最多消掉其中一个奇异方向，因此

$$
\|A_t^R-A_t^G\|_2\ge\alpha_t,
\qquad
\|A_t^R-A_t^G\|_F\ge\sqrt{127}\,\alpha_t.
\tag{10a}
$$

式 $(10a)$ 是 transition operator 的下界，不是 mixer NMSE 下界。实际输出还取决于
source state 能量、未来 read 和输出边界，所以实现同时报告直接下界和把 floor
残差放回 source trace 后的 observable output NMSE，绝不把二者混为一谈。

## 3. 先生成 activation oracle，再拟合有限维参数

式 $(3)$ 给出的信号随 token 变化，而 target 只能用 time-mix、静态投影和低秩 control subspace 生成它们。因而参数编译分成两层：

1. 从 source 真实前向中保存 $\bar q_t,\bar k_t,v_t,d_t,\beta_t$、gate、state read 和 mixer output；
2. 用式 $(3)$ 生成精确的 $r_t,\kappa_t,u_t,\delta_t,n_t,a_t$ activation oracle；
3. 再把 oracle 投影到 native 参数化。

write gauge $\mu_t$ 原则上不应预先固定。对每个 head，在 calibration set 上希望最小化

$$
\min_{\mu_t>0}
\mathcal E_{\kappa}(\mu_tk_t)
+
\mathcal E_u(\beta_tv_t/\mu_t),
\tag{11}
$$

其中两项分别是 native Key 与 Value signal compiler 的 output-weighted 回归残差。这样，动态 normalization 的尺度会被放到更容易拟合的一侧。

当前实现比较两个完整候选：`unity` 使用 $\mu_t=1$；`balanced` 使用

$$
\mu_t=
\operatorname{clip}_{[1/4,4]}
\sqrt{\frac{\|\beta_tv_t\|_2}{\|k_t\|_2}}.
\tag{11a}
$$

后者在写入 outer product 不变的前提下平衡 Key 与 Value factor 的尺度。两个 gauge
都与 identity/observability basis 组成完整 TMix 候选，只按 development mixer
NMSE 选择；式 $(11a)$ 是确定性初始化，不宣称已经直接解出式 $(11)$ 的全局最优。

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

若保持 exact matrix-state oracle，不允许把 RWKV state 另作卷积 cache，则所有
compiler signal 都只能依赖 $Z_t=(x_t,x_{t-1})$。对任意 source post-activation
signal $s_t$，甚至允许任意非线性估计器时也有

$$
\inf_f\mathbb E\|s_t-f(Z_t)\|^2
=
\mathbb E\operatorname{Tr}\operatorname{Cov}(s_t\mid Z_t).
\tag{12a}
$$

这是条件期望作为 $L^2$ 正交投影的直接结果。当前实现用独立 development 上的
read/key/write projection residual 作为可计算代理，并把三者最大值报告为
`temporal_conditional_residual`；它是 compiler diagnostic，不冒充式 $(12a)$ 的
无条件精确估计。

## 4. 用完整轨迹联合校正 readout

teacher-forced 的单步 activation 拟合只能生成初值，最终校正必须在 free-running recurrence 上进行。写成
$S_t^G=S_{t-1}^GA_t^G+B_t^G$、
$S_t^R=S_{t-1}^RA_t^R+B_t^R$，并令
$E_t=S_t^R-S_t^G$、$\Delta A_t=A_t^R-A_t^G$、
$\Delta B_t=B_t^R-B_t^G$，则有精确误差递推

$$
E_t
=E_{t-1}A_t^R
+S_{t-1}^G\Delta A_t
+\Delta B_t,
\qquad
\Delta y_t
=
E_tr_t^R
+S_t^G\Delta r_t.
\tag{13}
$$

反复使用次乘性可得一个可计算的轨迹上界

$$
\|E_t\|_F
\le
\sum_{i=1}^t
\left(
\|S_{i-1}^G\|_F\|\Delta A_i\|_2
+\|\Delta B_i\|_F
\right)
\prod_{j=i+1}^t\|A_j^R\|_2,
\tag{13a}
$$

以及

$$
\|\Delta y_t\|_2
\le
\|E_t\|_F\|r_t^R\|_2
+\|S_t^G\|_2\|\Delta r_t\|_2.
\tag{13b}
$$

若后续 normalization/gate/output 边界在当前轨迹邻域的 Lipschitz 常数为 $L_t$，
完整 mixer output 误差再至多乘以 $L_t$。因此 decay floor 与 compiler residual
会按未来 transition norm 累积；逐层蒸馏的意义正是直接最小化这个轨迹误差，而非
重复学习式 $(3)$ 已经相等的 canonical recurrence。

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

- `ln_x.weight/bias`；
- gate up projection；
- `output`；
- `r_k`；
- read/write gauge 的静态近似。

source 对每个 D128 head 使用 RMSNorm，不减去通道均值；标准 RWKV7 的 `ln_x`
对每个 head 使用 GroupNorm，会减去一个 DC 方向。令

$$
C=I-\frac1{128}\mathbf1\mathbf1^\top.
$$

GroupNorm 满足 $G(z+c\mathbf1)=G(z)$，所以 $J_G(z)\mathbf1=0$，每个 target
head 的历史可见 Jacobian rank 至多为 127。带正 epsilon 的 RMSNorm Jacobian为

$$
J_R(z)=\frac1sI-\frac{zz^\top}{128s^3},
\qquad s^2=\|z\|^2/128+\epsilon,
$$

其径向特征值是 $\epsilon/s^3>0$，因此泛型满秩。固定当前 token、只改变历史
state 时，16 个 target heads 的局部历史输出 rank 至多为 2032；source 在 state
局部可达、gate/`out_proj` 可观测的条件下可以达到 2048。于是 canonical
GroupNorm target 不存在对全部可达历史的全局无损参数。

这不意味着 held-out NMSE 必然很高。实现从 source RMSNorm、SiLU gate 与
`out_proj` 的局部 Jacobian构造每个 head 的 observability Gram，取最小特征向量
$u_h$，再用确定性 Householder 变换把 $u_h$ 映射到 target DC 方向。identity 与
observability 两个完整 TMix 候选只按独立 development mixer NMSE 选择。

权威 native ABI 的 `r_k` residual 位于 GroupNorm 之后、gate 之前。在 gate 与
`output` 固定时最终 mixer output 对 `r_k` 线性；`r_k` 与 `output` 联合优化则是
双线性问题。因此实现交替进行 matrix-free CG 的 `r_k` 解和闭式 `output` ridge，
最多四轮，并对完整边界候选做原子 development 选择。

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
- `source_recurrence_raw_nmse` 比较数学 oracle raw read 与从 source
  `RMSNormGated` 输入边界直接捕获的真实 recurrent read；
- `source_trace_output_nmse` 使用 source RMSNorm、SiLU gate 和 `out_proj`，用于检查
  手写 recurrence trace 是否真正复现 source mixer；
- `source_trace_pre_output_nmse` 在 `out_proj` 之前做同一比较，避免输出投影的
  null space 掩盖 trace 错误；
- `decay_operator_floor_bound_mean/rms/max` 报告式 $(10a)$ 的 operator 下界，
  `decay_observable_oracle_nmse` 报告同一 floor 残差在 source 输出边界上的可见误差；
- `groupnorm_rank_tail_bound` 报告每 head 丢失一个局部方向时的 Jacobian谱尾比例；
  Householder Gram 从 calibration 中确定性抽取最多 2048 个 token，并用
  `groupnorm_observability_tokens` 明确报告样本数；
- `temporal_conditional_residual` 报告 independent development 上 read/key/write
  compiler residual 的最大值；
- `observable_structural_diagnostic_max` 只取 decay observable、GroupNorm tail 与
  temporal residual 三个归一化 diagnostic 的最大值；这些量不能相加，这个最大值
  也不是 frozen-final 完整 TMix 的严格全局下界；
- identity 与 observability basis、`r_k`/`output` prior 与 candidate 都分别报告
  calibration/development mixer NMSE、最终安装值和安装状态；
- native zero-step mixer NMSE 还包含 signal compiler、decay 可达域、GroupNorm、
  gate 和 output projection 的误差；
- frozen-final block NMSE 才是本阶段硬验收，且 final rows 不参与拟合、选参或重试。

layerwise runner 对每个 rank 固定使用 rows `0:8` 做 calibration、`8:16` 做
development、`16:24` 做 frozen-final、`24:` 做 optimizer-train。本阶段运行参数为
`--through-layer 3`，因此只处理 layer 0--3；完整 block NMSE 的硬门是 $10^{-3}$，
mixer NMSE 单独优化和报告但不单独阻断。

本文所述源码改动尚未执行静态检查、reference probe 或真实 GPU 验收；这里描述的是
当前实现 contract，不是已经获得的 layer 0--3 精度证据。

## 6. 完整迁移顺序

最终流程可以写成八步：

1. 在真实 layer input 上回放 GDN，保存归一化前后的 Q/K/V、$d$、$\beta$、gate、state read 和 mixer output。
2. 按式 $(3)$ 构造逐项相等的 RWKV activation oracle，并用它校验状态坐标与实现方向。
3. 按式 $(12)$ 初始化 time-mix 与静态 Q/K/V projection，并计算独立 development
   compiler residual。
4. 对 decay、erase 和 gate 做 weighted reduced-rank regression；decay 先用式 $(6)$ 初始化。
5. 按式 $(9)$、式 $(10)$ 将域外 decay 与结构化残差投影到 native recurrence。
6. 分别物化 identity/observability Householder basis 与 unity/balanced write
   gauge 的四个完整候选；每个候选都重新拟合真实 gate、GroupNorm affine、`r_k`
   和 `output`。`r_k`/`output` 交替求解，只有独立 development 完整 mixer NMSE
   有限且改善时才安装，否则原子恢复。
7. 只用独立 development full-mixer NMSE 选择 zero-step checkpoint；frozen-final
   在方法和 checkpoint 固定后只评估一次。
8. 冻结其余层，以真实 layer input 开始逐层蒸馏；完成当前层后再推进下一层。

GDN→RWKV7 的核心优势不是“无需训练”，而是状态递推已经有机器精度的解析锚点。zero-step 的任务因此非常明确：把 native 参数化留下的偏差压到尽可能小，再让逐层蒸馏修正最后的 signal compiler、decay 可达域和 readout 差异。
