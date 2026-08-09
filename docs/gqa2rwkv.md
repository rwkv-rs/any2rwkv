# 从 GQA 到 RWKV7：两个完整状态与一次对角修正

参考：

- [将 Softmax Attention 线性化为 Gated DeltaNet](https://spaces.ac.cn/archives/11823)
- [Sparse to Linear Attention](https://www.haoyizhu.site/blog/sparse-linear-attention/)

本文只讨论 `full_attention` 的 GQA→RWKV7。Qwen3.5-2B 的 source
Query head dimension 是 256，因此 target 也使用 256 维 native head；每个
source Query head 配两个完整的 \(256\times256\) 状态。GDN 保持它自己的原生
`128×128` geometry，不属于本文的改造范围。

我们的目标不是从随机 RWKV 开始模仿 Qwen，而是尽量把一个已经训练好的 GQA
层直接“编译”为 RWKV7：先找到 Softmax Attention 的递推近似，再把递推式写成
RWKV7 的状态更新，最后只对实际参数化留下的误差做闭式校正。这样得到的是第一步
optimizer 之前的解析初始化；如果还需要蒸馏，也只需从一个相当接近的起点出发。

## 1. 先把 Softmax 写成递推式

考虑一个 GQA group，它共享

$$
k_t,v_t\in\mathbb R^d,
$$

但服务于若干个不同的 Query heads。对任意查询 \(q\)，prefix attention 是

$$
o_t(q)=
\frac{\sum_{i\leq t}\exp(q^\top k_i/\sqrt d)v_i}
{\sum_{i\leq t}\exp(q^\top k_i/\sqrt d)}.
\tag{1}
$$

记当前 token 的 Softmax 权重为

$$
p_t(q)=
\frac{\exp(q^\top k_t/\sqrt d)}
{\sum_{i\leq t}\exp(q^\top k_i/\sqrt d)},
\tag{2}
$$

直接拆开分子、分母就有

$$
o_t(q)=(1-p_t(q))o_{t-1}(q)+p_t(q)v_t.
\tag{3}
$$

式 \((3)\) 是精确恒等式。麻烦只在于 \(p_t(q)\) 仍然依赖查询，不能直接作为
一个与未来查询无关的 RNN 状态。参考文章的关键启发，就是不要逐项拟合完整
Attention 矩阵，而是把这个查询依赖展开成“均值 + 线性算子”。

先做对称缩放

$$
q'_t=d^{-1/4}q_t,\qquad k'_t=d^{-1/4}k_t,
\tag{4}
$$

再定义 prefix 均值

$$
\bar k_t=\frac1t\sum_{i\leq t}k'_i,\qquad
\bar v_t=\frac1t\sum_{i\leq t}v_i,
\tag{5}
$$

以及中心化 Key 和 Value innovation

$$
\tilde k_t=k'_t-\bar k_t,\qquad
\Delta v_t=v_t-\bar v_{t-1}.
\tag{6}
$$

于是可以用一个矩阵 \(M_t\) 表示查询相关部分：

$$
\widehat o_t(q)=\bar v_t+M_tq'.
\tag{7}
$$

对应的 GDN 形式递推为

$$
\begin{aligned}
M_t={}&(1-\rho_t)M_{t-1}
+\rho_t\Delta v_t\tilde k_t^\top\\
&-\lambda\rho_t
(M_{t-1}\tilde k_t)\tilde k_t^\top,
\qquad \rho_t=\frac1t.
\end{aligned}
\tag{8}
$$

前三项分别是旧状态衰减、当前 Value-Key 写入和 rank-one erase；\(\lambda\)
控制一阶展开之后的 closure。它不是 Softmax 恒等式的一部分，而是修正
\(M_{t-1}\) 沿当前中心化 Key 方向过量延续的近似参数。当前 initializer 在
calibration 上分别完整编译

$$
\lambda\in\{0,1/16,1/8,1/4,1/2,1\},
$$

然后只用 development 上的真实 native mixer NMSE 选择有限候选。冻结 final
不参与 closure 选择。

这里存在不能靠换参数消除的有限秩边界。指数点积核
\(K(q,k)=\exp(q^\top k/\sqrt d)\) 在任意开集上有无限 feature rank，因为它的
Taylor 展开包含所有阶的对称张量幂。对任意经验 kernel matrix \(K\)，任何 rank
不超过 \(r=d+1=257\) 的近似 \(\widehat K\) 都满足 Eckart--Young 下界

$$
\frac{\|K-\widehat K\|_F^2}{\|K\|_F^2}
\ge
\frac{\sum_{j>257}\sigma_j(K)^2}{\sum_j\sigma_j(K)^2}.
\tag{8a}
$$

实现报告的 `empirical_kernel_tail_ratio` 就是式 \((8a)\) 在截断 calibration 或
development score matrix 上的谱尾。它是 relaxed kernel lower bound，不是完整
TMix output lower bound：Softmax 归一化、Value 投影与 `o_proj` 都可能让 kernel
误差落入输出不可见方向。

反过来，若 \(|q^\top k/\sqrt d|\le R<1\)，一阶 Taylor feature
\(1+q^\top k/\sqrt d\) 的逐项余项满足

$$
|e^s-(1+s)|\le \frac{e^R R^2}{2}.
\tag{8b}
$$

若同时 \(\|v_i\|\le V\)，对这个正的一阶 kernel 做归一化后有一个保守输出上界

$$
\|o(q)-\widehat o(q)\|
\le V e^{2R}R^2.
\tag{8c}
$$

式 \((8c)\) 只证明小 score 区间中的一阶 kernel candidate 是二阶误差；当前
centered affine recurrence 还带有在线均值、closure 与 native clamp，实际误差
仍必须由 exact teacher trace 直接测量，不能把式 \((8c)\) 当作实现保证。

## 2. 为什么正好需要两个完整的 \(256\times256\) 状态

式 \((7)\) 有两项：一个完整线性算子 \(M_t\)，以及一个 256 维偏置
\(\bar v_t\)。不妨取最后一个标准基

$$
e=(0,\ldots,0,1)^\top\in\mathbb R^{256},
\tag{9}
$$

并定义

$$
S_t^{(0)}=M_t,\qquad
S_t^{(1)}=\bar v_te^\top.
\tag{10}
$$

两路 read 分别取

$$
r_t^{(0)}=q'_t,\qquad r_t^{(1)}=e,
\tag{11}
$$

那么

$$
S_t^{(0)}r_t^{(0)}+S_t^{(1)}r_t^{(1)}
=M_tq'_t+\bar v_t
=\widehat o_t(q_t).
\tag{12}
$$

这就是两个状态的分工：

- 第一个状态保存完整的 \(256\times256\) 查询算子；
- 第二个状态用最后一列保存 prefix Value mean；
- 两个状态都完整保持 source head 的 256 维 Query 特征空间。

Qwen3.5-2B 有 8 个 Query heads，所以 `full_attention` target 使用 16 个
RWKV heads、`head_size=256`，也就是每个 Query head 对应一对状态。这里的
“两个”来自 affine operator 的齐次化，而不是把一个矩阵按元素数量拆成两半。

## 3. 把文章递推逐项编译到 RWKV7

RWKV7 单个 head 的状态更新可以写成

$$
S_t
=S_{t-1}\operatorname{Diag}(d_t)
-(S_{t-1}n_t)(n_t\odot a_t)^\top
+u_t\kappa_t^\top,
\tag{13}
$$

其中 \(n_t\) 是归一化 Key，第二项是 rank-one erase，第三项是 Value-Key
写入。对矩阵状态 \(S^{(0)}\)，令

$$
\begin{aligned}
d_t&=1-\rho_t,\\
\kappa_t^{(0)}&=\rho_t^\gamma\tilde k_t,\\
u_t^{(0)}&=\rho_t^{1-\gamma}\Delta v_t,\\
n_t^{(0)}&=\tilde k_t/\|\tilde k_t\|_2,\\
a_t^{(0)}&=
\lambda\rho_t\|\tilde k_t\|_2^2\boldsymbol 1.
\end{aligned}
\tag{14}
$$

因为

$$
u_t^{(0)}{\kappa_t^{(0)}}^\top
=\rho_t\Delta v_t\tilde k_t^\top,
\tag{15}
$$

所以任意 \(\gamma\in[0,1]\) 都给出同一个 oracle outer product。但它们投影到
真实 `key/value` 后并不等价，因此完整方案应在 calibration 内按 mixer output
选择 factorization gauge。当前实现先把中心化 Key 单位化，再把
\(\rho_t\|\tilde k_t\|\) 放进 Value write；这是一个固定且数值稳定的 gauge，
不是已经完成候选选择后的结论。

对均值状态 \(S^{(1)}\)，取

$$
\kappa_t^{(1)}=\rho_t e,\qquad
u_t^{(1)}=v_t,\qquad
r_t^{(1)}=e,
\tag{16}
$$

就得到标准 prefix mean 更新。RWKV7 的 decay 链接为

$$
d=\exp\left[-e^{-1/2}\sigma(w)\right],
\tag{17}
$$

其最小值约为 \(0.5452\)，而 \(t=1\) 请求的 decay 是 0。第二个状态只占最后一列，
所以可以在 \(e\) 方向增加 erase correction，抵消 mean state 的
`realized_decay - requested_decay`。这个修正不同时解决 matrix state 的各向同性
decay floor；matrix state 仍需裁剪并把该误差计入 native diagnostic。因此，式
\((8)\) 到无限制 DPLR oracle 可以逐项编译，但到 clamp-w native RWKV7 不是机器
精度无损映射。

## 4. 两个状态必须先求和，再过 target 输出边界

设两路状态读出为 \(z^{(0)},z^{(1)}\)，source GQA 的输出边界是

$$
y=W_o\left[g\odot(z^{(0)}+z^{(1)})\right].
\tag{18}
$$

因此不能对两路状态分别做 GroupNorm；那会破坏式 \((12)\) 的 affine 和。实际
module 按 `[query_head, state, channel]` 保存 interleaved states，先计算

$$
z_h=z_h^{(0)}+z_h^{(1)},
\tag{19}
$$

然后只对 8 个 D256 Query-head outputs 做一次逐 token GroupNorm。两路 state 的
current-token residual 也先按 Query head 求和，但它位于 GroupNorm 之后、gate
之前。因此 target 的真实边界是

$$
\widehat y
=\widehat W_o\left[
\widehat g\odot\left(
\operatorname{GN}_8(z)+\delta z_{r_k}
\right)
\right].
\tag{19a}
$$

不能把 `r_k` 提前送入 GroupNorm，也不能让两个 states 分别归一化。式 \((19a)\)
还明确暴露了 source 无 GroupNorm、target 有 GroupNorm 的结构差异；最后的 gate、
`r_k` 和 `output` 拟合负责在观测数据上尽量吸收它，而不是宣称边界代数无损。

这个 pair-sum 约定同时解决训练和推理的一致性：checkpoint 中仍然是标准
RWKV7 tensor，kernel 仍然处理 16 个独立的 \(256\times256\) states；target
module 在逐 token GroupNorm 之前把每对状态还原成一个 Query head 的读出。

## 5. 从动态 oracle 到静态权重

前面得到的是逐 token 的 \(r,w,k,v,a,g\) 目标，checkpoint 需要的是固定权重。
对 bias-free projection，统一求解

$$
W^*=
\arg\min_W
\|XW^\top-Y\|_F^2
+\lambda\|W-W_0\|_F^2.
\tag{20}
$$

`receptance` 和 `key` 的 oracle 是在 partial RoPE 之后定义的，所以 initializer
先对 matrix-state read/key 施加逆 RoPE，在 pre-RoPE 空间求式 \((20)\)。target
training forward 使用序列内位置，inference forward 使用 cache 中每条序列的
`elapsed`，再对两个 interleaved states 同时施加真实 Qwen3.5 partial RoPE。
mean-state 的 DC read/key 放在不参与旋转的最后一个坐标。这样训练、prefill 和
逐 token decode 使用同一位置语义；不允许靠固定 calibration 位置相关性让静态
projection 隐式记住 RoPE。构建器同时校验 `0 < rotary_dim < head_dim` 且
`rotary_dim` 为偶数；full-RoPE checkpoint 没有可用 DC 尾部，必须 fail closed，
不能套用这条双状态布局。

`receptance`、`key` 和 expanded `value` 使用 calibration 上固定正则的
time-mix/ridge 解；每个 closure 都从完整 TMix snapshot 重新拟合，不跨候选复用
projection。`output` 才使用 source `o_proj` 作为 relative-ridge prior，并在真实
native recurrence 产生的 pre-output 上重新求解。

GroupNorm affine 也不是固定为 1/0。实现先从真实 native recurrence 捕获 pair-sum
raw read，对每个 Query head/channel 闭式拟合 `ln_x.weight/bias`，使其逼近 exact
attention head output；calibration 与 development residual 分开报告。这个拟合
只能吸收当前轨迹上可见的 affine 差异，不能消除 GroupNorm 的 DC null direction。

其余分支按真实 RWKV7 参数化拟合：

- `w_lora`：反解式 \((17)\) 后拟合 `down → tanh → up+bias`；
- `a_lora`：对 erase logit 拟合 `down → up+bias`；
- `g_lora`：保持 `down → sigmoid → up`，并复制 source gate 的 head 对应关系；
- `output`：在完整 free-running recurrence、pair-sum、单次 GroupNorm、gate 和
  `r_k` 之后的真实 pre-output 上求解。

initializer 同时报告三个互不替代的 trajectory 指标：

- `reference_source_trace_output_nmse`：FP32 exact Softmax reference 经过 source gate/
  `o_proj` 后，相对真实 source attention forward 的数值差异；
- `affine_attention_nmse`：式 \((8)\) 的 affine recurrence 相对 exact causal
  Softmax teacher trace 的误差；
- `native_clamp_delta_nmse`：把同一 recurrence 的 decay/erase/write 放进
  clamp-w native 可达域后，相对 affine recurrence 新增的误差，并以 exact
  attention energy 归一化；
- `empirical_kernel_tail_ratio`：式 \((8a)\) 的 rank-257 relaxed kernel 谱尾，
  calibration 与 development 分开报告。

前者衡量 Softmax 有限状态近似，后者衡量 native 参数域；静态 projection 和
GroupNorm/output fitting 的误差则继续由真实 native mixer NMSE 衡量。任何一个
内部指标接近零，都不能代替完整 mixer 或 block 验收。

闭式 signal fit 只使用 calibration rows。最终边界以 source `o_proj` 为 prior，
交替拟合 `output` 与 `r_k`；在 development 上分别记录 prior、candidate 和实际
installed mixer NMSE，只有 candidate 有限且严格改善才原子安装，否则恢复完整
TMix snapshot。冻结 final rows 不参与任何拟合、方法选择或重试。

## 6. 用 `r_k` 补当前 token 的对角余项

即使 \(r,k,v,w,a,g,W_o\) 全部固定，标准 RWKV7 还保留一个很有用的现成参数
`r_k`。它贡献的当前 token 项为

$$
\delta z_{t,h}
=
\left[
\sum_j r_{t,h,j}k_{t,h,j}(r_k)_{h,j}
\right]v_{t,h}.
\tag{21}
$$

这个项正好对应“当前 token 的 Key-Value 写入尚未来得及经过长期状态传播”的
对角修正。更重要的是，在其他参数固定时，完整 mixer 输出对 \(r_k\) 是严格
线性的。于是可以直接求

$$
\min_\eta
\|Y_{\text{base}}+\mathcal A\eta-Y_{\text{Qwen}}\|_2^2
+\lambda\|\eta\|_2^2,
\qquad \eta=r_k.
\tag{22}
$$

当前实现不显式构造巨大设计矩阵，而用真实 gate、pair-sum 和 `output` 定义
\(\mathcal A\) 及其转置，再以 matrix-free conjugate gradient 求解。固定 gate
和 `output` 时，式 \((22)\) 对 `r_k` 严格线性；一旦 `output` 也同时未知，两者的
乘积使问题变成双线性，不能谎称为一次联合线性回归。因此实现先闭式 ridge
`output`，再固定它求 `r_k`，随后重新解 `output`，最多交替四轮。

整个交替候选只按 development 原子安装；当前还没有执行 frozen-final 或真实
PRO6000/BF16 复验，因此本文只说明算法与实现边界，不报告收益数字。旧
Helicopter 的 development/final 数字也不能作为本仓库当前验收证据。

## 7. 当前逐层对齐边界

当前 runner 解冻本层完整标准 RWKV7 TMix，而不是旧文档所说的仅优化 24,576 个
小向量。teacher、Cmix/MLP、decoder Norm、embedding、其它层和 value-residual
保持冻结；optimizer-train 只更新参数，development 负责选择 checkpoint，冻结
final 只在方法和参数冻结后评估一次。

每个 rank 的 rows `0:8`/`8:16`/`16:24`/`24:` 分别固定为 calibration、
development、frozen-final 和 optimizer-train。本阶段以 `--through-layer 3` 正常
停在第一层 GQA；完整 block NMSE 的硬门是 $10^{-3}$，mixer NMSE 单独优化和报告
但不单独阻断。

当前 initializer、target module 和 runner 分别位于
[`src/any2rwkv/qwen2rwkv/gqa2rwkv.py`](../src/any2rwkv/qwen2rwkv/gqa2rwkv.py)、
[`src/any2rwkv/qwen2rwkv/transformers/modeling_qwen2rwkv.py`](../src/any2rwkv/qwen2rwkv/transformers/modeling_qwen2rwkv.py)
和 [`src/any2rwkv/qwen2rwkv/align/train.py`](../src/any2rwkv/qwen2rwkv/align/train.py)。

## 8. 一条完整而可复现的迁移顺序

将上述推导压缩起来，GQA→RWKV7 的初始化顺序是：

1. 从真实 Qwen 层回放 post-RoPE Q/K、V、source gate 和 mixer output。
2. 按式 \((4)\) 至式 \((8)\) 构造 prefix mean 与完整 affine operator，
   在 calibration/development 上选择 closure \(\lambda\)。
3. 为每个 source Query head 建立两个完整 \(256\times256\) states，分别保存
   \(M_t\) 与 \(\bar v_te^\top\)。
4. 按式 \((14)\) 至式 \((17)\) 把两路状态递推编译为 RWKV7
   decay/erase/write；matrix state 使用单位化 centered Key，并把相应尺度放入
   Value write。
5. 在 pre-RoPE 空间闭式拟合 read/key；按真实 nonlinear/low-rank contract
   拟合 decay、erase、write 和 sigmoid gate。
6. 先 pair-sum 两路 raw read，再对 8 个 Query heads 做一次 GroupNorm；把两路
   raw read 对 exact attention 拟合一次 GroupNorm affine，再把两路 `r_k`
   residual pair-sum 后加在 GroupNorm 之后、gate 之前。
7. 交替求解 `output` ridge 与 matrix-free `r_k` CG，最多四轮；只在 development
   mixer NMSE 有限且严格改善时安装完整边界候选。
8. 把 tensor 物化进真实 head-size-256 module，以 BF16 kernel 在独立
   development/final rows 上验收；需要进一步对齐时，只解冻当前标准 TMix，
   再进入逐层蒸馏。

这条路线的要点其实很简单：先给 affine operator 足够的完整状态容量，再利用
RWKV7 已有的 rank-one transition 表达两路递推，并明确记录 decay floor、
Softmax→affine recurrence、静态 signal projection 和 native normalization 各自
留下的误差，而不会再把两状态 layout 错误混入结果。

本文所述源码改动尚未执行静态检查、reference probe 或真实 GPU 验收；这里描述的是
当前实现 contract，不是已经获得的 layer 3 精度证据。
