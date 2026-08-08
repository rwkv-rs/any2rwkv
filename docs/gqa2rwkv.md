# 从 GQA 到 RWKV7：两个完整状态与一次对角修正

参考：[将 Softmax Attention 线性化为 Gated DeltaNet](https://spaces.ac.cn/archives/11823)

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
控制一阶展开之后的 closure。它不是凭经验随便塞进去的门，而是专门修正
\(M_{t-1}\) 沿当前中心化 Key 方向的过量延续。当前 initializer 继承旧实验的
\(\lambda=0.25\) 作为固定初值；在本仓库重新完成 calibration/development
选择前，不能把它表述为当前数据上的最优值。

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
\tag{17}
$$

就得到标准 prefix mean 更新。RWKV7 的 decay 链接为

$$
d=\exp\left[-e^{-1/2}\sigma(w)\right],
\tag{18}
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
\tag{19}
$$

因此不能对两路状态分别做 GroupNorm；那会破坏式 \((12)\) 的 affine 和。实际
module 先计算

$$
\tilde z^{(0)}=\tilde z^{(1)}
=\frac{z^{(0)}+z^{(1)}}2,
\tag{20}
$$

再把 source gate 和 `o_proj` 的对应通道复制到两个 slots。两个 slots 经过
线性输出边界相加后，两个 \(1/2\) 恰好恢复式 \((19)\)。

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
\tag{21}
$$

`r_proj` 和 `k_proj` 的目标是在 partial RoPE 之后定义的，所以先施加逆 RoPE，
在 pre-RoPE 空间求式 \((21)\)，验证时再施加真实 RoPE。第二个状态的 DC read
放在不参与旋转的最后一个坐标。

\(W_0\) 也不应一刀切地取零。对 `r_proj/k_proj/v_proj/o_proj`，分别在
calibration 内划分 fit/selection，独立比较

$$
W_0=0
\quad\text{和}\quad
W_0=W_{\text{source projection}},
\tag{22}
$$

然后用选中的相对 ridge 在全部 calibration rows 上重解。这样，少量样本没有
覆盖到的 hidden directions 可以保留 source 权重，而已经改变语义的动态信号
也不会被 source prior 强行拉回去。

其余分支按真实 RWKV7 参数化拟合：

- `w_lora`：反解式 \((18)\) 后拟合 `down → tanh → up+bias`；
- `a_lora`：对 erase logit 拟合 `down → up+bias`；
- `g_lora`：保持 `down → sigmoid → up`，并复制 source gate 的 head 对应关系；
- `o_proj`：在完整 free-running recurrence 的 pair-sum 输出上求解。

所有充分统计量按 row/time 流式累计；ridge 选择只看 calibration-selection，
adaptive development 只负责接受或拒绝完整候选，冻结 final rows 不参与任何
拟合和方法选择。

## 6. 用 `r_k` 补当前 token 的对角余项

即使 \(r,k,v,w,a,g,W_o\) 全部固定，标准 RWKV7 还保留一个很有用的现成参数
`r_k`。它贡献的当前 token 项为

$$
\delta z_{t,h}
=
\left[
\sum_j r_{t,h,j}k_{t,h,j}(r_k)_{h,j}
\right]v_{t,h}.
\tag{23}
$$

这个项正好对应“当前 token 的 Key-Value 写入尚未来得及经过长期状态传播”的
对角修正。更重要的是，在其他参数固定时，完整 mixer 输出对 \(r_k\) 是严格
线性的。于是可以直接求

$$
\min_\eta
\|Y_{\text{base}}+\mathcal A\eta-Y_{\text{Qwen}}\|_2^2
+\lambda\|\eta\|_2^2,
\qquad \eta=r_k.
\tag{24}
$$

可选的完整实现可以不显式构造巨大设计矩阵，而用真实 gate、pair-sum 和
`output` 定义 \(\mathcal A\) 及其转置，再以 matrix-free conjugate gradient
求解。当前本仓库尚未移植这一步，`r_k` 初始化为零；在实现、development 选择
和 frozen-final 复验完成前，不宣称存在这项收益。

上式描述的是可选的闭式校正问题。只有当前实现真实求解并在 frozen-final split
复验后，才可以报告对应的数值；旧 Helicopter 的 development/final 数字不能作为
本仓库当前 PRO6000/BF16 转换的证据。

## 7. 当前逐层对齐边界

当前 runner 解冻本层完整标准 RWKV7 TMix，而不是旧文档所说的仅优化 24,576 个
小向量。teacher、Cmix/MLP、decoder Norm、embedding、其它层和 value-residual
保持冻结；optimizer-train 只更新参数，development 负责选择 checkpoint，冻结
final 只在方法和参数冻结后评估一次。

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
4. 按式 \((14)\) 至式 \((18)\) 把两路状态递推编译为 RWKV7
   decay/erase/write，并选择可投影性最好的 write-factor gauge。
5. 在 pre-RoPE 空间闭式拟合 read/key，在 source projection prior 与零 prior
   之间逐 projection 选择；按真实 nonlinear/low-rank contract 拟合其余分支。
6. 在 pair-sum 后的完整 mixer output 上求 `o_proj`，不对两路 affine states
   分别做 GroupNorm。
7. 将 `r_k` 初始化为零；若以后实现式 \((24)\)，必须通过 development 选择并在
   frozen-final 上复验。
8. 把 tensor 物化进真实 head-size-256 module，以 BF16 kernel 在独立
   development/final rows 上验收；需要进一步对齐时，只解冻当前标准 TMix，
   再进入逐层蒸馏。

这条路线的要点其实很简单：先给 affine operator 足够的完整状态容量，再利用
RWKV7 已有的 rank-one transition 表达两路递推，并明确记录 decay floor、
Softmax→affine recurrence、静态 signal projection 和 native normalization 各自
留下的误差，而不会再把两状态 layout 错误混入结果。
