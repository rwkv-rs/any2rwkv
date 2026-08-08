# GQA → RWKV7

Qwen3.5-2B 的 6 个 full-attention 层固定使用每个 source query head 两个 D256 state：

- Taylor/delta associative matrix state；
- prefix value mean state。

保留 8 query heads、2 KV heads、4:1 group map、Q/K RMSNorm、partial RoPE 与
`1/sqrt(256)` scale。初始化时构造 exact causal Softmax trace；平均 attention lag 决定
decay，attention concentration 决定 write strength，Q/K 保持 direction 并拟合 row scale。
所有 ridge 使用

```text
lambda = 1e-4 * trace(X^T X) / feature_dim
```

两个 state 的 recurrent raw read 与 diagonal bonus先相加为 `8×256`，再执行 8-group
GroupNorm、source-compatible gate 和 output projection。共同的 2048 维 `value_base` 在
两条 write path 之前承载 value residual。

Softmax hazard reference 可以精确递推，但有限 RWKV state 与 native low-rank 参数化仍是
初始化近似；最终由 greedy layerwise NMSE 和全模型 teacher-logits KL 收敛。不实现
KV-repeat、单状态、curvature/gauge scan、output-refit candidate 或 polish 分支。
