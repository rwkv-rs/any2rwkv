# GDN → RWKV7

Qwen3.5-2B 的 18 个 GDN 层使用唯一映射。对每个 D128 head，source recurrence

```text
S_t = d_t S_(t-1) - beta_t d_t (S_(t-1) k_t) k_t^T + beta_t v_t k_t^T
```

对应 RWKV7 的 decay、低秩 erase、write 与 read activation 子空间：

```text
r = L2(q) / sqrt(128)
k = L2(k)
a = beta * d
v = beta * v_source
```

native decay link 不可到达的 source decay 裁剪到最近边界。current/previous-token
mix 与线性投影执行固定三轮 diagonal alternating least squares；full-rank map 只做一次
truncated SVD，固定 decay/a/value/gate rank 为 128/128/64/224。最后只做一次无 bias
ridge，把 target pre-output 映射到 source mixer output。

这是 recurrence activation 子空间的解析对应；source conv、SiLU、gated RMSNorm 与 native
RWKV7 参数化意味着完整 block 仍需 greedy layerwise 对齐。不存在候选扫描、held-out gate、
rollback 或第二条初始化路径。`v0/v1/v2` 始终存在，但数学迁移和逐层阶段的
`value_residual_scale` 严格为零。
