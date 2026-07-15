---
title: Markov、Reciprocal 与前后向势函数
description: 澄清 Markov、reciprocal 与 reference bridge mixture 的差别，并建立前后向势函数和双向动力学。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 4
slug: b4-markov-reciprocal-dynamics
tags:
  - schrodinger-bridge
  - reciprocal-process
  - markov-process
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 区分三类路径结构，讨论 endpoint factorization、时变势和前后向 drift。
---
## 1. 三个经常被混在一起的对象

给定 reference Markov law `R`，以下三句话并不等价：

1. `P` 使用 `R` 的 point-conditioned bridges；
2. `P` 是 reciprocal；
3. `P` 是 Markov。

更不能由其中任一句直接推出 `P` 是 entropy-minimizing Schrödinger Bridge。

B2 已证明，固定 endpoint coupling `gamma` 后的 KL-optimal lift 是

```text
P_gamma = integral R^{xy} gamma(dxdy).                 (1.1)
```

所有这类 mixtures 都共享 reference conditional bridges。Jamison 的结构理论说明：
它们至少是 reciprocal；只有 endpoint law 相对于 `R_0T` 具有适当 product
factorization 时，mixture 才进一步是 Markov。

本章的主线是

```text
endpoint coupling
 -> reciprocal bridge mixture
 -> factorization test
 -> Markov two-sided transform
 -> forward/backward kernels and drifts.               (1.2)
```

## 2. Markov 与 reciprocal 的条件独立性

Markov 性是单向时间的局部条件独立：给定现在，未来与过去独立。以三个时刻
`s<t<u` 表示，

```text
Law(X_u | X_[0,t]) = Law(X_u | X_t).                   (2.1)
```

reciprocal 性是区间内外的双边条件独立。对 `s<t`，给定 `(X_s,X_t)` 后，区间
`[s,t]` 内部与外部独立。有限五时刻版本可写成

```text
X_2 independent of (X_0,X_4) given (X_1,X_3).          (2.2)
```

[Jamison 1974](https://doi.org/10.1007/BF00532864 "官方论文页面")
Lemma 1.2 证明 Markov implies reciprocal，但 converse 一般失败。reciprocal
允许 global endpoint information 同时影响整个区间；这种双边依赖不必能由单个
current state 屏蔽。

历史上，Jamison 和 Léonard 将 reciprocal formulation 归于 Bernstein 1932。
本教程只转述这个 attribution，不在未核验 Bernstein 原文时声称独立优先权。

## 3. Endpoint mixture 为什么保持 reciprocal

设 reference endpoint law 为 `R_0T`，conditional bridges 为 `R^{xy}`。对任意
endpoint coupling `gamma<<R_0T`，

```text
dP_gamma/dR
 = dgamma/dR_0T(X_0,X_T).                              (3.1)
```

density tilt 只依赖 global endpoints。考虑任意子区间 `[s,t]`：给定
`X_s,X_t` 后，reference Markov factorization 将 interval interior 与 exterior
分开。乘上的 endpoint factor属于 exterior，不改变 interior conditional law。
因此 mixture 保持 reciprocal conditional independence。

Jamison Theorem 2.1 在其 sigma-compact Hausdorff canonical setup 中给出正式构造
和唯一性。这里的唯一性是：给定 endpoint coupling 与 reciprocal transition
family 后，path law 唯一；它不说同一 endpoint marginals 只有一个 coupling。

## 4. 何时 bridge mixture 仍为 Markov

若 endpoint coupling 可写成

```text
gamma(dxdy)
 = f_0(x) g_T(y) R_0T(dxdy)/Z,                         (4.1)
```

则

```text
dP/dR=f_0(X_0)g_T(X_T)/Z.                              (4.2)
```

terminal factor可由 conditional expectation 向后传播，initial factor 可沿 reference
backward kernels 向前传播，从而得到 time-inhomogeneous Markov factorization。

在 Jamison Theorem 3.1 的 common sigma-finite domination 与 strictly positive
transition density 条件下，converse 也成立：bridge mixture 是 Markov 当且仅当
endpoint law 具有 (4.1) 的 q-times-product form。Theorem 3.2 在 sigma-compact
metric state 与 continuous strictly positive kernel 下进一步给出：固定 endpoint
marginals 时，恰有一个 coupling 具有该 Markov factorization。

\`\`恰有一个'' 指 coupling/product measure；factors 仍有 gauge：

```text
(f_0,g_T) -> (c f_0,c^{-1}g_T).                        (4.3)
```

## 5. Reciprocal 但非 Markov 的最小反例

在二状态 endpoint table 上，从 positive factorized `gamma*` 出发，加入保持
row/column marginals 的 cycle

```text
Delta = epsilon [[ 1,-1],
                 [-1, 1]].                             (5.1)
```

令 `gamma_alt=gamma*+Delta`。两者拥有相同 endpoint marginals，但不同 endpoint
couplings。相对 reference endpoint law 的 ratio matrix

```text
M=gamma/R_0T                                             (5.2)
```

对 `gamma*` 为 rank one；cycle perturbation 一般使 `det(M)!=0`。因此
`P_gamma_alt` 仍共享 exact reference bridges并为 reciprocal，却不再 Markov。

`reciprocal_markov_checks.py` 精确枚举 `2^5=32` 条路径：

- 两种 laws 的 reciprocal conditional mutual information约 `1e-16`；
- factorized law 的 Markov CMI `1.21e-16`；
- perturbed law 的 Markov CMI `7.02e-3`；
- shared conditional bridge error `2.78e-17`；
- factorized ratio determinant `0`，alternative determinant `0.947`。

CMI 是 finite fixture 的诊断，不是一般定义。正式 converse 仍由 Jamison theorem
承担。

![相同 endpoint marginals 下 factorized 与 cycle-perturbed coupling 的比较](/images/bridge/B4_reciprocal_markov.png)

**图 5.1：** 两个 coupling 具有相同 endpoint marginals，也诱导相同 reference
bridges；cycle perturbation 破坏 density ratio 的 rank-one factorization，并使
Markov conditional mutual information 变为正，而 reciprocal 诊断仍为数值零。

## 6. Two-sided factors 与中间 marginals

先在 finite time-inhomogeneous Markov chain 上工作。reference forward kernel 为

```text
K_t(i,j)=R(X_{t+1}=j|X_t=i),                           (6.1)
```

reference marginals 为 `r_t`。定义 propagated terminal factor

```text
g_t(i)=E_R[g_T(X_T)|X_t=i]
      =sum_j K_t(i,j)g_{t+1}(j).                       (6.2)
```

reference backward kernel 是

```text
Kbar_t(j,i)
 =R(X_t=i|X_{t+1}=j)
 =r_t(i)K_t(i,j)/r_{t+1}(j),                          (6.3)
```

不是普通 matrix transpose。initial factor 传播为

```text
f_{t+1}(j)=sum_i Kbar_t(j,i)f_t(i).                    (6.4)
```

由 endpoint transform 得中间 marginal

```text
p_t(i)=r_t(i)f_t(i)g_t(i)/Z.                           (6.5)
```

这就是 SB 文献常见的 forward/backward potentials product。`f_t,g_t` 的方向必须
按 conditional expectation 定义，不能靠名称猜测。

## 7. 前向和反向 kernels

forward transformed kernel 为

```text
K_t^P(i,j)
 =K_t(i,j) g_{t+1}(j)/g_t(i).                          (7.1)
```

backward transformed kernel 为

```text
Kbar_t^P(j,i)
 =Kbar_t(j,i) f_t(i)/f_{t+1}(j).                       (7.2)
```

(6.2)/(6.4) 分别保证 row normalization。沿 forward path 相乘时 `g` telescopes；
沿 backward path 相乘时 `f` telescopes。配合 transformed endpoint marginal，二者
重建同一个 path law。

`two_sided_transform_checks.py` 在正的三状态四步 chain 上验证：

- factor marginal error `1.67e-16`；
- forward kernel error `2.22e-16`；
- backward kernel error `4.44e-16`；
- path reconstruction error `2.08e-17`；
- gauge errors 不超过 `1.67e-16`。

## 8. 从 kernels 到 diffusion drifts

令 reference diffusion 使用 convention

```text
dX_t=b_t(X_t)dt+sigma_t(X_t)dW_t,
a_t=sigma_t sigma_t^T,
L_t phi=b_t dot grad phi+(1/2)a_t:Hess phi.             (8.1)
```

若 transition densities 和 factors 足够光滑，transformed forward drift 是

```text
b_t^{P,+}=b_t+a_t grad log g_t.                         (8.2)
```

对 reversed-clock process `Y_tau=X_{T-tau}`，reference reverse drift记为
`btilde^R`，则

```text
btilde_{T-t}^P=btilde_{T-t}^R+a_t grad log f_t.         (8.3)
```

若使用 Nelson original-time backward drift

```text
b_t^- = lim_{h downarrow 0}
        E[(X_t-X_{t-h})/h | X_t],                      (8.4)
```

则 `btilde_{T-t}=-b_t^-`。在 smooth density 情形，

```text
b_t^- = b_t^{P,+}-div a_t-a_t grad log p_t.             (8.5)
```

结合 `p_t=r_t f_t g_t/Z` 可回到 (8.3)。

这些 gradient formulas 需要 positive differentiable factors、well-posed martingale
problem/SDE 和 endpoint-limit control，不能从 finite kernel identity 形式外推。

## 9. Current、osmotic 与 finite entropy

[Cattiaux et al. 2023](https://doi.org/10.1214/22-AIHP1320 "官方论文页面")
使用的 backward-arrow velocity 是
`E[(X_{t-h}-X_t)/h|X_t]=-b^-`。因此其定义映射为

```text
v_current=(b^+ + b^-)/2,
v_osmotic=(b^+ - b^-)/2.                               (9.1)
```

在 reversible Kolmogorov reference 的 Hypotheses 1.10、Markov `P` 和
`H(P|R)<infinity` 下：

- Proposition 4.6 构造前后向 measurable velocities 并给两侧 kinetic entropy
  identities；
- Proposition 4.8 给 current continuity equation；
- Theorem 4.9 给 distributional time-reversal/relative score identity及 finite
  integrated Fisher information。

因此 low regularity 下可以谈 distributional osmotic score，但不能假设 pointwise
`grad log p_t` 光滑存在。B4 的 smooth formulas 是该 theorem 在更强条件下的
可读 corollary，不替代 theorem 本身。

## 10. Structural zeros 与 converse 失败

Jamison 的 clean iff 使用 strictly positive kernel。若 reference support 有 zeros：

- `gamma/R_0T` 在 support 外无定义；
- factorization 可能只在 connected support components 上成立；
- factors/coupling 的 uniqueness 可能改变；
- Markov endpoint tilt 可能只有 intermediate-time weak factorization。

Föllmer--Gantert 1997
Example 2.17 给出 degenerate Markov mixture，其中 ordinary factorization/uniqueness
失败。Theorem 3.43 在 general setup 只保证

```text
factorization => entropy minimization => Markovity,     (10.1)
```

reverse arrows 需要 product domination 或 conditional regularity (3.42)。在
infinite-dimensional Brownian setting，essentially bounded endpoint density 恢复
equivalence；Proposition 5.31 又证明所有 finite `L^p` moments 仍不足以保证
measurable factorization。

所以 \`\`entropy optimizer is always f(X\_0)g(X\_T)R'' 不是无条件真理。

## 11. Doob transform 在 SB 中的位置

[Doob 1957](https://doi.org/10.24033/bsmf.1494 "官方论文页面") 研究
Green-space/Martin-boundary `h`-paths，是 one-sided transform 的经典概率论来源。
FPY 1993
Proposition 1 则严格构造 finite-time Markov bridges 和 non-homogeneous Doob kernel。

SB 的 two-sided law 可从 forward 方向看成 terminal `g_t` Doob transform，initial
factor进入 transformed initial law；也可从 backward 方向看成 `f_t` transform。
但 \`\`Doob transform'' 本身不包含 endpoint marginal optimization：B3 的
Schrödinger system 决定选择哪些 factors。

## 12. 常见错误

1. **mixture of Markov bridges is Markov。** 一般只保证 reciprocal。
2. **相同 endpoint marginals 等于相同 coupling。** cycle perturbation 直接反驳。
3. **把 backward kernel 写成 `K^T`。** 必须包含 reference marginals。
4. **混用 backward drift convention。** Cattiaux backward-arrow、Nelson `b^-` 与
   reversed-clock drift 有符号映射。
5. **把 smooth score formula 当 low-regularity theorem。** finite entropy theorem
   只保证 distributional derivative和 integrated control。
6. **在 structural zeros 下套 positive-kernel iff。** support geometry会改变结论。
7. **由 Markovity 推出 entropy optimality。** general converse 需要额外 regularity。

## 13. 小结

本章的结构可压缩为

```text
arbitrary endpoint coupling
 -> same reference bridges
 -> reciprocal law;

factorized coupling under stated support conditions
 -> Markov two-sided transform
 -> f_t/g_t marginals and kernels
 -> smooth or finite-entropy forward/backward dynamics.          (13.1)
```

exact SB 在 B3 的正 kernel条件下位于第二条链：entropy optimization 选择 factorized
coupling，因而 optimizer 同时为 reciprocal 与 Markov。其他 bridge mixtures 即使
使用 exact reference bridges，也不自动是 SB。

## 14. 研究式思考题

1. 对 non-rectangular support，怎样用 bipartite support components 重写
   factorization 与 gauge uniqueness？
2. 是否存在 Markov、entropy-minimizing、但无 measurable endpoint factors 的
   path law？Föllmer--Gantert 的反例改变了哪些直觉？
3. 若只知道全部 one-time marginals，能否检测 reciprocal 或 Markov 性？还需要
   哪些 multi-time statistics？
4. 对 degenerate diffusion matrix，(8.2)--(8.5) 中哪些 gradient directions可识别？
5. learned bridge matching 的一次 reciprocal/Markov projection 保留什么，丢失什么？

## 15. 前后章节链接

- B1：point-conditioned Brownian/Markov/diffusion bridge；
- B2：reference-bridge lift 与 dynamic/static entropy；
- B3：factor existence、Schrödinger system 与 entropy optimum；
- B5：Markov SB 与 entropic interpolation/OT；
- B6：forward/backward control energy；
- B9：reciprocal projection、Markovian projection 与 IMF。

完整条件矩阵见
`references/notes/derivations/bridge/b4_theorem_responsibility.md`，kernel/drift
推导见 `references/notes/derivations/bridge/two_sided_transform_kernels.md`。
