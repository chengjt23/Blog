---
title: 静态 Schrödinger Problem 与 Schrödinger System
description: 把路径空间问题约化为端点耦合，并推导 Schrödinger system、势函数与有限矩阵缩放结构。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 3
slug: b3-static-schrodinger-system
tags:
  - schrodinger-bridge
  - schrodinger-system
  - matrix-scaling
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦静态熵投影、正核条件、因子分解、存在唯一性边界与 finite matrix 示例。
---
## 1. 从 path law 到一个 endpoint matrix

B2 的 KL disintegration 已把 dynamic problem 化为

$$
\inf_{\gamma\in\Pi(\mu_0,\mu_T)} H(\gamma\mid R_{0T}). \tag{1.1}
$$

`R_0T` 是 reference 的 joint endpoint law，不一定是 product measure。B3 的问题是：

> 在什么条件下，(1.1) 的最优 coupling 相对于 `R_0T` 具有初端因子乘末端因子的
> density？这些因子怎样由边缘条件确定？

在有限正 kernel 中，答案是

$$
\gamma^*(i,j)=u_i\,R_{ij}\,v_j. \tag{1.2}
$$

`u,v` 满足两个 marginal equations，这就是 Schrödinger system。一般空间中，
(1.2) 的存在不能只靠形式 Lagrange multiplier；support、integrability、duality
attainment 和 measurable splitting 都可能失败。

## 2. Finite positive setup

令 `R=(R_ij)` 是严格正的有限 probability matrix，目标 marginals
`mu_i,nu_j` 也严格正且总质量为 1。可行 polytope 是

```text
Pi(mu,nu)
 = {gamma_ij>=0:
    sum_j gamma_ij=mu_i,
    sum_i gamma_ij=nu_j}.                               (2.1)
```

目标为

```text
H(gamma|R)=sum_ij gamma_ij log(gamma_ij/R_ij).          (2.2)
```

因为 `R_ij>0`，所有 positive feasible couplings entropy finite。strict positivity
也让 optimum 落在相对 interior；若某个 optimum entry 为零，可沿 positive feasible
direction 增加该 entry，`x log x` 在零点的右导数为负无穷，得到矛盾。

因此可以在 positive interior 使用普通 Lagrange multipliers。

## 3. Lagrange/KKT factorization

对 row/column constraints 引入 multipliers `alpha_i,beta_j`：

```text
L(gamma,alpha,beta)
 = sum_ij gamma_ij log(gamma_ij/R_ij)
   + sum_i alpha_i(mu_i-sum_j gamma_ij)
   + sum_j beta_j(nu_j-sum_i gamma_ij).                 (3.1)
```

stationarity 给

```text
log(gamma_ij/R_ij)+1-alpha_i-beta_j=0.                 (3.2)
```

令

```text
u_i=exp(alpha_i-1/2),
v_j=exp(beta_j-1/2),                                   (3.3)
```

得到

```text
gamma_ij=u_i R_ij v_j.                                 (3.4)
```

`1/2` 的分配没有意义；任何 constant 可从一侧移到另一侧。真正可观察的是 product
`u_i v_j`。

代入边缘约束：

```text
u_i (R v)_i = mu_i,
v_j (R^T u)_j = nu_j.                                 (3.5)
```

这就是有限 Schrödinger system。

## 4. Gauge 与 factors 的唯一性

若 `(u,v)` 解 (3.5)，则对任意 `c>0`

```text
(u,v) -> (c u,c^{-1}v)                                 (4.1)
```

给出相同 coupling。这个 gauge 不是非唯一 path law；它只是同一 rank-one density
的两种 factor coordinates。

在 strict-positive complete bipartite support 上，若

```text
u_i v_j = utilde_i vtilde_j,   all i,j,                (4.2)
```

固定任一 `j_0` 可得 `utilde_i/u_i=v_{j_0}/vtilde_{j_0}` 与 `i` 无关，因此 factors
只差 (4.1)。若 support disconnected，同样论证只在每个 support component 成立，
可能出现多个 gauge degrees of freedom。

## 5. KL Pythagorean identity 与 coupling 唯一性

设 `gamma*=diag(u)Rdiag(v)` 满足目标 marginals。对任意
`gamma in Pi(mu,nu)`，

```text
log(gamma*/R)=log u_i+log v_j.                         (5.1)
```

展开：

```text
H(gamma|R)-H(gamma|gamma*)
 = sum_ij gamma_ij log(gamma*_ij/R_ij)
 = sum_i mu_i log u_i + sum_j nu_j log v_j.            (5.2)
```

右侧只依赖 marginals，令 `gamma=gamma*` 可识别为 `H(gamma*|R)`，因此

```text
H(gamma|R)
 = H(gamma|gamma*) + H(gamma*|R).                      (5.3)
```

KL 非负性给 `gamma*` 最优；等号只有 `gamma=gamma*`，所以 coupling 唯一。这个
证明比只写 strict convexity 更有用，因为它同时给 objective gap，并固定 KL direction。

## 6. Cross-ratio 诊断

ratio matrix

```text
M_ij=gamma*_ij/R_ij=u_i v_j                           (6.1)
```

是 rank one。任意 indices 满足

```text
M_ij M_kl = M_il M_kj.                                (6.2)
```

在 `2x2` 情形，等价于 `det(M)=0`。它是 B4 finite mixture 是否 factorized 的
方便诊断；一般空间的 Markov iff theorem 仍需 Jamison 条件，数值 rank test 不是
定义。

## 7. 从 static coupling 回到 path law

若 `gamma*` 解 (1.1)，B2 给 dynamic optimizer

```text
P* = integral R^{xy} gamma*(dxdy).                     (7.1)
```

在 factorized 情形，

```text
dP*/dR=f_0(X_0)g_T(X_T)/Z.                             (7.2)
```

B4 证明它是 two-sided Markov transform，并给中间 marginals

```text
p_t=r_t f_t g_t/Z.                                    (7.3)
```

因此以下三层必须分开：

1. B2 entropy chain rule 说明只需优化 endpoint coupling；
2. B3 factor theorem 说明 endpoint density何时是 product；
3. B4 Markov theorem 说明 product tilt怎样改变 dynamics。

## 8. Schrödinger 1932 与 Fortet 1940

Schrödinger 1932
Section VII 写出从 particle migration likelihood 得到的 coupled integral equations，
但没有给一般 existence/uniqueness proof。

Fortet 1940 在实数区间
上系统研究

```text
phi(x) integral g(x,y)psi(y)dy = omega_1(x),
psi(y) integral g(x,y)phi(x)dx = omega_2(y).            (8.1)
```

Hypotheses I 要求 nonnegative measurable integrable data 与相等正质量；
Hypotheses II 加 continuous bounded-above kernel、每个 section a.e. positive、
continuous marginals。

Fortet 给两个不同存在路线：

- Theorem I：额外 reciprocal-denominator integrability；
- Theorem II：kernel 属于 class `(B)`；
- Theorem III：在 standing assumptions 下，positive Borel solution gauge-unique。

不能把两个 alternative existence条件和 uniqueness theorem 压成“continuous
positive kernel 总有唯一解”。Fortet 原文也不处理 abstract Polish state 或
general structural zeros。

## 9. Jamison 的 positive-kernel theorem

[Jamison 1974](https://doi.org/10.1007/BF00532864 "官方论文页面")
Theorem 3.2 在 sigma-compact metric state、common sigma-finite dominating measure
和 continuous strictly positive transition density 下，证明给定 probability
marginals 存在唯一 q-times-product coupling。

它同时连接 B3 与 B4：product coupling 是 fixed-marginal reciprocal family 中
唯一的 Markov member。

Jamison 报告 Beurling 1960 在 locally compact/product-measure setup 下的更一般
结果。Beurling 原文当前 closed，本教程只写“Jamison reports that Beurling proves
...”，不把后人摘要伪装成直接 theorem 摘录。

## 10. Léonard 的两条一般 factorization 路线

[Léonard 2014](https://doi.org/10.3934/dcds.2014.34.1533 "官方论文页面")
Theorems 2.8 与 2.12 不应合并。

Theorem 2.8 的主要条件包括：

- common endpoint reference marginal `m`；
- `R_0T` 对 `m tensor m` 的 weighted lower bound 与另一 weighted integrability；
- 两个 measures 之间至少一个 absolute-continuity direction；
- finite endpoint entropies 和对应 moments；
- constraint internality。

它给 positive factors。

Theorem 2.12 改用 Markov reference 与 intermediate-time conditional domination，
保留相应 entropy/moment assumptions，不再要求 internality，并允许 factors 为
nonnegative、在 positive-measure sets 上为零。这不是“任意 structural zeros 都
不影响 factorization”。

## 11. Factorization、optimality 与 Markovity并非无条件等价

Föllmer--Gantert 1997
Theorem 3.43 在一般 setup 中只无条件给出

```text
factorization
 => entropy minimization
 => Markovity.                                           (11.1)
```

reverse arrows 需要 product domination 或 conditional regularity。对
infinite-dimensional product Brownian reference，essentially bounded positive endpoint
density 恢复完整 equivalence；但 Proposition 5.31 构造 density 属于每个 finite
`L^p`、仍没有 measurable endpoint splitting 的 Markov entropy minimizer。

这揭示 measurable factorization 不是形式 algebra。即使 `log phi` 可由 endpoint
sums approximation，也未必分裂成两个 measurable functions。

## 12. Structural zeros 的四层边界

教程区分：

1. finite strictly-positive matrix：本章完整证明；
2. continuous positive kernel：Fortet/Jamison 的明示条件；
3. factors 可为零：Léonard Theorem 2.12 的 Markov conditional assumptions；
4. degenerate/nonrectangular support：ordinary factorization/uniqueness 可能失败。

Föllmer--Gantert Example 2.17 给出 degenerate Markov mixture，其中 ordinary
factorization 与 fixed-marginal uniqueness 失败。Propositions 5.7/5.31 进一步说明
density boundedness或 finite moments 不替代 reference regularity。

因此，遇到 zeros 时正确问题不是“把零加一个很小常数再套 theorem”，而是先问：

- target marginals 是否与 support compatible；
- support graph 有几个 connected components；
- dual/factors 在各 component 上是否可定义；
- coupling uniqueness 与 factor gauge 分别怎样变化。

## 13. 最小 scaling 验证

`finite_schrodinger_system_checks.py` 对 positive `3x3` reference 运行交替缩放：

```text
u <- mu/(Rv),
v <- nu/(R^T u).                                       (13.1)
```

15 iterations 后：

- row residual `4.11e-15`；
- column residual `2.78e-17`；
- rank-one singular-value ratio `8.16e-17`；
- gauge error `2.78e-17`；
- max Pythagorean error `1.30e-17`；
- tested nonzero perturbations 的 minimum entropy gap `2.30e-4`。

这里用 iteration 只为构造可检查的 solution。Fortet/IPF/Sinkhorn 的历史、
convergence、support conditions 和 numerical stability 由 B7 负责。

![Reference coupling、最优 coupling 与 rank-one density ratio](/images/bridge/B3_schrodinger_system.png)

**图 13.1：** 正 reference matrix 经 Schrödinger scaling 后得到满足目标 marginals
的唯一 coupling；逐点比值 `gamma*/R_0T` 呈外积结构 `u v^T`。数值来自上述
`3x3` finite fixture。

## 14. 常见错误

1. **把 Lagrange multiplier 当一般 existence proof。** infinite-dimensional dual
   attainment 和 measurable splitting 可能失败。
2. **声称 factors 唯一。** 唯一的是 coupling；factors 有 gauge。
3. **把 internality 和 Markov positivity route 合并。** Léonard 2.8/2.12 是不同 theorem。
4. **忽略 reference 是 joint endpoint law。** `R_0T` 不必是 product。
5. **由 Markovity 推出 product factors。** general converse 需要 regularity。
6. **把 positive theorem 套到 structural zeros。** support geometry改变可行性与唯一性。
7. **把 Fortet iteration 直接称现代 Sinkhorn。** 变量、投影解释和 convergence
   theorem 需要单独翻译。

## 15. 小结

B3 的经典闭环是

```text
path KL reduction
 -> static endpoint entropy
 -> multiplicative factors
 -> Schrödinger marginal system
 -> unique coupling + gauge factors
 -> Markov two-sided path law.                          (15.1)
```

finite positive 情形可完整证明。一般空间中，正确做法不是省略 assumptions，而是
选择适合 support/reference 的 theorem，并明确 factorization、optimality、Markovity
之间哪些箭头实际成立。

## 16. 研究式思考题

1. 对 disconnected finite support graph，factor gauge 有多少维？coupling 何时唯一？
2. 能否构造 static minimizer存在但 ordinary endpoint factors 不存在的简单例子？
3. Léonard internality 在 finite coupling polytope 中对应什么几何条件？
4. Föllmer--Gantert Proposition 5.31 为什么说明 `L^p` control 不是 measurable
   splitting 的替代？
5. 若 reference 是 sigma-finite 而非 probability，entropy 的 normalization 与
   Pythagorean identity如何调整？

## 17. 前后章节链接

- B2：dynamic/static KL reduction 与 reference bridge lift；
- B4：factorized coupling 的 Markov iff 与前后向 kernels；
- B5：Gibbs kernel、entropic OT 与 epsilon；
- B7：Fortet/IPF/Sinkhorn 计算；
- B14：support、measurable factorization 与 inexact algorithm开放问题。

假设对照见
`references/notes/derivations/bridge/b3_assumption_support_audit.md`，finite proof 见
`references/notes/derivations/bridge/finite_schrodinger_system_derivations.md`。
