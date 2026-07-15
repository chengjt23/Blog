---
title: 经典计算：Fortet、IPF 与 Sinkhorn
description: 从 Fortet 迭代到 IPF 与 Sinkhorn，解释有限状态精确投影、结构零、可行性和数值稳定化。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 7
slug: b7-fortet-ipf-sinkhorn
tags:
  - schrodinger-bridge
  - sinkhorn
  - ipf
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 finite exact IPF/Sinkhorn、support 条件、log-domain 实现与停止准则。
---
## 1. B3 给出解的形状，B7 负责怎样算出来

B3 的 finite positive Schrödinger system 是

```text
gamma*_ij = u_i R_ij v_j,
sum_j gamma*_ij = mu_i,
sum_i gamma*_ij = nu_j.                               (1.1)
```

把一个 factor 固定，另一个由 marginal constraint 直接确定：

```text
u <- mu/(R v),
v <- nu/(R^T u).                                      (1.2)
```

这里除法逐分量进行。反复应用 (1.2) 就得到 finite IPF/Sinkhorn scaling。
这一简单 recursion 包含四个必须分开的层次：

- 一次更新为何是 exact KL projection；
- 完整循环为何收敛到 joint projection；
- reference 有 zeros 时怎样改变 feasibility 与 factorization；
- 浮点下溢时怎样不制造伪 zeros。

## 2. 一次 marginal correction 是 exact I-projection

令当前正 coupling 为 `Q`，行约束集为

```text
C_mu={P>=0: P 1=mu}.                                  (2.1)
```

考虑正确方向的 projection

```text
minimize KL(P|Q) over P in C_mu.                      (2.2)
```

按行加 Lagrange multipliers，stationarity 给出每行一个乘法常数，因而

```text
P_ij=Q_ij mu_i/(sum_j Q_ij).                          (2.3)
```

列 projection 同理。这正是 [Csiszár--Shields 2004](https://doi.org/10.1561/0100000004 "官方论文页面")
printed pp. 459--460 和
[Benamou et al. 2015](https://doi.org/10.1137/141000439 "官方论文页面")
Proposition 1 的 marginal specialization。

注意 KL orientation 是 `KL(new|old)`。反转成 `KL(old|new)` 不会导出 (2.3)。

## 3. Pythagorean identity 与完整循环

对任意同时满足目标 rows 和 columns 的 feasible `P`，一次 exact half-step 满足

```text
KL(P|Q_old)
 = KL(P|Q_new)+KL(Q_new|Q_old).                       (3.1)
```

因此每次 correction 都消耗一个非负 KL increment。对两个 finite affine marginal
families，若 intersection 非空且初始 `R` 为正，Csiszár--Shields Theorem 5.1
给出 cyclic I-projections 收敛到唯一 joint I-projection：

```text
Q^(k) -> argmin_{P in C_mu intersection C_nu} KL(P|R). (3.2)
```

`finite_schrodinger_system_checks.py` 独立验证 positive `3x3` 例子的 factorization、
gauge、Pythagorean identity 和 marginal residual。15 iterations 后 row residual
为 `4.11e-15`，column residual 为 `2.78e-17`。

## 4. Half-step、cycle 与 fixed point

一次 row projection 只保证 row marginal；随后 column projection 可能再次破坏 rows。
因此必须区分

```text
one half-step != one full cycle != exact fixed point. (4.1)
```

在 fixed point，(1.2) 同时成立，coupling 对 gauge

```text
(u,v)->(c u,c^{-1}v)                                  (4.2)
```

不变。factors 的数值漂移可能只是 gauge 变化，不能只凭 `u,v` 的绝对大小判断
coupling 是否收敛。

## 5. Fortet iteration 与现代 matrix scaling 不是同一段代码

Fortet 1940 从连续
Schrödinger integral system 出发，先把问题化成 auxiliary fixed point

```text
h=Omega(h),                                             (5.1)
```

再用 max/min envelopes 构造 successive approximations。其 pp. 86--94 的证明
承担 positive-kernel existence/uniqueness 和迭代历史，但没有直接写现代 finite
row/column normalization 或 KL projection。

因此可以说 Fortet 方法与 IPF/Sinkhorn 共享 alternating scaling 的结构祖先，
却不能把 (1.2) 原样标成 \`\`Fortet 1940 algorithm''。变量翻译、state space 和
convergence proof 都不同。

## 6. IPF、Sinkhorn--Knopp 与 entropic OT 的责任边界

现代文献常把 (1.2) 统称为 Sinkhorn algorithm。更精确的来源分工是：

- finite marginal I-projection 与 cyclic convergence：Csiszár--Shields；
- nonnegative square matrix 的 support theorem：[Sinkhorn--Knopp 1967](https://doi.org/10.2140/pjm.1967.21.343 "官方论文页面")；
- entropic OT 的 Gibbs kernel/Bregman formulation：Benamou et al. 与 Cuturi；
- log-domain、epsilon scaling 和 stopping caveats：[Schmitzer 2019](https://doi.org/10.1137/16M1106018 "官方论文页面")。

Sinkhorn 1964 与 Csiszár 1975 在当前 corpus 中只有正式 metadata/后续原文回溯，
所以正文不伪称读过其 proof，也不做未经来源支持的 first-priority claim。

## 7. Structural zeros：三个矩阵条件

对 nonnegative square matrix，Sinkhorn--Knopp 的三个条件不能合并：

| 条件                   | 含义                                        | 结论                                                      |
| -------------------- | ----------------------------------------- | ------------------------------------------------------- |
| support              | 至少有一条 positive diagonal                   | alternating normalization 收敛到某个 doubly stochastic limit |
| total support        | 每个 positive entry 都位于一条 positive diagonal | 存在 positive diagonal scaling，且等于该 limit                 |
| fully indecomposable | 不可经 permutation 分成相应 block form           | 两个 scaling factors 除 gauge 外唯一                          |

scaled matrix 的唯一性和 factors 的唯一性也不是同一结论。

考虑

```text
A=[[1,1],
   [0,1]].                                              (7.1)
```

它有 support，但 entry `(0,1)` 不在任何 positive diagonal 上，故没有 total
support。迭代后的 matrices 仍收敛到 doubly stochastic limit，但这个原正 entry
趋于零，不能由有限 positive diagonal factors 在极限精确表示。

## 8. 非均匀 marginals 还需要 feasibility

Sinkhorn--Knopp theorem 是 square/unit-marginal 结论，不能直接替代一般
transportation polytope 的可行性条件。例如

```text
R=diag(1,1),
mu=(0.8,0.2),
nu=(0.2,0.8).                                         (8.1)
```

diagonal support 强迫 row masses 与 column masses 对应相等，因此没有 coupling
能同时满足这两个 targets。每个完整 cycle 的最后一次 column correction 都可使
column residual 为零，但 row residual 始终为 `0.6`。

这也解释了为什么 stopping criterion 必须同时检查两侧，而不能只检查刚更新的
marginal。

## 9. Gibbs kernel 与 small-epsilon underflow

entropic OT 中

```text
K_ij=exp(-C_ij/epsilon).                               (9.1)
```

数学上只要 `C_ij` 有限，`K_ij>0`。但在 IEEE float64 中，当
`C_ij/epsilon` 大于约 `745`，entry 会下溢成零。普通更新于是把数值 conditioning
错误地变成 structural-zero problem。

`log_sinkhorn_checks.py` 使用一个 `3x3` fixture；在 `epsilon=0.005` 时，两个
off-diagonal entries 下溢，naive iteration 产生 nonfinite value，而 stable version
达到 `4.23e-14` marginal residual。

## 10. Log-domain potentials

写

```text
u_i=exp(alpha_i/epsilon),
v_j=exp(beta_j/epsilon).                               (10.1)
```

则

```text
log gamma_ij=(alpha_i+beta_j-C_ij)/epsilon,            (10.2)

alpha_i=epsilon log mu_i
        -epsilon logsumexp_j((beta_j-C_ij)/epsilon),

beta_j=epsilon log nu_j
       -epsilon logsumexp_i((alpha_i-C_ij)/epsilon).   (10.3)
```

`logsumexp(z)=max(z)+log sum exp(z-max(z))` 避免先构造极小 kernel。Schmitzer
equation (3.1) 给出同一 max-subtracted principle；其 Algorithm 2 的 absorption
variant 在 exact arithmetic 中保持原 iterates，只控制数值范围。

potentials 也有 additive gauge：

```text
(alpha,beta)->(alpha+kappa,beta-kappa).                (10.4)
```

recenter gauge 可抑制无意义漂移，但不能修复真正 infeasible support。

## 11. Epsilon scaling 是 continuation，不是现成复杂度定理

实践中常取

```text
epsilon_0>epsilon_1>...>epsilon_target                 (11.1)
```

并用上一阶段的 dual potentials warm-start 下一阶段。Schmitzer Algorithm 3
明确把它作为实用 heuristic。Theorem 20/Proposition 24 只在 atomic assumptions
及 exact previous optimizer 等条件下控制部分稳定性；Remark 9 指出实际上一阶段
只有 approximate optimizer，因此一般 end-to-end logarithmic complexity proof
仍有缺口。

B7 可以推荐 continuation，但不能声称任意 schedule 都得到统一 `O(log(1/epsilon))`
复杂度。

## 12. Residual、objective 与 rate

以下指标回答不同问题：

```text
marginal residual: constraints 满足到什么程度；
primal objective gap: 当前 cost/KL 离最优值多远；
duality gap: primal 与某个 dual certificate 的差；
coupling error: 当前 matrix 离 optimizer 多远；
potential error: 选定 gauge 后 factors 离 dual optimum 多远。          (12.1)
```

small marginal residual 不推出其他四项都小。Schmitzer Remark 5 给出相应警告。
本章也不从 residual plot 拟合 ordinary Sinkhorn 的 linear rate；当前一手来源只
足以支持 qualitative convergence。其 modified asymmetric algorithm 的 bound
不能改名为 standard Sinkhorn rate。

## 13. 原创诊断图

`b7_sinkhorn_diagnostics.py` 同时运行 positive、support-not-total-support 与
infeasible 三个 fixtures。

![Positive scaling、structural zero 与 infeasible support 的 Sinkhorn 诊断](/images/bridge/B7_sinkhorn_diagnostics.png)

**图 13.1：** 左：positive kernel 在不同 `epsilon` 下都达到机器精度，但更小
`epsilon` 的 conditioning 更差；中：matrix 有 support 而无 total support，原正
entry 与 residual 一同趋零，limit 不由 finite positive factors attained；右：
infeasible zero pattern 中 column residual 可为零而 row residual 固定为 `0.6`。

具体检查结果为：

- positive fixtures 的 final residual 不超过 `1.11e-16`；
- structural-zero entry 从 `1.67e-1` 降到 `6.25e-5`；
- corresponding marginal residual 为 `6.25e-5`；
- infeasible fixture 的 row/column residual 分别为 `0.6` 与 `0`。

这些结果说明 theorem boundary 和 stopping risk，不证明一般 convergence rate。

## 14. Exact 与 learned/approximate update 的接口

B8--B10 会用 regression、Monte Carlo 或 neural parameterization 替代某些 exact
projections。比较时必须先写清基线：

```text
exact half-step
 -> exact full cycle
 -> finite-cycle residual
 -> learned estimator error
 -> sampler/discretization error.                    (14.1)
```

一次有限数据 SGD 不等于 (2.2) 的 exact I-projection；即使 population loss 的
minimizer 正确，function approximation 与 optimization 仍会破坏 exact marginal
correction。

## 15. 常见错误

1. **反转 KL direction。** closed-form scaling 解的是 `KL(new|old)` projection。
2. **把一次 half-step 当 complete SB。** 它只修正一个 marginal。
3. **混合 support、total support 与 full indecomposability。** 三者承担不同结论。
4. **把 square/unit-marginal theorem 外推到任意 marginals。** 还需 support compatibility。
5. **把 underflow 当数学 structural zero。** stable arithmetic 应先恢复正 kernel。
6. **把 log-domain 当加速 theorem。** 它控制 range，不自动改善 conditioning。
7. **把 marginal residual 当 objective/duality gap。** stopping signals 不可互换。
8. **从实验曲线声称 linear rate。** quantitative theorem 需要独立一手来源。

## 16. 小结

B7 的主线可压缩为

```text
Schrodinger factors
 -> exact marginal I-projections
 -> cyclic KL convergence
 -> support/feasibility audit
 -> log-domain stable arithmetic
 -> conservative residual and rate reporting.         (16.1)
```

positive finite matrix 是最干净的 baseline，却不是整个算法理论。真正可靠的实现
必须同时知道何时 fixed point 存在、zeros 是数学结构还是浮点伪影、检查了哪类
residual，以及 finite iteration 与 exact projection 相差多少。

## 17. 研究式思考题

1. 对 general marginals，怎样用 bipartite cuts 表述 zero-pattern feasibility？
2. support 但非 total support 时，coupling 收敛而 factors 发散的 gauge-free解释是什么？
3. 怎样构造可计算的 duality gap，使它与 marginal residual 分责？
4. epsilon continuation 的 approximate warm start 缺口需要哪种 stability theorem 补齐？
5. learned half-step 若只近似满足 marginal，Pythagorean telescoping 还保留什么？

## 18. 前后章节链接

- B3：Schrödinger system、factor gauge 与 positive-kernel existence；
- B5：Gibbs kernel、entropic OT 与 small-noise variational limit；
- B6：control formulation 与 HJB factors；
- B8：learned IPF/SDE updates；
- B9/B10：reciprocal/Markov fitting 与 simulation-free approximations；
- B14：support、rate 与 approximate projection 的开放问题。

完整定理责任表见
`references/notes/bridge/b7_algorithm_theorem_responsibility.md`。
