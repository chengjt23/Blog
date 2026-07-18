---
title: 精确 Bridge 怎样被计算出来
description: 从边缘缩放与 KL 投影出发，解释 IPF、Sinkhorn、可行性和稳定计算。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 3
slug: b3-how-exact-bridge-is-computed
tags:
  - schrodinger-bridge
  - sinkhorn
  - ipf
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 exact I-projection、cyclic IPF、support 条件、log-domain stabilization 与停止准则。
---
第二章得到有限状态下的精确结构：

$$
\gamma^*_{ij}
=
u_iQ_{ij}v_j,
$$

其中 $Q=(Q_{ij})$ 是 reference endpoint coupling，$u=(u_i)$、$v=(v_j)$ 是需要满足目标 marginals $\mu,\nu$ 的 Schrödinger factors。它们服从

$$
u_i(Qv)_i=\mu_i,
\qquad
v_j(Q^\top u)_j=\nu_j.
$$

这组方程已经给出解的形状，却没有直接给出 $u,v$。计算的核心观察非常简单：固定一个 factor 后，另一个 factor 可以由对应的 marginal constraint 直接解出。于是

$$
\boxed{
u\leftarrow
\frac{\mu}{Qv},
\qquad
v\leftarrow
\frac{\nu}{Q^\top u},
}
$$

其中除法逐分量进行。反复交替这两步，就是有限状态下的 IPF/Sinkhorn scaling。

可靠的 Bridge solver 还必须说明：half-step 为什么是精确 KL projection、循环何时收敛、reference 有 zeros 时是否可行，以及小 regularization 下怎样避免数值下溢。

## 1. 1940—2015：一次 marginal correction 为什么是精确 I-projection

先忘掉 factors，只看一张当前 coupling $Q=(Q_{ij})$。假设它严格为正并已归一化。现在只要求新的 coupling $P=(P_{ij})$ 具有目标 row marginal $\mu$：

$$
\mathcal C_\mu
=
\left\{
P\ge0:
\sum_jP_{ij}=\mu_i
\right\}.
$$

按正确的 KL 方向对当前 coupling 做 information projection：

$$
\boxed{
P^*
=
\arg\min_{P\in\mathcal C_\mu}
\operatorname{KL}(P\Vert Q).
}
$$

按行引入 Lagrange multipliers。Stationarity condition 说明，同一行中的所有 entries 只能乘上同一个常数，因此

$$
\boxed{
P^*_{ij}
=
Q_{ij}
\frac{\mu_i}{\sum_kQ_{ik}}.
}
$$

它只是把第 $i$ 行整体缩放到目标质量 $\mu_i$，并保留该行内部的 conditional distribution：

$$
\frac{P^*_{ij}}{\mu_i}
=
\frac{Q_{ij}}{\sum_kQ_{ik}}.
$$

所以一次 row scaling 的统计含义是：替换起点 marginal，同时保留“给定起点后终点怎样分布”的 conditional law。Column scaling 完全对称。

Csiszár 与 Shields 在 2004 年教程 [*Information Theory and Statistics: A Tutorial*](https://doi.org/10.1561/0100000004 "官方论文页面") 中系统整理了 finite I-projection 与 cyclic projection；Benamou、Carlier、Cuturi、Nenna 与 Peyré 在 2015 年论文 [*Iterative Bregman Projections for Regularized Transportation Problems*](https://doi.org/10.1137/141000439 "官方论文页面") 中，则把同一 projection geometry 用于一类 regularized transport constraints。

KL 的方向不能反转。Scaling 解的是

$$
\operatorname{KL}(\text{new}\Vert\text{old})
$$

的 projection；最小化 $\operatorname{KL}(\text{old}\Vert\text{new})$ 不会导出相同的乘法更新。

## 2. 1967—2004：为什么两个 half-steps 能收敛到共同约束

一次 row projection 只保证 row marginal；紧接着进行 column projection 时，rows 通常会再次偏离。因此必须区分：

$$
\text{one half-step}
\ne
\text{one full cycle}
\ne
\text{exact fixed point}.
$$

设 $Q^{(0)}=Q$。一次完整 cycle 可以写成

$$
Q^{(2n+1)}
=
\operatorname{Proj}_{\mathcal C_\mu}^{\mathrm{KL}}
\left(
Q^{(2n)}
\right),
$$

$$
Q^{(2n+2)}
=
\operatorname{Proj}_{\mathcal C_\nu}^{\mathrm{KL}}
\left(
Q^{(2n+1)}
\right),
$$

其中 $\mathcal C_\nu$ 是具有目标 column marginal $\nu$ 的集合。

对任意同时满足 rows 与 columns 的 feasible coupling $P$，一次 exact half-step 满足 KL Pythagorean identity：

$$
\boxed{
\operatorname{KL}(P\Vert Q_{\mathrm{old}})
=
\operatorname{KL}(P\Vert Q_{\mathrm{new}})
+
\operatorname{KL}(Q_{\mathrm{new}}\Vert Q_{\mathrm{old}}).
}
$$

最后一项非负，因此每次 projection 都消耗一部分相对于可行解的 KL gap。对有限 affine marginal families，在 intersection 非空且初始 reference 为正等条件下，cyclic I-projections 收敛到唯一 joint projection：

$$
\boxed{
Q^{(k)}
\longrightarrow
\arg\min_{P\in\mathcal C_\mu\cap\mathcal C_\nu}
\operatorname{KL}(P\Vert Q).
}
$$

这就是 IPF 收敛到 Schrödinger coupling 的几何原因。它不是因为 row residual 与 column residual “看起来越来越小”，而是因为每个 half-step 都是相对熵意义下的精确投影。

Sinkhorn 与 Knopp 在 1967 年论文 [*Concerning Nonnegative Matrices and Doubly Stochastic Matrices*](https://doi.org/10.2140/pjm.1967.21.343 "官方论文页面") 中研究 nonnegative square matrices 的 scaling 与 support conditions。现代文献常把上述更新统称为 Sinkhorn algorithm，但历史责任需要分开：Sinkhorn–Knopp 主要承担 matrix support theorem，general marginal I-projection 与 cyclic convergence 由 information-projection 文献承担。

Factors 仍然具有 gauge freedom：

$$
(u,v)
\mapsto
(cu,c^{-1}v),
\qquad c>0.
$$

因此 $u,v$ 的绝对数值可能持续漂移，而 coupling

$$
\operatorname{diag}(u)Q\operatorname{diag}(v)
$$

已经收敛。判断算法时应检查 coupling 与 marginals，而不能只比较 factors 的范数。

## 3. 1940—2021：从 finite scaling 到 path-space IPF

Fortet 在 1940 年论文 Résolution d'un système d'équations de M. Schrödinger（补充材料暂未公开） 中从连续 Schrödinger system 构造 successive approximations，是 alternating-factor 思想的重要来源，但尚未使用现代 matrix scaling 或 KL projection 语言。

Finite IPF 的 path-space 版本更能显示 Bridge 的本质。令初始 path law 为

$$
P^{(0)}=R.
$$

随后交替满足终端与初始约束：

$$
\boxed{
P^{(2n+1)}
=
\arg\min_{P_T=\mu_T}
\operatorname{KL}
\left(
P\Vert P^{(2n)}
\right),
}
$$

$$
\boxed{
P^{(2n+2)}
=
\arg\min_{P_0=\mu_0}
\operatorname{KL}
\left(
P\Vert P^{(2n+1)}
\right).
}
$$

一次 terminal half-bridge update 有闭式结构。若当前 law 是 $Q$，其 terminal marginal 为 $Q_T$，则

$$
\boxed{
\frac{dP^{\mathrm{new}}}{dQ}
=
\frac{d\mu_T}{dQ_T}(X_T).
}
$$

这个 density ratio 只依赖终点，因此新的 path law 把 terminal marginal 换成 $\mu_T$，同时保留给定 $X_T$ 后的 conditional path law。Initial half-step 则使用对应的 $X_0$ density ratio。

这与 finite row/column scaling 是同一个操作：

$$
\text{replace one marginal}
+
\text{preserve the corresponding conditional law}.
$$

当 path law 是 Markov diffusion 时，每次 half-step 也可以重新表示为反向或前向 dynamics；但“可以表示”不等于“已经能够高效计算”。Path-space IPF 仍可能要求 transition densities、conditional expectations 或大量路径样本，真正的高维困难正是在这里出现。

Cuturi 在 NeurIPS 2013 论文 Sinkhorn Distances: Lightspeed Computation of Optimal Transport（补充材料暂未公开） 中推动了 Gibbs kernel scaling 在大规模 entropic transport 中的应用，但这不构成任意 path-space IPF 的统一复杂度结论。

## 4. 1967—2019：structural zeros、support 与 feasibility 为什么不能忽略

严格正 kernel 是最友好的情况。若 $Q_{ij}=0$，relative entropy 会禁止 candidate 在该位置放置正质量，因此 zero pattern 是可行集的一部分，而不是普通数值细节。

对 square nonnegative matrix，Sinkhorn–Knopp 理论区分三个条件：

| 条件                   | 含义                                        | 主要结论                                                   |
| -------------------- | ----------------------------------------- | ------------------------------------------------------ |
| support              | 至少存在一条 positive diagonal                  | alternating normalization 可收敛到 doubly stochastic limit |
| total support        | 每个 positive entry 都位于某条 positive diagonal | limit 可由 positive diagonal scaling 得到                  |
| fully indecomposable | 不能通过 permutation 分成相应 block form          | scaling factors 除 gauge 外唯一                            |

这三个条件不能互换。Scaled matrix 的唯一性、factors 的唯一性和迭代是否保留所有原 positive entries，也是不同结论。

一般的目标 marginals 还需要满足 support compatibility。例如 diagonal reference 强迫质量只能沿 diagonal 移动；若目标 row 与 column masses 互换，就不存在可行 coupling。算法仍可能让刚更新的一侧 residual 暂时为零，却无法同时满足两端。

所以 stopping criterion 必须同时检查两个 marginals。只检查刚刚更新的一侧，会把 infeasible oscillation 误判为收敛。

![Positive scaling、structural zeros 与 infeasible support](/images/bridge/B7_sinkhorn_diagnostics.png)

## 5. 2013—2019：为什么稳定实现必须进入理论主线

Entropic OT 常使用 Gibbs kernel

$$
K_{ij}
=
\exp\!\left(
-\frac{C_{ij}}{\varepsilon}
\right),
$$

其中 $C_{ij}$ 是 transport cost，$\varepsilon>0$ 是 regularization scale。数学上只要 $C_{ij}<\infty$，就有 $K_{ij}>0$。但在 float64 中，当 $C_{ij}/\varepsilon$ 过大，正数会 underflow 成机器零。

这会把 numerical conditioning 错误地伪装成 structural-zero problem。Naive scaling 可能出现 division by zero、nonfinite factors，甚至错误地宣称目标 marginals 不可行。

稳定做法是转入 log-domain。写

$$
u_i
=
\exp\!\left(
\frac{\alpha_i}{\varepsilon}
\right),
\qquad
v_j
=
\exp\!\left(
\frac{\beta_j}{\varepsilon}
\right).
$$

则 coupling 的 log-density 为

$$
\log\gamma_{ij}
=
\frac{
\alpha_i+\beta_j-C_{ij}
}{\varepsilon}.
$$

Scaling updates 随后用 $\operatorname{logsumexp}$ 计算 row 与 column normalizers，而不直接形成 $K_{ij}$。

$\operatorname{logsumexp}$ 先减去最大值再求 exponentials，避免直接构造极小 kernel。Schmitzer 在 2019 年论文 [*Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems*](https://doi.org/10.1137/16M1106018 "官方论文页面") 中系统讨论 absorption、stabilization 与 sparse scaling。

Log-domain 解决的是数值范围，不会自动改善问题本身的 conditioning，也不能修复真正 infeasible 的 support。类似地，epsilon scaling 是 continuation strategy，不是无需条件的 convergence-rate theorem。

一个可靠实现至少要分别报告 row residual、column residual、objective 或 dual gap 以及 iteration budget。Marginal residual 只能说明 constraints 的满足程度，不等于 objective gap；有限迭代达到小 residual，也不等于已经证明 exact fixed point。

现在可以回答本章标题：

> 精确 Bridge 的计算从两个交替的 KL I-projections 开始：每个 half-step 替换一个 marginal 并保留相应 conditional law；在可行、正则和稳定计算的条件下，完整迭代循环收敛到 Schrödinger system 的共同 fixed point。

整条计算链是

$$
\text{Schrödinger factors}
\longrightarrow
\text{marginal I-projections}
\longrightarrow
\text{cyclic IPF}
\longrightarrow
\text{support audit}
\longrightarrow
\text{stable arithmetic}.
$$

## 本章论文索引

| 时间   | 论文                                                                                      | 本章中的作用                                                              |
| ---- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 1940 | Fortet, *Résolution d'un système d'équations de M. Schrödinger*                         | 建立连续 Schrödinger system 的 fixed-point 与 successive-approximation 路线 |
| 1967 | Sinkhorn & Knopp, *Concerning Nonnegative Matrices and Doubly Stochastic Matrices*      | 给出 nonnegative matrix scaling 的 support 条件                          |
| 2004 | Csiszár & Shields, *Information Theory and Statistics: A Tutorial*                      | 系统整理 finite I-projection、Pythagorean identity 与 cyclic convergence  |
| 2013 | Cuturi, *Sinkhorn Distances*                                                            | 推动 Gibbs kernel scaling 在大规模 entropic transport 中的应用                |
| 2015 | Benamou et al., *Iterative Bregman Projections for Regularized Transportation Problems* | 将 marginal scaling 表述为 Bregman projections                          |
| 2019 | Schmitzer, *Stabilized Sparse Scaling Algorithms...*                                    | 给出 log-domain stabilization、absorption 与 sparse scaling 路线          |
