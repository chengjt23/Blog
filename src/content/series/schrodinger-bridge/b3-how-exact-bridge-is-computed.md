---
title: 这个解怎样变成一条可计算的随机过程
description: 让端点 factors 沿参考过程传播，并说明 Doob transform、控制解释与 IPF/Sinkhorn 如何落地。
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
  - stochastic-control
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 forward/reverse dynamics、随机控制、边缘缩放、support 条件与 log-domain stabilization。
---
第二章得到的端点耦合有一个很简洁的形状：参考耦合在起点和终点两侧各乘一个因子。这个“左右缩放”很容易被误解成纯粹的矩阵技巧，其实它同时决定了中间每个时刻的分布、前向与反向动力学，以及最终使用什么算法求解。

从端点 factors 到一条可采样的随机过程，中间没有新的模型假设。所需的只是让这两个因子沿参考动力学传播。

## 1. 端点因子会沿参考过程传播

把第二章有限状态中的 $u_i,v_j$ 写到一般状态空间上。设 $f_0(x)$ 是只依赖初始状态的正函数，$g_T(y)$ 是只依赖终止状态的正函数。最优路径律相对于参考路径律 $R$ 的密度具有形式

$$
\boxed{
\frac{dP^*}{dR}
=
\frac{f_0(X_0)g_T(X_T)}{Z},
}
$$

其中 $Z>0$ 是归一化常数。把常数乘到 $f_0$ 再从 $g_T$ 中除掉，不会改变 $P^*$；这正是有限状态 factors 的尺度自由度。

现在让两个端点函数随参考过程传播。对任意时刻 $t\in[0,T]$，定义

$$
g_t(x)
=
\mathbb E_R
\left[
g_T(X_T)\mid X_t=x
\right],
$$

$$
f_t(x)
=
\mathbb E_R
\left[
f_0(X_0)\mid X_t=x
\right].
$$

$g_t$ 把终端信息向较早时刻传回，$f_t$ 把初始信息向较晚时刻传去。若参考过程在时刻 $t$ 的密度记为 $r_t(x)$，最优过程的密度便是

$$
\boxed{
p_t(x)
=
\frac{r_t(x)f_t(x)g_t(x)}{Z}.
}
$$

这个乘积公式值得多看一眼。边界条件不是只在 $t=0,T$ 两个时刻起作用；它们经参考过程传播后，在每一个中间时刻共同改变概率质量。

## 2. 同一对势函数决定前向和反向动力学

若 $R$ 是 Markov process，终端势函数 $g_t$ 会直接改变前向 transition kernel。对 $0\le s<t\le T$，记 $R_{s,t}(x,dy)$ 为参考过程从 $x$ 转移到 $dy$ 的 kernel，则

$$
\boxed{
P^*_{s,t}(x,dy)
=
R_{s,t}(x,dy)
\frac{g_t(y)}{g_s(x)}.
}
$$

由于

$$
g_s(x)
=
\int
g_t(y)R_{s,t}(x,dy),
$$

右侧自动归一化。这就是 Doob transform：它没有改变可走的参考路径，只重新分配了这些路径的概率。反向 transition kernel 则由 $f_t$ 作对称的变换。

在 smooth diffusion 情形，设参考过程满足

$$
dX_t
=
b_t(X_t)\,dt
+
\sigma_t(X_t)\,dW_t,
\qquad
a_t=\sigma_t\sigma_t^\top,
$$

其中 $b_t$ 是参考漂移，$\sigma_t$ 是扩散系数，$a_t$ 是 diffusion matrix，$W_t$ 是标准 Brownian motion。若 $g_t$ 足够光滑且为正，最优前向漂移为

$$
\boxed{
b_t^{*,+}(x)
=
b_t(x)
+
a_t(x)\nabla\log g_t(x).
}
$$

采用反向时间参数时，反向漂移相对于参考反向漂移的修正由

$$
a_t(x)\nabla\log f_t(x)
$$

给出。前向与反向模型不是两套独立生成器，它们是同一条路径律 $P^*$ 的两种条件表示。

Jamison 在 1974 年论文 [*Reciprocal Processes*](https://doi.org/10.1007/BF00532864 "官方论文页面") 中说明，任意参考条件桥的端点 mixture 都保留双侧条件结构，因此通常是 reciprocal process；但任意 mixture 未必 Markov。乘法密度 $f_0(X_0)g_T(X_T)$ 正是最优 mixture 在相应 positivity 条件下重新成为 Markov process 的关键。

![Reciprocal endpoint mixture 与 Markov factorization](/images/bridge/B4_reciprocal_markov.png)

这个区别不是术语洁癖。只匹配每个时刻的边缘分布，可能已经足够生成快照；要恢复 transition kernel 和完整路径律，却必须保留更强的结构。

## 3. 路径空间 KL 也可以读成控制能量

前向漂移多出的

$$
a_t\nabla\log g_t
$$

不只是 Doob transform 的结果，也是一条最小能量控制。

仍以参考 diffusion 为起点。设候选过程保持同一个扩散通道，只把漂移改为

$$
b_t(X_t)+a_t(X_t)\beta_t,
$$

其中 $\beta_t$ 是适应于当前信息的控制场。在 absolute continuity 与 finite-energy 条件下，Girsanov/Föllmer theory 给出

$$
\boxed{
\operatorname{KL}(P\Vert R)
=
\operatorname{KL}(P_0\Vert R_0)
+
\frac12
\mathbb E_P
\int_0^T
\beta_t^\top
a_t(X_t)
\beta_t
\,dt.
}
$$

若参考初始分布已经等于目标 $\mu_0$，第一项为零。最小化路径空间 KL 于是等价于：在所有能把 $\mu_0$ 驱动到 $\mu_T$ 的控制中，选择二次能量最小者。

对扩散强度为 $\varepsilon$ 的 Brownian reference，有 $a_t=\varepsilon I$。若用实际漂移修正 $u_t=\varepsilon\beta_t$，上式变为

$$
\operatorname{KL}(P\Vert R)
=
\frac1{2\varepsilon}
\mathbb E_P
\int_0^T
\|u_t\|^2\,dt.
$$

最优控制正是

$$
\boxed{
u_t^*(x)
=
\varepsilon\nabla\log g_t(x).
}
$$

Lehec 在 2013 年论文 [*Representation Formula for the Entropy and Functional Inequalities*](https://doi.org/10.1214/11-AIHP464 "官方论文页面") 中严格讨论了 Wiener-space entropy 与 canonical Föllmer drift。Chen、Georgiou 与 Pavon 在 2021 年综述 [*Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge*](https://doi.org/10.1137/20M1339982 "官方论文页面") 中，则系统连接了 Schrödinger system、stochastic control 与 transport。

这里的限制必须说清楚：KL 与控制能量的等式只覆盖能够通过同一扩散通道改变测度的过程。若漂移修正不在 $\operatorname{Range}(\sigma_t)$ 中，或者 martingale problem 不适定，不能机械套用这条公式。

## 4. 精确计算就是交替修正两个边缘

动力学公式告诉我们解是什么样，仍没有直接给出端点 factors。回到有限状态矩阵 $Q=(Q_{ij})$，目标边缘写成 $\mu,\nu$，需要求

$$
\gamma^*
=
\operatorname{diag}(u)
Q
\operatorname{diag}(v)
$$

使其行和等于 $\mu$、列和等于 $\nu$。固定 $v$ 时，$u$ 可由行约束直接解出；固定 $u$ 时，$v$ 可由列约束解出：

$$
\boxed{
u\leftarrow\frac{\mu}{Qv},
\qquad
v\leftarrow\frac{\nu}{Q^\top u},
}
$$

其中除法逐分量进行。交替重复这两步，就是有限状态下的 IPF/Sinkhorn scaling。

这两行更新之所以可靠，不只是因为 residual 通常会下降。设当前耦合为 $Q$，只要求新耦合 $P$ 具有目标行边缘 $\mu$。正确的 information projection 是

$$
P^*
=
\arg\min_{\substack{
P\ge0\\
\sum_jP_{ij}=\mu_i
}}
\operatorname{KL}(P\Vert Q).
$$

它的闭式解是

$$
\boxed{
P^*_{ij}
=
Q_{ij}
\frac{\mu_i}{\sum_kQ_{ik}}.
}
$$

整行被乘上同一个常数，所以给定起点后的条件分布保持不变。列修正完全对称。一次 half-step 只满足一侧边缘；一次完整 cycle 也通常还没有到 fixed point。

对任何同时满足两侧边缘的可行耦合 $P$，精确 half-step 满足 KL Pythagorean identity：

$$
\operatorname{KL}(P\Vert Q_{\mathrm{old}})
=
\operatorname{KL}(P\Vert Q_{\mathrm{new}})
+
\operatorname{KL}(Q_{\mathrm{new}}\Vert Q_{\mathrm{old}}).
$$

最后一项非负，每次投影都会消耗一部分相对于可行解的 KL gap。Csiszár 与 Shields 在 2004 年教程 [*Information Theory and Statistics: A Tutorial*](https://doi.org/10.1561/0100000004 "官方论文页面") 中系统整理了 finite I-projection 与 cyclic projection；Benamou、Carlier、Cuturi、Nenna 与 Peyré 在 2015 年论文 [*Iterative Bregman Projections for Regularized Transportation Problems*](https://doi.org/10.1137/141000439 "官方论文页面") 中把这套几何用于 regularized transport constraints。

Fortet 1940 年的 successive approximations 是连续 Schrödinger system 的早期迭代路线；Sinkhorn 与 Knopp 在 1967 年论文 [*Concerning Nonnegative Matrices and Doubly Stochastic Matrices*](https://doi.org/10.2140/pjm.1967.21.343 "官方论文页面") 中研究了非负矩阵 scaling 与 support 条件。今天常说的 Sinkhorn algorithm，实际汇合了这些不同来源。

路径空间中的 half-step 具有同样含义。若当前路径律为 $Q$，终端边缘为 $Q_T$，把终端边缘替换为 $\mu_T$ 的精确投影满足

$$
\boxed{
\frac{dP^{\mathrm{new}}}{dQ}
=
\frac{d\mu_T}{dQ_T}(X_T).
}
$$

这个密度比只依赖终点，所以给定 $X_T$ 后的条件路径律保持不变。下一步再对初始边缘作同样修正，便得到 path-space IPF。

## 5. 零值和数值稳定性不是实现细节

若参考矩阵中的 $Q_{ij}=0$，任何对 $Q$ 具有有限 KL 的候选耦合都不能在该位置放置正质量。这个零值表示参考模型禁止相应迁移，不是可以随手加一个小常数抹掉的数值瑕疵。

Sinkhorn–Knopp 理论区分 support、total support 与 fully indecomposable。它们分别影响是否存在 doubly stochastic limit、是否能由正对角 scaling 得到，以及 factors 是否除尺度外唯一。一般目标边缘还要与参考 support 相容；例如对角参考矩阵不允许质量跨状态移动，目标边缘若要求交换质量，问题本身就不可行。

![Positive scaling、structural zeros 与 infeasible support](/images/bridge/B7_sinkhorn_diagnostics.png)

另一类零值来自浮点数。Entropic OT 常使用 Gibbs kernel

$$
K_{ij}
=
\exp\!\left(-\frac{C_{ij}}{\varepsilon}\right),
$$

其中 $C_{ij}$ 是 transport cost，$\varepsilon>0$ 是 entropy scale。数学上，只要 $C_{ij}<\infty$，就有 $K_{ij}>0$；数值上，当 $C_{ij}/\varepsilon$ 很大时，float64 会把它下溢成零。

稳定实现通常写

$$
u_i=\exp\!\left(\frac{\alpha_i}{\varepsilon}\right),
\qquad
v_j=\exp\!\left(\frac{\beta_j}{\varepsilon}\right),
$$

并在 log-domain 中计算

$$
\log\gamma_{ij}
=
\frac{\alpha_i+\beta_j-C_{ij}}{\varepsilon}.
$$

行列归一化由 $\operatorname{logsumexp}$ 完成，不再直接形成极小的 $K_{ij}$。Cuturi 在 NeurIPS 2013 论文 Sinkhorn Distances: Lightspeed Computation of Optimal Transport（补充材料暂未公开） 中推动了 Gibbs kernel scaling 的大规模应用；Schmitzer 在 2019 年论文 [*Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems*](https://doi.org/10.1137/16M1106018 "官方论文页面") 中系统讨论了 absorption、stabilization 与 sparse scaling。

Log-domain 只能避免数值下溢，不能修复真实的 support 不可行，也不会自动改善问题的 conditioning。我不会信任一个只展示漂亮样本的 Bridge solver；至少还应看到两侧 marginal residual、objective 或 dual gap，以及明确的 iteration budget。

## 文献索引

| 时间   | 论文                                                                                      | 本章采用的内容                                                    |
| ---- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1940 | Fortet, *Résolution d'un système d'équations de M. Schrödinger*                         | 连续 Schrödinger system 的 successive approximations          |
| 1967 | Sinkhorn & Knopp, *Concerning Nonnegative Matrices and Doubly Stochastic Matrices*      | 非负矩阵 scaling 的 support 条件                                  |
| 1974 | Jamison, *Reciprocal Processes*                                                         | 连接 endpoint factorization、reciprocal structure 与 Markovity |
| 2004 | Csiszár & Shields, *Information Theory and Statistics: A Tutorial*                      | 整理 finite I-projection 与 cyclic convergence                |
| 2013 | Cuturi, *Sinkhorn Distances*                                                            | 推动 Gibbs kernel scaling 的大规模应用                             |
| 2013 | Lehec, *Representation Formula for the Entropy and Functional Inequalities*             | 给出 Wiener entropy 与 Föllmer drift 的能量表示                    |
| 2015 | Benamou et al., *Iterative Bregman Projections for Regularized Transportation Problems* | 将边缘 scaling 表述为 Bregman projections                        |
| 2019 | Schmitzer, *Stabilized Sparse Scaling Algorithms...*                                    | 给出 log-domain stabilization 与 sparse scaling               |
| 2021 | Chen, Georgiou & Pavon, *Stochastic Control Liaisons...*                                | 连接 Schrödinger Bridge、控制与 transport                        |
