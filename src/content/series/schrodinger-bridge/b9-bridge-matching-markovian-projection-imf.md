---
title: Bridge Matching、Markovian Projection 与 IMF
description: >-
  拆解 bridge matching、Markovian projection 与 iterative Markovian
  fitting，明确边缘匹配和路径律投影的差别。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 9
slug: b9-bridge-matching-markovian-projection-imf
tags:
  - schrodinger-bridge
  - bridge-matching
  - markovian-projection
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: >-
  聚焦 population conditional expectation、mimicking marginals、reciprocal/Markov
  投影与 IMF 迭代。
---
## 1. 一次 bridge regression 到底做了什么

给定 reference Markov law `Q`，固定 endpoints 后的 conditional law 记为
`Q^{xy}`。任取 endpoint coupling `gamma`，可以构造

$$
\Pi_\gamma=\int Q^{xy}\,\gamma(dx\,dy). \tag{1.1}
$$

它保留 reference bridges，因此属于 `Q` 的 reciprocal class；但 B4 已证明，一般
`Pi_gamma` 不为 Markov，也不一定是 Schrödinger Bridge。

Bridge Matching 的核心问题不是“怎样再采一次 bridge”，而是：能否从这个
reciprocal mixture 的 local conditional drift，构造一个 Markov diffusion，并保留
哪些统计对象？随后再把它投回 reciprocal class，迭代是否收敛到 SB？

本章始终区分：

```text
exact Markovian projection,
exact reciprocal projection,
exact IMF iteration,
population regression,
finite learned DSBM.                                   (1.2)
```

## 2. Reciprocal class 与 Markov class

固定 reference `Q`，其 reciprocal class 写为

```text
R(Q)={gamma Q_|0,T : gamma is an endpoint coupling}.  (2.1)
```

这里 `Q_|0,T` 表示 endpoint-conditioned reference bridge kernel。`R(Q)` 中所有
laws 共享 pinned bridges，但 endpoint couplings 可以不同。

Markov class `M` 则由 local transition/conditional-independence structure 定义。两个
集合的交点不是自动单点；在目标 endpoints、support 与 classical SB assumptions
加入后，Schrödinger Bridge 才是相应 `M intersection R(Q)` 中的 distinguished law。

## 3. Markovian projection 保留所有 one-time marginals

设 reciprocal mixture 的 local bridge drift correction 是 random target `Y_t`。在
Brownian/diffusion setting，population regression 给

$$
b_M(t,x)=\mathbb{E}[Y_t\mid X_t=x]. \tag{3.1}
$$

[Shi et al. 2023](https://doi.org/10.52202/075280-2717 "官方论文页面")
Proposition 2 在其 Appendix A1--A3 条件内，把该 law 识别为 reverse-KL Markovian
projection：

```text
proj_M(Pi)=argmin_{M in Markov class} KL(Pi|M).        (3.2)
```

它保留

```text
(proj_M Pi)_t=Pi_t       for every t,                 (3.3)
```

而不是只保留 `t=0,T`。但 (3.3) 不保证 endpoint joint law、pinned bridges 或完整
path law 相同。

## 4. Gyöngy mimicking theorem 的精确责任

[Gyöngy 1986](https://doi.org/10.1007/BF00699039 "官方论文页面")
Theorem 4.6 在 bounded adapted coefficients 与 uniform ellipticity 下，用条件
drift/covariance 构造一个 weak SDE，使每个固定时间的 marginal 与原 Itô process
相同。

这个 theorem 证明 marginal mimicking，不证明：

- original process 与 mimicking process 有同一 path law；
- `(X_0,X_T)` coupling 不变；
- conditional bridges 不变；
- 新 SDE 自动 unique 或 strong Markov。

因此“对 drift 取条件期望”只是 projection construction 的 probabilistic核心，不是
所有 path-law 结论的总称。

## 5. Brunick--Shreve 的 weak existence 与 well-posedness 边界

[Brunick--Shreve 2013](https://doi.org/10.1214/12-AAP881 "官方论文页面")
Theorem 3.6 在 local integrability 与 continuous updating map 条件下，把 mimicking
扩展到 selected path functionals。Corollaries 3.7/3.13 分开承担 ordinary state
marginals 与额外 well-posedness/strong Markov 条件。

可靠链条应写成

```text
conditional coefficient identity
 + weak existence
 + claimed SDE class的 uniqueness/well-posedness
 => a well-defined Markovian projected law.            (5.1)
```

缺少最后一步时，coefficient formula 可能对应多个 weak laws。

## 6. Reciprocal projection 保留 endpoint coupling

对任意 path law `P`，相对于 `Q` 的 reciprocal projection 是

```text
proj_R(Q)(P)=P_0T Q_|0,T.                             (6.1)
```

它精确保留 `P_0T`，并把 conditional paths 换回 `Q` 的 pinned bridges。在 IMF 的
class slots 中，Markov step 是 `KL(reciprocal|Markov)`，reciprocal step 是
`KL(Markov|reciprocal)`；下面的通用定义仍统一写成 `KL(input|candidate)`：

```text
proj_R(Q)(P)=argmin_{Pi in R(Q)} KL(P|Pi).             (6.2)
```

因此两个 operators 的 invariant 为：

| Operator    | 保留                                   | 一般改变                               |
| ----------- | ------------------------------------ | ---------------------------------- |
| `proj_M`    | every one-time marginal              | endpoint coupling、bridges、path law |
| `proj_R(Q)` | endpoint joint law、reference bridges | intermediate marginals、Markovity   |

“marginal preserving”和“bridge preserving”不能互换。

## 7. Exact Iterative Markovian Fitting

从一个具有目标 endpoint marginals 的 reciprocal law `P_0` 开始：

```text
P_{2n+1}=proj_M(P_{2n}),
P_{2n+2}=proj_R(Q)(P_{2n+1}).                          (7.1)
```

第一步保留所有 marginals，第二步保留 endpoint coupling，所以 exact IMF 每一步都
保持目标 endpoints。若 `P*` 同时 Markov、属于 `R(Q)` 且具有目标 endpoints，则
它是 fixed point。

但

```text
SB is a fixed point != every IMF sequence converges to SB. (7.2)
```

Shi et al. Lemma 6、Proposition 7 与 Theorem 8 的 asymptotic reverse-KL convergence
还需要 A1--A3、每步 finite KL、Proposition 5，以及 Appendix C.6 中的 tightness、
closure/coercivity 与 lower-semicontinuity responsibility。

## 8. Population bridge matching

平方损失的基本 identity 是

```text
argmin_v E|Y-v(X_t)|^2=E[Y|X_t].                      (8.1)
```

将 `Y` 取为 reference bridge drift，population optimum 恢复 (3.1)。所以 exact
object 到 learned loss 的链条是

```text
bridge conditional drift
 -> conditional expectation
 -> projected Markov drift
 -> SDE law under well-posedness.                      (8.2)
```

这说明 matching loss 学的是哪个 local object，却还没有说明有限 network 能否达到
population optimum。

## 9. Learned DSBM 多出的误差

实际 DSBM 把 (8.1) 换成 finite data、finite function class 与 SGD，再用离散 SDE
产生 samples。误差至少包括：

```text
conditional bridge sampling,
Monte Carlo regression,
function approximation,
optimization,
forward/backward alternation,
time discretization.                                  (9.1)
```

Propositions 9--10 给出论文所述 population/rich-class interfaces，不是一般
finite-sample nonasymptotic theorem。forward/backward alternation 可以减轻 endpoint
bias，但不会把 approximate drift 自动升级为 exact `proj_M`。

## 10. Gaussian projection 的独立推导

对 Brownian reference，先取任意 nondegenerate Gaussian endpoint coupling

```text
(X_0,X_T)~N((m_0,m_T),[[V_0,C_0T],[C_0T,V_T]]),       (10.1)
```

再条件 endpoints 采 Brownian bridge：

```text
X_t=(1-lambda)X_0+lambda X_T
    +sigma sqrt(T lambda(1-lambda))Z.                  (10.2)
```

bridge forward drift target 为

```text
Y_t=(X_T-X_t)/(T-t).                                  (10.3)
```

Gaussian conditional expectation 得到 affine projected drift

```text
b_M(t,x)=a_t x+c_t.                                   (10.4)
```

把 (10.4) 代入 mean/variance ODE，可精确恢复 (10.2) 的每个 one-time Gaussian
law。但 projected endpoint covariance 为

```text
Cov_M(X_0,X_T)=V_0 exp(integral_0^T a_t dt),           (10.5)
```

一般不等于原 `C_0T`。

## 11. 数值与原创图示

`b9_gaussian_projection_checks.py` 验证：

- mean/variance ODE errors 为 `3.98e-11` / `4.76e-10`；
- 500,000 samples OLS slope/intercept errors 为 `1.20e-3` / `1.46e-3`；
- original/projected endpoint covariance 为 `-0.18` / `0.75446`。

![Gaussian bridge matching 与 Markovian projection](/images/bridge/B9_markovian_projection.png)

**图 11.1：** 左：reciprocal mixture 与 Markov projection 共享的 one-time
mean/variance；中：bridge drift target 的 conditional-mean regression；右：projection
显著改变 endpoint coupling。图为固定 seed 的原创计算。

这张图证明 finite Gaussian fixture 的 object boundary，不替代一般 mimicking theorem。

## 12. Coupling initialization 不是无关 sampler detail

`Pi_0T` 决定 reciprocal mixture。independent、reference、entropic、learned paired
或 aligned coupling 会改变第一轮 path law 和 finite-iteration trajectory，而不只改变
estimator variance。

[Somnath et al. 2024](https://arxiv.org/abs/2302.11419 "官方论文页面")
使用 paired/aligned data，并假设 observed pairing 对应目标 `pi*`。若 pairs noisy、
partial 或来自另一个 data-generating mechanism，学得的是该 observed coupling 的
bridge mixture，不会自动恢复只由 marginals/reference 定义的 SB。

## 13. Recent IMF rates 怎样读

[Sokolov--Korotin 2025](https://arxiv.org/abs/2508.02770 "官方论文页面")
在 finite-state、discrete-time、everywhere-positive setting 给 explicit geometric
reverse-KL rate。

[Gentiloni--Conforti--Durmus 2025](https://arxiv.org/abs/2510.20871 "官方论文页面")
在连续 diffusion、log-concavity/regularity 与 sufficiently large horizon 等 H1--H7
regimes 给 exact-IMF rates。

两者仍是 frontier preprints，且都控制 exact operators，不包含 learned drift、finite
sample 或 Euler errors。finite theorem 不能外推连续 diffusion，continuous theorem
也不能删除 large-`T`/convexity assumptions。

## 14. IPF、IMF 与 IPMF

| Procedure | Exact steps                                     | Primary role                         |
| --------- | ----------------------------------------------- | ------------------------------------ |
| IPF       | endpoint marginal KL projections                | 修正 endpoint marginals                |
| IMF       | `proj_M` / `proj_R`                             | 在 Markov 与 reciprocal structures 间拟合 |
| IPMF      | reciprocal、Markovian 与 endpoint projections 的组合 | 同时处理 structure 与 marginals           |

[Kholkin et al. 2026](https://arxiv.org/abs/2410.02601 "官方论文页面") Theorems
3.2--3.3 给 restricted Gaussian/bounded-support results；论文标记的一般结论仍是
conjectural，不能从缩写相似性借用 IPF/IMF theorem。

## 15. 常见错误

1. **把 mimicking 说成 path-law equality。** theorem 只保留指定 marginals/functionals。
2. **把 endpoint marginals 当 endpoint coupling。** `proj_M` 可以改变 joint law。
3. **把 fixed point 当 convergence theorem。** 还需 closure、tightness 与 KL assumptions。
4. **把 population regression 当 finite neural projection。** sampling/approximation/optimization 尚未控制。
5. **把 aligned pairs 当无害 variance reduction。** 它们指定 coupling。
6. **把 finite-state rate 外推 continuous diffusion。** theorem regimes 不同。
7. **混用 IPF、IMF 与 IPMF。** operators 与 invariants 不同。

## 16. 小结

B9 的可靠链条是

```text
reciprocal bridge mixture
 -> exact Markovian projection preserving all marginals
 -> exact reciprocal projection preserving endpoint coupling
 -> exact IMF under theorem assumptions
 -> population bridge matching
 -> learned and discretized DSBM.                     (16.1)
```

从左到右，每一步都增加新的 approximation responsibility。一次 regression 可以得到
正确 local target，却不能仅凭 endpoint samples 或 one-time marginals 证明已经得到
指定 reference 下的 exact Schrödinger Bridge。

## 17. 研究式思考题

1. 哪些额外 statistics 与 one-time marginals 一起足以识别 endpoint coupling？
2. 若 `proj_M` SDE weak solution 不唯一，operator 应怎样定义才不含歧义？
3. approximate projection error 用哪种 divergence 才能沿 IMF iteration 累积？
4. noisy aligned pairs 改变的是 target coupling 还是仅 regression variance？
5. IMF 的 large-horizon contraction 与 small-noise conditioning 是否存在冲突？

## 18. 前后章节链接

- B4：reciprocal/Markov 与 endpoint factorization；
- B7：exact IPF/Sinkhorn projection baseline；
- B8：learned IPF、FBSDE 与 regression error stack；
- B10：simulation-free generalized matching 与 IPMF；
- B12：same marginals/different path laws；
- B14：inexact IMF/IPMF convergence 的开放 guarantee matrix。

完整 operator responsibility 见
`references/notes/derivations/bridge/b9_projection_operator_responsibility.md`。
