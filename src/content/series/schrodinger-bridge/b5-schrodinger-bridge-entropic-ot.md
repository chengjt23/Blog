---
title: Schrödinger Bridge 与 Entropic Optimal Transport
description: 系统比较 Schrödinger Bridge 与熵正则最优传输，分开有限噪声恒等式、动态作用量和零噪声极限。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 5
slug: b5-schrodinger-bridge-entropic-ot
tags:
  - schrodinger-bridge
  - optimal-transport
  - entropic-ot
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 覆盖静态 Gibbs kernel、动态粘性作用量、Sinkhorn divergence 与 OT 极限的适用条件。
---
## 1. 两个相似问题，两个不同起点

最优传输从“把质量从 `mu_0` 搬到 `mu_T`”出发。若 cost 是 `c(x,y)`，静态
Kantorovich 问题是

$$
\inf_{\gamma\in\Pi(\mu_0,\mu_T)}\int c(x,y)\,\gamma(dx\,dy). \tag{1.1}
$$

它选择 endpoint coupling `gamma`，但不先给出两端之间的随机动力学。

Schrödinger problem 的起点不同。它先给 reference path law `R`，再在满足双端
边缘的 path laws 中寻找对 `R` 改动最小者：

$$
\inf\left\{H(P\mid R):P_0=\mu_0,\;P_T=\mu_T\right\}. \tag{1.2}
$$

因此，SB 的基本输入不是 scalar cost，而是整条 reference dynamics。只有当
`R_0T` 具有 Gibbs/heat-kernel 结构时，(1.2) 的 endpoint 部分才可改写成常见
entropic OT。这个限定是本章最重要的边界：

```text
general Schrödinger problem
    != automatically Euclidean entropy-regularized OT.
```

## 2. Path-space problem 精确降到 endpoint coupling

令 `R_0T` 是 `R` 的 endpoint law，`R^{xy}` 是给定 `X_0=x,X_T=y` 的 reference
conditional bridge。任一可行 `P` 可分解为 endpoint coupling `gamma=P_0T` 和
conditional laws `P^{xy}`。相对熵 chain rule 给出

```text
H(P|R)
 = H(gamma|R_0T)
   + integral H(P^{xy}|R^{xy}) gamma(dxdy).                       (2.1)
```

第二项非负。固定 `gamma` 后，唯一的最优选择是

```text
P^{xy}=R^{xy},      gamma-a.e.                                   (2.2)
```

所以 dynamic problem 精确化为

```text
inf_{gamma in Pi(mu_0,mu_T)} H(gamma|R_0T).                       (2.3)
```

并由最优 endpoint coupling `gamma*` 重建

```text
P* = integral R^{xy} gamma*(dxdy).                               (2.4)
```

这里没有 small-noise approximation。(2.1)--(2.4) 对每个固定 reference 都是
finite-noise identity。B2 负责一般 path-space 条件，B3 负责 factorization；本章
只使用已经核验的结构。

式 (2.4) 也说明 SB 的随机性来自哪里：即使 endpoint coupling 已确定，路径仍按
reference pinned bridges 波动。经典 OT 只给 endpoint coupling 或 displacement
paths；SB 还保留了 reference 在端点条件下的 path fluctuations。

## 3. Gibbs endpoint reference 如何产生 entropic OT

先看有限正情形。令

```text
M_ij = mu_i nu_j,
Q_ij = M_ij exp(-c_ij/epsilon),
Z_epsilon = sum_ij Q_ij,
R_ij = Q_ij/Z_epsilon.                                          (3.1)
```

对任意 `gamma in Pi(mu,nu)`，直接展开相对熵：

```text
epsilon H(gamma|R)
 = <c,gamma>
   + epsilon H(gamma|mu tensor nu)
   + epsilon log Z_epsilon.                                    (3.2)
```

最后一项与 `gamma` 无关。因此，最小化 Gibbs reference KL 等价于最小化

```text
J_MI(gamma)
 = <c,gamma> + epsilon H(gamma|mu tensor nu).                    (3.3)
```

若用 Shannon entropy `H_Sh(gamma)=-sum gamma log gamma`，则固定 marginals 下

```text
H(gamma|mu tensor nu)
 = -H_Sh(gamma)+H_Sh(mu)+H_Sh(nu).                               (3.4)
```

所以常见目标

```text
J_Sh(gamma)=<c,gamma>-epsilon H_Sh(gamma)                        (3.5)
```

与 (3.3) 仍有相同 minimizer，但两者数值相差
`epsilon[H_Sh(mu)+H_Sh(nu)]`。若再与 `epsilon H(gamma|R)` 比较，还要加
`epsilon log Z_epsilon`。这些常数不影响单个 fixed-marginal optimization，却会
影响跨数据集、跨 `epsilon` 或跨实现报告 objective value。

对一般 `R_0T`，只有在选定 base measure `M` 且
`R_0T << M` 时，才可定义

```text
c_R(x,y)=-epsilon log(dR_0T/dM)(x,y).                            (3.6)
```

base 的选择会改变 cost 与常数；reference structural zeros 对应 support constraints
或 infinite cost。不能先写一个 Euclidean cost，再把任意 reference dynamics 塞进
同一公式。

## 4. Schrödinger factors 与离散 scaling

在 positive finite kernel 下，B3 的 entropy projection 有唯一 coupling

```text
gamma*_ij = u_i R_ij v_j.                                       (4.1)
```

代入 (3.1)，固定的 `mu,nu,Z_epsilon` 可吸收到 factors，得到熟悉形式

```text
gamma* = diag(a) exp(-c/epsilon) diag(b).                        (4.2)
```

`(a,b)` 只有 reciprocal gauge：`(a,b)` 与 `(ka,k^{-1}b)` 表示相同 coupling。
边缘方程可用 B7 的 alternating scaling 求解。

对任意其他可行 coupling，KL Pythagorean identity 给

```text
J_MI(gamma)-J_MI(gamma*)
 = epsilon H(gamma|gamma*).                                     (4.3)
```

有限正情形下，(4.3) 同时给出唯一性和严格 objective gap。它不提供一般空间
existence，也不提供 Sinkhorn 的 iteration rate；这两个责任分别属于 B3 与 B7。

## 5. 从 endpoint coupling 到 dynamic action

### 5.1 Classical Benamou--Brenier baseline

对二次 cost，classical OT 可写成 continuity-equation action：

```text
partial_t rho + div(rho v)=0,
rho_0=mu_0, rho_T=mu_T,                                         (5.1)

W_2^2(mu_0,mu_T)
 = inf integral_0^1 integral |v_t(x)|^2 rho_t(x) dx dt.         (5.2)
```

[Benamou--Brenier 2000](https://doi.org/10.1007/s002110050002 "官方论文页面")
中的 `v` 是 deterministic transport velocity。它不是受控 diffusion 的完整 forward
drift，也不是 osmotic velocity。

### 5.2 Brownian control action

若 reference 是 Brownian diffusion，candidate law 通过 control velocity 改变 drift，
Girsanov/Föllmer identity 在相应 absolute-continuity、finite-energy 与 martingale
problem 条件下给

```text
H(P|R)
 = H(P_0|R_0)
   + (1/2) E_P integral |beta_t|_a^2 dt.                         (5.3)
```

其中 `a=sigma sigma^T`，drift correction 是 `a beta`。scalar
`a=sigma^2 I` 时，(5.3) 等价于

```text
(1/(2sigma^2)) E integral |control velocity|^2 dt.              (5.4)
```

本教程固定 generator convention

```text
L phi=b dot grad phi+(1/2)a:Hess phi.                            (5.5)
```

因此 Doob drift correction 是 `a grad log g`。若来源把 SDE 写成
`sqrt(2a)dW`，公式会多一个因子 2，必须先换 convention。

### 5.3 Current--osmosis decomposition

[Cattiaux et al. 2023](https://doi.org/10.1214/22-AIHP1320 "官方论文页面")
在 reversible Kolmogorov reference、Hypotheses 1.10、Markov `P` 和
`H(P|R)<infinity` 下给出严格分解。

若 `b^+` 是 forward drift，`b^-` 是 Nelson original-time backward drift，则

```text
v_current=(b^+ + b^-)/2,
v_osmotic=(b^+ - b^-)/2.                                       (5.6)
```

current velocity 驱动 distributional continuity equation

```text
partial_t rho + div_m(rho v_current)=0.                         (5.7)
```

osmotic relative velocity 由 marginal density 的 distributional score 决定，并有
finite integrated Fisher information。Definitions 6.1 与 Proposition 6.2 给出

```text
H(P_[0,t] | R_[0,t]^{P_0})
 = F(P_t)-F(P_0)
   + integral_0^t [current action + I_a(P_s|m)] ds,              (5.8)
```

其中

```text
F(mu)=H(mu|m)/2,
I_a(mu|m)=(1/2) integral |grad log(dmu/dm)|_a^2 dmu.             (5.9)
```

式 (5.8) 是 finite-entropy viscous dynamic formulation。它把 classical kinetic
action 与 osmotic/Fisher penalty 分开，而不是把 controlled drift 直接称为
Benamou--Brenier velocity。

## 6. Small-noise limit 不是一句“epsilon 趋于零”

finite-noise identity、Gamma convergence、minimum convergence、optimizer
convergence 和 interpolation convergence 是五个不同结论。

### 6.1 Léonard 的 Gamma-convergence 路线

[Léonard 2012](https://doi.org/10.1016/j.jfa.2011.11.026 "官方论文页面")
在 Polish path space、conditional LDP 和 coercive rate assumptions 下证明 scaled
entropy functionals Gamma-converge 到 static/path action。

但 constrained minima 使用 recovery terminal marginals

```text
mu_T^k -> mu_T.                                                (6.1)
```

Theorems 3.3/3.6 不是“任意固定 `mu_T` 对每个 `k` 都无条件成立”。finite limiting
value 与 relative compactness 给 cluster points；只有 limiting optimizer 唯一时，
才能把 subsequential statement 升级为 full-sequence convergence。path law 与
interpolation convergence 还需要 relevant geodesics 唯一。

### 6.2 Mikami 的 Brownian h-path 路线

[Mikami 2004](https://doi.org/10.1007/s00440-004-0340-4 "官方论文页面") 使用 noise
amplitude `sqrt(epsilon)`。主要 theorem 的 terminal marginal 是 Gaussian-smoothed
`P_{1,epsilon}`，不是原始 `P_1`。fixed-terminal variant 需要额外 density 和
log-integrability assumptions。

在 finite second moments 下，endpoint couplings tight，cluster points 具有
cyclically monotone support；若初始 law 有 density，quadratic OT optimizer 唯一，
full sequence 才收敛。过程 path/control convergence 又需要更强条件。

### 6.3 参数因子

Mikami 使用 cost `|x-y|^2` 与 action `integral |u|^2`；Léonard 的 Schilder rate
使用 `|x-y|^2/2` 与 `(1/2)integral |dot omega|^2`。二者 minimizer 相同，数值值
差因子 2。`epsilon` 只有在 heat-kernel exponent 已固定后才能解释为 Brownian
variance。

## 7. Entropic interpolation 与 displacement interpolation

finite `epsilon` 时，SB 的 marginal curve

```text
t -> P_t^epsilon                                             (7.1)
```

称为 entropic interpolation。它来自最优 endpoint coupling 与 noisy reference
bridges 的混合。classical displacement interpolation 则沿 optimal transport
geodesics 推送质量。

二者在有限噪声时不是同一对象。Léonard 2012 的 Brownian example 说明：在
optimizer 和 geodesics 唯一等条件下，small-noise path laws 收敛到 geodesic-
supported law，相应 one-time marginals 才收敛到 displacement interpolation。

因此，以下推理不成立：

```text
SB uses quadratic Brownian cost
    => every finite-noise SB marginal curve is an OT geodesic.             (7.2)
```

osmotic/Fisher action正是 finite-noise curve 偏离 deterministic geodesic 的结构性
来源之一。

## 8. Regularization bias 与 Sinkhorn divergence

raw regularized OT value

```text
OT_epsilon(alpha,beta)
 = inf_pi integral C d pi
   + epsilon H(pi|alpha tensor beta)                            (8.1)
```

一般满足 `OT_epsilon(alpha,alpha)>0`，并可能在 measure fitting 中产生 shrinkage
bias。现代 debiased Sinkhorn divergence 定义为

```text
S_epsilon(alpha,beta)
 = OT_epsilon(alpha,beta)
 - OT_epsilon(alpha,alpha)/2
 - OT_epsilon(beta,beta)/2.                                    (8.2)
```

Feydy et al. 2019
在 compact metric space、Lipschitz cost 与 positive universal Gibbs kernel 等条件
下证明它 nonnegative、separates points 并 metrizes weak convergence。

`S_epsilon` 是由三个不同 marginal pairs 的 static values 组合成的 scalar
discrepancy。它不是 coupling、path law 或某个固定 reference 下的 SB optimizer。
Cuturi 2013 和 Peyré--Cuturi 早期章节中 \`\`Sinkhorn divergence'' 的历史用法还可能
指 regularized cost/approximation，引用时必须消歧。

## 9. 最小数值检查

`finite_entropic_ot_checks.py` 在 `3x3` positive Gibbs kernel 上验证：

- scaling 55 次后 marginal residual `7.99e-15`；
- Gibbs-reference KL 与 `J_MI` 常数 identity error `1.11e-16`；
- `J_MI` 与 Shannon form 的常数 error `1.78e-15`；
- Pythagorean gap error `1.65e-16`；
- `gamma*/R` rank-one ratio `2.70e-17`。

这些结果验证 finite algebra 和实现，不证明 Brownian Gamma convergence。

B7 的 log-domain fixture 进一步说明 small `epsilon` 的数值问题：在
`epsilon=0.005` 时普通 kernel 将必要正 entries 下溢为零并产生非有限迭代；
log-domain continuation 的 marginal residual 为 `4.23e-14`。数值稳定性和
数学 small-noise limit 是两条独立问题。

![不同 entropy temperature 下的最优 endpoint couplings](/images/bridge/B5_entropic_couplings.png)

**图 9.1：** 在固定 marginals 与 cost 下，较小 `epsilon` 使 coupling 集中到低成本
entries，较大 `epsilon` 则产生更弥散的 coupling。该有限图示说明 regularization
形态变化，不承担 small-noise Gamma-convergence 的证明。

## 10. 常见错误

1. **把 general SB 直接写成 Euclidean entropic OT。** 先检查 `R_0T` 相对于
   哪个 base 具有何 density。
2. **把 finite-noise identity 当作 zero-noise theorem。** (3.2) 是代数恒等式；
   Gamma/minimizer/path convergence 需要额外 LDP、tightness 和 uniqueness。
3. **忽略 moving endpoint marginals。** Léonard 的 constrained theorem 使用
   recovery sequence，不能删掉上标 `k`。
4. **混合 cost/action 的因子 2。** 必须同时报告 noise amplitude、heat exponent、
   cost 和 action convention。
5. **把 current velocity 当 controlled forward drift。** 前者驱动 marginal
   continuity equation；后者还含 osmotic component。
6. **把 Sinkhorn divergence 当 SB。** 它是 debiased scalar discrepancy。
7. **把稳定算法当极限定理。** log-domain 防下溢，不证明 `epsilon -> 0` 的
   variational convergence。

## 11. 小结

SB 与 entropic OT 的关系可整理为四步：

```text
reference path law R
 -> endpoint reference R_0T and unchanged conditional bridges
 -> Gibbs/heat-kernel representation when available
 -> entropic OT, dynamic action and small-noise OT limit.         (11.1)
```

第一步是 path-space probability，第二步是 exact entropy disintegration，第三步
需要 reference structure，第四步包含有限噪声和极限两类不同结论。保持这四层，
才能同时解释为什么 SB 与 OT 紧密相连，又为什么二者不能无条件互换。

## 12. 研究式思考题

1. 若 `R_0T` 有 structural zeros，怎样修改 (3.6) 的 cost 表示和可行 coupling
   集？factorization 是否仍存在？
2. Léonard 的 moving recovery marginals 在什么附加条件下可替换为固定
   `mu_T`？需要哪种 recovery construction？
3. 当 classical OT optimizer 不唯一时，entropic regularization 是否选择唯一
   limiting optimizer？选择原则依赖 reference 吗？
4. 对 state-dependent `a(x)`，如何把 Proposition 6.2 的 weighted current/
   Fisher action与几何 transport metric 对齐？
5. `S_epsilon` 的 debiasing 改善 measure fitting，但它是否仍能解释为某个单一
   path-space inference problem？若不能，缺失的对象是什么？

## 13. 前后章节链接

- B3：static factorization、Schrödinger system 和一般存在唯一性；
- B4：two-sided factors、前后向 kernels 与 current/osmotic convention；
- B6：Girsanov、control energy、Föllmer drift 和 Gaussian closed form；
- B7：IPF/Sinkhorn、log-domain stabilization 和 convergence；
- B12：与 score diffusion、flow matching 的对象级比较。

统一符号见
`references/notes/derivations/bridge/b3_b7_notation_audit.md`；有限恒等式见
`references/notes/derivations/bridge/finite_entropic_ot_identity.md`。
