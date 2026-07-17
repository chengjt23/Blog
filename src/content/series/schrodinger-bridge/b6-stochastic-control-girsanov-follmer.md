---
title: 随机最优控制、Girsanov 与 Föllmer Drift
description: 从路径相对熵到二次控制能量，梳理 Girsanov、Föllmer drift、HJB 与线性高斯 steering 的边界。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 6
slug: b6-stochastic-control-girsanov-follmer
tags:
  - schrodinger-bridge
  - stochastic-control
  - girsanov
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖 Wiener 与非退化扩散下的控制表示、可达性条件和 same-channel 验证。
---
## 1. 同一个二次能量，至少四种不同结论

Schrödinger problem 最常见的控制写法是

$$
\min_{P}\ H(P\mid R)\quad\text{subject to}\quad P_0=\mu_0,\;P_T=\mu_T. \tag{1.1}
$$

当 candidate 只通过 reference 的噪声通道改变 drift 时，(1.1) 常被改写成

$$
\min_u\ \frac{1}{2}\,\mathbb{E}_P\!\left[\int_0^T\lVert u_t\rVert^2\,dt\right]. \tag{1.2}
$$

但从 (1.1) 到 (1.2) 不是一次形式代换。必须区分：

1. 给定 control 能否通过 Girsanov 产生 `P<<R`；
2. 给定 realization 只给 entropy upper bound，还是给 exact equality；
3. 每个有限熵 `P` 是否具有 canonical drift；
4. canonical weak law 是否能在指定 Brownian basis 上强实现。

本章先在 Wiener space 闭合这四层，再讨论双端 factors、HJB 和一般 diffusion 的
可安全范围。

## 2. Wiener space 上的 energy upper bound

令 `R` 为从零出发的 Wiener measure，`B` 为 Brownian motion，`u` 为适应过程，

```text
U_t = integral_0^t u_s ds,
X_t = B_t+U_t.                                          (2.1)
```

若 `P=Law(X)`，则 [Lehec 2013](https://doi.org/10.1214/11-AIHP464 "官方论文页面")
Proposition 1 给出

```text
H(P|R) <= (1/2) E integral_0^T |u_t|^2 dt.             (2.2)
```

不等号的原因不是 Girsanov \`\`损失了常数''，而是不同 controls 可能在 path-law
层产生同一个 `P`。右侧依赖 realization，左侧只依赖 law。因而任取一个 neural
drift 并最小化样本能量，不能直接宣称已经精确最小化 path KL。

若 reference 初始 law 为 `R_0`、candidate 初始 law 为 `P_0`，对应分解为

```text
H(P|R)
 = H(P_0|R_0)+H(P(.|X_0)|R(.|X_0) | P_0).             (2.3)
```

控制能量负责第二项；初始 mismatch 不能隐藏进 drift。

## 3. Canonical Föllmer drift 给出 equality

反过来，设 `P=F R`。令

```text
F_t = E_R[F | F_t^X]
    = 1 + integral_0^t v_s dot dX_s,                  (3.1)
u_t = 1_{F_t>0} v_t/F_t.                              (3.2)
```

Girsanov 使

```text
X_t - integral_0^t u_s ds                            (3.3)
```

在 `P` 下成为 Brownian motion；对 `log F_t` 应用 Itô 公式得到

```text
H(P|R)=(1/2) E_P integral_0^T |u_t|^2 dt.             (3.4)
```

这就是 canonical Föllmer drift。Lehec Theorem 2 还通过 localization 覆盖
infinite-entropy 边界，所以 Novikov 只是从给定 control 构造 martingale 的常用
充分条件，不是 (3.4) 所有成立情形的必要条件。

这里的结论首先是 canonical path law 上的 weak representation。Lehec Theorems
4/7 只有在 Föllmer SDE pathwise unique 或更强的 regular density 条件下，才把它
提升为指定 Brownian basis 上的 strong minimum-energy realization。

## 4. Half-bridge 与 two-sided control

若只固定 terminal law `nu=rho R_T`，Wiener half-bridge 为

```text
dP^half/dR = rho(X_T).                                 (4.1)
```

Lehec Lemmas 10--11 给出

```text
H(P^half|R)=H(nu|R_T),
u_t=grad log Q_{T-t}rho(X_t)                           (4.2)
```

（后一式需要额外光滑与正性）。它只满足一个 endpoint constraint。

双端 Schrödinger Bridge 则由 B3/B4 的 factors 给出

```text
dP*/dR = f_0(X_0) g_T(X_T)/Z.                         (4.3)
```

从 forward 方向看，`f_0` 改变 initial law，传播后的 terminal potential

```text
g_t(x)=E_R[g_T(X_T)|X_t=x]                            (4.4)
```

产生 feedback。对 variance `epsilon` 的 Brownian reference，
[Chen--Georgiou--Pavon 2021](https://doi.org/10.1137/20M1339982 "官方论文页面")
equations (4.20)--(4.22) 给出

```text
dX_t = u*(t,X_t)dt + sqrt(epsilon)dW_t,
u* = epsilon grad log g_t,                             (4.5)
J(P*)=E integral_0^T |u*|^2/(2 epsilon)dt.             (4.6)
```

`g_T` 不是预先给定的 terminal cost；它和 forward factor 一起由 `mu_0,mu_T`
的 split boundary system 决定。

## 5. Same-channel diffusion 与 range 条件

考虑

```text
dX_t=[b(t,X_t)+sigma(t,X_t)u_t]dt
     +sigma(t,X_t)dW_t,
a=sigma sigma^T.                                       (5.1)
```

`u` 是 noise-coordinate control，physical drift correction 为 `sigma u`。因此
quadratic energy 是 `|u|^2`；若 `a` 可逆，也可写成 physical correction 的
`a^{-1}` norm。

漂移改变量若不属于 `Range(sigma)`，便不能由同一噪声的 measure change 产生。
在 degenerate case 中，直接把 `a^{-1}` 换成 pseudoinverse 仍不够；还要证明 range、
measurability、martingale 和 absolute-continuity 条件。本章不陈述任意退化 diffusion
的 blanket Girsanov theorem。

## 6. 非退化 state-dependent diffusion 的有限熵定理

[Cattiaux et al. 2023](https://doi.org/10.1214/22-AIHP1320 "官方论文页面")
给出本章使用的一般 diffusion 层。在其 Hypotheses 1.10 下，reference 是可逆
Kolmogorov diffusion，`a` 为可逆 `C^1` matrix field，并有相应 growth、nonexplosion
和 martingale-problem uniqueness 条件。对 Markov `P` 且 `H(P|R)<infinity`，
equation (4.4) 与 Proposition 4.6 给出

```text
H(P|R)
 = H(P_0|R_0)
   +(1/2) E_P integral |beta_forward^{P|R}|_a^2 dt

 = H(P_T|R_T)
   +(1/2) E_P integral |beta_backward^{P|R}|_a^2 dt.   (6.1)
```

这些 relative velocities 可以是 measurable/distributional objects，不要求正文先
假设经典 pointwise score。该 theorem 覆盖一类真正 state-dependent 的非退化
diffusions，但不应外推到任意 time-dependent、degenerate 或 non-Markov law。

## 7. Forward、backward、current 与 osmotic

使用 Nelson original-time backward drift

```text
b_t^- = lim_{h downarrow 0}
        E[(X_t-X_{t-h})/h | X_t],                     (7.1)
```

定义

```text
v_current=(b^+ + b^-)/2,
v_osmotic=(b^+ - b^-)/2.                              (7.2)
```

Cattiaux 原文的 backward-arrow velocity 是 `-b^-`，引用时必须先做此符号映射。
在足够光滑的常系数情形，

```text
b^+ = b + a grad log g_t,
b^- = b_R^- - a grad log f_t,                         (7.3)
```

而一般有限熵 theorem 只保证相应 distributional score 与 integrated Fisher
information。B4 负责 two-sided kernel/drift 结构；B6 只解释其控制能量含义。

## 8. 从 density control 到 HJB verification

对 same-channel model，考虑 classical density-control problem

```text
J(rho,u)=integral_0^T integral
         [(1/2)|u|^2+V]rho dx dt,                     (8.1)
```

约束为

```text
partial_t rho+div[(b+sigma u)rho]
 =(1/2)partial_ij(a_ij rho),
rho_0=mu_0, rho_T=mu_T.                               (8.2)
```

令 `lambda` 满足

```text
partial_t lambda+b dot grad lambda
 +(1/2)a:Hess(lambda)
 +(1/2)|sigma^T grad lambda|^2-V=0.                   (8.3)
```

在光滑性、可积性和边界项消失的 classical regime 中，分部积分与平方配方给出

```text
J(rho,u)
 = integral lambda(T)dmu_T-integral lambda(0)dmu_0
   +(1/2)integral_0^T integral
      |u-sigma^T grad lambda|^2 rho dx dt.             (8.4)
```

所以任何命中两端的可行流在

```text
u*=sigma^T grad lambda                                (8.5)
```

时最优。关键限定是“命中两端”：一份 HJB solution 本身并不证明 terminal marginal
已满足。

## 9. Hopf--Cole 与 Schrödinger factors

令

```text
phi=exp(lambda)>0.                                    (9.1)
```

则 (8.3) 线性化为

```text
partial_t phi+b dot grad phi+(1/2)a:Hess(phi)=V phi,  (9.2)
u*=sigma^T grad log phi.                              (9.3)
```

再令 `hat_phi=rho/phi`，得到 adjoint forward equation，并满足

```text
phi_0 hat_phi_0=mu_0,
phi_T hat_phi_T=mu_T.                                 (9.4)
```

CGP 2021 Proposition 5.1/Theorem 5.2 与独立推导
`references/notes/derivations/bridge/b6_hjb_hopf_cole_control.md` 共同承担这个
classical verification。Hopf--Cole 解释了为什么 quadratic control 产生线性
Schrödinger system；它不替代 factors 的存在唯一性 theorem，也不自动建立
density flow 与唯一 path law 的等价。

## 10. Scalar Gaussian bridge 的闭式检查

取一维 Brownian reference

```text
dX_t=sqrt(epsilon)dW_t,
X_0 ~ N(m_0,s_0^2),
X_T ~ N(m_T,s_T^2).                                   (10.1)
```

Gaussian SB 仍为 Gaussian Markov process，其 optimal drift 为 affine：

```text
u*(t,x)=alpha_t x+beta_t.                             (10.2)
```

均值与方差满足

```text
dot m_t=alpha_t m_t+beta_t,
dot S_t=2 alpha_t S_t+epsilon.                        (10.3)
```

现有推导
`references/notes/derivations/bridge/gaussian_schrodinger_bridge.md` 从 endpoint
Gaussian KL、quadratic factor 和 covariance ODE 独立构造该解。脚本
`references/code/bridge/gaussian_schrodinger_bridge_checks.py` 验证：

- endpoint mean/variance 命中误差为数值精度；
- endpoint KL 与 control energy 的差为 `3.82e-14`；
- Gaussian Markov covariance identity 成立；
- forward/backward score convention 一致。

这是一维解析 fixture，不是 nonlinear 或 matrix theorem 的数值证明。

## 11. Linear Gaussian covariance steering

对同通道线性系统

```text
dX_t=A(t)X_t dt+B(t)u_t dt+B(t)dW_t,                  (11.1)
```

以及正定 Gaussian endpoints，quadratic `lambda` 把 HJB 化为 matrix Riccati
equation，feedback 为线性函数。
[Chen--Georgiou--Pavon 2016](https://doi.org/10.1109/TAC.2015.2457784 "官方论文页面")
在 controllability 条件下，用 coupled Lyapunov/Riccati boundary system 构造
nonsingular branch，并证明相应 optimal feedback 的存在与唯一性。

必须保留三个范围限制：control 与 noise 使用同一 `B` 通道；endpoint covariances
正定；controllability 不可省略。scalar fixture 只是该 matrix theorem 的 Brownian
特例。

## 12. FBSDE 在本章的位置

HJB/PDE 可在足够正则时通过 nonlinear Feynman--Kac 转写为 forward--backward SDE，
随后再用 neural networks 参数化未知函数。但这是新的 theorem 与 approximation
层，至少增加：terminal/boundary consistency、BSDE well-posedness、time
discretization、Monte Carlo、function approximation 和 optimization error。

B6 不从公式相似性推出 general FBSDE representation，也不把 finite neural loss
等同于 (1.1)。这些来源、likelihood identity 和误差责任由 B8 专门审计。

## 13. 常见错误

1. **把 energy upper bound 写成 equality。** 任意 realization 先只有 (2.2)。
2. **忽略 initial entropy。** `P_0!=R_0` 时必须加 `H(P_0|R_0)`。
3. **把 half-bridge 当 two-sided SB。** terminal tilt 不自动保持指定 initial law。
4. **把 physical drift 当 noise-coordinate control。** 两者相差 `sigma`，能量范数也随之改变。
5. **在 degenerate diffusion 中无条件使用 pseudoinverse。** range 与 AC 必须先证明。
6. **由 HJB 解直接宣称 bridge 已解。** 还需 split boundary feasibility 与 realization。
7. **把 density flow 当唯一 path law。** one-time marginals 不决定 multi-time correlations。
8. **把 FBSDE/neural parameterization 当 classical equivalence theorem。** 表示与近似各有额外假设。

## 14. 小结

B6 的可靠逻辑链是

```text
path-law entropy
 -> chosen-control upper bound
 -> canonical Follmer equality
 -> two-sided potential feedback
 -> classical HJB/Hopf--Cole verification
 -> Gaussian/covariance-steering specializations.      (14.1)
```

Wiener theory、有限熵非退化 diffusion theorem 与 classical density control 是三层
不同证据。它们在重叠条件内一致，却不能拼成一个对任意 diffusion、control 或
neural solver 都成立的无条件结论。

## 15. 研究式思考题

1. 同一个 path law 在两个不同 Brownian bases 上实现时，为什么 control energy 可能不同？
2. 对 degenerate `sigma`，如何用 range condition 表述可允许的 drift perturbation？
3. split boundary HJB 与给定 terminal cost 的 ordinary HJB 在 well-posedness 上有何差异？
4. Cattiaux 的 measurable relative velocity 怎样在 smooth regime 退化为 pointwise score drift？
5. 当 control 与 noise channels 不同，minimum-energy steering 还保留哪些 SB 解释？

## 16. 前后章节链接

- B2：path-space KL、initial/conditional entropy 与 endpoint constraints；
- B3/B4：Schrödinger factors、two-sided transform 与 forward/backward drifts；
- B5：Brownian control action、current--osmosis 与 entropic OT；
- B7：exact proportional fitting 怎样计算 factors；
- B8：FBSDE、likelihood training 与 neural approximation；
- B13：控制、transport 和 sampling 三类应用何时需要 exact SB。

完整条件矩阵见
`references/notes/derivations/bridge/b6_girsanov_responsibility.md`。
