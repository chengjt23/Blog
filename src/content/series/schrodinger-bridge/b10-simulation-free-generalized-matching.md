---
title: Simulation-Free、Generalized Matching 与 IPMF
description: 比较 simulation-free matching、广义桥匹配和 IPMF，追踪每种方法替换了哪一个投影或采样算子。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 10
slug: b10-simulation-free-generalized-matching
tags:
  - schrodinger-bridge
  - simulation-free
  - ipmf
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖 SF2M、generalized matching、adversarial/light 方法与 IPMF 的 operator-level 关系。
---
B8 的神经方法仍保留 IPF 外循环：模拟当前扩散、拟合反向过程，再从另一端模拟并
拟合回来。B9 改从 reciprocal/Markov projection 理解这个过程。本章继续追问：
能否省掉训练时的路径模拟、昂贵 projection，或难以计算的 conditional control？

2024--2026 的答案不是一个统一算法：

```text
SF2M       用 analytic conditional bridges 替代 learned-process simulation；
GSBM       把目标推广到 kinetic cost + state/running cost；
LightSB-M  直接投影到 Schrödinger-bridge family；
ASBM       用 adversarial transition fitting 近似 Markov projection；
IPMF       把 IPF marginal resets 嵌入 bidirectional IMF。
```

因此比较这些方法时，需要逐项写清 target path law、exact population operator、
保持的 invariant、可采样 estimator，以及 finite network 最终近似了哪一层。

## 1. Operator、estimator 与 learned process

本章把每种方法拆成四层：数学目标、probability-law/population exact operator、
可训练 estimator，以及有限网络/样本/优化/离散实现。只有前两层的假设成立时，
才能使用 exact theorem。例如

```text
exact Markov projection theorem
  != population adversarial identity
  != finite discriminator/generator convergence.               (1.1)
```

类似地，`P_0=mu_0,P_1=mu_1` 只说明两端边缘正确，不说明 `P` 是否最小化
`KL(P||R)`，也不说明 endpoint coupling 或 pinned bridges 与 reference 一致。

## 2. SF2M：训练时不模拟 learned process

### 2.1 Conditional paths 到 marginal fields

设 latent variable `z~q`，每个 `z` 对应可采样 conditional path `p_t(x|z)`：

```text
p_t(x) = integral p_t(x|z) q(z) dz.                             (2.1)
```

若 conditional probability-flow velocity `u_t^o(x|z)` 与 score
`s_t(x|z)=grad log p_t(x|z)` 可计算，Bayes averaging 给出

```text
u_t^o(x) = E[u_t^o(X_t|Z) | X_t=x],
s_t(x)   = E[s_t(X_t|Z)   | X_t=x].                            (2.2)
```

score identity 由 (2.1) 求导得到；flow identity 则把 conditional continuity
equations 按 `q(z)` 积分。相应 population regression 是

```text
L(theta)=E[||v_theta(t,X_t)-u_t^o(X_t|Z)||^2
           +lambda(t)^2||s_theta(t,X_t)-s_t(X_t|Z)||^2].       (2.3)
```

平方损失的 minimizer 正是 (2.2) 的 conditional expectations。
[SF2M note](https://arxiv.org/abs/2307.03672 "官方论文页面")
记录的 Theorem 3.2 更精确地证明 conditional/unconditional loss gradients 相同；
它要求 interpolation density 为正，并把结论放在 global population optimum 层。

### 2.2 Brownian bridge targets

取 `z=(x_0,x_1)`，reference diffusion scale 为 `sigma`。Brownian bridge 有

```text
p_t(.|x_0,x_1)
=N((1-t)x_0+t x_1, sigma^2 t(1-t) I).                           (2.4)
```

记 `m_t^z=(1-t)x_0+t x_1,w_t=sigma^2t(1-t)`，则

```text
s_t(x|z)=-(x-m_t^z)/w_t,
u_t^o(x|z)=(x_1-x_0)+(w_t'/(2w_t))(x-m_t^z).                   (2.5)
```

训练循环只抽 `z,t,X_t` 并回归 (2.5)，不积分当前 learned SDE。“simulation-free”
只限定 **learned-process simulation during training**；它没有取消 endpoint/coupling
samples、analytic bridge samples、minibatch OT 或 inference ODE/SDE integration。

### 2.3 Coupling 定义 target

若 `q(x_0,x_1)` 只有给定边缘，Theorem 3.3 保证 population process 恢复相应
Brownian-bridge mixture marginals 并输送 `q_0` 到 `q_1`，但不同 coupling 仍产生
不同 interior marginals。

标量 Gaussian 例子中，endpoint variances 为 `v_0,v_1`、covariance 为 `c` 时

```text
V_t=(1-t)^2v_0+t^2v_1+2t(1-t)c+sigma^2t(1-t).                  (2.6)
```

改变 `c` 不改变 endpoint marginals，却改变中间 scores 与 flows。独立推导见
Gaussian ledger（补充材料暂未公开）。

![同端点边缘下的 coupling 与 SF2M population regression](/images/bridge/B10_sf2m_population_identity.png)

**图 2.1：** 两个 endpoint coupling 具有相同边缘，却诱导不同的中间方差、
conditional score 与 flow target；虚线为解析 population field，散点/实线为有限样本回归。

说明代码为每种 coupling 使用 400000 个样本。exact Gaussian normal equations
误差为 `0`；Monte Carlo 最大 coefficient error 为 `1.12e-2`，最大 marginal moment
error 为 `1.60e-3`。两种 coupling 在 `t=0.5` 的 marginal variance 相差 `0.4`。

要升级为 exact Brownian-reference SB，必须使用精确 entropic-OT endpoint coupling；
SF2M Proposition 3.4 承担这一步。实际 minibatch OT 只是 population coupling 的
近似，所以 coupling estimation 是独立误差层。

## 3. GSBM：推广的是问题，不只是 parameterization

GSBM 在 kinetic stochastic control 中加入 state 或 population-dependent running
cost，抽象目标为

```text
min E integral_0^1 [1/2||u_t(X_t)||^2+V_t(X_t;rho_t)]dt
s.t. dX_t=u_t(X_t)dt+sigma dW_t,
     X_0~rho_0, X_1~rho_1.                                    (3.1)
```

`V_t` 可表达 obstacle、几何表面或 mean-field interaction。它一般不是相对普通
Brownian reference 的同一个 `KL(P||R)`；若要翻译成 tilted reference，还需额外验证
normalization 与绝对连续性。

[GSBM note](https://arxiv.org/abs/2310.02233 "官方论文页面")
把方法分成两个 population stages：Proposition 1 识别 Stage 1 unique minimizer，
Proposition 2 把 Stage 2 写成 conditional SOC。二次 state cost 下 Lemma 3 给出
analytic Gaussian conditional path；一般 cost 则依赖 spline/Gaussian solver 或
path-integral importance reweighting。

因此成本从 unconditional path fitting 转移到许多 conditional subproblems；一般
cost 下仍有 solver approximation、support mismatch 和 weight degeneracy。

Theorem 5 只证明 exact stages 的 objective 单调不增：

```text
L(theta^(n+1)) <= L(theta^n).                                  (3.2)
```

Theorem 6 只给 `global optimum => fixed point`。二者都没有证明 global convergence、
fixed-point uniqueness 或 rate；finite CondSOC solver 还会破坏 exact descent premise。

## 4. Light/Optimal SBM：任意 coupling theorem 的真正原因

给定 endpoint plan `pi`，用 Brownian pinned bridges 构造 reciprocal mixture `T_pi`。
令 `S` 为具有起点 `rho_0` 的 **exact Schrödinger bridges family**，定义

```text
proj_S(T_pi)=argmin_{S in S} KL(T_pi||S).                       (4.1)
```

[Light/Optimal note](https://arxiv.org/abs/2402.03207 "官方论文页面")
中的 Theorem 3.1 证明，对任意 `pi`，(4.1) 的 population optimum 都是目标 SB。
这与 SF2M 的 arbitrary-coupling boundary 不矛盾：

```text
SF2M:       fit the marginal fields of T_pi;
Optimal SBM: project T_pi over the exact SB family S.          (4.2)
```

coupling-independence 来自 projection domain 恰好是 exact SB family。Theorems
3.2--3.3 再把 projection 化成 regression，并连接 EgNOT/LightSB objective。

实践中 `S` 被 finite Gaussian-mixture Schrödinger potential 代替，再用 SGD 优化。
搜索空间不再是全部 exact SBs，所以 theorem 不自动保证 finite model 的
coupling-independence 或 exact optimality。

## 5. ASBM：exact D-IMF 与 adversarial fit

离散 reciprocal projection 保留 endpoint coupling 并换回 reference pinned bridges；
Markov projection 保留 Markov factorization/one-time marginals，但通常改变 endpoint
joint law。exact D-IMF 交替

```text
q^(2l+1)=proj_M(q^(2l)),
q^(2l+2)=proj_R(q^(2l+1)).                                     (5.1)
```

[ASBM note](https://doi.org/10.52202/079017-2845 "官方论文页面")
中 Propositions 3.3/3.5 证明两步是 KL projections；Theorem 3.1 把
`R intersection M` 识别为 discrete SB，Theorem 3.6 给出 exact D-IMF KL convergence。

ASBM 实际用 adversarial divergence estimation 与 conditional generator 拟合
transition：

```text
exact proj_M
  -> population adversarial equality at optimal networks
  -> finite alternating GAN training.                           (5.2)
```

Theorem 3.6 只位于第一行。directional fit 精确保留 conditioning-side endpoint，
另一端一般只近似；forward/backward alternation 缓解 endpoint bias，但不消除
discriminator、generator、optimization、finite-grid 与 sampler errors。

## 6. IPMF：把 bidirectional IMF 展开成 IPF + IMF

IPMF 指出实用 bidirectional IMF heuristic 还包含 endpoint marginal replacement，
即 IPF half-steps。一轮 exact IPMF 可写成

```text
proj_R -> proj_1 o proj_M -> proj_R -> proj_0 o proj_M.         (6.1)
```

`proj_R/proj_M` 改善 reciprocal--Markov consistency；`proj_0/proj_1` 强制重设
一个 endpoint marginal。它们的 invariant 不同：

| operator              | 保持/强制                                   | 可能改变                               |
| --------------------- | --------------------------------------- | ---------------------------------- |
| reciprocal projection | endpoint joint law                      | interior conditional law、Markovity |
| Markov projection     | Markov factorization、one-time marginals | endpoint coupling、reciprocity      |
| IPF endpoint reset    | 一个 endpoint marginal 与 conditional law  | 另一端 marginal、joint law             |

[IPMF note](https://arxiv.org/abs/2410.02601 "官方论文页面")中的 Theorem 3.2 在指定
Gaussian regimes 下证明 parameter 与 bidirectional-KL exponential convergence；
Theorem 3.3 在 bounded-support marginals 下证明 discrete/continuous IPMF endpoint
couplings 的 weak convergence。

高维 Gaussian rate 需要论文规定的 noise regime；bounded-support weak convergence
不等于一般 KL rate。任意 start、dimension 与 regularization 的 general convergence
仍被论文明确标为 conjecture。learned DSBM/ASBM 又把 (6.1) 的 law operators 换成
regression/adversarial fits，因此还要承担 outer truncation 与 stepwise projection error。

## 7. 五种方法省掉了什么

| 方法        | 数学目标                                  | 被替换的成本                                   | 仍需的 primitive                    | exact 结论                                | learned 边界                  |
| --------- | ------------------------------------- | ---------------------------------------- | -------------------------------- | --------------------------------------- | --------------------------- |
| SF2M      | chosen mixture；entropic coupling时为 SB | training learned-SDE simulation、IPF loop | coupling + analytic bridge       | population identity                     | empirical OT、network、solver |
| GSBM      | kinetic + running cost control        | 部分 unconditional bridge loop             | CondSOC/importance sampling      | exact-stage descent；optimum fixed point | solver、weights、network      |
| LightSB-M | projection onto exact SB family       | 多轮 reciprocal/Markov fitting             | bridge samples + potential model | one population projection               | finite mixture + SGD        |
| ASBM      | discrete IMF/SB                       | explicit Markov density projection       | adversarial transition fit       | exact D-IMF KL convergence              | GAN、endpoint bias、grid      |
| IPMF      | combined IPF/IMF fixed point          | 重组 outer operators                       | exact/learned projection steps   | bounded convergence regimes             | finite loop + learned steps |

完整表见
B10 operator matrix。

## 8. 误差账本与选择原则

现代方法的误差至少有七层：

```text
coupling error
 + conditional-sampler/control-solver error
 + population projection mismatch
 + finite function-class error
 + statistical error
 + optimization/outer-iteration error
 + time-discretization error.                                  (8.1)
```

SF2M 把成本放在 coupling 与 analytic conditional sampling；GSBM 放进 CondSOC；
LightSB-M 放进 potential family；ASBM 放进 adversarial projection；IPMF 解释 outer
sequence，但不自动控制 learned inner steps。

选择方法时应先问手中有哪些 exact primitives。analytic bridges 与可靠 coupling
可得时，SF2M 很自然；state cost 属于问题定义时，需要 GSBM 类 generalized control；
能接受受限 potential family 时，LightSB-M 提供一次 projection；transition 只能
隐式采样时，ASBM 提供 adversarial approximation；已有 bidirectional IMF solver 时，
IPMF 给出更精确的 outer-operator 解释。

## 9. 常见错误

- **“simulation-free 就完全不采样。”** 它仍采 endpoint pairs、times 与 analytic
  bridges，推理仍需积分 learned dynamics。
- **“任意 coupling 都得到 SB。”** SF2M 中 arbitrary coupling 定义不同 mixture；
  exact SB 需要 entropic-OT coupling。Optimal SBM 的结论来自 exact projection domain。
- **“objective 下降就是 global convergence。”** GSBM Theorems 5--6 只给 monotonicity
  与 optimum fixed-point inclusion。
- **“D-IMF theorem 证明 GAN 收敛。”** ASBM Theorem 3.6 不覆盖 learned projection。
- **“IPMF 已证明一般分布收敛。”** 当前 theorem 只覆盖指定 Gaussian 与 bounded-support
  regimes，更一般情况仍是 conjecture。

## 10. 小结与思考题

本章统一的不是 loss，而是责任语言：先写 path-law objective 与 exact operator，
再写 estimator，最后才讨论 network 与 benchmark。进入 B11--B14 后仍应沿用这套
顺序。

1. 推导 Gaussian entropic-OT endpoint covariance，并与 arbitrary `c` 下的 (2.6)
   比较；哪一步使用 Brownian reference regularization？
2. 构造 objective 单调下降却收敛到非全局 fixed point 的有限维例子，说明 GSBM
   Theorems 5--6 为什么不能合并成 global theorem。
3. 若 adversarial Markov projection 每轮只有 divergence error `epsilon_n`，exact
   D-IMF 的 KL argument 需要怎样的 stability/summability 条件？
4. 比较 weak convergence、forward KL、reverse KL 与 endpoint Wasserstein error；
   IPMF bounded-support theorem 可直接推出哪些结论？
