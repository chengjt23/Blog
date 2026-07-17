---
title: 连续时间统一：SDE、Reverse SDE 与 Probability-Flow ODE
description: 用连续时间框架统一 VP、VE、reverse-time SDE、probability-flow ODE、likelihood 与路径差异。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 4
slug: d4-continuous-time-sde
tags:
  - diffusion
  - sde
  - probability-flow-ode
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 覆盖连续时间边缘演化、时间反演和确定性流，明确相同 marginal 不等于相同 path law。
---
D3 得到了一个非常强的接口：只要能在每个噪声时刻估计 marginal score

$$
s_t(x)=\nabla_x\log p_t(x),
$$

就能写出朝高密度区域移动的局部方向。但“局部方向”还不是完整的生成过程。NCSN 在一列噪声尺度上运行 Langevin dynamics，DDPM 沿离散 Markov chain 逐步去噪；它们看起来仍是两套算法。

连续时间视角把这两条路线放进同一个对象中：

$$
dX_t=f(X_t,t)dt+G(X_t,t)dW_t.
$$

forward SDE 规定数据怎样变成易采样的噪声，time-reversal theorem 说明怎样借助 score 反演这条随机过程，而 probability-flow ODE 则给出一条确定性 density flow。三者的关系可以先压缩成一句话：

> reverse SDE 试图恢复 forward SDE 的反向 **path law**；probability-flow ODE 只复现它在每个固定时刻的 **one-time marginal**。

这句话里的每个名词都不能省。本章会从 Itô formula 完整推到 Fokker--Planck equation，再从 probability current 推出 reverse drift 和 probability-flow velocity；同时会明确指出，这种 PDE 推导只能建立公式直觉，不能替代严格的 path-space 时间反演定理。

本章仍分三层阅读。第一次阅读可抓住 SDE、score、reverse SDE 与 ODE 的关系；第二次阅读完整跟随 generator、Fokker--Planck、VP/VE/sub-VP marginal 和 likelihood 推导；研究层则需留意 state-dependent diffusion、time-reversal assumptions、endpoint convention、learned-score error 与数值误差。

***

## 1. 先区分三个对象：path、transition 与 marginal

设 $(X_t)_{t\in[0,T]}$ 是一个随机过程。至少有三个不同层次的对象：

1. **sample path**：一次随机试验得到的整条曲线 $t\mapsto X_t(\omega)$；
2. **transition kernel**：从 $X_s=x_s$ 出发到 $X_t$ 的条件分布 $p_{st}(x_t\mid x_s)$；
3. **one-time marginal**：忘掉此前路径后，单独观察 $X_t$ 得到的分布 $p_t$。

它们的强弱关系不是对称的。完整 path law 决定所有有限维 joint distributions，因而决定 transitions 和 marginals；只知道每个 $p_t$，通常不能还原样本如何在时刻之间耦合。

一个最简单的反例是：对所有 $t$，都令 $X_t\sim\mathcal N(0,1)$。这并没有说明 $X_t$ 是同一个随机变量保持不动、相互独立地重采样，还是一个 stationary Ornstein--Uhlenbeck process。三者可有相同 marginal，却有完全不同的 transition 和 path regularity。

本章讨论的 forward Itô SDE 是

$$
\boxed{
dX_t
=
f(X_t,t)dt
+
G(X_t,t)dW_t,
}
$$

其中：

- $X_t\in\mathbb R^d$；
- $f:\mathbb R^d\times[0,T]\to\mathbb R^d$ 是 drift；
- $W_t$ 是 Brownian motion；
- $G$ 是 diffusion coefficient；
- diffusion matrix 定义为

  $$
  \boxed{a(x,t)=G(x,t)G(x,t)^\top.}
  $$

小时间 $\Delta t>0$ 下，Euler--Maruyama 的局部形式是

$$
X_{t+\Delta t}
\approx
X_t+f(X_t,t)\Delta t
+G(X_t,t)\sqrt{\Delta t}\,Z,
\qquad Z\sim\mathcal N(0,I).
$$

因此 drift 控制 conditional mean 的一阶变化，$a$ 控制 conditional covariance 的一阶变化。注意：生成模型文献常写 $G(x,t)=g(t)I$。这是重要特例，但不是一般公式。

### 1.1 连续时间不是把下标换成实数

DDPM 的一步 variance $\beta_k$ 在连续极限中必须随 step size 缩放。若 $t_k=k\Delta t$，则典型对应为

$$
\beta_k\approx\beta(t_k)\Delta t.
$$

此时

$$
\prod_{k:t_k\le t}(1-\beta_k)
\longrightarrow
\exp\left[-\int_0^t\beta(s)ds\right].
$$

如果保持每步 $\beta_k$ 不变却让 $\Delta t\to0$，累计噪声会以错误速度爆炸。所谓“DDPM 的 continuous limit”包含 scaling assumption，不只是把整数 $k$ 改写成实数 $t$。

***

## 2. 从 Itô generator 到 Fokker--Planck equation

SDE 描述随机路径，Fokker--Planck equation 描述这些路径诱导的 density 怎样演化。它是 reverse SDE 与 probability-flow ODE 的共同入口。

### 2.1 Itô formula 与 generator

取二次连续可微 test function $\varphi:\mathbb R^d\to\mathbb R$。Itô formula 给出

$$
d\varphi(X_t)
=
\nabla\varphi(X_t)^\top dX_t
+
\frac12 dX_t^\top\nabla^2\varphi(X_t)dX_t.
$$

利用 Itô multiplication rules

$$
dt^2=0,
\qquad
dt\,dW_t=0,
\qquad
dW_t dW_t^\top=I\,dt,
$$

代入 SDE：

$$
d\varphi(X_t)
=
\left[
f\cdot\nabla\varphi
+
\frac12 a:\nabla^2\varphi
\right]dt
+
\nabla\varphi^\top GdW_t.
$$

这里

$$
a:\nabla^2\varphi
=
\sum_{i,j}a_{ij}\partial_i\partial_j\varphi.
$$

于是 time-dependent generator 为

$$
\boxed{
\mathcal L_t\varphi
=
f\cdot\nabla\varphi
+
\frac12a:\nabla^2\varphi.
}
$$

在 stochastic integral 可积、期望为零时，

$$
\frac d{dt}\mathbb E[\varphi(X_t)]
=
\mathbb E[(\mathcal L_t\varphi)(X_t)].
$$

若 $X_t$ 有 density $p_t$，左边可写成

$$
\int \varphi(x)\partial_t p_t(x)dx.
$$

右边则是

$$
\int
\left[
f_i\partial_i\varphi
+
\frac12a_{ij}\partial_i\partial_j\varphi
\right]p_tdx,
$$

这里重复指标求和。

### 2.2 两次 integration by parts

对 drift term 积分一次：

$$
\int f_i p_t\,\partial_i\varphi\,dx
=
-\int\varphi\,\partial_i(f_ip_t)dx,
$$

对 diffusion term 积分两次：

$$
\int a_{ij}p_t\,\partial_i\partial_j\varphi\,dx
=
\int\varphi\,\partial_i\partial_j(a_{ij}p_t)dx.
$$

忽略边界项需要条件，例如 $\varphi$ compactly supported，或 $p_t,f,a$ 及其相关导数有足够 decay；在 bounded domain 上则需要相应 boundary/flux conditions。由 test-function identity 得到 generator 的 adjoint：

$$
\boxed{
\partial_t p_t(x)
=
-\sum_i\partial_i[f_i(x,t)p_t(x)]
+
\frac12
\sum_{i,j}\partial_i\partial_j[a_{ij}(x,t)p_t(x)].
}
$$

简写为

$$
\boxed{
\partial_t p_t
=
-\nabla\cdot(fp_t)
+
\frac12\nabla^2:(ap_t).
}
$$

这个公式最容易被错误简化。只有在特殊条件下 second-order term 才能写成 $a\Delta p_t/2$。当 $a=a(x,t)$ 随 state 变化时，导数同时作用于 $a$ 和 $p_t$：

$$
\nabla^2:(ap_t)
=
\sum_{i,j}\partial_i\partial_j(a_{ij}p_t),
$$

不能把 $a$ 从导数中直接提出。

### 2.3 弱推导告诉了我们什么

上面的推导从 test function 出发，首先建立的是 weak PDE identity。要升级为经典 pointwise PDE，还要增加 density smoothness、coefficient regularity 与可积性。这里已经出现本章反复使用的分层：

- **代数层**：形式展开是否正确；
- **PDE 层**：density 是否以 weak/classical sense 解方程；
- **path-law 层**：是否存在唯一的 stochastic process 实现这些 transitions。

Fokker--Planck equation控制第二层，不会自动解决第三层。

***

## 3. Probability current：把 density 演化写成守恒律

定义 matrix field 的 divergence

$$
[\nabla\cdot(ap_t)]_i
=
\sum_j\partial_j(a_{ij}p_t).
$$

再定义 probability current

$$
\boxed{
J_t
=
fp_t
-
\frac12\nabla\cdot(ap_t).
}
$$

Fokker--Planck equation 就变成 continuity equation：

$$
\boxed{
\partial_t p_t=-\nabla\cdot J_t.
}
$$

直觉上，$p_t$ 是“概率质量密度”，$J_t$ 是单位时间穿过单位面积的净概率流。drift 产生 advective current $fp_t$，Brownian diffusion 产生第二项。

利用 product rule，

$$
\frac{\nabla\cdot(ap_t)}{p_t}
=
\nabla\cdot a
+
a\nabla\log p_t,
$$

其中

$$
[\nabla\cdot a]_i=\sum_j\partial_j a_{ij}.
$$

因此在 $p_t>0$ 处，current per unit density 是

$$
\frac{J_t}{p_t}
=
f
-
\frac12\nabla\cdot a
-
\frac12a\nabla\log p_t.
$$

这个速度稍后正是 probability-flow ODE。它也显示 score 不是凭空加入 sampler 的经验项：score 把 diffusion 导致的 density-dependent flux 写成局部向量场。

***

## 4. 时间反演：为什么 reverse drift 中出现 score

现在定义新时间

$$
\tau=T-t,
\qquad
Y_\tau=X_{T-\tau}.
$$

若 $q_\tau$ 是 $Y_\tau$ 的 marginal，则

$$
q_\tau(x)=p_{T-\tau}(x),
$$

从而

$$
\partial_\tau q_\tau
=
-\partial_t p_t\big|_{t=T-\tau}
=
\nabla\cdot J_{T-\tau}.
$$

假设 reversed process 仍有 diffusion matrix $a(x,T-\tau)$，drift 记为 $\bar f(x,\tau)$。它自己的 current 是

$$
\bar J_\tau
=
\bar f q_\tau
-
\frac12\nabla\cdot(aq_\tau),
$$

且满足

$$
\partial_\tau q_\tau=-\nabla\cdot\bar J_\tau.
$$

要让概率流沿原路反向，取

$$
\bar J_\tau=-J_{T-\tau}.
$$

于是

$$
\bar f p
-
\frac12\nabla\cdot(ap)
=
-fp
+
\frac12\nabla\cdot(ap),
$$

得到

$$
\boxed{
\bar f(x,\tau)
=
-f(x,T-\tau)
+
\frac{\nabla\cdot[a(x,T-\tau)p_{T-\tau}(x)]}{p_{T-\tau}(x)}.
}
$$

展开 product rule：

$$
\boxed{
\bar f
=
-f
+
\nabla\cdot a
+
a\nabla\log p.
}
$$

这就是 **forward-in-reversed-time** convention：$\tau$ 从 $0$ 增长到 $T$，$Y_0=X_T$，$Y_T=X_0$。

### 4.1 原时间标签、负 $dt$ 的写法

diffusion-model 论文更常保留原标签 $t$，直接从 $T$ 积分到 $0$。因为 $d\tau=-dt$，drift 符号相反：

$$
\boxed{
dX_t
=
\left[
f
-
\nabla\cdot a
-
a\nabla\log p_t
\right]dt
+
Gd\bar W_t,
\qquad dt<0.
}
$$

两套写法描述同一反演过程，但不能把第一套的 drift 和第二套的步进方向拼在一起。离散实现中，若 $\Delta t<0$，noise scale 是

$$
\sqrt{|\Delta t|},
$$

不是没有实数意义的 $\sqrt{\Delta t}$。

![两种等价的 reverse-time convention：新时钟使用正 d tau，原标签使用负 dt](/images/diffusion/d4_time_conventions.png)

### 4.2 Song 公式为何少了一项

若

$$
G(x,t)=g(t)I,
\qquad
a(x,t)=g(t)^2I,
$$

则 $a$ 不依赖 $x$，所以 $\nabla\cdot a=0$。negative-$dt$ 写法简化为

$$
\boxed{
dX_t
=
\left[
f(X_t,t)
-
g(t)^2\nabla_x\log p_t(X_t)
\right]dt
+
g(t)d\bar W_t,
\qquad dt<0.
}
$$

这就是 Score SDE 主文中的公式。它没有错，但它是 **scalar、state-independent diffusion** 的特例。把它原样用于一般 $G(x,t)$ 会漏掉 $\nabla\cdot a$。

### 4.3 这里还不是严格的 path-law proof

上述 current matching 是极有用的推导，但它只展示了一个 candidate reverse drift。原因有两层：

第一，Fokker--Planck 只约束 current 的 divergence。若加入满足 $\nabla\cdot K=0$ 的场，marginal PDE 不变；仅由 one-time densities 不能识别 canonical reversed transitions。

第二，即使写出了正确系数，还需要证明 SDE/martingale problem 的存在性、reversed filtration 下的 semimartingale structure，以及所得 process 的 finite-dimensional distributions 确实等于原 process 的反向 joint law。

严格定理需要具体 assumptions。一个现代边界是 Cattiaux、Conforti、Gentil 与 Léonard 在 finite path-space relative entropy 条件下给出的 time-reversal theorem：在其 reference diffusion、matrix regularity、growth、invertibility 与 finite-entropy assumptions 下，reversed drift 以上述形式在相应 almost-everywhere/finite-energy 意义成立。经典 smooth elliptic setting 也可得到相同公式，但所需条件不同。

因此，本章使用以下措辞边界：

- Fokker--Planck/current matching **推导公式并建立直觉**；
- time-reversal theorem **保证反向 path law 的合法性**；
- 本章不重证 measure-theoretic theorem；
- 对 $t\to0$ score singularity、degenerate diffusion 或 learned non-score field，不能无条件套用。

***

## 5. 三类前向 SDE：VP、VE 与 sub-VP

Score SDE 的统一不意味着所有 forward processes 相同。VP、VE 与 sub-VP 选择不同 drift/diffusion，因而有不同 perturbation geometry、terminal approximation 和 likelihood/sampling behavior。

记

$$
B(t)=\int_0^t\beta(s)ds.
$$

### 5.1 VP SDE：DDPM 的连续版本

Variance-Preserving SDE 为

$$
\boxed{
dX_t
=
-\frac12\beta(t)X_tdt
+
\sqrt{\beta(t)}dW_t.
}
$$

它是 time-varying Ornstein--Uhlenbeck process。为完整推导 transition，使用 integrating factor

$$
M(t)=e^{B(t)/2}.
$$

因为 $M$ 是确定性函数，product rule 给出

$$
d[M(t)X_t]
=
M(t)\sqrt{\beta(t)}dW_t.
$$

从 $0$ 积分到 $t$：

$$
X_t
=
e^{-B(t)/2}X_0
+
\int_0^t
e^{-[B(t)-B(s)]/2}
\sqrt{\beta(s)}dW_s.
$$

条件在 $X_0=x_0$ 上，第二项是零均值 Gaussian。由 Itô isometry，covariance coefficient 为

$$
\begin{aligned}
\int_0^t
e^{-[B(t)-B(s)]}\beta(s)ds
&=
e^{-B(t)}
\int_0^t e^{B(s)}\beta(s)ds\\
&=
e^{-B(t)}[e^{B(t)}-1]\\
&=
1-e^{-B(t)}.
\end{aligned}
$$

因此

$$
\boxed{
X_t\mid X_0=x_0
\sim
\mathcal N\left(
e^{-B(t)/2}x_0,
[1-e^{-B(t)}]I
\right).
}
$$

与 D1 的离散记号逐项对应：

$$
\boxed{
\bar\alpha(t)
=
e^{-B(t)},
\qquad
a(t)=e^{-B(t)/2},
\qquad
b(t)=\sqrt{1-e^{-B(t)}}.
}
$$

离散乘积到连续指数的对应是

$$
\log\bar\alpha_k
=
\sum_{j=1}^k\log(1-\beta_j)
\approx
-\sum_{j=1}^k\beta(t_j)\Delta t
\to
-\int_0^t\beta(s)ds.
$$

VP 的 “variance preserving” 更准确地说是：若 $\operatorname{Cov}(X_0)=I$，则 $\operatorname{Cov}(X_t)=I$；若进一步有 $X_0\sim\mathcal N(0,I)$，所有 marginals 都仍是 $\mathcal N(0,I)$。它不是说对任意初始 scale，variance 都会保持原值。

### 5.2 VE SDE：NCSN 的连续版本

Variance-Exploding SDE 没有 drift：

$$
\boxed{
dX_t
=
\sqrt{\frac{d[\sigma(t)^2]}{dt}}dW_t.
}
$$

直接积分：

$$
X_t
=
X_0
+
\int_0^t
\sqrt{\frac{d[\sigma(s)^2]}{ds}}dW_s.
$$

所以

$$
\operatorname{Var}(X_t\mid X_0)
=
\int_0^t\frac{d[\sigma(s)^2]}{ds}ds
=
\boxed{\sigma(t)^2-\sigma(0)^2}.
$$

这里有一个容易被代码隐藏的 endpoint convention。若理论上取 $\sigma(0)=0$，conditional standard deviation 就是 $\sigma(t)$。Score SDE 官方代码使用 finite $\sigma_{\min}>0$，同时 `marginal_prob` 返回 $\sigma(t)$。这相当于把起点理解为已经带有 $\sigma_{\min}$ noise 的 endpoint；若坚持从 raw data 在 $t=0$ 严格积分，则新增 variance 应为

$$
\sigma(t)^2-\sigma_{\min}^2.
$$

两种 convention 都能使用，但 training perturbation、初始 distribution 与 sampler endpoint 必须成套对齐。本章代码同时实现 raw-data variance 和官方 finite-endpoint 报告方式，以便显式检查两者相差 $\sigma_{\min}^2$。

### 5.3 sub-VP SDE：相同 mean，更小 conditional variance

sub-VP SDE 定义为

$$
\boxed{
dX_t
=
-\frac12\beta(t)X_tdt
+
\sqrt{
\beta(t)[1-e^{-2B(t)}]
}dW_t.
}
$$

drift 与 VP 相同，所以 conditional mean 仍是

$$
\mathbb E[X_t\mid X_0=x_0]
=
e^{-B(t)/2}x_0.
$$

令 conditional covariance coefficient 为 $P(t)$。linear SDE covariance ODE 给出

$$
P'(t)
=
-\beta(t)P(t)
+
\beta(t)[1-e^{-2B(t)}],
\qquad
P(0)=0.
$$

乘 integrating factor $e^{B(t)}$：

$$
\begin{aligned}
P(t)
&=
e^{-B(t)}
\int_0^t
e^{B(s)}\beta(s)[1-e^{-2B(s)}]ds\\
&=
e^{-B(t)}
\int_0^t
\beta(s)[e^{B(s)}-e^{-B(s)}]ds\\
&=
e^{-B(t)}
\left[e^{B(t)}+e^{-B(t)}-2\right]\\
&=
[1-e^{-B(t)}]^2.
\end{aligned}
$$

因此 conditional **standard deviation** 是

$$
\boxed{
\operatorname{Std}(X_t\mid X_0)
=
1-e^{-B(t)}.
}
$$

不要把这行误读成 variance。因为 $0\le 1-e^{-B}\le1$，它小于 VP 的 $\sqrt{1-e^{-B}}$，这也是 “sub-variance preserving” 名称的来源。

![VP、VE 与 sub-VP 的 conditional mean、noise scale 和 SNR；VE 曲线采用 raw-data endpoint](/images/diffusion/d4_sde_families.png)

图中 VP 与 sub-VP 的 mean 曲线完全重合，但小 $t$ 时 noise scale 不同；VE 保留 signal coefficient 1，同时 noise scale 增长。所谓 “VP/VE 路线统一” 是它们都进入同一 SDE/reversal machinery，不是说它们的 marginals 或 numerical conditioning 相同。

***

## 6. 连续时间训练接口：D3 的 score 怎样接入 SDE

对具有 tractable transition $p_{0t}(x_t\mid x_0)$ 的 forward SDE，continuous denoising score matching 写成

$$
\boxed{
\mathcal L(\theta)
=
\mathbb E_{t\sim\rho}
\lambda(t)
\mathbb E_{X_0}
\mathbb E_{X_t\mid X_0}
\left[
\left\|
s_\theta(X_t,t)
-
\nabla_{x_t}\log p_{0t}(X_t\mid X_0)
\right\|^2
\right].
}
$$

若 transition 是 affine Gaussian

$$
X_t=m(t)X_0+s(t)\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

则 conditional target 可解析计算：

$$
\boxed{
\nabla_{x_t}\log p_{0t}(x_t\mid x_0)
=
-\frac{x_t-m(t)x_0}{s(t)^2}
=
-\frac{\epsilon}{s(t)}.
}
$$

D3 已证明，在 population optimum 下，这个随机 conditional target 的 conditional expectation 是 marginal score：

$$
\mathbb E\left[
\nabla_{x_t}\log p_{0t}(X_t\mid X_0)
\middle|X_t=x
\right]
=
\nabla_x\log p_t(x).
$$

于是：

- VP transition 连接 DDPM 的 $\epsilon$-prediction；
- VE transition 连接 NCSN 的 multi-noise DSM；
- reverse SDE 与 PF ODE 都消费同一个 time-dependent marginal score interface。

这里不应提前把所有 $\lambda(t)$ 混成一个“理论最优权重”。sampling distribution $\rho(t)$、loss weight $\lambda(t)$、network parameterization 和 numerical time grid 是不同设计维度，D5 会系统处理。D4 只固定 population target。

***

## 7. Probability-flow ODE：同一 density evolution 的确定性实现

考虑确定性 ODE

$$
\frac{dX_t}{dt}=v(X_t,t).
$$

在 flow 足够规则时，其 density 满足 continuity equation

$$
\partial_t p_t=-\nabla\cdot(vp_t).
$$

若令 ODE current 与 SDE current 完全相同，

$$
v_{\mathrm{PF}}p_t=J_t,
$$

则二者满足相同的 marginal PDE。由第 3 节公式：

$$
\boxed{
v_{\mathrm{PF}}(x,t)
=
f(x,t)
-
\frac12\nabla\cdot a(x,t)
-
\frac12a(x,t)\nabla_x\log p_t(x).
}
$$

在 scalar、state-independent case 中：

$$
\boxed{
\frac{dX_t}{dt}
=
f(X_t,t)
-
\frac12g(t)^2\nabla_x\log p_t(X_t).
}
$$

reverse SDE 的 score coefficient 是 $1$，PF ODE 是 $1/2$。这不是随意减半：reverse SDE 自己仍有 Brownian diffusion，需要完整 score correction 才能反转 probability current；ODE 没有 diffusion，全部 current 都由 deterministic velocity 承担，所以得到半系数。

### 7.1 为什么它们只共享 one-time marginals

若 initial law 相同、PDE solution 唯一且 score exact，则对每个固定 $t$，

$$
\boxed{
\mathcal L(X_t^{\mathrm{SDE}})
=
\mathcal L(X_t^{\mathrm{PF}})
=
p_t.
}
$$

但以下对象一般不同：

- transition kernel $p_{st}(x_t\mid x_s)$；
- 多时刻 joint distribution；
- 给定起点后的 conditional randomness；
- sample-wise trajectory；
- quadratic variation。

如果 $a$ 非零，SDE path 的 quadratic variation 满足

$$
[X]_t=\int_0^t a(X_s,s)ds
$$

（按坐标/矩阵意义理解），而足够光滑的 ODE path 有

$$
[X]_t=0.
$$

所以即使每个时刻的直方图完全一致，两种 path law 也不可能相同。

![同一初始 mixture 在 VP SDE 与 probability-flow ODE 下的路径和固定时刻 marginals](/images/diffusion/d4_sde_pf_paths.png)

图中两组 trajectories 从同一批初始点出发。左侧路径持续接受 Brownian noise，右侧每个起点沿确定性 ODE 移动；下方固定时刻的经验 marginal 近似重合。图只作数值说明，有限 Euler step 和有限样本不会产生 theorem 级的 exact equality。

### 7.2 PF ODE 也不是唯一的 same-marginal flow

continuity equation只要求

$$
\nabla\cdot(vp_t)=-\partial_t p_t.
$$

若 vector field $u$ 满足 $\nabla\cdot(up_t)=0$，则 $v+u$ 给出相同 marginal evolution。probability-flow ODE 是通过 **匹配 SDE current** 选出的 canonical construction，不是仅由 $(p_t)_{t\in[0,T]}$ 唯一决定的所有可能 deterministic coupling。

更进一步，用 learned $s_\theta$ 替代 exact score 后，learned reverse SDE 和 learned PF ODE 各自诱导新的 PDE；一般不能再声称二者的 endpoint model distributions 完全相同。

***

## 8. 一个 state-dependent 例子：漏掉 divergence 会发生什么

一维中取

$$
a(x)=1+x^2,
\qquad
p(x)=\mathcal N(0,1),
\qquad
s(x)=\partial_x\log p(x)=-x,
$$

并令

$$
f(x)=\frac12(x-x^3).
$$

先计算

$$
\partial_x[a(x)p(x)]
=
2xp(x)+(1+x^2)(-x)p(x)
=
(x-x^3)p(x).
$$

因此 current 为

$$
J(x)
=
f(x)p(x)
-
\frac12\partial_x[a(x)p(x)]
=0.
$$

这是一个 nontrivial state-dependent diffusion，但 $p$ 是 stationary 且 process reversible。一般 probability-flow velocity 为

$$
\begin{aligned}
v_{\mathrm{PF}}
&=
f-\frac12\partial_xa-\frac12as\\
&=
\frac12(x-x^3)-x+\frac12x(1+x^2)\\
&=0.
\end{aligned}
$$

也就是说，SDE sample paths 随机运动，stationary PF ODE path 原地不动；marginal 均保持 $\mathcal N(0,1)$。这是 “same marginal, different path” 的解析例子。

再看 reverse drift 的 negative-$dt$ 形式：

$$
f-\partial_xa-as
=
\frac12(x-x^3)-2x+(1+x^2)x
=
-f.
$$

由于 $dt<0$，换到正的 reversed time 后 drift 又变回 $f$，符合 reversibility。

若错误省略 $\partial_xa=2x$，PF velocity 会变成 $x$，不再为零；reverse drift 也会错误。这个例子说明 divergence term 不是高阶装饰，而是 state-dependent noise 引起的系统性 probability flux correction。

***

## 9. 从公式到算法：三种最小接口

本章不比较高阶 solver，但需要把公式准确映射成可执行接口。以下伪代码保留最关键的时间方向、系数和随机性。

### 9.1 Reverse SDE：原时间标签版本

设 grid

$$
T=t_N>t_{N-1}>\cdots>t_0=\varepsilon,
$$

其中 $\varepsilon\ge0$；实践中常取小正数以避开 endpoint singularity。先采样

$$
x_N\sim\pi,
$$

再对 $k=N,N-1,\ldots,1$ 执行：

```text
dt = t[k-1] - t[k]                         # dt < 0
a  = G(x, t[k]) @ G(x, t[k]).T
b  = f(x, t[k]) - div(a) - a @ score(x, t[k])
z  ~ Normal(0, I)
x  = x + b * dt + G(x, t[k]) @ z * sqrt(abs(dt))
```

在常见 $G=g(t)I$ 情况下，`div(a)=0`，且 `a @ score = g(t)^2 * score`。Euler--Maruyama 只是最低阶离散；predictor--corrector、ancestral discretization 和更高阶 SDE solver 的差异留到 D6。

这里有两个独立 approximation：

- $x_N\sim\pi$ 是否真的等于 forward terminal $p_T$；
- `score` 是否真的等于 $\nabla\log p_t$。

即使 time step 趋于零，这两项也不会自动消失。

### 9.2 Probability-flow ODE：同一网络，另一种 dynamics

对 scalar state-independent diffusion，定义

$$
v_\theta(x,t)
=
f(x,t)-\frac12g(t)^2s_\theta(x,t).
$$

生成时同样从 $x_T\sim\pi$ 出发，但确定性地从 $T$ 积到 $\varepsilon$：

```text
dx/dt = v_theta(x, t)
x_T ~ prior
x_epsilon = ODESolve(v_theta, x_T, T -> epsilon)
```

给定相同 $x_T$、solver 和 tolerance，输出是确定的。它允许 adaptive ODE solver 和 exact change-of-variables likelihood，但“deterministic”不等于“数值上容易”：靠近低噪声端时 score curvature 可能很大，ODE 可以 stiff，函数调用次数也可能上升。

### 9.3 新时间 $\tau$ 的 reverse SDE

如果实现希望所有 step size 为正，可定义 $Y_\tau=X_{T-\tau}$，使用

$$
dY_\tau
=
\left[
-f
+
\nabla\cdot a
+
a s_{T-\tau}
\right]d\tau
+
Gd\widetilde W_\tau,
\qquad d\tau>0.
$$

这与上一算法完全等价。工程上应在 API 名称中固定 convention，例如 `reverse_drift_original_time` 与 `reverse_drift_new_time`，而不是依赖调用者猜测符号。

***

## 10. Probability-flow ODE 为什么能计算 likelihood

PF ODE 不只提供 deterministic sampler，还把 score model 连接到 continuous normalizing flow。

### 10.1 Instantaneous change of variables

设

$$
\dot x_t=v(x_t,t).
$$

density continuity equation 是

$$
\partial_t p_t
=
-\nabla\cdot(vp_t)
=
-p_t\nabla\cdot v
-v\cdot\nabla p_t.
$$

沿 ODE trajectory 使用 chain rule：

$$
\begin{aligned}
\frac d{dt}\log p_t(x_t)
&=
\partial_t\log p_t(x_t)
+
\nabla\log p_t(x_t)^\top\dot x_t\\
&=
-\nabla\cdot v
-v\cdot\nabla\log p_t
+v\cdot\nabla\log p_t\\
&=
\boxed{-\nabla\cdot v(x_t,t)}.
\end{aligned}
$$

从 data endpoint $0$ 正向积分到 prior endpoint $T$：

$$
\log p_T(x_T)-\log p_0(x_0)
=
-\int_0^T\nabla\cdot v(x_t,t)dt.
$$

所以

$$
\boxed{
\log p_0(x_0)
=
\log p_T(x_T)
+
\int_0^T\nabla\cdot v(x_t,t)dt.
}
$$

若 terminal model density $p_T$ 取 tractable prior $\pi$，就可以通过一个 augmented ODE 同时积分

$$
\frac{dx_t}{dt}=v_\theta(x_t,t),
\qquad
\frac{dA_t}{dt}=\nabla\cdot v_\theta(x_t,t),
$$

最后输出

$$
\log p_\theta^{\mathrm{ODE}}(x_0)
=
\log\pi(x_T)+A_T.
$$

也可以令 accumulator 满足 $d\ell_t/dt=-\nabla\cdot v$，再按相反符号组合；实现时应先固定“accumulate divergence”还是“accumulate log-density change”。

### 10.2 高维瓶颈：Jacobian trace

divergence 是 Jacobian trace：

$$
\nabla\cdot v
=
\operatorname{Tr}\left(\frac{\partial v}{\partial x}\right).
$$

对 $d$-维网络显式构造 $d\times d$ Jacobian 代价很高。Hutchinson identity 使用满足

$$
\mathbb E[\epsilon]=0,
\qquad
\mathbb E[\epsilon\epsilon^\top]=I
$$

的 random vector，例如 Gaussian 或 Rademacher noise。对任意 square matrix $A$：

$$
\begin{aligned}
\mathbb E[\epsilon^\top A\epsilon]
&=
\mathbb E[\operatorname{Tr}(\epsilon^\top A\epsilon)]\\
&=
\mathbb E[\operatorname{Tr}(A\epsilon\epsilon^\top)]\\
&=
\operatorname{Tr}(A).
\end{aligned}
$$

因此

$$
\boxed{
\nabla\cdot v
=
\mathbb E_\epsilon
\left[
\epsilon^\top
\frac{\partial v}{\partial x}
\epsilon
\right].
}
$$

vector--Jacobian product 可由 reverse-mode autodiff 计算，无需 materialize full Jacobian。

### 10.3 为什么一次 ODE solve 内要固定 $\epsilon$

FFJORD 的实现思想是在一次 solve 中采样一次 $\epsilon$，随后每次 vector-field evaluation 都复用它：

$$
\int_0^T
\epsilon^\top
\frac{\partial v(x_t,t)}{\partial x}
\epsilon\,dt.
$$

如果 adaptive solver 每次调用右端函数时都重新采样，右端不再是固定的 deterministic function；solver 的 local-error model 会被额外 Monte Carlo noise 打乱。固定 noise 不会消除 estimator variance，但会使一次 solve 的 augmented dynamics 自洽。

### 10.4 “可计算 likelihood”不等于“训练就是 MLE”

必须区分四个命题：

1. 给定 regular ODE vector field，可以使用 instantaneous change of variables 定义并计算其 model density；
2. 用 learned score 构造 PF vector field，可以得到 $p_\theta^{\mathrm{ODE}}$；
3. ordinary weighted DSM 可高效估计 score risk；
4. training objective 是否直接最小化 $-\log p_\theta^{\mathrm{ODE}}(x)$。

前两项不推出第四项。Song、Durkan、Murray 与 Ermon 的 maximum-likelihood 工作表明，在相应 regularity assumptions 下选择

$$
\lambda(t)=g(t)^2
$$

可给出 approximate reverse-SDE model 的 KL/NLL upper-bound route：

$$
D_{\mathrm{KL}}(p_0\|p_\theta^{\mathrm{SDE}})
\le
J_{\mathrm{SM}}(\theta;g^2)
+
D_{\mathrm{KL}}(p_T\|\pi).
$$

但这不是“每个训练 batch 都通过 ODE solve 精确最大化 $p_\theta^{\mathrm{ODE}}$”。当 score approximate 时，$p_\theta^{\mathrm{SDE}}$ 与 $p_\theta^{\mathrm{ODE}}$ 也不是未经证明就相同的 model distribution。likelihood weighting 的详细 variance 与 parameterization 问题留给 D5，path-space bound 留给 D12。

***

## 11. 说明代码与官方实现怎样对应

配套代码 d4\_continuous\_time\_sde.py（补充材料暂未公开） 不训练网络，而使用解析 Gaussian mixture 和 linear fields 检查本章接口：

- `VPSDE`、`VESDE`、`SubVPSDE`：drift、diffusion 与 closed-form marginal；
- `GaussianMixture1D.vp_score`：VP 下 exact time-dependent marginal score；
- `reverse_drift_original_time` 与 `reverse_drift_new_time`：两套时间 convention；
- `probability_flow_drift`：scalar PF ODE；
- `state_dependent_stationary_terms`：第 8 节 divergence 例子；
- `exact_divergence` 与 `hutchinson_divergence`：trace 的 exact/stochastic 版本；
- `integrate_flow_and_log_density`：固定 trace noise 的 augmented flow；
- `check_identities`：VP/sub-VP/VE endpoint、时间符号、negative-$dt$ noise 和 likelihood 检查。

运行：

```bash
# 本地验证脚本暂未公开
```

当前脚本的代表性数值结果为：

```text
VP marginal max error: 1.110e-16
sub-VP variance max error: 0.000e+00
VE finite-endpoint offset error: 3.174e-14
state-dependent PF velocity max error: 7.105e-15
state-dependent reverse-drift max error: 1.421e-14
reverse-time convention max error: 0.000e+00
negative-dt noise-scaling max error: 0.000e+00
Hutchinson trace max error: 0.000e+00
likelihood accumulator max error: 1.721e-15
Euler flow endpoint max error: 2.013e-04
```

最后一项刻意保留有限步 Euler error；它说明“解析 identity 正确”和“numerical trajectory exact”是两件事。

本地保存的 Score SDE 官方实现固定在 commit `0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c`，Apache-2.0 license。关键映射为：

| 官方文件                    | 本章对象                                             |
| ----------------------- | ------------------------------------------------ |
| sde\_lib.py（补充材料暂未公开）   | reverse SDE/PF 半系数、VP/VE/sub-VP                  |
| sampling.py（补充材料暂未公开）   | Euler--Maruyama、predictor--corrector、ODE sampler |
| likelihood.py（补充材料暂未公开） | Hutchinson divergence 与 augmented ODE            |
| losses.py（补充材料暂未公开）     | continuous DSM 与 likelihood weighting            |

教程代码重建数学接口，不复制完整研究代码，也不承担 large-image sampling 或 solver benchmark。

***

## 12. 从 exact equations 到实际模型：误差必须分层

“reverse SDE/PF ODE 在 exact score 下成立”常被压缩成“训练一个 score network 就能精确反演”。中间至少有以下误差层。

| 层次                   | 问题                                                        | 即使 step size 趋零是否仍存在 |
| -------------------- | --------------------------------------------------------- | -------------------- |
| data transform       | dequantization、normalization、latent encoder 改变了什么 density | 是                    |
| terminal mismatch    | forward $p_T$ 与 chosen prior $\pi$ 不同                     | 是                    |
| time truncation      | 实际只积分到 $\varepsilon>0$，未到 $0$                             | 是，除非控制极限             |
| score estimation     | $s_\theta\ne\nabla\log p_t$                               | 是                    |
| model identity       | learned SDE 与 learned ODE induced distributions 是否相同      | 一般不同                 |
| path discretization  | Euler/solver 对 reverse dynamics 的误差                       | 否，需收敛条件              |
| ODE tolerance        | adaptive solver 的 local/global error                      | 可减小但有成本              |
| trace estimation     | Hutchinson finite-sample variance                         | 是，除非增加样本/结构          |
| likelihood reporting | bits/dim、Jacobian、dequantization bound 是否一致               | 是                    |

### 12.1 Exact score equality不能机械延伸到 learned score

forward SDE 的 PF ODE 公式使用真实 $p_t$ 的 score。替换成 $s_\theta$ 后，SDE 和 ODE 分别解各自的 evolution equation。二者仍共享 network 参数，不等于共享 exact marginal family。

甚至当 $s_\theta$ 的 average $L^2(p_t)$ error 很小，低密度区域的 pointwise error、Lipschitz behavior 和沿 sampler 自身分布访问到的 error 仍可能重要。从 score risk 到 endpoint distribution distance，需要额外的 moment、regularity、initialization、early-stopping 与 discretization assumptions。这部分是 D12 的主题。

### 12.2 Terminal Gaussian通常是近似，不是定义真理

VP 在 $B(T)$ 足够大时有

$$
X_T
=
e^{-B(T)/2}X_0
+
\sqrt{1-e^{-B(T)}}\epsilon,
$$

因而接近 $\mathcal N(0,I)$，但有限 $B(T)$ 下仍保留 data signal。VE 在大 $\sigma(T)$ 下是 data 与大 Gaussian noise 的 convolution，也不自动等于一个事先选择的 centered Gaussian。初始化误差必须作为独立项，而不是藏在“足够 noisy”里。

### 12.3 Deterministic sampler 不是 automatic exact transport

PF ODE 在 exact field、exact initial law 与 exact integration 下复现 marginals。现实中 learned field 可能 non-conservative，terminal law 可能 mismatch，solver 也有 tolerance。因此“一一映射”“可逆 ODE”“同 marginal theorem”不能替代 end-to-end error analysis。

***

## 13. 技术演进：问题怎样推动方案变化

本章的历史不是按模型名罗列，而是一条连续的问题链。

| 时间          | 当时的问题                                                       | 解决方案                                                       | 新暴露的边界                                          |
| ----------- | ----------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| 1982        | forward diffusion 能否写成 reverse-time diffusion               | Anderson 的 reverse-time diffusion equation models 成为经典入口   | 一般条件与符号需要严格区分                                   |
| 2000s--2020 | score matching、NCSN 与 DDPM 各有训练/采样语言                        | noisy score、annealed Langevin 与 variational chain 分别发展     | 两条路线仍显得分裂                                       |
| 2021        | 如何统一 SMLD/NCSN、DDPM 与多种 sampler                             | Score SDE：VE/VP/sub-VP、reverse SDE、PF ODE、PC sampler       | exact theorem 与 learned/numerical error需分层      |
| 2019--2021  | ODE density 的 Jacobian trace 太昂贵；score loss与 likelihood关系不清 | FFJORD/Hutchinson trace；likelihood-weighted score matching | stochastic trace、ODE/SDE model identity 与 MLE边界 |
| 2023--2025  | 教程公式常省略 assumptions，general matrix 情况易写错                    | finite-entropy time reversal 与现代 technical tutorial        | convergence、degeneracy、low-regularity仍是研究问题     |

Anderson 1982 在本项目中只承担经过 DOI/卷期核验的历史职责，因为当前没有合法开放全文；本章的严格公式边界由 Cattiaux et al. 2023 的公开作者版本和 Tang--Zhao 2025 的正式 technical tutorial 交叉承担。

Score SDE 的关键贡献也不应被缩成“提出了一个新 sampler”。它把几个此前分离的接口统一起来：

$$
\text{forward perturbation}
\longrightarrow
\text{continuous DSM}
\longrightarrow
\text{reverse SDE}
\longleftrightarrow
\text{PF ODE}
\longrightarrow
\text{sampling / likelihood}.
$$

统一之后，新问题从“公式是否存在”转向“选哪条路径、怎样离散、怎样训练、误差怎样传播”。这正是 D5 和 D6 的起点。

***

## 14. 常见错误与边界检查

### 错误 1：把 $p_t$ 当成 $p_{0t}(x_t\mid x_0)$

reverse drift 使用 aggregate marginal score $\nabla\log p_t(x)$，不是单个 clean sample 对应的 conditional score。DSM 只是在 population regression 中把后者变成前者。

### 错误 2：一般 diffusion matrix 下仍写 $f-g^2s$

一般 negative-$dt$ reverse drift 是

$$
f-\nabla\cdot a-a s_t.
$$

只有 $a=g(t)^2I$ 与 state 无关时 divergence 才消失。

### 错误 3：只翻转 drift $f\mapsto-f$

Brownian diffusion 改变 density flux；time reversal 需要由 score 与 $\nabla\cdot a$ 修正。简单负号只有特殊 reversible coordinate interpretation 下才成立。

### 错误 4：混用 $d\tau>0$ 和 $dt<0$

new-time drift 是 $-f+\nabla\cdot a+as$，original-label drift 是 $f-\nabla\cdot a-as$。二者差一个整体负号，因为 $d\tau=-dt$。

### 错误 5：对负 $dt$ 写 $\sqrt{dt}$

反向 Euler noise scale 是 $\sqrt{|dt|}$。Brownian increment 的 covariance 与正的 elapsed time 对应。

### 错误 6：Fokker--Planck matching 已经证明 path-law reversal

它只证明 candidate coefficients 产生正确 marginal PDE；martingale problem、reversed filtration、existence/uniqueness 与 joint-law identity 还需要严格 theorem。

### 错误 7：SDE 与 PF ODE 是同一个 stochastic process

它们在 exact conditions 下共享 fixed-time marginals，不共享 transitions、joint path law 或 quadratic variation。

### 错误 8：PF ODE 是唯一能产生这些 marginals 的 flow

加入满足 $\nabla\cdot(up_t)=0$ 的 current-free component 仍可保持 density PDE。PF ODE 是 current-matching construction。

### 错误 9：sub-VP 的 $1-e^{-B(t)}$ 是 variance

它是 conditional standard deviation；variance 是 $[1-e^{-B(t)}]^2$。

### 错误 10：VE 代码里的 $\sigma(t)$ 与 raw-data SDE 自动完全一致

finite $\sigma_{\min}$ 时，两种 endpoint convention 相差 $\sigma_{\min}^2$ variance。必须检查 initial perturbation definition。

### 错误 11：PF ODE 可算 likelihood，所以 ordinary DSM 就是 exact MLE

likelihood evaluation、SDE-model likelihood bound、ODE-model likelihood 与直接 likelihood optimization 是不同命题。

### 错误 12：Hutchinson unbiased 就意味着一次估计 exact

unbiased 是对 trace-noise 取期望的结论；单次 estimate 有 variance，且还叠加 ODE solver、truncation 和 data-transform error。

### 错误 13：确定性 ODE 一定比随机 SDE 更快

速度取决于 stiffness、tolerance、network evaluation、time grid 与目标精度。determinism 与 NFE 没有单调关系。

### 错误 14：学到近似 score 后，SDE/ODE equality 自动保留

exact-score proof 用的是 forward marginal $p_t$。arbitrary learned field 改变两种 dynamics 的 evolution equations，endpoint equality 需要另证。

***

## 15. 本章给后续章节准备了什么

D4 已把 D1--D3 的离散对象放进同一连续时间语言：

$$
\bar\alpha(t)=e^{-\int_0^t\beta(s)ds},
\qquad
s_t=\nabla\log p_t,
$$

$$
\text{forward SDE}
\xrightarrow{\text{time reversal}}
\text{reverse SDE},
\qquad
\text{forward SDE}
\xrightarrow{\text{current matching}}
\text{PF ODE}.
$$

接下来：

- D5 会区分 $\epsilon,x_0,v,$ score parameterizations，解释 loss weighting、log-SNR 与 preconditioning；
- [D6](/blog/diffusion/d6-sampling-solvers/) 会把 Euler--Maruyama、ancestral sampling、DDIM、predictor--corrector 和 modern ODE/SDE solvers放在同一 numerical map 中；
- D9 会从 PF ODE/continuity equation 进入 flow matching、rectified flow 与少步生成；
- D12 会回到 terminal、score、truncation 与 discretization error 的严格界；
- Bridge 篇会再次使用 forward/backward drift、path-space entropy 和 time reversal，但目标变为同时满足两个 endpoint constraints。

***

## 16. 章节小结

- Itô SDE $dX_t=fdt+GdW_t$ 的 diffusion matrix 是 $a=GG^\top$；path、transition kernel 与 one-time marginal 是不同层次；
- Itô generator 为 $\mathcal L_t\varphi=f\cdot\nabla\varphi+\tfrac12a:\nabla^2\varphi$；对 test-function identity 做 integration by parts 得到 Fokker--Planck equation；
- state-dependent diffusion 的 second-order term 是 $\tfrac12\nabla^2:(ap)$，不能擅自简化成 $a\Delta p/2$；
- probability current 为 $J=fp-\tfrac12\nabla\cdot(ap)$，density 满足 $\partial_tp=-\nabla\cdot J$；
- forward-in-reversed-time drift 为 $-f+\nabla\cdot a+a\nabla\log p$；原时间标签、$dt<0$ 的 drift 为 $f-\nabla\cdot a-a\nabla\log p$；
- Song 的 $f-g(t)^2s_t$ 是 scalar state-independent diffusion 特例；一般公式必须保留 $\nabla\cdot a$；
- Fokker--Planck/current matching建立 reverse formula 的直觉，不替代带 existence、regularity、entropy/ellipticity 等条件的 path-law time-reversal theorem；
- VP SDE 的 transition mean/std 是 $e^{-B/2}x_0$ 与 $\sqrt{1-e^{-B}}$，并满足 $\bar\alpha(t)=e^{-B(t)}$；
- VE 从 raw endpoint 积分产生 variance $\sigma(t)^2-\sigma(0)^2$；finite $\sigma_{\min}$ code convention 必须单独说明；
- sub-VP 与 VP mean 相同，conditional standard deviation 是 $1-e^{-B(t)}$；
- continuous DSM 用 tractable conditional score 监督，population optimum 是 marginal score；VP 与 VE 分别容纳 DDPM 与 NCSN 路线；
- PF ODE velocity 为 $f-\tfrac12\nabla\cdot a-\tfrac12a\nabla\log p_t$，在 exact conditions 下与 SDE 共享 fixed-time marginals；
- SDE 与 PF ODE 不共享 transitions、joint path law、conditional randomness 或 quadratic variation；
- ODE instantaneous density change 是 $d\log p_t(x_t)/dt=-\nabla\cdot v$，Hutchinson identity用 vector--Jacobian product近似高维 trace；
- tractable PF-ODE likelihood 不等于 ordinary score objective 已在直接执行 exact maximum-likelihood training；
- terminal mismatch、time truncation、score error、SDE/ODE discretization、trace variance 与 data transform 是不同误差层。

***

## 17. 思考题

1. **PDE 不唯一与 path reversal。** 构造二维 divergence-free field $K$，使 $\nabla\cdot K=0$。说明仅从 $\partial_tp=-\nabla\cdot J$ 为什么不能区分 $J$ 与 $J+K$。还需要什么 path-space 信息才能识别真正的 reversed transition？

2. **两套时间 convention。** 从 $Y_\tau=X_{T-\tau}$ 出发，逐项推导 drift 与 Brownian increment 的变化。解释为什么 drift 变号，但 diffusion covariance 仍乘正的 $d\tau$ 或 $|dt|$。

3. **state-dependent divergence。** 对一维一般 $a(x)>0$ 和目标 stationary density $p(x)$，求使 stationary current $J=0$ 的 drift。再比较省略 $a'(x)$ 后得到的错误 stationary equation。

4. **same marginal, different coupling。** 除本章 stationary 例子外，构造一对连续过程，使它们在每个 fixed time 有相同 Gaussian marginal，却有不同 covariance $\operatorname{Cov}(X_s,X_t)$。这怎样直接否定“same marginal implies same process”？

5. **VP discrete limit。** 令 $\beta_k=\beta(t_k)\Delta t+r_k$，给出一组对 remainder $r_k$ 的条件，使 $\prod_k(1-\beta_k)\to e^{-\int\beta}$。若 $r_k=O(\Delta t)$ 而非 $o(\Delta t)$，极限会怎样改变？

6. **VE endpoint convention。** 设训练用 $x_t=x_0+\sigma(t)\epsilon$，但 sampler 被解释为从 raw $X_0$ 对 finite-$\sigma_{\min}$ SDE 积分。写出两者 conditional variance 与 target score 的差异，并讨论 $t\to0$ 时相对误差。

7. **PF ODE 的非唯一性。** 在二维 isotropic Gaussian density path 上，寻找一个满足 $\nabla\cdot(up_t)=0$ 的旋转场 $u$。加入该场后 marginals 不变，但 trajectories 发生什么变化？

8. **approximate score 的分叉。** 把 $s_t$ 替换为 $s_t+e_t$，分别写出 learned reverse SDE 与 learned PF ODE 的 Fokker--Planck/continuity operators。为什么误差系数分别为 $1$ 和 $1/2$ 不能单独推出哪个 endpoint error 更小？

9. **Hutchinson variance。** 对一般 matrix $A$，比较 Gaussian 与 Rademacher $\epsilon$ 下 $\epsilon^\top A\epsilon$ 的 variance。Jacobian 的 diagonal/off-diagonal structure怎样影响 estimator？

10. **likelihood sign check。** 对线性 flow $\dot x=cx$ 在 $d$ 维中直接求 $x_T$、Jacobian determinant 和 density change，再与 $\int_0^T\nabla\cdot v\,dt=dcT$ 对照，验证本章 likelihood 公式的正负号。

11. **ODE stiffness。** 对 VP PF ODE，score Hessian 会进入 vector-field Jacobian。讨论 $t\to0$ 时窄 data modes 可能怎样增加 stiffness，以及 early stopping $\varepsilon>0$ 同时引入了哪类 statistical bias。

***

## 18. 本章来源与继续阅读

1. Yang Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021. 本地论文（补充材料暂未公开）；结构化笔记（补充材料暂未公开）；固定版本官方代码（补充材料暂未公开）。
2. Patrick Cattiaux, Giovanni Conforti, Ivan Gentil, Christian Léonard. *Time Reversal of Diffusion Processes under a Finite Entropy Condition*. AIHP, 2023. 作者版论文（补充材料暂未公开）；[严格边界笔记](https://doi.org/10.1214/22-AIHP1320 "官方论文页面")。
3. Simo Särkkä, Arno Solin. *Applied Stochastic Differential Equations*. Cambridge University Press, 2019. 本地研究副本说明与章节笔记（补充材料暂未公开）。本地 PDF 受 personal-use 条款约束，不作为教程附件再发布。
4. Wenpin Tang, Hanyang Zhao. *Score-based Diffusion Models via Stochastic Differential Equations -- a Technical Tutorial*. Statistics Surveys, 2025. 本地论文（补充材料暂未公开）；[结构化笔记](https://doi.org/10.1214/25-SS152 "官方论文页面")。
5. Will Grathwohl et al. *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*. ICLR, 2019. 本地论文（补充材料暂未公开）；结构化笔记（补充材料暂未公开）。
6. Yang Song, Conor Durkan, Iain Murray, Stefano Ermon. *Maximum Likelihood Training of Score-Based Diffusion Models*. NeurIPS, 2021. 本地论文（补充材料暂未公开）；结构化笔记（补充材料暂未公开）。
7. Brian D. O. Anderson. *Reverse-Time Diffusion Equation Models*. Stochastic Processes and their Applications, 1982. 当前仅作 [metadata 与获取边界记录](https://doi.org/10.1016/0304-4149\(82\)90051-5 "官方论文页面")，不使用来源可疑的转载全文承担 theorem 细节。

完整独立代数复核见 SDE/reverse/PF ODE derivation ledger（补充材料暂未公开），高精度数值交叉检查见 sde\_identity\_checks.py（补充材料暂未公开），本章 claim--evidence mapping 见 D4 chapter source packet（补充材料暂未公开）。
