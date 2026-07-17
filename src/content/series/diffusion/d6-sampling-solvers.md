---
title: 采样算法演进：从 DDPM 到高阶 Solver
description: 系统比较随机与确定性轨迹、Euler 与 Heun、多步法、DPM-Solver、采样网格和误差预算。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 6
slug: d6-sampling-solvers
tags:
  - diffusion
  - sampling
  - ode-solver
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦给定 denoiser 后的有限步数值路径、NFE 与误差来源，不把 formal order 当作端到端保证。
---
训练结束后，我们手里只有一个接收带噪样本和噪声水平的网络。DDPM 可以用它走一千个随机步，DDIM 可以换成几十个确定性步，DPM-Solver 可以把同一个网络接进指数积分器，EDM 又可以换一组 $\sigma$ 网格、Heun correction 和随机 churn。模型参数没有改变，生成轨迹、计算量和误差却都变了。

这件事揭示了 diffusion sampling 最重要的结构：**模型给出局部的 denoising/score 信息，sampler 决定怎样把这些局部信息连接成从噪声到数据的有限计算路径。** 一个 sampler 至少包含 trajectory law、prediction conversion、solver formula、sampling grid 和 stochasticity policy。论文名称往往把它们捆在一起，理解时却必须拆开。

![同一 denoiser 上的确定性与随机轨迹](/images/diffusion/d6_sampler_trajectories.png)

本章有三层目标。第一层建立 DDPM、DDIM、ODE/SDE sampler 的直觉地图；第二层完整推导 Euler、Heun、multistep 与 DPM-Solver 的指数积分公式；研究层则讨论 formal order 的条件、model error、guidance、stochastic contraction、grid optimization，以及为什么 NFE 降到 1--4 后问题会从“改 solver”转向“改训练”。

***

## 1. Sampler 不是模型：先拆五个独立决定

沿用 D5 的线性 Gaussian marginal：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

VP 情形满足 $\alpha_t^2+\sigma_t^2=1$。网络可能输出 $\hat\epsilon_\theta$、$\hat x_\theta$、$\hat v_\theta$ 或 score；D5 已给出它们的转换和端点条件。本章把这些输出封装为一个 model wrapper，再把 sampler 拆成五层：

| 层                    | 要回答的问题         | 典型选择                                               |
| -------------------- | -------------- | -------------------------------------------------- |
| model output         | 网络给出的量是什么？     | $\epsilon,x_0,v,$ score, denoiser                  |
| trajectory law       | 连续或离散路径是否随机？   | DDPM chain、reverse SDE、PF ODE、DDIM $\eta$-family   |
| solver formula       | 一步怎样近似积分？      | Euler、Heun、LMS、exponential integrator              |
| sampling grid        | 在哪些噪声水平调用网络？   | uniform time、log-SNR、EDM polynomial、optimized grid |
| stochasticity policy | 在何时、以多大方差注入噪声？ | ancestral noise、Langevin corrector、churn、Restart   |

![Sampler 名称背后的五个设计列](/images/diffusion/d6_sampler_design_matrix.png)

“换成 Karras sigmas”只换了 grid，不会自动把 Euler 变成 EDM Heun；把 `prediction_type` 从 epsilon 改成 data 是 model wrapper 的变化，也不等于换了 trajectory law。后面所有算法都按这五列比较。

### 1.1 一次网络调用才是一次 NFE

Number of Function Evaluations（NFE）统计实际调用 denoiser/score network 的次数。系数计算、Gaussian noise draw、target conversion 都不计 NFE。它比“循环多少步”更接近主要计算成本，但仍不是完整硬件指标：classifier-free guidance 可以把 conditional/unconditional 拼成一次 batched forward，记作 1 NFE，FLOPs、显存和延迟却可能接近翻倍。

因此至少要同时报告：solver intervals、NFE、是否使用 CFG batch，以及必要时的 wall-clock latency。

***

## 2. 从 DDPM 到 DDIM：训练 marginal 不锁死采样 path

DDPM 的 forward process 是 Markov chain，训练时却可以直接采样

$$
q(x_t\mid x_0)
=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I).
$$

令

$$
\alpha_t=\sqrt{\bar\alpha_t},
\qquad
\sigma_t=\sqrt{1-\bar\alpha_t}.
$$

从 epsilon predictor 恢复 clean prediction：

$$
\boxed{
\hat x_0(x_t,t)
=\frac{x_t-\sigma_t\hat\epsilon_\theta(x_t,t)}{\alpha_t}.
}
$$

DDPM ancestral sampler 使用 learned Gaussian reverse transition，在每个相邻步加入与 posterior variance 匹配的随机噪声。直觉上，这似乎要求 sampling 必须沿训练的完整 Markov chain 逆行。DDIM 的关键观察是：DDPM 的常用训练目标由各时刻 conditional marginals 决定，并没有唯一规定跨时刻怎样 coupling。

### 2.1 DDIM family

对任意 reverse subsequence step $t\to s<t$，定义

$$
\tilde\sigma_{t\to s}^2
=\eta^2
\frac{\sigma_s^2}{\sigma_t^2}
\left(1-\frac{\alpha_t^2}{\alpha_s^2}\right).
$$

DDIM update 为

$$
\boxed{
x_s
=\alpha_s\hat x_0
+\sqrt{\sigma_s^2-\tilde\sigma_{t\to s}^2}\,
\hat\epsilon_\theta(x_t,t)
+\tilde\sigma_{t\to s}z,
\quad z\sim\mathcal N(0,I).
}
$$

这里三项职责很清楚：

- $\alpha_s\hat x_0$：向预测数据移动；
- 第二项：保留与当前 prediction 一致的方向；
- 最后一项：控制新的随机性。

当 $\eta=0$ 时，$\tilde\sigma=0$：

$$
\boxed{x_s=\alpha_s\hat x_0+\sigma_s\hat\epsilon_\theta.}
$$

初始 latent 和网络固定后，整条轨迹确定。这不表示轨迹 exact，也不表示所有初始 latent 会收缩到同一 mode；“deterministic sampler”只表示给定初值后没有额外随机数。

### 2.2 $\eta=1$ 何时才恢复 DDPM

取原始相邻步 $t=k,s=k-1$，记 per-step variance 为

$$
\beta_k=1-\frac{\bar\alpha_k}{\bar\alpha_{k-1}}.
$$

将 $\eta=1$ 代入 DDIM variance：

$$
\begin{aligned}
\tilde\sigma_{k\to k-1}^2
&=\frac{1-\bar\alpha_{k-1}}{1-\bar\alpha_k}
\left(1-\frac{\bar\alpha_k}{\bar\alpha_{k-1}}\right)\\
&=\frac{1-\bar\alpha_{k-1}}{1-\bar\alpha_k}\beta_k
=\tilde\beta_k.
\end{aligned}
$$

这正是 DDPM conditioned posterior variance；展开 $\hat x_0$ 后，均值也恢复 epsilon-parameterized DDPM mean。

这个结论有两个不能删除的条件：**原始相邻完整网格**与**匹配的 variance convention**。在只保留几十个时刻的 coarse subsequence 上取 $\eta=1$，得到的是相应 skipped transition，不是原始一千步 DDPM path law。

### 2.3 Same marginals 不等于 same paths

DDIM construction 可以保持训练所需的 $q(x_t\mid x_0)$，同时改变 $q(x_s\mid x_t,x_0)$。因此：

$$
\text{same one-time marginals}
\centernot\Rightarrow
\text{same transition kernels or joint path law}.
$$

这与 D4 的 PF ODE/reverse SDE 关系完全一致：它们在 exact score 下共享 $p_t$，却有不同 quadratic variation 和路径分布。

***

## 3. 从离散 update 到连续求解问题

D4 已推导 scalar state-independent diffusion 下的 reverse SDE：

$$
d x_t
=\left[f(x_t,t)-g^2(t)\nabla_x\log p_t(x_t)\right]dt
+g(t)d\bar W_t,
$$

以及 probability-flow ODE：

$$
\frac{dx_t}{dt}
=f(x_t,t)-\frac12g^2(t)\nabla_x\log p_t(x_t).
$$

采样时从 noisy endpoint 向 data endpoint 积分。把 exact score 替换为 learned score 后，首先得到一个 learned vector field；再选择有限网格和 numerical solver。这里已经出现三类不同误差：

1. terminal distribution 近似误差；
2. learned score/denoiser 的 model error；
3. 对 learned differential equation 作有限步求解的 discretization error。

solver order 只直接描述第三项。

### 3.1 两类 predictor--corrector 不要混写

Score SDE 的 predictor--corrector（PC）通常是：predictor 推进 reverse SDE/ODE，随后用若干 Langevin corrector steps 在固定 noise level 改善 marginal sampling。每个 corrector 都要重新调用 score 并注入噪声。

UniPC、SA-Solver 里的 corrector 则是数值积分意义的 endpoint correction：利用 predictor 产生的 endpoint model output 和历史 buffer 重估当前积分。若 endpoint evaluation 本来就会供下一步使用，steady phase 可以不增加 NFE。

名称相同，Markov kernel、随机性和 NFE 完全不同。

***

## 4. 最小数值分析：Euler 与 Heun 为什么是一阶和二阶

EDM 用 additive Gaussian noise coordinate：

$$
p(x;\sigma)=p_{\mathrm{data}}*\mathcal N(0,\sigma^2I).
$$

最优 L2 denoiser 满足 Tweedie identity：

$$
\nabla_x\log p(x;\sigma)
=\frac{D(x;\sigma)-x}{\sigma^2}.
$$

取 $\sigma(t)=t,s(t)=1$，PF ODE 简化为

$$
\boxed{
\frac{dx}{dt}=F(x,t):=\frac{x-D(x;t)}t.
}
$$

采样从大 $t$ 到小 $t$，所以一步 $h=t_{i+1}-t_i<0$。

### 4.1 Euler

Euler 只在起点评估一次：

$$
x_{i+1}^{E}=x_i+hF(x_i,t_i).
$$

沿 exact trajectory 作 Taylor expansion：

$$
x(t_i+h)
=x_i+hF_i
+\frac{h^2}{2}
\left(\partial_tF+\partial_xF\,F\right)_i
+O(h^3).
$$

Euler 丢掉了 $h^2$ 项，所以 one-step local truncation error 是 $O(h^2)$。在 Lipschitz/stability 条件下，约 $1/|h|$ 个 local errors 累积为 global error $O(h)$：Euler 是 global first order。

### 4.2 Heun

Heun 先作 Euler predictor：

$$
\tilde x_{i+1}=x_i+hF(x_i,t_i),
$$

再评估 endpoint slope 并取梯形平均：

$$
\boxed{
x_{i+1}^{H}
=x_i+\frac h2
\left[F(x_i,t_i)+F(\tilde x_{i+1},t_i+h)\right].
}
$$

展开第二个 slope：

$$
F(\tilde x_{i+1},t_i+h)
=F_i+h(\partial_tF+\partial_xF\,F)_i+O(h^2).
$$

代回后正好匹配 exact solution 的 $h^2/2$ 项，因此 local error $O(h^3)$，global order 2。

代价是非终点 interval 需要两次 network evaluation。EDM Algorithm 1 在最终 $t_{i+1}=0$ 时不能计算 $(x-D)/t$，回退 Euler。若有 $N$ 个 intervals，典型计数是

$$
\boxed{
\mathrm{NFE}_{\mathrm{Euler}}=N,
\qquad
\mathrm{NFE}_{\mathrm{Heun}}=2N-1.
}
$$

![按步数与按 NFE 比较 Euler、Heun 和 LMS](/images/diffusion/d6_solver_error_nfe.png)

图中使用可解析二维 Gaussian-mixture denoiser。左图固定 interval count，Heun 看起来明显占优；右图改用真实 NFE 后，比较关系发生变化。它不是图像质量 benchmark，而是提醒我们：以“20 steps”比较 sampler，如果没有说明每步几个 stages，并不公平。

***

## 5. 复用历史：Adams、LMS 与 PNDM

ODE 的 integral form 为

$$
x_{n+1}=x_n+\int_{t_n}^{t_{n+1}}F(x(t),t)dt.
$$

single-step Runge--Kutta 在当前 interval 内增加 stages；linear multistep method（LMS）则缓存之前已经支付过的 model outputs。用最近 $p$ 个 slopes 插值：

$$
F(x(t),t)
\approx
\sum_{j=0}^{p-1}F_{n-j}L_j(t),
$$

其中 Lagrange basis

$$
L_j(t)=\prod_{m\ne j}
\frac{t-t_{n-m}}{t_{n-j}-t_{n-m}}.
$$

积分得到

$$
\boxed{
x_{n+1}
\approx x_n+
\sum_{j=0}^{p-1}
\left[\int_{t_n}^{t_{n+1}}L_j(t)dt\right]F_{n-j}.
}
$$

uniform grid、四个 history values 时得到 Adams--Bashforth coefficients：

$$
\boxed{
x_{n+1}=x_n+\frac h{24}
(55F_n-59F_{n-1}+37F_{n-2}-9F_{n-3}).
}
$$

### 5.1 非均匀网格不能照抄常数系数

若 $\{t_i\}$ 是 log-SNR grid、EDM polynomial grid 或 AYS optimized grid，intervals 不等距。此时 $L_j$ 的 nodes 改变，权重

$$
w_{n,j}=\int_{t_n}^{t_{n+1}}L_j(t)dt
$$

必须按实际 grid 重算。配套代码的 `lagrange_integral_weights` 直接对 polynomial basis 积分，并验证任意四个不等距 nodes 对三次多项式仍 exact。

### 5.2 PNDM 的问题与边界

PNDM 认为把 generic RK/LMS 直接套到 raw diffusion ODE，没有充分利用 DDIM transfer 所编码的 diffusion structure。它把一步拆成 pseudo gradient estimate 和 diffusion-aware transfer：PRK 用于 warmup，PLMS 随后复用历史 epsilon predictions。

这里容易产生一个错误推断：使用四个历史系数，所以 PNDM 是 global fourth order。论文实际证明的是 third-order local error 与 second-order convergence；manifold/pseudo transfer、历史 state error 和具体 assumptions 决定最终 claim。教程必须保留论文自己的阶数表述。

### 5.3 Startup 不是免费细节

第一个 step 没有四个 history values。LMS 必须用 Euler、lower-order formulas 或 Runge--Kutta warmup 逐渐填满 buffer。steady phase 每步 1 NFE 不等于整个 sampler 从第一步起都具有最高阶，也不等于 startup 没有额外 NFE。

***

## 6. 精确传播线性项：DEIS 与 exponential integrator

diffusion PF ODE 通常具有 semi-linear structure：

$$
\frac{dx}{dt}=A(t)x+B(t)F_\theta(x,t).
$$

generic Euler/Heun 同时近似 linear 和 nonlinear parts。但 $A(t)$ 完全由 noise schedule 决定，常常可以解析积分。variation of constants 给出

$$
\boxed{
x_t=\Psi(t,s)x_s
+\int_s^t\Psi(t,\tau)B(\tau)F_\theta(x_\tau,\tau)d\tau,
}
$$

其中 $\Psi(t,s)$ 是 linear homogeneous system 的 transition matrix。exponential integrator 精确计算 $\Psi$，只近似包含 neural network 的 integral。

DEIS 将 network integrand 在历史 nodes 上作 polynomial interpolation，并积分

$$
\int_s^tK(t,\tau)L_j(\tau)d\tau.
$$

这统一了 exponential Euler、Runge--Kutta 式 single-step 和 temporal Adams--Bashforth multistep variants。deterministic DDIM 可看作 VP exponential-integrator family 的 zero-order integrand approximation。

重要的是：**高效不只来自“更高阶”，还来自改变了被近似的对象。** exact-integrate 已知 linear part，往往比给 black-box vector field 盲目增加 stages 更有效。

***

## 7. DPM-Solver：在 half-log-SNR 中积分 neural term

DPM-Solver 进一步选择内在坐标

$$
\boxed{
\lambda_{\mathrm{DPM}}(t)
=\log\frac{\alpha_t}{\sigma_t}.
}
$$

必须注意 D5 默认使用

$$
\lambda_{\mathrm{D5}}
=\log\mathrm{SNR}
=\log\frac{\alpha_t^2}{\sigma_t^2}
=2\lambda_{\mathrm{DPM}}.
$$

本节简写 $\lambda=\lambda_{\mathrm{DPM}}$。从 noisy $s$ 走向 cleaner $t$ 时，$\lambda_t>\lambda_s$，令 $h=\lambda_t-\lambda_s>0$。

### 7.1 Noise-prediction exact solution

DPM-Solver 将 PF ODE 写成 exact integral：

$$
\boxed{
x_t
=\frac{\alpha_t}{\alpha_s}x_s
-\alpha_t
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat\epsilon_\theta(x_\lambda,\lambda)d\lambda.
}
$$

若在一步内冻结 network output：

$$
\hat\epsilon_\theta(x_\lambda,\lambda)
\approx\hat\epsilon_s,
$$

则积分可解析：

$$
\begin{aligned}
x_t
&=\frac{\alpha_t}{\alpha_s}x_s
-\alpha_t(e^{-\lambda_s}-e^{-\lambda_t})\hat\epsilon_s\\
&=\boxed{
\frac{\alpha_t}{\alpha_s}x_s
-\sigma_t(e^h-1)\hat\epsilon_s}.
\end{aligned}
$$

这就是 DPM-Solver-1。DPM-Solver-2/3 对 neural integrand 作更高阶 Taylor/interpolation 近似，同时继续精确保留 linear propagation。

### 7.2 为什么一阶式就是 deterministic DDIM transfer

从

$$
x_s=\alpha_s\hat x_s+\sigma_s\hat\epsilon_s
$$

代入一阶式，并使用

$$
\sigma_t e^h
=\sigma_t
\frac{\alpha_t/\sigma_t}{\alpha_s/\sigma_s}
=\frac{\alpha_t\sigma_s}{\alpha_s},
$$

得到

$$
x_t=\alpha_t\hat x_s+\sigma_t\hat\epsilon_s.
$$

这正是 DDIM $\eta=0$ 的 transfer。两者不是偶然相似，而是同一个 frozen-integrand exponential-Euler update 在不同记号中的表达。

***

## 8. DPM-Solver++：为什么 guided sampling 偏向 data prediction

大 classifier-free guidance scale 会显著放大 model output 和 vector-field curvature。DPM-Solver++ 的出发点是：noise-prediction high-order formulas 在这一 regime 容易不稳，因此改为围绕 data predictor 组织 exact solution：

$$
\boxed{
x_t
=\frac{\sigma_t}{\sigma_s}x_s
+\sigma_t
\int_{\lambda_s}^{\lambda_t}
e^\lambda
\hat x_\theta(x_\lambda,\lambda)d\lambda.
}
$$

冻结 $\hat x_\theta\approx\hat x_s$：

$$
\begin{aligned}
x_t
&=\frac{\sigma_t}{\sigma_s}x_s
+\sigma_t(e^{\lambda_t}-e^{\lambda_s})\hat x_s\\
&=\boxed{
\frac{\sigma_t}{\sigma_s}x_s
+\alpha_t(1-e^{-h})\hat x_s}.
\end{aligned}
$$

对一致的 $(\hat x_s,\hat\epsilon_s)$，它同样化为 deterministic DDIM transfer。差异从高阶开始：noise/data forms 近似不同 integrands，其 derivatives、guidance amplification 和 endpoint conditioning 不同。

DPM-Solver++ 常见两类实现：

- 2S：second-order single-step，需要 intermediate evaluation；
- 2M：second-order multistep，缓存历史 output，steady phase 每步约 1 NFE。

dynamic thresholding 是额外的 data-prediction stabilization，不是 convergence theorem 的一部分；latent-space 模型是否适合直接 thresholding 也要单独判断。完整 guidance 推导留给 D7。

***

## 9. UniPC：用已经支付的 endpoint evaluation 做 correction

classical predictor--corrector 常在 predictor 后额外调用网络，NFE 近乎翻倍。UniPC 的目标是：在 exponential-integrator 框架中，让 predictor 产生 endpoint state；对这个 endpoint 作一次 evaluation，而该 output 同时可作为下一步的起点 output；随后用它和 history differences 修正当前积分。

设一步长度

$$
h_i=\lambda_{t_i}-\lambda_{t_{i-1}},
$$

历史差分写成

$$
D_m
=\epsilon_\theta(\tilde x_{s_m},s_m)
-\epsilon_\theta(\tilde x_{t_{i-1}},t_{i-1}).
$$

UniPC 通过 Vandermonde-like coefficient system 匹配 exponential-integrator 的 $\varphi$-functions：

- UniP-$p$：$p$-order predictor；
- UniC-$p$：在任意 $p$-order solver 后加入 endpoint corrector。

这里必须区分三句话：

1. UniP-$p$ 在论文 assumptions 下具有 global order $p$；
2. UniC-$p$ 的 local accuracy 可达到 $p+1$；
3. 某个低 NFE benchmark 的 FID 改善是经验结果。

它们不是同一个 claim。“corrector no extra NFE”也依赖 endpoint output 复用和具体调度；startup、final denoise 与特殊 pipeline 仍需逐次计数。

***

## 10. EDM：solver、grid 与 stochasticity 拆成三项

EDM 不只提出 Heun。它的重要方法论是把 noise coordinate、network preconditioning、training proposal、loss weight、sampling grid 和 solver 拆开。本章只处理 sampling 侧。

### 10.1 Polynomial sigma grid

取 $N$ 个正 noise levels：

$$
\boxed{
\sigma_i
=\left[
\sigma_{\max}^{1/\rho}
+\frac{i}{N-1}
(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})
\right]^\rho,
\quad i=0,\ldots,N-1,
}
$$

再追加 $\sigma_N=0$。$\rho$ 只决定 NFE 在 noise axis 上的分配，不改变 Heun formula。Appendix 的 truncation 分析提示 $\rho\approx3$ 可较均衡 local error，图像实验却偏好更大的 5--10，正文常用 7。数值误差均衡不等于 perceptual metric optimum。

![Uniform、log-uniform 与 EDM polynomial sampling grids](/images/diffusion/d6_sampling_grids.png)

左图中三组 grid 使用相同节点数，分配位置完全不同；右图显示 local interval widths 也随之改变。训练时的 log-normal $p_{\mathrm{train}}(\sigma)$ 不在这张图里，因为它是另一个测度。

### 10.2 Stochastic churn

EDM Algorithm 2 在选定 noise window 内先临时增大噪声：

$$
\hat\sigma_i=\sigma_i(1+\gamma_i),
$$

$$
\hat x_i
=x_i
+\sqrt{\hat\sigma_i^2-\sigma_i^2}\,
S_{\mathrm{noise}}\epsilon_i,
\quad
\epsilon_i\sim\mathcal N(0,I),
$$

其中

$$
\gamma_i=
\begin{cases}
\min(S_{\mathrm{churn}}/N,\sqrt2-1),
&\sigma_i\in[S_{\min},S_{\max}],\\
0,&\text{otherwise}.
\end{cases}
$$

随后从 $(\hat x_i,\hat\sigma_i)$ 用同一个 Euler/Heun solver 走到 $\sigma_{i+1}$。所以 churn 是 layered stochasticity policy，不是 Euler--Maruyama 的别名，也不改变 training objective。

***

## 11. Formal order 为什么经常不能预测低 NFE 质量

把 endpoint error 概念性拆成

$$
\boxed{
e_{\mathrm{total}}
=e_{\mathrm{init}}
+e_{\mathrm{model}}
+e_{\mathrm{wrapper}}
+e_{\mathrm{grid}}
+e_{\mathrm{disc}}
+e_{\mathrm{stability/roundoff}}.
}
$$

ODE order $p$ 通常只说明 exact smooth vector field、stable scheme 和 $h\to0$ 时

$$
e_{\mathrm{disc}}=O(h^p).
$$

它不控制 model bias，也不保证 5--10 个大步已经进入 asymptotic regime。

![Formal solver order 与总误差地板](/images/diffusion/d6_error_budget.png)

图是明确标注的 schematic decomposition，不是经验曲线。它展示了为什么更高阶 solver 在中等 NFE 可以快速降低 discretization error，却最终撞上 model/wrapper floor。

### 11.1 常见 order reduction 来源

- high-noise/low-noise endpoints 附近 derivatives 快速变化；
- CFG 放大 curvature，使 nominal step 过大；
- multistep history 携带之前的 state/model errors；
- irregular grid 导致 coefficient system 病态；
- clipping、thresholding 和 piecewise wrapper 使 vector field 不够光滑；
- final Euler fallback 降低局部阶；
- finite precision 在大系数相消时放大 roundoff。

因此实践中低阶 2/3 往往比“任意高阶”更稳健。UniPC、DPM-Solver-v3 等论文也在低 NFE 使用 lower/pseudo-order 或减少 corrector，而不是盲目增大 $p$。

### 11.2 ODE order 与 SDE order 不是同一种量

SDE numerical analysis 至少区分：

- strong error：$\mathbb E\|X_T-X_T^h\|$；
- weak error：$|\mathbb E\varphi(X_T)-\mathbb E\varphi(X_T^h)|$。

一般 Euler--Maruyama strong order 为 $1/2$、weak order 为 1；additive noise 等特殊结构可以更高。不能把 ODE Heun 的 global second order、SDE strong order 和 FID 改善统称为“二阶采样质量”。

***

## 12. 随机 sampler 的另一条主线：SA-Solver 与 Restart

ODE sampler 在低 NFE 常因离散误差较小而占优；NFE 增大后，model error 可能成为主导，而 stochasticity 有机会收缩已经积累的误差。2023 年的工作开始系统研究这条轴，而不只把噪声当作 DDPM 遗留设置。

### 12.1 SA-Solver：连续控制 stochasticity

SA-Solver 定义一族 same-marginal reverse SDE：

$$
d x_t
=\left[
f(t)x_t
-\frac{1+\tau^2(t)}2g^2(t)
\nabla_x\log p_t(x_t)
\right]dt
+\tau(t)g(t)d\bar W_t.
$$

- $\tau=0$：PF ODE；
- $\tau=1$：standard reverse SDE；
- 中间值：连续调节随机性。

在 DPM half-log-SNR coordinate 中，论文解析计算 stochastic convolution 的 Gaussian variance，再用 Lagrange history 近似带指数权的 data-prediction integral，形成 stochastic Adams predictor--corrector。

在其 regularity assumptions 下：

$$
\text{SA-Predictor strong global error}
=O\left(\sup_t\tau(t)h+h^s\right),
$$

$$
\text{SA-Corrector strong global error}
=O\left(\sup_t\tau(t)h+h^{\hat s+1}\right).
$$

随机项的 $O(\tau h)$ 说明：增加 interpolation order 不会无条件把整个 stochastic scheme 提升到任意强阶。

### 12.2 Restart：把随机性集中成 forward jumps

Restart 的动机是分离两件事：reverse ODE 保持较小 discretization error；解析 forward noising 负责 error contraction。在区间 $[t_{\min},t_{\max}]$ 内重复：

1. 用已知 perturbation kernel 从 $t_{\min}$ 加噪回 $t_{\max}$，0 NFE；
2. 再用 deterministic ODE solver 从 $t_{\max}$ 走回 $t_{\min}$。

其 informal bound 具有结构

$$
W_1(p_{t_{\min}}^{\mathrm{Restart}(K)},p_{t_{\min}})
\lesssim
B(1-\lambda)^K E_{\mathrm{in}}
+(K+1)O(\delta+\epsilon_{\mathrm{approx}})
(t_{\max}-t_{\min}).
$$

第一项随重复次数 $K$ 指数收缩，第二项因重复 reverse solves 线性增加。因此存在 task-dependent sweet spot，不是“加噪越多越好”。正文 theorem 是 informal 版本，严格引用必须保留 appendix assumptions；upper bound 也不直接等于 FID ordering。

***

## 13. 公式之外的优化：EMS、AYS 与 learned solver 边界

当基础 solver 已经成熟，后续工作开始优化“对哪个函数积分”“在哪里评估”和“是否学习更新系数”。

### 13.1 DPM-Solver-v3：model-specific integrand

DPM-Solver-v3 不再固定 noise/data parameterization，而把 ODE 写成

$$
\frac{dx_\lambda}{d\lambda}
=\left(\frac{\dot\alpha_\lambda}{\alpha_\lambda}-\ell_\lambda\right)x_\lambda
-N_\theta(x_\lambda,\lambda).
$$

先选

$$
\ell_\lambda^*
=\arg\min_{\ell_\lambda}
\mathbb E_{p_\lambda^\theta}
\|\nabla_xN_\theta(x_\lambda,\lambda)\|_F^2,
$$

再用 scaling $s_\lambda$ 和 bias $b_\lambda$ 最小化 transformed integrand 的一阶 derivative residual：

$$
(s_\lambda^*,b_\lambda^*)
=\arg\min_{s_\lambda,b_\lambda}
\mathbb E_{p_\lambda^\theta}
\|f_\theta^{(1)}-s_\lambda f_\theta-b_\lambda\|_2^2.
$$

这些 empirical model statistics（EMS）由冻结模型 trajectories 和 Jacobian-vector products 离线估计。它减少 model-specific first-order discretization term，但有 preprocessing cost，也依赖 model、guidance 和 schedule。pseudo-order 在极低 NFE 更稳定，却明确放弃相应 formal high-order guarantee。

### 13.2 Align Your Steps：直接优化 nodes

AYS 把 stochastic solver 的一步解释为某个 linearized SDE 的 exact solution。对共享 diffusion coefficient $g(t)$ 的两个 SDE，Girsanov argument 给出

$$
D_{\mathrm{KL}}(P_1\|P_2)
\le
\frac12
\mathbb E_{P_1^{\mathrm{paths}}}
\int_0^T
\frac{\|f_1-f_2\|^2}{g^2(t)}dt.
$$

将 true learned SDE 和 solver-specific linearized SDE 代入，得到可按 intervals 相加的 KL upper bound：

$$
\mathrm{KLUB}(t_0,\ldots,t_n)
=\sum_{i=1}^n\mathrm{KLUB}(t_{i-1},t_i).
$$

于是固定 endpoints 和 NFE 后，可以搜索中间 nodes。一个漂亮的解析例子是 additive Gaussian data $p_{\mathrm{data}}=\mathcal N(0,c^2I)$ 与 Euler/DDIM；KL 最优 grid 满足

$$
\boxed{
t_i^*=c\tan\left[
\left(1-\frac{i}{n}\right)\arctan\frac{t_{\min}}{c}
+\frac{i}{n}\arctan\frac{t_{\max}}{c}
\right].
}
$$

最优 nodes 随 data scale $c$ 改变，因此不存在与数据、模型、solver 都无关的 universal best schedule。KLUB 只是 upper bound，不是 FID；从 stochastic derivation 迁移到 ODE solver 是论文的 empirical observation。

### 13.3 Bespoke Non-Stationary Solvers：D6 到 D9 的边界

BNS 考虑随 step 变化、可使用全部历史 states/velocities 的 family：

$$
x_{i+1}=X_i c_i+U_i d_i,
$$

并证明在一般条件下可紧凑写为

$$
x_{i+1}=a_i z_{\mathrm{init}}+U_i b_i.
$$

这里用 $z_{\mathrm{init}}$ 表示 source noise；BNS 原文把它记为 $x_0$，与本教程的 clean-data $x_0$ 不同。其 taxonomy 包含 generic RK、exponential RK、multistep 及 scale-time variants。随后用高精度 teacher trajectories 优化少量 grid/coefficients。这已经是 solver distillation，不是纯 training-free sampling；本章只保留 family/taxonomy，训练 objective、teacher error 和 1--4 NFE 竞争留给 D9。

***

## 14. 一张实践决策表：给定 pretrained model 怎样选 sampler

不存在脱离模型和预算的永久排名，但可以按 failure mode 做初步选择：

| 约束/现象                   | 优先检查                                              | 合理起点                                | 不要直接得出的结论                 |
| ----------------------- | ------------------------------------------------- | ----------------------------------- | ------------------------- |
| 只允许约 10 NFE             | prediction wrapper、log-SNR endpoints              | DPM-Solver++ 2M / UniPC 低阶          | 形式阶越高越好                   |
| 无 guidance、NFE 20--50   | grid 与 NFE 公平性                                    | EDM Heun 或稳定 multistep              | 20 Heun steps 等于 20 NFE   |
| 大 CFG 下过曝/发散            | data prediction、lower-order final、thresholding 边界 | DPM-Solver++ 低阶                     | 问题一定来自训练                  |
| 增加 NFE 后质量早早平台          | model/terminal error floor                        | stochastic churn、SDE 或 Restart 作为诊断 | 继续减小步长必然改善                |
| 换 grid 收益大于换公式          | local error 分配                                    | Karras/optimized schedule           | grid 属于 training schedule |
| custom irregular grid   | multistep coefficients                            | 按实际 nodes 重算 Lagrange integrals     | 可照抄 AB 常数                 |
| img2img/inpainting 中途开始 | scheduler state、begin index                       | 使用经过验证的 framework wrapper           | 只截断 `timesteps` 即可        |

真实系统还要考虑 dynamic thresholding、latent scaling、final denoise、model input preconditioning 和 pipeline 的 step-index state。diffusers 中的 scheduler 比论文伪代码长很多，主要就是在管理这些接口，而不是“推导了另一个定理”。

***

## 15. 配套代码：二维 analytic denoiser，而不是小型 benchmark

d6\_sampling\_solvers.py（补充材料暂未公开） 使用六分量二维 Gaussian mixture。若 clean component 为

$$
x_0\mid k\sim\mathcal N(\mu_k,\tau^2I),
$$

观测

$$
x=x_0+\sigma\epsilon,
$$

则 component posterior mean 为

$$
\mathbb E[x_0\mid x,k]
=\mu_k+
\frac{\tau^2}{\tau^2+\sigma^2}(x-\mu_k),
$$

再按 noisy component likelihood 加权，就得到 exact mixture denoiser。这样所有 sampler 共享同一个无训练误差的 model，图中的 state error 主要反映 discretization、grid 与 stochasticity。

代码实现：

- `SamplerResult`：trajectory、noise levels、NFE 与 metadata；
- `CountedDenoiser`：每次真实 model call 计数；
- `ddim_sampler`：VP $\eta$-family；
- `euler_sampler`、`heun_sampler`、`lms_sampler`；
- `dpm_solver_first_order_noise/data`；
- `edm_sigma_grid`、`edm_sampler` 与 churn；
- high-resolution RK4 只作教学参考轨迹；
- DPM/DDIM algebra、nonuniform polynomial exactness 与 NFE 内置检查。

运行：

```bash
# 本地验证脚本暂未公开
```

它会生成本章五张图。`--no-figures` 只运行 algebra/NFE checks。这里没有训练模型、FID 或“谁是 SOTA”的主张。

***

## 16. 历史主线：每一代 sampler 修复了什么

| 时间                   | 旧问题                       | 新方案                                     | 暴露的新边界                               |
| -------------------- | ------------------------- | --------------------------------------- | ------------------------------------ |
| 2020 DDPM            | 可训练但 reverse chain 很长     | ancestral posterior-like steps          | sampling path 与训练 grid 紧耦合           |
| 2021 DDIM            | 为什么必须走同一 Markov chain     | non-Markovian family、subsequence、$\eta$ | 少步一阶误差                               |
| 2021 Score SDE       | 离散模型缺少连续统一                | reverse SDE、PF ODE、PC                   | generic solver 未利用结构                 |
| 2022 PNDM            | black-box RK 按 NFE 效率低    | pseudo transfer、PRK/PLMS                | startup 与实际阶数                        |
| 2022/23 DEIS         | 已知 linear part 被重复近似      | exponential integrator                  | integrand/grid coefficients 耦合       |
| 2022 DPM-Solver      | 约 10 NFE 仍需结构化求解          | half-log-SNR exact integral             | guidance 下稳定性                        |
| 2022/25 DPM-Solver++ | noise form 在大 guidance 不稳 | data form、2S/2M                         | wrapper/thresholding 依赖              |
| 2022 EDM             | framework 组件混杂            | sigma ODE、Heun、grid、churn 拆分            | 经验 grid 与 NFE 成本                     |
| 2023 UniPC           | corrector 常增加 NFE         | endpoint reuse                          | arbitrary order 不等于 few-step optimum |
| 2023 SA/v3/Restart   | model error 和随机性未系统处理     | controlled SDE、EMS、error contraction    | preprocessing 与新超参数                  |
| 2024 AYS/BNS         | formula 不再是唯一瓶颈           | optimize grid / learn solver            | 从 training-free 走向 distillation      |

技术演进不是“新算法取代旧算法”的单线排名，而是不断把 error responsibilities 拆得更细：先解锁 path，再利用 semi-linear structure，再优化 parameterization 和 grid，最后开始学习 solver 本身。

***

## 17. 常见错误

1. **“DDIM $\eta=1$ 在任何跳步网格都等于原始 DDPM。”** 只有相邻完整网格和匹配 variance convention 才成立。
2. **“Deterministic sampler 没有随机性，所以不能生成多样样本。”** 多样性仍来自随机初始 latent；deterministic 只表示条件于初值后轨迹固定。
3. **“PF ODE 与 reverse SDE 共享 marginals，所以路径可互换。”** transition kernels、joint law 和 quadratic variation 不同。
4. **“四个 history values 就是四阶 PNDM。”** PNDM 论文声称 third-order local、second-order convergence。
5. **“20 steps 就是 20 NFE。”** Heun、RK 和额外 correctors 可能每步多次调用网络。
6. **“更高 formal order 自动改善 FID。”** order 只控制 exact smooth field 的 asymptotic discretization error。
7. **“SDE 的一阶与 ODE 的一阶是同一个定义。”** strong、weak 与 deterministic global order 不同。
8. **“换 Karras sigmas 就是换成 EDM sampler。”** 只换 grid；solver 和 preconditioning 仍可不同。
9. **“训练 noise proposal 和 sampling grid 都是 schedule，所以可以共用。”** 前者是训练 Monte Carlo measure，后者是数值求解 nodes。
10. **“UniPC corrector 永远零成本。”** 依赖 endpoint output 复用、startup 和 pipeline 实现。
11. **“Restart forward jump 不耗 NFE，所以 Restart 不增加成本。”** 每次额外 reverse ODE cycle 的 network calls 都要累计。
12. **“AYS 优化了 KL，所以必然优化 FID。”** 它优化 solver-specific KL upper bound；metric transfer 需经验验证。
13. **“DPM-Solver-v3 是完全 training-free 且无预处理。”** 它冻结模型，但需要离线估计 EMS。
14. **“BNS 只学少量参数，所以仍是普通 training-free solver。”** 它用 teacher trajectories 优化 solver，是 solver distillation。
15. **“diffusers 同名 scheduler 与原论文逐行等价。”** framework 还包含 target conversion、endpoint、state 和兼容性策略，且默认值随版本变化。

***

## 18. 章节小结

- diffusion network 提供局部 denoising/score 信息，sampler 决定 finite path；
- model output、trajectory law、solver、grid 和 stochasticity 必须分层；
- DDIM 说明相同训练 marginals 不唯一决定 path；
- $\eta=0$ 是 deterministic DDIM，$\eta=1$ 恢复 DDPM 需要严格的相邻网格条件；
- Euler global 一阶，Heun global 二阶，但 Heun 通常约花两倍 NFE；
- LMS 用历史换 stages；nonuniform grid 必须重算 interpolation weights；
- PNDM/DEIS 把 diffusion structure 纳入 transfer/exponential integration；
- DPM-Solver 在 $\log(\alpha/\sigma)$ 中精确传播 linear part，DPM-Solver++ 改用 data integrand 提升 guided stability；
- UniPC 复用 endpoint output 做 correction，但 local/global order 与经验质量要分开；
- EDM 把 sigma grid、Heun 和 churn 拆成独立组件；
- formal order 不消除 model error、endpoint mismatch、guidance instability 或 error floor；
- SA-Solver、Restart、DPM-Solver-v3 与 AYS 分别处理随机积分、累计误差收缩、model-specific integrand 和 node allocation；
- 当 NFE 降到 1--4，solver 自由度逐渐耗尽，问题自然转向 D9 的 distillation、consistency 和 learned flow。

***

## 19. 研究式思考题

1. DDIM training objective 只依赖 fixed-time marginals。还能构造哪些不同 path couplings 保持同一 objective？哪些会改变 finite-network sampling error？
2. 从 DDIM $\eta=0$ 的一步式出发，在哪些 regularity 和 grid refinement 条件下可严格得到 PF ODE 极限？
3. exponential integrator 精确传播 linear part 后，剩余 neural integrand 的 stiffness 怎样量化？能否从 pretrained model 自动诊断？
4. 对相同 NFE，single-step second order 与 multistep high order 在含 model error 时谁更稳定？应设计什么可识别实验区分 history contamination？
5. CFG 改变的是 vector field 而非 solver theorem。能否根据 guidance scale 自适应选择 order 和 grid，而不额外训练？
6. AYS 优化 grid，DPM-Solver-v3 优化 integrand parameterization。二者能否写成一个 joint bilevel optimization？
7. stochasticity 可以收缩 accumulated error，也会破坏已形成细节。如何把这两项写成可估计的 optimal-control objective？
8. NFE 忽略 batch size 和 hardware utilization。一个跨 sampler 公平的成本指标应如何组合 NFE、FLOPs、latency、memory 和 energy？
9. 若 learned score 的 error 集中在某些 noise levels，最优 grid 应增加还是减少这些区域的 evaluations？答案如何依赖 error 是 bias 还是 variance？
10. BNS taxonomy 把许多 solver 包含进同一 linear-combination family。expressivity inclusion 为什么不推出可优化性或跨 NFE 泛化？
11. 终点 Euler fallback 只影响最后一步，为什么仍可能显著影响 perceptual quality？它与 decoder/data scaling 有什么交互？
12. 什么证据足以判断“继续改 solver”已不如“改训练/蒸馏”？请给出可操作的 error-budget 诊断标准。

***

## 20. 资料与实现入口

核心一手资料的页码、版本、正式发表状态和 theorem boundary 见 D6 chapter source packet（补充材料暂未公开）。其中包括：

- [DDIM source note](https://arxiv.org/abs/2010.02502 "官方论文页面")；
- [PNDM source note](https://arxiv.org/abs/2202.09778 "官方论文页面")；
- [DEIS source note](https://arxiv.org/abs/2204.13902 "官方论文页面")；
- [DPM-Solver source note](https://arxiv.org/abs/2206.00927 "官方论文页面")；
- [DPM-Solver++ source note](https://doi.org/10.1007/s11633-025-1562-4 "官方论文页面")；
- [UniPC source note](https://arxiv.org/abs/2302.04867 "官方论文页面")；
- [EDM source note](https://arxiv.org/abs/2206.00364 "官方论文页面")；
- [SA-Solver](https://arxiv.org/abs/2309.05019 "官方论文页面")、[DPM-Solver-v3](https://arxiv.org/abs/2310.13268 "官方论文页面")、[Restart](https://arxiv.org/abs/2306.14878 "官方论文页面")、[AYS](https://arxiv.org/abs/2404.14507 "官方论文页面") 与 [BNS](https://arxiv.org/abs/2403.01329 "官方论文页面") notes；
- k-diffusion / diffusers implementation synthesis（补充材料暂未公开）；
- 独立推导台账（补充材料暂未公开）与数值检查（补充材料暂未公开）。

上一章：[D5. 参数化、Schedule 与训练设计空间](/blog/diffusion/d5-parameterization-training-design/)。下一章进入 [D7. 条件生成、Guidance、反演与编辑](/blog/diffusion/d7-guidance-conditioning-editing/)，完整解释 classifier guidance、classifier-free guidance、inversion 与 posterior sampling，以及它们为何会改变本章 solver 的稳定性。
