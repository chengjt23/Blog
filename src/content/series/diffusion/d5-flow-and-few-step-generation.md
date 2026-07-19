---
title: 为什么 Diffusion 可以加速，以及它与 Flow 的边界
description: 区分 solver、distillation、Flow Matching、Rectified Flow 与 finite flow map 三类加速思路。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 5
slug: d5-flow-and-few-step-generation
tags:
  - diffusion
  - flow-matching
  - distillation
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 说明固定场的数值积分、教师过程压缩和新生成动力学之间的区别，并划清 Diffusion、Flow 与 Bridge 的边界。
---
把一千步采样压到十步、四步甚至一步，表面上都叫“加速”，数学上却可能是三件不同的事：用更好的 solver 积分同一个向量场；训练 student 模仿 teacher 的有限时间转移；或者改学一条更适合短路径生成的新动力学。

如果不区分这三种变化，很容易把“采样器更好”“模型被蒸馏”和“目标函数已经换成 Flow”混为一谈。

## 1. Solver 加速没有改变网络学到的对象

设训练好的 probability-flow ODE 为

$$
\frac{dX_t}{dt}
=
v_\theta(X_t,t).
$$

$v_\theta(x,t)$ 是局部 velocity：它回答状态此刻应往哪里移动。给定时间网格 $T=t_N>\cdots>t_0=0$，数值 solver 通过多次查询 $v_\theta$ 近似整条轨迹。

最简单的一阶更新是

$$
x_{t_{k-1}}
\approx
x_{t_k}
+
(t_{k-1}-t_k)
v_\theta(x_{t_k},t_k).
$$

步长变大时，轨迹曲率和网络误差都会使这一近似变差。更高阶 solver 会在一个区间内组合多次场评估，从而更准确地追踪同一 ODE；它不会改变 $v_\theta$ 本身。

Song、Meng 与 Ermon 在 2021 年论文 [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502 "官方论文页面") 中表明，DDPM denoiser 不绑定唯一的随机 reverse chain。确定性 DDIM 可以沿时间子序列采样，在不重新训练模型的情况下显著减少步骤。

Lu 等人在 2022 年论文 [*DPM-Solver*](https://arxiv.org/abs/2206.00927 "官方论文页面") 中进一步利用 diffusion ODE 的半线性结构与 log-SNR 坐标，构造一至三阶 exponential integrator。它减少的是离散化误差和 network function evaluations，而不是把 local field 变成一次调用即可得到终点的函数。

![局部向量场需要 solver 积分，finite flow map 则直接描述跨时间转移](/images/diffusion/d9_local_field_vs_flow_map.png)

因此，一步 Euler 采样通常不是“一步生成模型”。它只是拿一个只接受局部监督的网络，跨过了一个远大于训练中局部近似尺度的区间。

## 2. Distillation 学习的是教师的有限转移

Salimans 与 Ho 在 ICLR 2022 论文 [*Progressive Distillation for Fast Sampling of Diffusion Models*](https://arxiv.org/abs/2202.00512 "官方论文页面") 中提出另一条路线。先固定一个 deterministic teacher sampler，让 teacher 从时刻 $t$ 连走两步到达 $s$；student 则学习用一步到达同一个 teacher endpoint。

若 teacher 的两步复合映射记为

$$
\Phi^{\mathrm{teacher}}_{t\to s}(x_t),
$$

student 的目标不再只是瞬时 derivative，而是满足

$$
\Phi^{\mathrm{student}}_{t\to s}(x_t)
\approx
\Phi^{\mathrm{teacher}}_{t\to s}(x_t).
$$

训练完成后，把 student 升格为新 teacher，再把步数减半。反复进行便得到 $N\to N/2\to N/4$ 的 progressive distillation。每轮都会改变模型参数和训练 target，因此它不是把同一个网络交给更高阶 solver。

Song、Dhariwal、Chen 与 Sutskever 在 ICML 2023 论文 [*Consistency Models*](https://arxiv.org/abs/2202.00512 "官方论文页面") 中把有限转移进一步写成 endpoint consistency。设 $x_t,x_s$ 位于同一条 probability-flow ODE trajectory，并保留一个靠近数据端的边界时刻 $\varepsilon>0$。Consistency function 满足

$$
\boxed{
f(x_t,t)
=
f(x_s,s)
=
x_\varepsilon.
}
$$

网络不必显式输出整条 trajectory，只需把同轨迹上的不同点映射到同一 data-side endpoint。Consistency distillation 可以使用预训练 diffusion teacher 构造相邻时间对；consistency training 则从 clean sample 和共享噪声构造训练关系，但仍依赖 chosen path、边界条件与离散极限。

一步结果的上限因而受 teacher path、student capacity、训练时覆盖的时间间隔和 distillation error 共同限制。它不是 solver order 提高后自然得到的定理。

## 3. Flow Matching 先选概率路径，再回归速度

Flow Matching 改变得更彻底：它直接把训练对象写成生成 ODE 的 velocity。

设 $p_0$ 是简单 base distribution，$p_1=p_{\mathrm{data}}$ 是数据分布，并先选择 endpoint coupling

$$
(X_0,X_1)\sim\pi.
$$

再为每对端点规定一条 conditional probability path。以最简单的线性插值为例，

$$
X_t
=
(1-t)X_0+tX_1,
\qquad
t\in[0,1],
$$

它的 sample-wise velocity 为

$$
U_t
=
X_1-X_0.
$$

不同 endpoint pairs 可能在同一个位置 $x$ 相交，单个 Markov ODE 不能同时沿所有 sample lines 前进。平方回归的 population optimum 会取条件平均：

$$
\boxed{
v^*(x,t)
=
\mathbb E
\left[
U_t\mid X_t=x
\right].
}
$$

这个 marginal velocity 满足 continuity equation

$$
\partial_t p_t(x)
+
\nabla\cdot
\left(
p_t(x)v^*(x,t)
\right)
=0,
$$

因此由 ODE $\dot X_t=v^*(X_t,t)$ 产生同一组 marginal densities。

Lipman、Chen、Ben-Hamu、Nickel 与 Le 在 ICLR 2023 论文 [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2209.03003 "官方论文页面") 中系统化了这一训练方式。他们用 tractable conditional path 与 conditional velocity 替代难以直接计算的 marginal field，并证明在相应 positivity 与 integrability 条件下，conditional 和 marginal Flow Matching objectives 具有相同的 population gradient。

![Conditional Flow Matching 把 sample-wise velocity 回归成只依赖当前状态的 marginal field](/images/diffusion/d9_conditional_flow_matching.png)

Flow Matching 省掉了训练时模拟当前生成 ODE 的需要，却仍然学习 local velocity；推理时通常还需要 solver。Path、endpoint coupling、time weighting 和 conditional target 都是建模选择，不由 “Flow Matching” 这个名称唯一决定。

## 4. Rectified Flow 真正改变的是 coupling

Liu、Gong 与 Liu 在 ICLR 2023 论文 [*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*](https://arxiv.org/abs/2209.03003 "官方论文页面") 中研究 Rectified Flow。它同样从

$$
X_t=(1-t)X_0+tX_1
$$

出发，并最小化

$$
\int_0^1
\mathbb E
\left[
\left\|
X_1-X_0-v_\theta(X_t,t)
\right\|^2
\right]dt.
$$

总体最优解仍是 $\mathbb E[X_1-X_0\mid X_t=x]$。因此“训练样本沿直线插值”不意味着 learned ODE 的每条轨迹都是直线：当许多 sample lines 相交时，条件平均会重新连接这些路径。

Rectified Flow 的 reflow 才进一步改变 coupling。先用当前 ODE 从 base samples 生成 endpoints，得到模型自身的配对 $(Z_0,Z_1)$；再对这些配对做一次新的直线插值训练。因为新的 coupling 更接近当前 deterministic flow 建立的对应关系，轨迹交叉会减少，生成路径往往更直，也更适合粗步积分。

![Reflow 用模型生成的 endpoint pairing 重新训练，从而减少轨迹交叉](/images/diffusion/d9_coupling_reflow.png)

Reflow 不是免费的 solver trick。它包含生成新配对、再次训练，有时还要追加 distillation。Rectified Flow 也不自动等于 optimal transport；直线、非交叉、较低 convex transport cost 与 Wasserstein-optimal map 是不同结论。

## 5. 从局部场到 finite flow map

一个 local velocity $v(x,t)$ 回答“现在往哪里走”。真正的一步模型更想直接回答“从时刻 $s$ 到时刻 $t$ 会到哪里”。把这个 two-time flow map 记为

$$
\Phi_{s\to t}(x_s)=x_t.
$$

对 well-posed ODE，精确 flow maps 满足

$$
\Phi_{s\to s}=\operatorname{Id},
\qquad
\Phi_{t\to r}\circ\Phi_{s\to t}
=
\Phi_{s\to r}.
$$

Boffi、Albergo 与 Vanden-Eijnden 在 2024 年预印本 [*Flow Map Matching with Stochastic Interpolants*](https://arxiv.org/abs/2406.07507 "官方论文页面") 中直接研究 two-time map；Frans 等人在 2024 年 [*One Step Diffusion via Shortcut Models*](https://arxiv.org/abs/2406.07507 "官方论文页面") 中让网络同时接收 desired step size，并用小步 map 的复合构造大步 bootstrap target。2025 年的 MeanFlow 则学习区间平均 velocity，而不是把 instantaneous Flow Matching velocity 直接拿来走一步。

这些工作仍在快速演进，但共同点很明确：要稳定跨越大时间间隔，训练目标必须包含 finite transition、composition 或 average displacement 的信息。只训练 local derivative，再在推理时突然要求一次远距离跳跃，通常信息不足。

实际选择时，已有可靠 diffusion/score model、只想减少 NFE，可以先改 solver；希望保留 teacher 行为并做少步部署，适合 distillation 或 consistency；愿意重新选择 probability path 并训练 velocity，才需要 Flow Matching 或 Rectified Flow；真正关心双端约束下、相对于 reference dynamics 的随机 path law，才进入 Schrödinger Bridge。

最后一项的区别来自目标本身。Schrödinger 在 1931/1932 年论文 Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique（补充材料暂未公开） 中提出的 Bridge 问题，后来被整理为在双端约束下最小化 path-space KL；generic Flow Matching 则先选择 coupling/path，再回归产生这些 marginals 的 velocity。两者可以共享网络和 regression identity，却不是同一个变分问题。

一步生成不是某个单独公式带来的结果。它要求模型从“局部应该怎样走”转向“跨过一段时间最终会到哪里”，并为这次改变付出 teacher、额外训练、coupling design 或 self-consistency 的代价。

## 文献索引

| 时间   | 论文                                                                                | 本章采用的内容                                            |
| ---- | --------------------------------------------------------------------------------- | -------------------------------------------------- |
| 2021 | Song, Meng & Ermon, *Denoising Diffusion Implicit Models*                         | 说明同一 denoiser 可沿确定性、稀疏时间网格采样                       |
| 2022 | Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*   | 用 student 一步匹配 teacher 两步并逐轮减半                     |
| 2022 | Lu et al., *DPM-Solver*                                                           | 以 diffusion ODE 结构加速固定模型的数值积分                      |
| 2023 | Song et al., *Consistency Models*                                                 | 学习同轨迹点到 data-side endpoint 的一致映射                   |
| 2023 | Lipman et al., *Flow Matching for Generative Modeling*                            | 用 conditional path regression 学习 marginal velocity |
| 2023 | Liu, Gong & Liu, *Flow Straight and Fast*                                         | 提出 Rectified Flow、reflow 与 coupling straightening  |
| 2024 | Boffi, Albergo & Vanden-Eijnden, *Flow Map Matching with Stochastic Interpolants* | 直接学习 two-time flow map                             |
| 2024 | Frans et al., *One Step Diffusion via Shortcut Models*                            | 用 step-conditioned bootstrap 学习大步转移                |
| 2025 | Geng et al., *Mean Flows for One-step Generative Modeling*                        | 学习区间 average velocity                              |
