---
title: 一步生成究竟需要学习什么
description: >-
  区分 local velocity、solver、teacher transition、endpoint consistency 与 finite flow
  map，说明一步生成改变了什么学习对象。
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
scope: >-
  聚焦 Flow Matching、Rectified Flow、distillation、consistency 与 flow map 的核心对象，并划清
  Diffusion、Flow 与 Schrödinger Bridge 的边界。
---
第四章把一个现代 Diffusion 系统拆成 objective、guidance、representation、architecture 与 sampler。沿着这条分工继续追问，最直接的速度问题是：为什么一个已经训练好的网络还要被调用几十次？

快速 sampler 的回答是在不改变网络所学函数的前提下，用更合适的 trajectory、solver 和 grid 减少 numerical error。可是当预算降到一次或两次 network evaluation 时，问题发生了性质变化：

> 网络应该继续给出当前位置的瞬时方向，还是直接预测跨越一段时间后的有限结果？

设状态 $x_t\in\mathbb R^d$ 满足 ordinary differential equation

$$
\frac{d x_t}{dt}
=
b_t(x_t),
$$

其中 $b_t(x)\in\mathbb R^d$ 是时刻 $t$ 的 local velocity field。给定两个时刻 $s,t$，记

$$
\Phi_{s,t}(x_s)=x_t
$$

为该 ODE 的 finite flow map：它直接回答“从 $s$ 的状态出发，到 $t$ 在哪里”。Local field $b_t$ 与 finite map $\Phi_{s,t}$ 由同一动力学联系，却不是同一种函数接口。前者通常需要 solver 反复查询，后者若能被准确学习，才天然支持一次大跨度跳转。

本章沿这一区分梳理历史。2021—2022 年的快速 solver 暴露了局部场的单步极限；2022—2023 年的 Flow Matching 与 Rectified Flow 改写了局部 velocity 的训练方式和 path geometry；2022—2025 年的 progressive distillation、Consistency Models 与 flow-map 方法开始学习教师转移、固定端点或任意两时刻间的有限映射。少步生成的核心转向并不是继续压缩同一套采样步数，而是改变网络被要求近似的数学对象。

## 1. 2021—2022：更好的 solver 为什么不会自动变成一步生成

Song、Meng 与 Ermon 在 ICLR 2021 论文 [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502 "官方论文页面") 中说明，训练好的 DDPM denoiser 可以沿 deterministic trajectory 和较粗 subsequence 采样。Lu 等人在 NeurIPS 2022 论文 [*DPM-Solver*](https://arxiv.org/abs/2206.00927 "官方论文页面") 中又利用 diffusion ODE 的 semilinear structure，提高每次 model evaluation 的数值效率。两者解决的都是同一个问题：**给定已经学到的 local field，怎样更准确地积分。**

为了看清这种改进的边界，令 $h$ 是从 $t$ 到 $t+h$ 的时间跨度。Euler solver 只查询起点的 local velocity：

$$
\Psi_h^{\mathrm{Euler}}(x_t)
=
x_t+h\,b_t(x_t).
$$

这里 $\Psi_h^{\mathrm{Euler}}$ 是 numerical one-step operator。Exact flow map 则为

$$
\Phi_{t,t+h}(x_t)
=
x_t
+
\int_t^{t+h}
b_\tau(x_\tau)\,d\tau.
$$

积分中的 $x_\tau$ 是真实中间轨迹。两式之差来自整段区间中 time 与 state 共同造成的 field variation。即使 $b_t(x_t)$ 在起点被网络精确预测，只要轨迹弯曲或速度快速变化，一条长度为 $h$ 的切线也不会自动落在弧线终点。

下图左侧用二维旋转场展示这个差别：黑点处的 local tangent 是正确的，但一步 Euler 沿切线到达红叉，exact map 则沿蓝色圆弧到达另一位置。右侧把两类误差分开：增加 Euler/Heun evaluations 会降低 solver error；直接学习 finite map 则是在逼近图中的 exact-map oracle。Oracle 只是本文的教学对象，不代表现实网络能无误差获得该映射。

![局部 velocity、数值积分与 finite flow map](/images/diffusion/d9_local_field_vs_flow_map.png)

少步生成因此出现了另一条路线。Salimans 与 Ho 在 ICLR 2022 论文 [*Progressive Distillation for Fast Sampling of Diffusion Models*](https://arxiv.org/abs/2202.00512 "官方论文页面") 中不再只换 solver，而是让 student 的一个大步匹配 deterministic teacher 的两个小步。若 $t>t'>t''$ 是三个相邻 noise times，抽象地说，训练目标要求

$$
\boxed{
\Psi^{\mathrm{student}}_{t\to t''}(x_t)
\approx
\Psi^{\mathrm{teacher}}_{t'\to t''}
\circ
\Psi^{\mathrm{teacher}}_{t\to t'}(x_t).
}
$$

论文实际使用 DDIM update 的代数反演构造 prediction target，而不是简单平均两次 teacher noise estimates。训练完成后，student 被提升为下一轮 teacher，再把步数继续减半。因此 progressive distillation 改变了 learned function 和 checkpoint；一步质量是多轮 teacher、target 与 optimization error 累积后的经验结果，不是高阶 solver theorem。

本节的结论是：solver acceleration 与 learned transition compression 是两条轴。前者保持 local field 不变，减少离散误差；后者用额外训练让网络吸收一段 teacher trajectory。将 NFE 从 50 降到 10 可能主要依赖 solver，将它继续降到 1 往往需要学习新的有限时间对象。

## 2. 2022—2023：Flow Matching 如何把不可见的 marginal velocity 变成可训练目标

Diffusion 的 probability-flow ODE 已经说明，只要知道正确 velocity，就可以用 deterministic transport 连接 noise 与 data。但它并没有规定 velocity 必须通过 score 再换算得到。Albergo 与 Vanden-Eijnden 的 ICLR 2023 论文 *Building Normalizing Flows with Stochastic Interpolants*，以及 Lipman、Chen、Ben-Hamu、Nickel 与 Le 在 ICLR 2023 论文 [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2209.03003 "官方论文页面")，共同推动了一个问题：能否先选择一条 probability path，再直接回归生成该路径的 velocity，而不在训练内反复求解 ODE？

本节采用 flow 文献常见的时间方向。令 $\tau\in[0,1]$，$p_0=p_{\mathrm{base}}$ 是易采样分布，$p_1=p_{\mathrm{data}}$ 是数据分布。若 density path $p_\tau(x)$ 由 velocity $u_\tau(x)$ 搬运，它们满足 continuity equation

$$
\boxed{
\partial_\tau p_\tau(x)
+
\nabla\cdot
\left[
p_\tau(x)u_\tau(x)
\right]
=0.
}
$$

这里 $\nabla\cdot$ 是对 state $x$ 的 divergence。该方程只描述 probability mass 怎样被局部速度搬运，不含随机 diffusion term。

最直接的 Flow Matching objective 是

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{\tau,\,X_\tau\sim p_\tau}
\left[
\left\|
v_\theta(X_\tau,\tau)
-u_\tau(X_\tau)
\right\|^2
\right],
$$

其中 $v_\theta$ 是 neural velocity field。困难在于，虽然我们可以设计 endpoints 和 conditional paths，marginal field $u_\tau(x)$ 往往需要对许多可能的 endpoint pairs 做 mixture integral，无法逐样本直接计算。

Conditional Flow Matching 的关键不是新的 continuity equation，而是 conditional-expectation projection。设 $(X_0,X_1)$ 来自一个选定的 endpoint coupling，并用可微 interpolation 构造随机路径 $X_\tau$。单个 training sample 的 tangent $\dot X_\tau=dX_\tau/d\tau$ 可以直接计算。定义

$$
\boxed{
u_\tau(x)
=
\mathbb E
\left[
\dot X_\tau
\mid
X_\tau=x
\right].
}
$$

这个条件均值正是随机路径的 Eulerian marginal velocity。平方损失的 Pythagorean decomposition 给出

$$
\begin{aligned}
\mathbb E
\left[
\|v_\theta(X_\tau,\tau)-\dot X_\tau\|^2
\right]
&=
\mathbb E
\left[
\|v_\theta(X_\tau,\tau)-u_\tau(X_\tau)\|^2
\right]\\
&\quad+
\mathbb E
\left[
\|\dot X_\tau-u_\tau(X_\tau)\|^2
\right].
\end{aligned}
$$

第二项与 $\theta$ 无关，因为 residual $\dot X_\tau-u_\tau(X_\tau)$ 在给定 $X_\tau$ 后条件均值为零。Lipman 等人 2023 在 positivity 与 integrability 条件下证明 FM 与 CFM 具有相同 parameter gradients；上式是本文用 conditional expectation 对这一结论所作的教学性重写。

下图左侧显示随机 endpoint pairs 产生的 sample-wise tangents：它们可计算，却因 pairing 而高度随机。右侧显示在同一个位置附近取条件平均后得到的 marginal field。网络学到的不是某一根灰色连线的方向，而是所有能经过当前位置的 conditional paths 的平均速度。

![Conditional Flow Matching 从随机 tangent 投影到 marginal velocity](/images/diffusion/d9_conditional_flow_matching.png)

Flow Matching 的贡献是把一个 inaccessible marginal target 变成可采样的 conditional regression target，并允许研究者自由设计 path、coupling 与 time weighting。它留下的边界同样重要：普通 FM 网络仍输出 local velocity $v_\theta(x,\tau)$，生成时仍需 ODE solver；选择了 Flow Matching loss，并不等于已经得到 exact one-step map。

## 3. 2022—2023：Rectified Flow 的关键为何在于 coupling，而不只是直线插值

Flow Matching 把 conditional path 变成设计变量后，下一个问题是：什么样的 path 更适合少步积分？Liu、Gong 与 Liu 在 2022 年公开、ICLR 2023 发表的论文 [*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*](https://arxiv.org/abs/2209.03003 "官方论文页面") 中提出 Rectified Flow，把 endpoint coupling 与 trajectory straightness 放在中心。

令 $X_0\sim p_{\mathrm{base}}$、$X_1\sim p_{\mathrm{data}}$，并令 $\pi$ 表示二者的 joint distribution，即 endpoint coupling。Rectified Flow 对每个 pair 采用 linear interpolation

$$
X_\tau
=
(1-\tau)X_0+\tau X_1,
\qquad
\dot X_\tau
=
X_1-X_0.
$$

于是训练 local field $v_\theta(x,\tau)$ 的 objective 是

$$
\mathcal L_{\mathrm{RF}}(\theta)
=
\mathbb E_{\tau,\,(X_0,X_1)\sim\pi}
\left[
\left\|
v_\theta(X_\tau,\tau)
-(X_1-X_0)
\right\|^2
\right].
$$

在 population optimum，

$$
\boxed{
v^*(x,\tau)
=
\mathbb E
\left[
X_1-X_0
\mid
X_\tau=x
\right].
}
$$

这个公式揭示了“straight”一词的边界。每个 conditional segment 都是直线，但若许多 segment 在 state-time space 中交叉，同一 $(x,\tau)$ 会对应不同 directions；单值 neural field 只能取条件平均。由 $v^*$ 生成的 ODE 可以保持 linear interpolation 的 one-time marginals，却不会逐条复现最初抽到的直线 pairing。

Rectified Flow 的 reflow 步骤试图改变 coupling。它先用当前 deterministic model 从 base samples 生成 endpoints，得到 model-induced pairs；再把这些 pairs 的直线 chords 作为新训练路径。下图从左到右展示：任意 coupling 的直线大量交叉，当前 flow 给出一组弯曲但确定的 endpoint pairs，而 reflow 对这些 model-induced pairs 重新画直线。观察重点是 pairing 被替换了，而不是单纯把 interpolation formula 又写了一遍。

![Rectified Flow 与 reflow 如何改变 endpoint coupling](/images/diffusion/d9_coupling_reflow.png)

更贴近当前 deterministic coupling 的训练 pairs 往往能减少 crossings 和 curvature，使粗步 solver 更有效。但 reflow 需要先生成 pairs，再训练新模型；实际一步系统还可能包含额外 distillation。论文关于 convex transport cost 的结论也不能升级为“Rectified Flow 总能得到 optimal transport map”：straightness、non-crossing、较低 convex cost 与 Wasserstein optimality 是不同命题。

因此 Rectified Flow 仍属于 local-field learning，只是通过 coupling 与 path design 改善场的几何条件。它为一步生成创造了更容易压缩的 teacher trajectory，却不保证原始模型用一次 Euler step 就能保持多步质量。

## 4. 2023—2025：Consistency 与 flow map 如何直接学习有限转移

Progressive distillation 学的是特定 teacher sampler 的大步转移。Song、Dhariwal、Chen 与 Sutskever 在 ICML 2023 论文 [*Consistency Models*](https://arxiv.org/abs/2202.00512 "官方论文页面") 中进一步提出 endpoint consistency：不要求网络输出某个局部 derivative，而是让同一 probability-flow ODE trajectory 上的所有状态映射到相同 data-side endpoint。

沿用 diffusion noise time，令 $t\in[t_{\min},T]$，其中 $T$ 是 noise boundary，$t_{\min}>0$ 是靠近数据端的 cutoff。若 $\Phi_{t,t_{\min}}$ 是 exact probability-flow map，定义 consistency function

$$
f^*(x_t,t)
:=
\Phi_{t,t_{\min}}(x_t).
$$

若 $x_s,x_t$ 位于同一条 exact trajectory，则

$$
\boxed{
f^*(x_t,t)
=
f^*(x_s,s),
\qquad
f^*(x,t_{\min})
=x.
}
$$

第一项是 trajectory invariance，第二项是 boundary condition。Consistency Distillation 用 pretrained score model 与 solver 构造相邻 trajectory pairs，再让 online model 匹配 stop-gradient 或 EMA target；Consistency Training 不需要 pretrained diffusion teacher，但仍依赖 chosen corruption path、shared clean/noise coupling、boundary parameterization 与细网格极限。因而 “from scratch” 不等于“不需要路径假设”。

Endpoint consistency 只回答“回到固定 data boundary 在哪里”。若希望同一个网络跳过任意时间区间，就需要 two-time flow map。Boffi、Albergo 与 Vanden-Eijnden 在 2024 年预印本 [*Flow Map Matching with Stochastic Interpolants*](https://arxiv.org/abs/2406.07507 "官方论文页面") 中显式研究 $\Phi_{s,t}$。在 ODE 解存在且唯一时，exact maps 满足

$$
\boxed{
\Phi_{s,s}
=
\operatorname{Id},
\qquad
\Phi_{m,t}\circ\Phi_{s,m}
=
\Phi_{s,t}.
}
$$

这里 $s,m,t$ 是依次经过的三个时刻，第二式表示 direct jump 与 two sub-jumps 必须一致。还可以用 average velocity $u(x,s,t)$ 参数化 finite map：

$$
\Phi_{s,t}(x)
=
x+(t-s)u(x,s,t).
$$

$u$ 表示整个区间的平均位移率，而不是起点 local velocity；只有在足够光滑且 $t\to s$ 时，才有 $u(x,s,t)\to b_s(x)$。

2024 年 Frans 等人的 *One Step Diffusion via Shortcut Models* 让网络接收 step size，并用两个短 shortcut 的 composition bootstrap 一个长 shortcut；2025 年 Geng 等人的 *Mean Flows for One-step Generative Modeling* 则用包含 total derivative 的 identity，从 local Flow Matching signal 学习 average velocity。这些近期方法的共同方向是把 interval 本身变成网络输入或训练约束，但 approximate neural maps 并不会自动满足 exact composition law，自生成 target 还会带来 compounding error 与 distribution shift。

下图左侧比较 direct map 与两段 composition；中间说明 average velocity 只有在 interval 收缩时才趋近 instantaneous velocity；右侧用解析例子显示 MeanFlow identity 必须包含沿 trajectory 的 spatial derivative。三幅图共同支持一个结论：finite map 不是把 local velocity 换个名字，而是需要额外的 interval information 与 consistency constraint。

![flow-map composition、average velocity 与 MeanFlow identity](/images/diffusion/d9_flowmap_shortcut_meanflow.png)

从 progressive distillation 到 consistency，再到 arbitrary flow map，learned object 依次从“一个更大的 teacher step”扩展为“固定端点映射”和“任意两时刻转移”。表达能力在增加，监督也更难：teacher bias、boundary error、composition error 与 off-distribution queries 都可能被压进一次看似便宜的 forward。

## 5. 一步生成的结果究竟能说明什么，以及 Diffusion、Flow 与 Schrödinger Bridge 的边界

“一步生成”首先只是 inference interface：从一个 noise sample 出发，经过一次主生成网络调用得到 output。它不自动意味着总计算最低。Conditional/unconditional guidance 可能需要两份 predictions，student 可能比多步 teacher 更大，decoder 与 text encoder 仍有成本，distillation 还把大量计算前移到了训练和 pair generation。

一步 sample quality 也只验证输出层面的部分性质。较好的 FID 或视觉锐度不证明 learned map 复现 teacher trajectory，不证明 exact likelihood、invertibility 或 mode coverage，也不保证 inversion、editing 和 intermediate-state control 与多步模型同样可靠。Solver error、teacher/model error、finite-map approximation error、guidance error 与 decoder error 必须分别记账；把 NFE 写成 1 不会让这些误差消失。

这条历史线还解释了三个常被混在一起的框架。Song 等人在 ICLR 2021 论文 Score-Based Generative Modeling through Stochastic Differential Equations（补充材料暂未公开） 中从 chosen forward SDE 出发，学习各时刻 marginal score，再构造 reverse-time SDE 或 probability-flow ODE。这里被固定的是 noising/reference dynamics，训练目标估计的是 score。

Flow Matching 则先选择 endpoint coupling、conditional interpolation 或 marginal path，再回归搬运这条 path 的 velocity。Albergo、Boffi 与 Vanden-Eijnden 在 2023 年预印本、后续 JMLR 版本 *Stochastic Interpolants* 中进一步表明，一个随机 interpolant 可以同时导出 marginal velocity、score 以及共享 one-time marginals 的 ODE/SDE family；same marginals 仍不等于 same path law。

Schrödinger 在 1931 年讲演、1932 年论文 Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique（补充材料暂未公开） 中提出双端分布约束下的“最可能演化”问题。现代 path-space formulation 可参见 Léonard 2014 年综述 [*A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport*](https://doi.org/10.3934/dcds.2014.34.1533 "官方论文页面")：给定 reference path law $R$ 与两个 endpoint marginals $\mu_0,\mu_1$，在所有满足约束的 candidate path laws $P$ 中求

$$
\boxed{
P^*
=
\arg\min_{
P:\,P_0=\mu_0,\,
P_1=\mu_1
}
\operatorname{KL}(P\|R).
}
$$

这里 $P_t$ 是 path law $P$ 在时刻 $t$ 的 marginal，$\operatorname{KL}(P\|R)$ 是 path space 上的 relative entropy。与 generic Flow Matching 不同，Schrödinger Bridge 的 objective 会相对于 $R$ 选择 distinguished endpoint coupling 与完整 path law；仅仅选一条 interpolation、回归一个 velocity，并没有完成这项 entropy projection。

因此三者最简洁的边界是：

$$
\boxed{
\begin{aligned}
\text{Score Diffusion}
&:\ \text{fix noising dynamics, learn score},\\
\text{Flow Matching}
&:\ \text{choose probability path, learn velocity},\\
\text{Schrödinger Bridge}
&:\ \text{fix reference and endpoints, optimize path law}.
\end{aligned}
}
$$

这也是整篇 Diffusion 五章的收束。从“为什么先加噪”，到“如何学习 reverse conditional”，再到“score 怎样决定连续动力学”，第四章讨论怎样把理论落实为现代生成系统，本章则说明怎样把局部动力学压缩成有限转移。一步生成的真正门槛从来不只是少调用几次网络，而是能否用有限容量、有限监督与可控误差，学到跨越整段生成过程的函数。

## 本章论文索引

| 时间        | 论文                                                                                                      | 本章中的作用                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1931/1932 | Schrödinger, *Sur la théorie relativiste de l'électron...*                                              | 提出双端分布约束下最可能扩散演化的原始问题                                                  |
| 2014      | Léonard, *A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport*, DCDS | 给出 path-space entropy minimization 的现代系统表述                             |
| 2021      | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR          | 建立 score、reverse SDE 与 probability-flow ODE 的统一接口                      |
| 2021      | Song et al., *Denoising Diffusion Implicit Models*, ICLR                                                | 表明 denoiser 不绑定唯一 reverse chain，并支持 deterministic subsequence sampling |
| 2022      | Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*, ICLR                   | 将两个 deterministic teacher steps 压缩为一个 student step                     |
| 2022      | Lu et al., *DPM-Solver*, NeurIPS                                                                        | 展示固定 local field 下 solver acceleration 的代表路线                           |
| 2022/2023 | Liu et al., *Flow Straight and Fast*, arXiv / ICLR                                                      | 提出 Rectified Flow、coupling rectification 与 reflow                      |
| 2022/2023 | Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants*, arXiv / ICLR       | 从随机 interpolation 构造可回归的 marginal velocity                             |
| 2022/2023 | Lipman et al., *Flow Matching for Generative Modeling*, arXiv / ICLR                                    | 建立 FM/CFM objective 与 conditional-path gradient equivalence            |
| 2023      | Song et al., *Consistency Models*, ICML                                                                 | 学习沿 probability-flow trajectory 不变的 data-side endpoint map             |
| 2023—2025 | Albergo, Boffi & Vanden-Eijnden, *Stochastic Interpolants*, arXiv / JMLR                                | 区分 interpolant、marginal fields 与共享 marginals 的 ODE/SDE laws            |
| 2024      | Boffi et al., *Flow Map Matching with Stochastic Interpolants*, arXiv                                   | 将学习对象扩展为任意两时刻间的 finite flow map                                        |
| 2024      | Frans et al., *One Step Diffusion via Shortcut Models*, arXiv                                           | 用 step-conditioned composition bootstrap 学习长区间 shortcut                |
| 2025      | Geng et al., *Mean Flows for One-step Generative Modeling*, technical report                            | 用 total-derivative identity 学习 two-time average velocity               |
