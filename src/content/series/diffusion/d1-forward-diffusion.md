---
title: 为什么生成模型要先学会毁掉数据
description: 沿 2015—2024 年的论文演进，解释前向扩散、直接加噪、SNR 与 noise schedule 为什么让生成问题变得可学习。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 1
slug: d1-forward-diffusion
tags:
  - diffusion
  - forward-process
  - noise-schedule
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 聚焦已知前向破坏过程如何制造多噪声尺度的训练问题，并明确已知 forward transition 不等于已知 reverse transition。
---
今天谈到 Diffusion，人们很容易从 2020 年的 DDPM 开始：给图像加噪，让网络预测噪声，再从 Gaussian noise 一步步生成图像。但这会漏掉一个更有意思的问题：

> 为什么研究者会想到，先设计一条“毁掉数据”的路径，反而能够解决生成问题？

这个想法并不是突然出现的。2015 年，Sohl-Dickstein 等人在 ICML 论文 [*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*](https://arxiv.org/abs/1503.03585 "官方论文页面") 中建立了 diffusion probabilistic model 的基本框架；2020 年，Ho、Jain 和 Abbeel 在 NeurIPS 论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中把它变成了真正有竞争力的图像生成方法；此后 2021 年的 Improved DDPM 与 VDM、2022 年的 EDM、2024 年关于 terminal SNR 的修正，又不断回答同一个问题：**数据究竟应该怎样被破坏，才能让逆向学习既有效又稳定？**

本章沿着这条历史线索，只讨论前向过程。核心不是记住一组噪声参数及其公式，而是理解每次改动在解决什么问题。

## 1. 2015：从非平衡热力学借来一条生成路径

2015 年的生成建模面临一个长期矛盾：简单分布容易采样和计算，却无法描述真实数据；足够灵活的分布虽然表达能力强，却经常伴随困难的归一化、推断或采样。Sohl-Dickstein 等人在 ICML 2015 论文中把这个问题概括为生成模型的**灵活性与可计算性冲突**。

他们借用了非平衡统计物理中“系统逐渐走向平衡态”的语言。这里的借用需要准确理解：统计物理没有直接给出神经生成模型的训练算法，它提供的是一种构造思路——不要试图一步把简单分布变成复杂数据，而是先定义一条容易分析的渐进过程。

论文规定一条固定的 forward Markov chain。设数据位于 $d$ 维空间，$X_0$ 表示从真实数据分布中抽到的干净样本，$X_t$ 表示经过 $t$ 次加噪后的随机变量，$T$ 是总加噪步数。本文用大写 $X_t$ 表示随机变量，用小写 $x_t$ 表示它的一个具体取值。

记 $q$ 为这条人为规定的前向过程所诱导的概率分布，$q_{\mathrm{data}}$ 为未知的真实数据分布。Markov 假设表示：给定当前状态 $X_{t-1}$ 后，下一状态 $X_t$ 不再依赖更早的状态。于是整条前向路径的联合密度可以分解为

$$
q(x_{0:T})
=q_{\mathrm{data}}(x_0)
\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
$$

这里 $x_{0:T}$ 是整条轨迹 $(x_0,x_1,\ldots,x_T)$ 的简写，而 $q(x_t\mid x_{t-1})$ 是第 $t$ 步的 transition density。

在 Gaussian 情形下，Sohl-Dickstein et al. 2015 的 Appendix Table 1 使用

$$
\boxed{
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right),
}
$$

其中，$\mathcal N(\mu,\Sigma)$ 表示均值为 $\mu$、协方差为 $\Sigma$ 的 Gaussian 分布，$I$ 是 $d\times d$ 单位矩阵；$\beta_t\in(0,1)$ 是第 $t$ 步加入的噪声方差。由于协方差是 $\beta_t I$，每个坐标在这一步加入相同强度且彼此独立的噪声。

从这个条件分布采样，等价于写成

$$
X_t
=\sqrt{1-\beta_t}\,X_{t-1}
+\sqrt{\beta_t}\,\varepsilon_t,
\qquad
\varepsilon_t\overset{\mathrm{iid}}{\sim}\mathcal N(0,I),
$$

其中 $\varepsilon_t$ 是第 $t$ 步新采样的标准 Gaussian noise，$\mathrm{iid}$ 表示不同时间步的噪声相互独立且同分布。第一项保留上一时刻的信号，第二项加入新噪声。

平方根保证 signal 与 noise 按**方差**相加：如果 $X_{t-1}\sim\mathcal N(0,I)$，那么 $X_t$ 的协方差为

$$
(1-\beta_t)I+\beta_t I=I.
$$

所以 $X_t$ 仍服从 $\mathcal N(0,I)$，标准 Gaussian 因而是这条链的平稳分布。

为什么不直接令 $X_1=\varepsilon$，一步把数据变成噪声？因为这样会立即切断输入与数据的全部联系。给定 $\varepsilon$ 后恢复 $X_0$，仍然是原来的无条件生成问题。2015 年论文的关键判断是：把破坏拆成许多小步后，相邻状态仍高度相关，困难的全局生成可以被重新组织为许多局部逆转。

但早期 diffusion model 并没有因此立即流行。它的 reverse chain 很长，样本质量也尚未显示出压倒性优势。更重要的是，“小步逆过程可以用简单分布族逼近”并不表示任意有限步的真实

$$
q(x_{t-1}\mid x_t)
$$

都自动成为已知 Gaussian；它仍然依赖未知的数据边缘分布。2015 年解决的是“怎样规定一条可训练的路径”，还没有完全解决“网络应当以什么目标学习逆转”。

## 2. 2020：DDPM 把前向路径变成了训练数据生成器

五年后，Ho、Jain 和 Abbeel 在 NeurIPS 2020 的 DDPM 论文中沿用了同一类 Gaussian forward chain，却重新组织了参数化、训练目标和网络系统。DDPM 的历史意义不是第一次提出 diffusion，而是证明这套框架能够生成当时具有竞争力的高质量图像。

对前向过程而言，DDPM 最重要的计算接口是论文 Eq. (4)：任意时刻的带噪样本都可以直接从 $x_0$ 构造，不必真的执行前面 $t$ 次转移。

为了描述多步累积效果，DDPM 沿用 2020 年论文的记号，定义

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

其中，$\alpha_t$ 是第 $t$ 步保留的 signal power 比例；$\bar\alpha_t$ 是从第 1 步到第 $t$ 步累计保留的 signal power。横线不是平均号，而表示时间上的累积乘积。

下面用归纳法推导任意时刻的分布。假设在 $t-1$ 时刻可以写成

$$
X_{t-1}
=\sqrt{\bar\alpha_{t-1}}X_0
+\sqrt{1-\bar\alpha_{t-1}}\,\varepsilon',
$$

其中 $\varepsilon'\sim\mathcal N(0,I)$，并且与第 $t$ 步新加入的 $\varepsilon_t$ 独立。将它代入第 $t$ 步后，$X_0$ 前的 signal amplitude 变为

$$
\sqrt{\alpha_t\bar\alpha_{t-1}}
=\sqrt{\bar\alpha_t},
$$

而两项独立 Gaussian noise 的方差可以直接相加，总方差为

$$
\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)
=1-\bar\alpha_t.
$$

因此得到 DDPM 2020 Eq. (4)：

$$
\boxed{
q(x_t\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right),
}
$$

这个条件分布说明：给定干净样本 $x_0$ 后，$X_t$ 的条件均值是 $\sqrt{\bar\alpha_t}x_0$，条件协方差是 $(1-\bar\alpha_t)I$。前者衡量还剩多少原始信号，后者衡量已经累积了多少噪声。

用 reparameterization 写成可直接计算的形式，就是

$$
\boxed{
X_t
=\sqrt{\bar\alpha_t}X_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
}
$$

这里的 $\varepsilon$ 是把前 $t$ 步所有独立噪声合并后得到的一份标准 Gaussian noise；它不是某一个固定时间步的 $\varepsilon_t$。

这条公式改变了训练方式。一次训练样本的构造可以完全写成四个随机变量：

$$
\begin{aligned}
X_0 &\sim q_{\mathrm{data}},\\
t &\sim \operatorname{Uniform}\{1,\ldots,T\},\\
\varepsilon &\sim \mathcal N(0,I),\\
X_t &=\sqrt{\bar\alpha_t}X_0
      +\sqrt{1-\bar\alpha_t}\,\varepsilon.
\end{aligned}
$$

第一行从数据集采样干净样本；第二行选择本次训练使用的噪声时刻；第三行采样独立噪声；第四行直接得到对应的带噪样本。这里的 uniform time sampling 是 DDPM 2020 的训练选择，不是前向扩散公式本身强制要求的唯一方案。

模型由此可以在一次训练中看到同一数据分布的许多破坏程度。前向过程不再只是一个物理类比，而成为一台可以无限制造 $(x_0,x_t,\varepsilon)$ 训练样本的机器。

这也是 DDPM 与“一次从噪声映射到图像”的根本差别：它人为规定了数据与噪声之间的中间状态，使网络可以在不同信息尺度上接受监督。2020 年论文随后利用可解析的 $q(x_{t-1}\mid x_t,x_0)$，把逆过程训练转化为 noise-prediction regression；那是下一章的主题。

## 3. “数据变成 Gaussian”究竟是什么意思

DDPM 之后，许多介绍会把前向过程简写成一句“不断加噪，最后数据变成标准 Gaussian”。这句话源自 2015 与 2020 年论文采用的 Gaussian chain，但如果不区分条件分布与整体边缘，就会把近似写成错误的等号。

根据 DDPM 2020 Eq. (4)，给定一个确定的干净样本 $x_0$ 时，

$$
q(x_t\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right)
$$

确实是 Gaussian。这个式子描述的是“从某一个固定 $x_0$ 出发，加噪后会落在哪里”。

为了描述整个数据集在时刻 $t$ 的分布，记 $q_t$ 为随机变量 $X_t$ 的无条件边缘密度。它需要对所有可能的起点 $X_0$ 做平均：

$$
q_t(x_t)
=\mathbb E_{X_0\sim q_{\mathrm{data}}}
\!\left[q(x_t\mid X_0)\right]
=\int q(x_t\mid x_0)
q_{\mathrm{data}}(x_0)\,dx_0.
$$

期望写法同时适用于连续和离散数据；当 $q_{\mathrm{data}}$ 具有密度时，它可以写成右侧积分。积分变量 $x_0$ 遍历数据空间，权重 $q_{\mathrm{data}}(x_0)$ 表示该区域的数据概率。这个 $q_t$ 通常是“缩放后的数据分布与 Gaussian kernel 的卷积”，而不是单个 Gaussian。

下图用一个二维多峰分布展示同一过程。每种颜色代表原数据的一个 mode；随着 $t$ 增大，各 mode 一边向原点收缩，一边被 Gaussian noise 展宽，最终颜色与模式结构都难以区分。

![Cosine schedule 下多峰数据的前向边缘分布](/images/diffusion/d1_forward_marginals.png)

下面的一维双峰例子是对上述公式的直接推演，用来说明论文构造的含义，并非额外引用某篇论文的新定理。设 $m>0$，并假设一维数据 $X_0$ 只可能取 $-m$ 与 $m$，且二者概率相同：

$$
P(X_0=-m)=P(X_0=m)=\frac12,
$$

那么

$$
q_t(x)
=\frac12\mathcal N\!\left(
-\sqrt{\bar\alpha_t}m,1-\bar\alpha_t
\right)
+\frac12\mathcal N\!\left(
\sqrt{\bar\alpha_t}m,1-\bar\alpha_t
\right).
$$

早期时刻仍能看到两个数据模式；随着 $\bar\alpha_t$ 下降，两个均值向零靠近，Gaussian smoothing 同时变强，双峰逐渐合并。只要 $\bar\alpha_t>0$，这个分布一般仍是 Gaussian mixture。只有当

$$
\bar\alpha_t\to0
$$

时，原始信号项消失，才有

$$
X_t\xrightarrow{d}\mathcal N(0,I).
$$

符号 $\xrightarrow{d}$ 表示“依分布收敛”：当累计 signal power $\bar\alpha_t$ 趋于零时，$X_t$ 的分布趋近标准 Gaussian。

因此，2015/2020 框架中的准确表述应当是：forward diffusion 逐渐降低 $X_t$ 中关于 $X_0$ 的信息，并让 aggregate marginal 接近一个简单 Gaussian prior，也就是逆向生成开始时希望采样的简单分布。有限离散步下通常是

$$
q_T\approx\mathcal N(0,I),
$$

而不是由定义自动得到精确等号。

这一区别后来产生了实际影响。定义终点的 signal-to-noise ratio 为

$$
\operatorname{SNR}_T
=\frac{\bar\alpha_T}{1-\bar\alpha_T}.
$$

分子 $\bar\alpha_T$ 是终点残留的 signal power，分母 $1-\bar\alpha_T$ 是终点累计 noise power。Lin 等人在 WACV 2024 论文 [*Common Diffusion Noise Schedules and Sample Steps are Flawed*](https://arxiv.org/abs/2305.08891 "官方论文页面") 中指出，一些常用 schedule 满足 $\operatorname{SNR}_T>0$，即最后训练时刻仍保留非零信号，但推理却从纯 Gaussian noise 启动，形成 terminal train–inference mismatch。他们提出把 terminal SNR 重缩放到精确零，并要求 sampler 从训练的最后时刻开始。这个工作说明：一句被早期教程当作直觉的“终点就是纯噪声”，在真实系统中需要被认真核对。

## 4. 2021—2022：研究重点转向“信息以多快速度消失”

DDPM 2020 在 $T=1000$ 个时间步上使用 linear schedule：令 $\beta_1=10^{-4}$、$\beta_T=0.02$，中间的 $\beta_t$ 线性插值。但 $\beta_t$ 只表示第 $t$ 个 transition 加入多少噪声，真正决定 $X_t$ 还保留多少原始信息的是累计量

$$
\bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s).
$$

这意味着“$\beta_t$ 线性增加”并不等于“数据难度线性变化”。2021 年之后，研究开始把注意力从单步方差转向累计信号强度。

第一个关键节点是 Nichol 与 Dhariwal 的 ICML 2021 论文 [*Improved Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2102.09672 "官方论文页面")。他们观察到，DDPM 的 linear schedule 会在低分辨率图像上过早破坏信息，于是直接设计累计 signal power 曲线。对离散时刻 $t\in\{0,1,\ldots,T\}$，论文定义

$$
\bar\alpha_t
=\frac{f(t)}{f(0)},
\qquad
f(t)=\cos^2\!\left(
\frac{t/T+s}{1+s}\frac{\pi}{2}
\right),
$$

其中 $f(t)$ 是辅助的 cosine-squared 曲线，$s=0.008$ 是避免起点噪声变化过于剧烈的小偏移量。由于 $\bar\alpha_t$ 表示累计 signal power，相邻两时刻的比值可以反推出 one-step signal power，进而得到

$$
\alpha_t=\frac{\bar\alpha_t}{\bar\alpha_{t-1}},
\qquad
\beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}.
$$

cosine schedule 的意义不是某个定理证明它全局最优，而是把信息破坏更均匀地分配到整个时间区间；论文也明确承认，相似形状可能同样有效。

第二个节点是 Kingma、Salimans、Poole 与 Ho 的 NeurIPS 2021 论文 [*Variational Diffusion Models*](https://arxiv.org/abs/2107.00630 "官方论文页面")。VDM 不再把 timestep 当作最自然的噪声坐标，而是先用一个更一般的条件边缘描述加噪：

$$
q(z_t\mid x)
=\mathcal N\!\left(
\alpha(t)x,\sigma(t)^2I
\right).
$$

这里 $x$ 是干净数据，$z_t$ 是时刻 $t$ 的带噪变量，$\alpha(t)$ 是累计 signal amplitude，$\sigma(t)$ 是累计 noise standard deviation。需要特别注意：VDM 的 $\alpha(t)$ 对应 DDPM 的 $\sqrt{\bar\alpha_t}$，并不是 DDPM 中表示 one-step signal power 的 $\alpha_t=1-\beta_t$。

在这组记号下，VDM 定义

$$
\operatorname{SNR}(t)
=\frac{\alpha(t)^2}{\sigma(t)^2},
\qquad
\gamma(t)=-\log\operatorname{SNR}(t).
$$

$\operatorname{SNR}(t)$ 是 signal power 与 noise power 的比值；$\gamma(t)$ 则是 negative log-SNR，随着噪声增加而上升。在 DDPM 的 variance-preserving 写法中，$\alpha(t)^2=\bar\alpha_t$、$\sigma(t)^2=1-\bar\alpha_t$，因此

$$
\operatorname{SNR}_t
=\frac{\bar\alpha_t}{1-\bar\alpha_t}.
$$

SNR 大表示输入仍接近数据，SNR 小表示输入几乎只剩噪声。VDM 的推进在于：它把不同 Gaussian diffusion specification、连续时间目标和 loss weighting 放进同一个 SNR/log-SNR 坐标中讨论，并允许学习单调的 noise schedule。

但 VDM 的结论不能被简化成“schedule 不重要”。它证明的某些连续时间等价需要相同的 SNR 端点、单调可逆的参数化和相应的网络重缩放；有限步离散误差、训练采样方差和神经网络优化仍会受到 schedule 影响。

第三个节点是 Karras 等人的 NeurIPS 2022 论文 [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364 "官方论文页面")。EDM 直接以 noise standard deviation $\sigma$ 标记 noisy marginal：

$$
p(x;\sigma)
=p_{\mathrm{data}}*\mathcal N(0,\sigma^2I).
$$

这里星号 $*$ 表示卷积：先从数据分布 $p_{\mathrm{data}}$ 采样干净数据，再加入标准差为 $\sigma$ 的 Gaussian noise，所得密度记为 $p(x;\sigma)$。EDM 以 $\sigma$ 作为直接的 noise-level coordinate，并把过去经常混称为“schedule”的对象拆开：

- forward/noise-level path 决定经过哪些 noisy distributions；
- training noise distribution 决定训练时哪些噪声级出现得更多；
- loss weighting 决定哪些噪声级贡献更大梯度；
- sampling grid 决定生成时把有限计算花在哪些噪声级；
- solver 决定如何在这些点之间推进。

下图把几种常见 schedule 放在同一坐标中比较。左图是一阶方差 $\beta_t$，中图是累计 signal power $\bar\alpha_t$，右图是 $\log_{10}\operatorname{SNR}_t$。同一条 schedule 在三种坐标下呈现完全不同的形状，这正是不能只看“$\beta_t$ 是否线性”的原因。

![不同 noise schedule 的单步方差、累计信号与 SNR 对比](/images/diffusion/d1_schedule_comparison.png)

这条 2020—2022 的演进线索说明，研究者最初问“每一步加多少噪声”，后来逐渐改问“信息沿什么坐标消失、训练和采样应怎样分配有限计算”。这比记住 linear、cosine 或某组默认参数更接近 schedule 研究的本质。

## 5. 这条历史线索最终解决了什么

从 2015 到 2024，前向扩散的演进可以压缩成四步。

2015 年，Sohl-Dickstein et al. 用一条渐进 Markov chain 连接数据分布与简单平稳分布，建立“固定破坏过程、学习逆过程”的生成框架。它解决了如何为复杂生成问题规定中间路径。

2020 年，Ho et al. 用 DDPM 的 closed-form marginal 与有效参数化，把这条路径变成可高效采样的训练接口。它解决了如何在任意噪声尺度制造监督数据。

2021—2022 年，Improved DDPM、VDM 与 EDM 逐步把单步 $\beta_t$、累计信号、SNR、训练分布、loss weighting 和采样网格拆成不同设计对象。它们解决了“schedule”概念长期含混的问题。

2024 年，Lin et al. 进一步指出 terminal SNR 与 sampler 起点必须和训练分布一致。它提醒我们，理论上的“接近纯噪声”在工程系统中可能仍留下可见偏差。

这些工作共同给出了 Diffusion 的第一个核心答案：

> 加噪之所以有用，不是因为 Gaussian noise 本身能够创造数据，而是因为一个已知、可直接采样、覆盖多种信息尺度的前向过程，能够持续制造逆向学习所需的训练问题。

这个答案同时划清了 forward diffusion 的职责边界。记 $q_{t-1}(x)$ 与 $q_t(x)$ 分别为随机变量 $X_{t-1}$ 和 $X_t$ 的无条件边缘密度。即使 forward transition $q(x_t\mid x_{t-1})$ 完全已知，对于已经观察到的带噪状态 $x_t$，Bayes rule 仍给出

$$
q(x_{t-1}\mid x_t)
=\frac{
q(x_t\mid x_{t-1})q_{t-1}(x_{t-1})
}{q_t(x_t)},
$$

分子包含未知边缘 $q_{t-1}(x_{t-1})$，分母 $q_t(x_t)$ 也由未知的 data distribution 推送而来。因此 forward transition 是 Gaussian，并不意味着无条件 reverse transition 已经可计算，更不意味着它在有限步下必然是 Gaussian。

换句话说，

$$
\boxed{
\text{known forward transition}
\ \not\Rightarrow\
\text{known unconditional reverse transition}.
}
$$

前向扩散完成的不是生成本身，而是把一个难以直接学习的 data distribution，改写成一族带有已知 corruption rule、明确 noise scale 和可直接采样 training pairs 的逆向问题。它提供了生成学习的坐标系与课程，却把“如何从这些 training pairs 恢复 reverse dynamics”留给模型学习。到这里，“为什么要先毁掉数据”已经得到完整回答。

## 本章论文索引

| 时间   | 论文                                                                                            | 本章中的作用                                                         |
| ---- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 2015 | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML | 建立 forward diffusion、learned reversal 与 path variational bound |
| 2020 | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS                                | 给出 DDPM 系统与任意时刻 direct noising 接口                              |
| 2021 | Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML                  | 用 cosine cumulative schedule 改善信息破坏速度                          |
| 2021 | Kingma et al., *Variational Diffusion Models*, NeurIPS                                        | 用 SNR/log-SNR 统一 forward schedule 与连续目标                        |
| 2022 | Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS   | 拆分 noise path、训练分布、权重、采样网格与 solver                             |
| 2024 | Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed*, WACV              | 指出 nonzero terminal SNR 与采样起点造成的 mismatch                      |
