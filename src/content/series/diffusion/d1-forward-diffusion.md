---
title: Diffusion Model 是什么
description: 从生成问题出发，介绍 forward noising、learned reverse process 与 Diffusion 的历史位置。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 1
slug: d1-forward-diffusion
tags:
  - diffusion
  - generative-modeling
  - forward-process
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 说明数据分布、简单噪声分布和生成路径之间的关系，并区分生成式 Diffusion 与普通去噪。
---
假设手里有一批图像。我们可以从中取样，却不知道这些图像在高维空间里服从怎样的概率密度。生成模型要做的，是学会从一个简单随机变量出发，再产生与这批数据相似的新样本。

Diffusion 的办法有些反直觉：先设计一条不断破坏数据的随机过程，把图像逐渐变成近似 Gaussian noise；再学习怎样沿相反方向，一步步把噪声变回数据。

真正需要学习的不是“怎样加噪”。加噪过程由建模者预先给定，未知的是它在数据分布上的逆向动力学。

## 1. 一条生成路径由两个过程组成

用随机变量 $X_0\in\mathbb R^d$ 表示数据样本，记其分布为 $p_{\mathrm{data}}$。前向过程依次产生 $X_1,\ldots,X_T$，其中 $T$ 是离散时间步数。它的路径分布写成

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}).
$$

这里 $q(x_t\mid x_{t-1})$ 是建模者指定的加噪 transition。经过足够多步后，终点分布 $q_T$ 被设计成接近一个容易采样的分布，通常是标准 Gaussian $\mathcal N(0,I_d)$，其中 $I_d$ 是 $d$ 维单位矩阵。

生成时从相反方向开始。先采样 $X_T\sim p_T$，其中 $p_T$ 通常直接取 $\mathcal N(0,I_d)$，再使用神经网络参数化的反向 transition：

$$
p_\theta(x_{0:T})
=
p_T(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
$$

$\theta$ 表示网络参数。最终输出是 $X_0$，而 $X_1,\ldots,X_T$ 是生成过程中引入的中间随机变量。

下图展示的是前向过程的边缘分布。应当注意的不是样本“越来越模糊”，而是原本复杂、多峰的分布怎样逐渐接近一个简单的单峰分布。

![数据分布在前向加噪过程中逐渐接近 Gaussian](/images/diffusion/d1_forward_marginals.png)

## 2. 为什么要先把数据破坏掉

直接从 Gaussian noise 一步跳到自然图像，需要模型同时决定全局结构、局部纹理和所有可能模式。Diffusion 把这次大跳跃拆成许多较小的随机 transition：噪声很强时先恢复粗略结构，接近数据端时再处理细节。

更重要的是，前向过程可以自动制造训练数据。取一个真实样本 $X_0$，按已知的 $q$ 加噪，就能得到任意时刻的 $X_t$。网络因此可以在许多噪声尺度上观察“被破坏后的样本来自哪里”，而不需要真实世界提供一条从噪声生成图像的配对轨迹。

不过，“拆成小步”不等于每一步的真实逆条件分布已经知道。前向 transition $q(x_t\mid x_{t-1})$ 由我们规定，反向的

$$
q(x_{t-1}\mid x_t)
$$

还依赖未知的数据分布。Diffusion 的学习问题正藏在这里：用 $p_\theta(x_{t-1}\mid x_t)$ 逼近这个未知逆过程。

所以我更愿意把前向扩散理解成一套训练问题的生成器。它规定数据怎样被逐步扰动，也决定网络在训练时会遇到哪些噪声尺度；它本身并没有完成生成。

## 3. 这条思路是怎样形成的

Sohl-Dickstein、Weiss、Maheswaranathan 与 Ganguli 在 ICML 2015 论文 [*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*](https://arxiv.org/abs/1503.03585 "官方论文页面") 中提出 diffusion probabilistic model。论文面对的是生成建模中长期存在的矛盾：简单分布容易求值和采样，却不能描述复杂数据；灵活模型又常常依赖昂贵的 MCMC 或难以处理的归一化常数。

他们借用非平衡热力学的语言，构造一条把数据逐渐推向简单平稳分布的 forward Markov chain，再用变分方法训练 reverse chain。2015 年工作的关键贡献是问题分解方式，而不是今天常见的 noise-prediction 网络。

Ho、Jain 与 Abbeel 在 NeurIPS 2020 论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中把 Gaussian forward process、可解析 posterior 与噪声预测参数化组合成 DDPM。Diffusion 从此不再只是一个有吸引力的概率模型，也开始在图像生成质量上成为有竞争力的方法。

Song、Sohl-Dickstein、Kingma、Kumar、Ermon 与 Poole 在 ICLR 2021 论文 Score-Based Generative Modeling through Stochastic Differential Equations（补充材料暂未公开） 中进一步说明，DDPM 与此前的 score-based model 可以看作连续时间 SDE 框架下的不同离散化。这个统一后来带来了 reverse-time SDE、probability-flow ODE 和更广泛的数值求解方法。

这三步对应三个不同问题：2015 年提出路径式生成模型，2020 年给出简单有效的离散训练方案，2021 年解释网络学习的向量场怎样决定连续时间生成动力学。

## 4. Diffusion 与普通去噪不是一回事

普通图像去噪通常给定一张受损图像，目标是恢复与它对应的干净图像。输入与输出之间存在具体配对，模型关心的是条件估计。

生成式 Diffusion 的起点则是一份随机噪声。它需要经过许多噪声尺度，使最终样本的整体分布接近 $p_{\mathrm{data}}$。训练中看到的 $(X_0,X_t)$ 配对由前向过程合成，不是现实中观察到的“这团噪声原来对应哪张图”。

两者确实共享 denoising regression：给定一个 noisy state，估计其中的干净信号或噪声。但单个 denoiser 只定义一个条件估计函数；只有把不同时间的 denoiser 与一条反向采样动力学组合起来，才得到生成模型。

这也解释了为什么模型不能只在一个固定噪声强度上训练。生成过程会依次经过从纯噪声到近数据的整段区间，网络必须知道当前处于哪个时间或噪声尺度。

## 5. 模型学到的是什么，又没有学到什么

理想情况下，反向模型产生的终点分布 $p_\theta(x_0)$ 接近 $p_{\mathrm{data}}$。这并不意味着中间路径复原了真实图像的形成历史。Forward process 是为了训练和生成而设计的数学路径，换一套噪声 schedule、状态空间或随机动力学，便会得到另一条同样可能生成正确边缘分布的路径。

终点 $q_T$ 也通常只是近似 Gaussian。若训练时的终端分布与采样时使用的 $p_T$ 不一致，模型会从没有充分见过的噪声状态起步。这个差异在步数有限或 schedule 设计不当时尤其明显。

因此，Diffusion 最核心的结构只有三件事：一条已知的前向扰动路径，一个需要学习的反向过程，以及一个从简单分布逐步生成数据的采样程序。Gaussian posterior 为什么可算、训练目标为什么会变成噪声预测、去噪网络为何等价于 score，都是在这三个对象确定之后才出现的问题。

Diffusion 给出的不是“数据怎样在现实中产生”的唯一解释，而是一条可以从简单噪声分布到达数据分布的可学习生成路径。

## 文献索引

| 时间   | 论文                                                                                       | 本章采用的内容                                            |
| ---- | ---------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 2015 | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*  | 提出 forward diffusion 与 learned reverse chain 的生成框架 |
| 2020 | Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*                            | 建立高质量图像生成所用的 DDPM 方案                               |
| 2021 | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* | 统一离散 Diffusion、score model 与连续时间 SDE               |
