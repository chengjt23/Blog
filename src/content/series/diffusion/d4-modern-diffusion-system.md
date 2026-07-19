---
title: 一套现代 Diffusion 系统怎样工作
description: >-
  说明 prediction target、guidance、latent representation、U-Net/DiT 与 sampler
  怎样共同工作。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 4
slug: d4-modern-diffusion-system
tags:
  - diffusion
  - guidance
  - architecture
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 解释训练误差尺度、条件方向、表示空间、网络架构与数值采样怎样共同决定质量和成本。
---
第三章已经给出了生成所需的数学对象：在每个噪声时刻估计 score，再用反向 SDE 或 ODE 把简单分布送回数据分布。实际系统仍有很大差异，因为网络可以预测不同变量、在不同表示空间中工作、用不同方式注入条件，并交给不同的数值方法采样。

这些选择不能只看成可互换的工程零件。它们会改变训练时的误差尺度、网络可见的信息、计算发生的位置，以及采样时真正积分的向量场。

## 1. 网络预测什么，会改变训练强调什么

沿用 variance-preserving path 的记号：

$$
X_t=a_tX_0+b_t\epsilon,
\qquad
a_t^2+b_t^2=1,
\qquad
\epsilon\sim\mathcal N(0,I_d).
$$

DDPM 2020 让网络预测 $\epsilon$。得到 $\hat\epsilon_\theta(X_t,t)$ 后，可以恢复 clean estimate

$$
\hat X_{0,\theta}
=
\frac{
X_t-b_t\hat\epsilon_\theta
}
{a_t}.
$$

反过来，若网络直接预测 $\hat X_{0,\theta}$，也能计算

$$
\hat\epsilon_\theta
=
\frac{
X_t-a_t\hat X_{0,\theta}
}
{b_t}.
$$

在 $a_t,b_t\neq0$ 时，两种输出可以代数转换；但同样大小的 network error 经除法缩放后，对另一变量的影响并不相同。特别是在低 SNR 端 $a_t\to0$ 时，从 $\epsilon$ 恢复 $X_0$ 会放大误差。

Salimans 与 Ho 在 2022 年论文 [*Progressive Distillation for Fast Sampling of Diffusion Models*](https://arxiv.org/abs/2202.00512 "官方论文页面") 中使用

$$
\boxed{
v_t
=
a_t\epsilon-b_tX_0
}
$$

作为 prediction target。由于

$$
\boxed{
X_0=a_tX_t-b_tv_t,
\qquad
\epsilon=b_tX_t+a_tv_t,
}
$$

$v$-prediction 在 VP path 两端都不需要除以趋近于零的系数。这使它更适合少步蒸馏，但不表示它对任意 interpolation 都是“真实速度”；上述 $v_t$ 是 VP 圆参数化下的特定变量。

Prediction target 还隐含 loss weighting。例如

$$
\|\epsilon-\hat\epsilon\|^2
=
\operatorname{SNR}(t)
\|X_0-\hat X_0\|^2.
$$

所以 uniform noise MSE 会以 SNR 加权 implied clean error。参数化可转换，不代表有限网络在相同采样分布和 loss 下会学到完全相同的函数。

![不同 prediction target 对训练误差尺度的影响](/images/diffusion/d5_parameterization_geometry.png)

## 2. Guidance 怎样改变条件生成方向

设 $y$ 是类别、文本或其他条件。Bayes rule 给出 conditional score decomposition：

$$
\nabla_x\log p_t(x\mid y)
=
\nabla_x\log p_t(x)
+
\nabla_x\log p_t(y\mid x).
$$

Dhariwal 与 Nichol 在 NeurIPS 2021 论文 [*Diffusion Models Beat GANs on Image Synthesis*](https://arxiv.org/abs/2105.05233 "官方论文页面") 中据此提出 classifier guidance。他们额外训练一个能读取 noisy state $(x_t,t)$ 的 classifier $p_\phi(y\mid x_t,t)$，并在采样时使用

$$
s_{\mathrm{guided}}(x_t,t,y)
=
s_\theta(x_t,t)
+
w\nabla_{x_t}
\log p_\phi(y\mid x_t,t),
$$

其中 $w\ge0$ 是 guidance scale。$w>1$ 会强调与条件一致的区域，通常提高 fidelity，却可能减少 diversity；这时采样目标也不再是未经 tempering 的原条件分布。

Classifier guidance 的麻烦是需要额外 noisy classifier。Ho 与 Salimans 在 2022 年论文 [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598 "官方论文页面") 中让同一个 denoiser 同时学习 conditional 与 unconditional prediction：训练时以一定概率把条件替换为空条件，采样时线性外推

$$
\boxed{
\epsilon_{\mathrm{cfg}}
=
\epsilon_{\mathrm{u}}
+
s
\left(
\epsilon_{\mathrm{c}}
-
\epsilon_{\mathrm{u}}
\right),
}
$$

其中 $\epsilon_{\mathrm{c}}$ 与 $\epsilon_{\mathrm{u}}$ 分别是 conditional 和 unconditional noise prediction，$s=1$ 对应普通 conditional model。

CFG 不需要外部 classifier，却仍需要 conditioning dropout 训练，并且通常要计算两次网络输出。高 scale 的向量场未必是某个 normalized density 的精确 score；把它理解为沿“更像条件、较不像无条件”的方向作外推更稳妥。

![Classifier-free guidance 对 conditional 与 unconditional 向量场作线性外推](/images/diffusion/d7_guidance_vector_fields.png)

## 3. 在 pixel space 还是 latent space 中扩散

Pixel-space Diffusion 直接令 $X_t\in\mathbb R^{H\times W\times C}$ 表示带噪图像。这样保留了全部像素信息，但网络在每个时间步都要处理高分辨率张量，计算量随空间尺寸迅速增长。

Rombach、Blattmann、Lorenz、Esser 与 Ommer 在 CVPR 2022 论文 [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752 "官方论文页面") 中先训练 autoencoder：

$$
z_0=\mathcal E(x_0),
\qquad
\hat x_0=\mathcal D(z_0),
$$

其中 $\mathcal E$ 把图像压缩成较小的 latent tensor，$\mathcal D$ 负责解码。Diffusion 随后作用于 $z_0$，而不是原始 pixels。

这项改变把感知压缩与生成建模拆开。Latent resolution 较低，denoiser 的训练和采样便宜得多；代价是生成模型只能在 autoencoder 保留下来的信息范围内工作。重建伪影、细小文字或高频纹理若已在第一阶段丢失，Diffusion 无法凭空恢复。

Latent space 也改变了噪声的含义。标准 Gaussian 在 latent coordinates 中是否对应自然的感知扰动，取决于 encoder 的尺度与几何；它不是把 pixel-space 理论原封不动地“压缩运行”。

## 4. U-Net 与 DiT 近似的是同一个时间条件函数

Ronneberger、Fischer 与 Brox 在 2015 年论文 [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597 "官方论文页面") 中提出的 U-Net 原本用于医学图像分割。Diffusion 继承的是多尺度 encoder–decoder 与同分辨率 skip connections，而不是原任务本身。

Ho 等人 2020 的 DDPM 把 U-Net 改成 time-conditioned denoiser：输入 noisy image 与 timestep embedding，经过 residual blocks、down/up sampling 和部分 attention layers，输出与输入形状相同的 noise、clean sample 或其他 prediction target。卷积和多尺度结构为局部纹理与空间层级提供了强归纳偏置。

Peebles 与 Xie 在 2023 年论文 [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2209.12152 "官方论文页面") 中提出 DiT。它把 latent image 切成 patches，映射为 token sequence，再用 Transformer blocks 处理；时间和类别条件通过 adaptive layer normalization 等方式进入网络。

DiT 的优势是架构规则、容易随宽度和深度扩展，并能使用全局 token interaction。代价是 dense attention 的主要交互成本随 token 数量近似二次增长，因此 patch size、latent resolution 与模型宽度必须一起考虑。

![U-Net 与 DiT 都接收 noisy state、time 和 condition，但内部信息传递方式不同](/images/diffusion/d8_unet_dit_interfaces.png)

U-Net 或 DiT 决定怎样近似 $s_\theta(x,t,y)$，却不决定 forward schedule、训练权重或采样器。同一骨干可以配合 DDPM、score SDE 或 Flow Matching；看到 Transformer 并不能判断模型学习的一定是 flow。

## 5. Sampler 使用已训练的场，不会替它补课

训练结束后，网络只提供局部信息：在给定 $(x_t,t)$ 时返回 score、noise 或 velocity。Sampler 决定在哪些时间点调用网络，以及怎样把这些局部预测积分成一条完整轨迹。

DDPM ancestral sampler 使用随机 reverse transition。Song、Meng 与 Ermon 在 2021 年论文 [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502 "官方论文页面") 中表明，同一个 DDPM denoiser 还可以配合一族 non-Markovian sampling processes；取确定性设置并跳过部分时间点，便得到常用 DDIM sampler。

Lu、Zhou、Bao、Chen、Li 与 Zhu 在 2022 年论文 [*DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*](https://arxiv.org/abs/2206.00927 "官方论文页面") 中进一步利用 diffusion ODE 的半线性结构，在 log-SNR 坐标下构造高阶 solver。

这些方法可以在不重新训练 denoiser 的情况下减少 network function evaluations，但它们只能更好地使用已有向量场。若 score model 在某些噪声区间有系统误差，或者 guidance 让向量场变得尖锐，solver 不会自动把错误模型修正成正确模型。

因此一套现代 Diffusion 系统的结果由四件事共同决定：预测变量和训练权重塑造网络学到的近似，guidance 改变条件采样方向，表示空间决定信息与计算发生在哪里，sampler 再把局部场积分成样本。任何一项的改动，都可能改变质量、可控性与成本。

## 文献索引

| 时间   | 论文                                                                              | 本章采用的内容                                             |
| ---- | ------------------------------------------------------------------------------- | --------------------------------------------------- |
| 2015 | Ronneberger et al., *U-Net*                                                     | 提供多尺度 encoder–decoder 与 skip connection 的架构来源       |
| 2020 | Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*                   | 建立 time-conditioned U-Net 与 noise prediction 系统     |
| 2021 | Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*              | 提出 classifier guidance                              |
| 2021 | Song, Meng & Ermon, *Denoising Diffusion Implicit Models*                       | 说明同一 denoiser 可配合不同 sampling process                |
| 2022 | Ho & Salimans, *Classifier-Free Diffusion Guidance*                             | 用 conditional dropout 与双预测实现 CFG                    |
| 2022 | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*  | 把 Diffusion 移到 perceptually compressed latent space |
| 2022 | Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* | 引入适合 VP path 与少步生成的 $v$-prediction                  |
| 2022 | Lu et al., *DPM-Solver*                                                         | 利用 diffusion ODE 结构构造训练无关高阶 solver                  |
| 2023 | Peebles & Xie, *Scalable Diffusion Models with Transformers*                    | 提出 DiT 的 patch-token 与 adaptive conditioning 架构     |
