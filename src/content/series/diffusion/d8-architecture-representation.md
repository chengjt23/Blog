---
title: 架构与表征：U-Net、Latent Diffusion 与 DiT
description: 从信息与计算瓶颈比较 U-Net、多尺度生成、latent diffusion、DiT、压缩和表征对齐。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 8
slug: d8-architecture-representation
tags:
  - diffusion
  - unet
  - dit
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦网络与表示空间的接口、复杂度和证据边界，不把系统收益全部归因于单一架构变化。
---
到目前为止，我们大多把 denoiser 写成一个抽象函数 $F_\theta(x_t,t,c)$。这个抽象让 DDPM、score、SDE、parameterization、solver 和 guidance 的数学关系变得清楚，却暂时隐藏了一个工程上同样根本的问题：**究竟用什么网络表示这个函数，又在哪个空间里表示 $x_t$？**

同一个扩散公式可以配 U-Net，也可以配 Transformer；可以直接处理像素，可以分级生成多种分辨率，也可以先把图像压缩到 latent。现代系统的改善往往同时改变这些选择，还改变数据、训练计算、text encoder、objective、sampler 和 guidance。如果把最终 FID 的变化全部归因于“新架构”，技术史就会变成品牌列表，研究结论也会失去可检验性。

本章的中心观点是：**架构设计本质上是在重新分配信息瓶颈与计算瓶颈。** U-Net 把计算分配到多个空间尺度；cascade 把分辨率负担分给多个模型；latent diffusion 把感知细节交给 autoencoder；DiT 把图像变成 token，并把计算集中到全局交互；DC-AE、linear attention 与 REPA 又分别重新分配压缩、交互和表征学习的职责。

![从 pixel/multiscale、latent representation 到 patch-token backbone 的问题—方案路线。](/images/diffusion/d8_architecture_routes.png)

本章分三层阅读。第一层掌握上图和三条主线：U-Net、latent、DiT。第二层推导 tensor shape、compression ratio、patch token 数、attention complexity 与 adaLN-Zero。研究层则追问：scaling curve 能否推出因果结论？REPA 是否证明了唯一瓶颈？高压缩 autoencoder 的系统收益能否独立归因于 $f$？

***

## 1. 先建立因果坐标系：一个模型名不是一个变量

把一个生成系统写成七元组：

$$
\mathcal S
=
(\mathcal R,\mathcal B,\mathcal O,\mathcal D,
\mathcal C_{train},\mathcal A_{sample},\mathcal G),
$$

其中：

| 轴                     | 含义                    | 例子                                                                  |
| --------------------- | --------------------- | ------------------------------------------------------------------- |
| $\mathcal R$          | representation        | pixel、VQ latent、KL latent、high-compression latent                   |
| $\mathcal B$          | backbone              | U-Net、U-ViT、DiT、MMDiT、linear DiT                                    |
| $\mathcal O$          | objective/path        | epsilon MSE、velocity、interpolant、rectified flow、REPA auxiliary loss |
| $\mathcal D$          | data and conditioning | dataset、caption、text encoder、crop/size metadata                     |
| $\mathcal C_{train}$  | training compute      | parameters、steps、batch、precision、optimizer、hardware                 |
| $\mathcal A_{sample}$ | sampler               | ancestral、DDIM、ODE solver、NFE、grid                                  |
| $\mathcal G$          | guidance              | CFG scale、interval、threshold/rescale                                |

若系统 A 与系统 B 在七个坐标上同时不同，那么“B 更好”是一个系统比较，不是架构因果结论。要把差异归因于 backbone，理想对照应尽量满足

$$
\mathcal S_A\setminus\{\mathcal B_A\}
\approx
\mathcal S_B\setminus\{\mathcal B_B\}.
$$

现实中很少能完全控制，但至少应逐项报告。后面讨论 DiT scaling、SiT、MMDiT、SANA 和 REPA 时，我们会反复使用这张因果账本。

***

## 2. Denoiser 首先是接口，然后才是网络

在线性 Gaussian corruption 下，

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
$$

网络可以预测

$$
F_\theta(x_t,t,c)in
\{\epsilon_\theta,x_{0,\theta},v_\theta,s_\theta\}.
$$

无论内部是卷积还是 attention，最外层都必须满足三个接口约束。

### 2.1 空间与通道接口

若 $x_t\in\mathbb R^{B\times C\times H\times W}$，常见 image-like target 也有相同的 $B,H,W$，输出通道由 target 和 variance parameterization 决定。Improved DDPM 若同时预测均值相关量和 variance 参数，输出通道可能加倍；这不是 backbone 自动知道的，而是 model head 与 objective 的契约。

### 2.2 时间接口

$t$ 可能是论文中的连续时间、离散 index、noise level $\sigma$ 或 log-SNR。time embedding 只是把传入的数编码成向量，不能自动修复 scheduler 与模型之间的 convention mismatch。

### 2.3 条件接口

$c$ 可以通过 additive embedding、adaptive normalization、cross-attention 或 joint attention 进入网络。D7 已经区分 architecture conditioning 与 sampling-time guidance；本章只问条件如何被网络计算，不重新推导 CFG。

因此，架构替换前首先要固定

$$
(\text{input shape},\text{time convention},\text{condition interface},
\text{target},\text{output shape}).
$$

这五项不一致时，所谓“只换 backbone”通常并不成立。

***

## 3. U-Net 的来源：先压缩空间，再恢复定位

U-Net 最初是生物医学图像分割网络，不是生成模型。它的可迁移结构是：contracting path 获取大感受野，expanding path 恢复空间分辨率，同尺度 lateral skip 保留精细定位信息。原论文与扩散实现的职责边界见 [U-Net source note](https://arxiv.org/abs/1505.04597 "官方论文页面")。

设第 $\ell$ 层 encoder feature 为

$$
h_\ell
\in\mathbb R^{B\times C_\ell\times H_\ell\times W_\ell},
\qquad
H_\ell=H/2^\ell,quad W_\ell=W/2^\ell.
$$

decoder 从更低分辨率得到

$$
u_\ell=U_\ell(u_{\ell+1})
\in\mathbb R^{B\times C^u_\ell\times H_\ell\times W_\ell}.
$$

若使用 concatenative skip，则

$$
[u_\ell,h_\ell]
\in
\mathbb R^{B\times(C^u_\ell+C_\ell)\times H_\ell\times W_\ell}.
$$

后续 block 必须把扩大的 channel axis 映射回目标宽度。若使用 additive skip，则先要求两路 shape 完全相同。只写“有 skip connection”并不足以定义网络。

原始 U-Net 因 valid convolution 需要 crop encoder feature；现代 diffusion U-Net 通常用 padding 保持同尺度对齐。继承的是 feature transport 思想，不是 crop 这个历史实现细节。

![Diffusion U-Net 的多尺度接口，以及 DiT 的 patch-token 接口。](/images/diffusion/d8_unet_dit_interfaces.png)

***

## 4. 为什么 multiscale 能省计算：先做一笔卷积账

一个 $k\times k$ convolution 在 $H_\ell\times W_\ell$ 上把 $C_{in}$ 映射为 $C_{out}$，leading multiply-add count 约为

$$
\operatorname{MAC}_{conv,\ell}
\approx
H_\ell W_\ell k^2 C_{in}C_{out}.
$$

空间长宽各减半，在 channel 不变时该项减为四分之一。因此在低分辨率层增加 channel，通常比在原图分辨率增加同样 channel 便宜。U-Net 不是简单“越深分辨率越低”，而是在高分辨率层处理局部细节，在低分辨率层用更宽 feature 处理全局结构，再通过 skip 合并。

但 MAC 不是 wall-clock runtime：activation memory、normalization、attention、kernel fusion、data movement 和硬件利用率都会改变结果。后面讨论 DiT GFLOPs 与 SANA latency 时也要保留同样边界。

***

## 5. DDPM 如何把 U-Net 变成多噪声 denoiser

普通 segmentation U-Net 只接收图像；diffusion U-Net 必须知道当前噪声状态。DDPM 使用 sinusoidal timestep embedding。令 $d=2m$，

$$
e_{2k}(t)=\cos(\omega_k t),
\qquad
e_{2k+1}(t)=\sin(\omega_k t),
\qquad k=0,\ldots,m-1,
$$

频率 $\omega_k$ 按对数尺度排列。每对满足

$$
e_{2k}(t)^2+e_{2k+1}(t)^2=1,
$$

所以原始偶数维 embedding 的平方范数为

$$
\|e(t)\|_2^2=m=d/2.
$$

随后 MLP 把 $e(t)$ 变成 block conditioning。一个简化 ResBlock 可以写成

$$
r_1=\operatorname{Conv}_1(\phi(\operatorname{Norm}(h))),
$$

$$
r_2=\operatorname{Conv}_2(\phi(\operatorname{CondNorm}(r_1,e(t)))),
$$

$$
h'=S(h)+r_2,
$$

其中 $S$ 是 identity 或 $1\times1$ channel projection。

DDPM 还在选定的低分辨率 feature map 上加入 self-attention，并使用 group normalization。关键不是这些组件名字，而是它们共同把单一图像网络变成了一个共享参数的函数族

$$
\{F_\theta(\cdot,t):t\in\mathcal T\}.
$$

共享参数提高了统计效率，也造成不同噪声区间之间的 gradient interference；D5 的 Min-SNR 等 weighting 方法处理的是 objective balance，不应误写成架构模块。

***

## 6. U-Net 中的 attention：在什么分辨率做全局交互

把 feature map

$$
h\in\mathbb R^{B\times C\times H_\ell\times W_\ell}
$$

展平为 $N_ell=H_\ell W_\ell$ 个 token。self-attention 为

$$
Q=hW_Q,qquad K=hW_K,qquad V=hW_V,
$$

$$
\operatorname{Attn}(h)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V.
$$

interaction table 的 shape 是

$$
N_\ell\times N_\ell,
$$

leading interaction cost 为 $O(N_\ell^2d)$。若长宽各减半，$N_\ell$ 减为四分之一，$N_\ell^2$ 减为十六分之一。这解释了早期 diffusion U-Net 为什么只在较低分辨率层放全局 attention。

attention 并不“让模型成为 Transformer”。U-Net 仍依靠显式 down/up sampling 和 convolution hierarchy；attention 只是其中一类 feature mixer。

***

## 7. ADM：更强架构与 classifier guidance 是两件事

Dhariwal--Nichol 2021 的工作常因 classifier guidance 被记住，但论文也系统改进了 unconditional/conditional diffusion backbone。其 [architecture source note](https://arxiv.org/abs/1505.04597 "官方论文页面") 核验了 residual blocks、attention resolutions、multi-head attention 与 adaptive group normalization 的职责。

ADM-style scale-shift normalization 可以写成

$$
\operatorname{AdaGN}(h,e)
=\operatorname{GN}(h)\odot[1+s(e)]+b(e).
$$

若 embedding projection 初始给出 $s=b=0$，该层退化为普通 group norm。与简单 additive bias 相比，scale 和 shift 能按 channel 改变 normalized feature 的幅度与位置。

这里必须分开两条因果线：

1. architecture improvement 改善 denoiser family；
2. classifier guidance 在 sampling 时修改 conditional score。

即使二者出现在同一论文和同一最终指标中，也不能把 guidance 增益写成 architecture 增益。更深网络和更多 attention 的消融也是特定数据、训练预算和实现下的经验结果，不是“增加 block 必然降低 FID”的定理。

***

## 8. 高分辨率像素建模的瓶颈

分辨率从 $H\times W$ 放大到 $2H\times2W$ 时：

- 像素数变为 $4$ 倍；
- 同 channel feature activation 约变为 $4$ 倍；
- 同分辨率 dense attention pair 数约变为 $16$ 倍；
- 为保持 global receptive field，网络深度、downsampling 或 attention placement 还可能变化。

因此问题不只是“参数够不够”，而是高分辨率 activation、全局结构形成与局部细节恢复如何分工。历史上形成了三种回应：

| 路线                 | 主要方案                               | 新问题                                      |
| ------------------ | ---------------------------------- | ---------------------------------------- |
| pixel cascade      | 低分辨率 base + 多级 super-resolution    | 上游样本误差与多模型部署                             |
| latent diffusion   | autoencoder 压缩后在 latent 建模         | reconstruction bottleneck 与 decoder bias |
| single pixel model | 调整 schedule/loss/architecture 直接建模 | 训练内存与 coupled design choices             |

这三种路线不是互斥的理论范式，而是不同的 compute allocation。

***

## 9. Cascaded Diffusion：把联合分布拆给多个模型

设 $x^{(0)},\ldots,x^{(K)}$ 表示从低到高的多个分辨率。cascade 定义

$$
p(x^{(0:K)})
=p_0(x^{(0)})
\prod_{k=1}^{K}p_k(x^{(k)}\mid x^{(k-1)}).
$$

最终高分辨率 marginal 是

$$
p(x^{(K)})
=\int p_0(x^{(0)})
\prod_{k=1}^{K}p_k(x^{(k)}\mid x^{(k-1)})
\,dx^{(0:K-1)}.
$$

采样则沿 factorization 顺序执行：先运行 base diffusion，再依次运行 conditional super-resolution diffusion。每一阶段只承担一部分分辨率跨度，但总 NFE、参数、模型装载和 latency 是各阶段的组合。

### 9.1 训练—推理条件错位

训练 upsampler 时，低分辨率 condition 常来自真实图像的 downsample：

$$
z_{train}\sim q_{down}(z\mid x_{data}).
$$

推理时，condition 来自上游生成模型：

$$
z_{test}\sim p_{base}(z).
$$

除非 $p_{base}=q_{data,low}$，否则

$$
q_{train}(z)\neq q_{test}(z).
$$

后一阶段会遇到训练时较少见的 artifact。继续级联时，误差可能积累；“可能”很重要，因为误差如何传播取决于 conditional model 的 robustness，不存在不加假设的统一倍增公式。

***

## 10. Conditioning augmentation：主动污染上游条件

Cascaded Diffusion 的关键方案是训练时扰动低分辨率 condition。Gaussian augmentation 可写成

$$
z_s
=\sqrt{\bar\alpha_s}z_0
+\sqrt{1-\bar\alpha_s}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

upsampler 学习

$$
p_\theta(x_{high}\mid z_s,s),
$$

其中 augmentation level $s$ 也作为条件输入。若训练时随机采样 $s$，一个模型就 amortize 了多个 corruption strengths；推理时再选择验证表现较好的 $s$。

从分布角度看，训练 condition 变为 mixture：

$$
q_{aug}(z)
=\int q(z_s\mid z_0,s)
q_{data}(z_0)q(s)
\,dz_0ds.
$$

目标不是让 $q_{aug}$ 精确等于未知的 $p_{base}$，而是扩大训练 support，使 upsampler 对合理的上游偏差更稳健。论文在其 cascade 上给出经验支持；这不是“任意上游 error distribution 都被正确边缘化”的定理。

原工作还区分 truncated 与 non-truncated augmentation。前者改变低分辨率 reverse chain 的截断接口，后者只对最终低分辨率样本再加 forward noise。两者训练和采样流程不同，不能只用同一个“加噪 condition”标签代替算法。

***

## 11. Imagen 与 simple diffusion：系统延续和反例路线

Imagen 延续 text-conditioned base + super-resolution cascade，并使用 noise conditioning augmentation。其 Efficient U-Net、text encoder、guidance、thresholding 与 cascade 同时构成系统。D7 已讨论 guidance 和 dynamic threshold；本章只保留 backbone 与 resolution factorization。

*simple diffusion* 则问：能否不用 latent 或 cascade，仍用一个标准 diffusion model 直接生成高分辨率图像？它的方案不是单一模块，而是组合：

1. 按 resolution 调整 noise schedule；
2. 使用 multiscale loss；
3. 主要扩展低分辨率 U-Net stage；
4. 在指定低分辨率 block 加 dropout；
5. 尽早 downsample/patch，避免昂贵的高分辨率 feature。

前两项属于 process/objective，后三项属于 architecture/regularization。论文结果说明“高分辨率逻辑上必须依赖 cascade/latent”这个强命题不成立，但不证明 single pixel model 在所有 compute、memory、data 和 quality 约束下都更优。详细边界见 [pixel/cascade source note](https://arxiv.org/abs/2106.15282 "官方论文页面")。

***

## 12. 从像素到 latent：把感知细节交给第一阶段

two-stage latent model 先训练 encoder/decoder：

$$
z=E_\phi(x),
\qquad
\hat x=D_\psi(z),
$$

再训练 generative model 逼近 latent distribution。若 decoder 是 deterministic map，生成图像分布是 pushforward：

$$
p_{image}=D_{\psi\#}p_\theta(z).
$$

这句话比“在 latent 中扩散”更精确：diffusion 不再直接学习像素细节的 distribution，而是学习 encoder 所定义的 representation；decoder 负责把 latent 还原成图像。

第一阶段的通用 objective 可以写成

$$
\mathcal L_{AE}
=\lambda_{rec}\mathcal L_{rec}(x,\hat x)
+\lambda_{perc}\mathcal L_{perc}(x,\hat x)
+\lambda_{adv}\mathcal L_{adv}
+\lambda_{reg}\mathcal R(z).
$$

四项分别控制像素重建、感知相似、局部真实感与 latent regularization。它们往往互相拉扯：perceptually plausible reconstruction 不一定逐像素准确，极强压缩也会抹掉下游 diffusion 无法恢复的信息。

VQGAN 使用离散 codebook。对 encoder output $\hat z_{ij}$，quantization 为

$$
z_{q,ij}
=\arg\min_{z_k\in\mathcal Z}
\|\hat z_{ij}-z_k\|_2.
$$

典型 codebook/commitment objective 包含 stop-gradient：

$$
\mathcal L_{VQ}
=\|\operatorname{sg}[\hat z]-z_q\|_2^2
+\beta\|\hat z-\operatorname{sg}[z_q]\|_2^2.
$$

VQGAN 再加入 perceptual 与 adversarial terms，目标是让压缩后的表示保留感知上重要的结构。LDM 随后把这个 first-stage 思想用于 diffusion。完整来源职责见 [latent representation source note](https://arxiv.org/abs/2012.09841 "官方论文页面")。

***

## 13. Latent Diffusion：扩散的是 encoder 定义的坐标

LDM 可以使用 vector-quantized latent，也可以使用 KL-regularized continuous latent。以后者为例，encoder 给出 approximate posterior

$$
q_\phi(z\mid x)
=\mathcal N\!\left(z;\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right),
$$

regularizer 约束它不要无限偏离简单 prior：

$$
\mathcal R_{KL}
=D_{KL}\!\left(q_\phi(z\mid x)\,\|\,\mathcal N(0,I)\right).
$$

冻结第一阶段后，在 latent 上加噪：

$$
z_t=\alpha_t z_0+\sigma_t\epsilon,
$$

并训练

$$
\mathcal L_{LDM}
=\mathbb E_{z_0,t,\epsilon,c}
\left[
w(t)\|\epsilon-epsilon_\theta(z_t,t,c)\|_2^2
\right].
$$

形式上它与 pixel DDPM 相似，差异却不只在 tensor 更小。$z_0$ 的 distribution、各 channel 的 scale、局部性和语义都由 encoder 决定。实现通常还会用固定 scale 重新标定 latent variance；否则原来为单位尺度设计的 noise schedule 与实际 latent 尺度可能不匹配。

LDM 对文本条件使用 cross-attention：

$$
Q=W_Q\varphi_i(z_t),
\qquad
K=W_K\tau(c),
\qquad
V=W_V\tau(c),
$$

$$
\operatorname{CrossAttn}(Q,K,V)
=\operatorname{softmax}(QK^\top/\sqrt d)V.
$$

D7 已经讨论这个条件接口及其编辑用途。本章关注的是 encoder--latent denoiser--decoder 的完整系统，不能把“有 cross-attention”当成 latent diffusion 的定义。

### 13.1 为什么不是精确 change of variables

若 $E,D$ 可逆且维数相同，可以写 Jacobian determinant；实际 perceptual autoencoder 通常降维且有损，因此不存在普通的 bijective density formula

$$
\log p_X(x)
=\log p_Z(E(x))+log|\det J_E(x)|.
$$

生成仍然定义良好：先采 $z\sim p_\theta$，再输出 $D(z)$。但 latent likelihood、reconstruction loss 与 pixel-space likelihood 不是同一个量。

***

## 14. 三种“压缩率”必须分开计算

设 RGB image 为 $H\times W\times3$，latent 为

$$
z\in\mathbb R^{(H/f)\times(W/f)\times c}.
$$

### 14.1 Spatial compression

$f$ 表示每条空间边缩小多少。例如 $f=8$ 意味着面积 token grid 缩小 $64$ 倍。它没有包含 latent channel $c$。

### 14.2 Element-count compression

图像与 latent 的元素数之比是

$$
R_{elem}
=\frac{3HW}{(HW/f^2)c}
=\frac{3f^2}{c}.
$$

因此 $f=8,c=4$ 时

$$
R_{elem}=48,
$$

而 $f=32,c=32$ 时

$$
R_{elem}=96.
$$

后者的 spatial factor 增大 $4$ 倍，但 channel 也增大 $8$ 倍，所以 element ratio 只增大 $2$ 倍。

### 14.3 Transformer token compression

若 latent 还用 patch side $p$，image-token 数为

$$
N
=\frac{H}{fp}\frac{W}{fp}
=\frac{HW}{(fp)^2}.
$$

在 $1024\times1024$ 图像上：

$$
N_{pixel,p=2}=262144,
$$

$$
N_{f=8,p=2}=4096,
$$

$$
N_{f=32,p=1}=1024.
$$

后两者 token 数相差 $4$ 倍，dense attention pair 数相差

$$
\left(\frac{4096}{1024}\right)^2=16.
$$

![Spatial factor、latent channels、patch size 对 element/token/attention pair 预算的不同影响。](/images/diffusion/d8_compression_token_budget.png)

这只是 interaction table 算术，不等于整网加速 $16$ 倍。projection、MLP、decoder、memory bandwidth、batch 和 kernel efficiency 都还在。

***

## 15. Autoencoder 的信息瓶颈：省下的计算由谁偿还

压缩把 denoiser 的负担转移给 encoder/decoder。若 $Z=E(X)$ 是 $X$ 的函数，data processing 直觉告诉我们：任何被 encoder 丢弃的区分信息都不能由只观察 $Z$ 的 diffusion 恢复。decoder 可以生成 plausible detail，却无法知道原图中已经丢失的那一个具体细节。

可以把第一阶段看成 rate--distortion trade-off：

$$
\min_{E,D}
\quad
\mathbb E[d(X,D(E(X)))]
+\lambda\,\operatorname{Rate}(E(X)).
$$

这里 distortion 可以是 pixel、perceptual 或 adversarial surrogate，rate 可以由 codebook、KL 或 latent capacity 控制。不同 distortion 定义会保留不同信息。

### 15.1 Reconstruction metric 不是 generation metric

reconstruction FID（rFID）比较真实图像与其重建；generative FID 比较真实图像与从 latent prior/diffusion 采样再 decode 的图像。前者主要检查 first stage，后者还包含 latent generator error：

$$
\text{generation error}
\not\equiv
\text{reconstruction error}.
$$

更好的 rFID 往往有帮助，却不能单独保证更好的 generation FID；反之，一个 decoder 也可能通过 perceptual prior 产生看似真实但不忠实的重建。

### 15.2 Tokenizer 与 denoiser 的职责边界

低压缩 autoencoder 把更多细节留给 denoiser，token 多但 first-stage distortion 小；高压缩 autoencoder 减少 token，却要求每个 token 承载更丰富的信息，并让 decoder 承担更多细节生成。不存在脱离数据、任务和 compute budget 的唯一最优 $f,c,p$。

***

## 16. SDXL：latent diffusion 的系统扩展

SDXL 没有抛弃 latent U-Net，而是在多个维度扩展系统：更大的 U-Net、重新分配的 transformer blocks、多个 text encoders、size/crop micro-conditioning，以及可选 refiner。其核心页面与实现映射见 [latent source note](https://arxiv.org/abs/2012.09841 "官方论文页面")。

### 16.1 Micro-conditioning

训练图像有不同原始尺寸、crop 和 aspect ratio。若这些 preprocessing 状态不告诉模型，同一个 crop 可能对应不同的摄影构图语义。SDXL 把原始尺寸和 crop offset 编码为额外条件：

$$
c_{size}=\operatorname{Embed}(H_{orig},W_{orig}),
$$

$$
c_{crop}=\operatorname{Embed}(y_{top},x_{left}),
$$

并与 timestep/text condition 组合。这是 data/conditioning interface 改进，不是 latent representation 本身。

### 16.2 Base 与 refiner

refiner 对 base 生成的 latent 在低噪声区间继续 denoise，类似 latent-space SDEdit workflow。它可能改善局部细节，但增加第二模型的参数、装载和 NFE。比较“SDXL with refiner”与单模型时，必须把这部分 sampling compute 计入。

### 16.3 因果边界

SDXL 的最终效果同时包含 architecture、conditioning、data curation、text encoders 与训练规模。技术报告支持“这些模块组成了 SDXL”，不支持“U-Net 变大三倍单独造成全部提升”。

***

## 17. DC-AE：把 patch compression 前移到 autoencoder

普通 latent diffusion 常先用 $f=8$ autoencoder，再在 DiT 中用 $p>1$ patch embedding 继续减少 token。DC-AE 的问题设定是：能否让 autoencoder 直接承担更多 spatial compression，使 denoiser 用更小 patch 甚至 $p=1$？

### 17.1 Space-to-channel 是可逆重排

factor $r$ 的 space-to-channel operation 为

$$
S_r:
\mathbb R^{B\times C\times H\times W}
\to
\mathbb R^{B\times Cr^2\times H/r\times W/r}.
$$

它只是 entry permutation，存在精确逆

$$
S_r^{-1}=\operatorname{channel\mbox{-}to\mbox{-}space}.
$$

DC-AE 在 down/up block 中加入这类 non-parametric shortcut，让 neural module 学 residual：

$$
y=S_r(x)+F_\theta(x)
$$

或对应的 upsampling 形式。shortcut 提供稳定的信息通路，但完整 autoencoder 仍可能因 channel matching、bottleneck 和 nonlinear layers 有损，不能把 $S_r$ 的可逆性推广到整个 $E,D$。

### 17.2 Decoupled high-resolution adaptation

高压缩 autoencoder 在低分辨率训练后直接推广到高分辨率，可能出现 generalization penalty。DC-AE 将 high-resolution latent adaptation 与较低分辨率 local refinement 分开，降低训练成本。这里“decoupled”描述 training strategy，不是概率独立性。

### 17.3 与 SANA 的系统组合

SANA 把高压缩 DC-AE、linear DiT、text encoder、training strategy 和 sampler 组合。其速度/质量结果是系统证据。若只写“$f=32$ 带来全部加速”，就遗漏了 latent channels、patch size、linear attention 和实现 kernel。

***

## 18. Patchify：从 feature grid 到 token sequence

给定

$$
x\in\mathbb R^{B\times C\times H\times W},
\qquad p\mid H,W,
$$

先 reshape 为

$$
\mathbb R^{B\times C\times(H/p)\times p\times(W/p)\times p},
$$

再 permute/flatten 得到

$$
P(x)
\in
\mathbb R^{B\times N\times(p^2C)},
\qquad N=HW/p^2.
$$

纯 patchify 是 entry permutation，因此

$$
P^{-1}(P(x))=x.
$$

但 Transformer 输入前通常还有

$$
h_0=P(x)W_{in}+b,
\qquad
W_{in}\in\mathbb R^{p^2C\times d}.
$$

若 $W_{in}$ 不 injective，这个 projection 不可逆。DiT 最终 head 预测每个 token 的 $p^2C_{out}$ 个值，再用几何 unpatchify 恢复 image-like output。不要把“patchify 可逆”误写成“整个 token encoder 可逆”。

***

## 19. U-ViT：保留 U 形信息流，不必保留空间金字塔

U-ViT 把 noisy image patches、time 和 condition 都表示为 token，并用 Transformer 处理。它从 U-Net 借来的核心不是 convolution/downsampling，而是 shallow-to-deep long skip。

设浅层和对应深层 token shape 都是

$$
h_s,h_d\in\mathbb R^{B\times N\times d}.
$$

U-ViT 拼接并投影：

$$
h'
=[h_d;h_s]W_{skip}+b,
\qquad
W_{skip}\in\mathbb R^{2d\times d}.
$$

token count $N$ 没有改变。它与 convolutional U-Net 都跨越网络深度运输 shallow features，却不共享同一个空间多尺度结构。官方实现保存浅层序列到 stack，再按对称顺序 pop；这个代码接口已在 [Transformer source note](https://arxiv.org/abs/2209.12152 "官方论文页面") 核验。

***

## 20. DiT：把 conditioning 设计变成可比较变量

DiT 在 latent patches 上使用标准 ViT blocks，并比较多种 condition injection：

| 方案                 | 条件如何进入                           | 主要计算影响                         |
| ------------------ | -------------------------------- | ------------------------------ |
| in-context tokens  | condition token 与 image token 拼接 | 增加 sequence length             |
| cross-attention    | image query attend condition K/V | 增加单独 attention sublayer        |
| adaptive LayerNorm | condition 生成 scale/shift         | 不增加 condition token pair table |
| adaLN-Zero         | 再增加 residual gates 并零初始化         | 改变 initialization/interface    |

令

$$
M(h;c,s,b)
=\operatorname{LN}(h)\odot[1+s(c)]+b(c).
$$

一个 DiT block 可写成

$$
h_1
=h+g_a(c)\odot
\operatorname{Attn}(M(h;c,s_a,b_a)),
$$

$$
h_2
=h_1+g_m(c)\odot
\operatorname{MLP}(M(h_1;c,s_m,b_m)).
$$

其中 $s_a,b_a,g_a,s_m,b_m,g_m$ 都来自 condition embedding 的一次 projection。

***

## 21. adaLN-Zero 为什么在初始化时是 identity

adaLN-Zero 将 modulation output layer 初始化为零，因此初始时

$$
s_a=b_a=g_a=s_m=b_m=g_m=0.
$$

代入第一条 residual：

$$
h_1
=h+0\odot\operatorname{Attn}(\operatorname{LN}(h))
=h.
$$

再代入第二条：

$$
h_2
=h_1+0\odot\operatorname{MLP}(\operatorname{LN}(h_1))
=h.
$$

所以每个 block 初始为 identity。这个结论与内部 attention/MLP 的随机权重无关，因为 residual branch 被 gate 截断。

但这只说明 initialization：训练第一步后 gate 可以非零，block 不再保持 identity。它也不同于 ControlNet 的 zero convolution：后者初始化一条外部 control residual path，adaLN-Zero 初始化 Transformer block 内的 condition-derived gates。

***

## 22. DiT scaling：相关曲线能说什么，不能说什么

对 token count $N$、width $d$、depth $L$，一个粗略 Transformer compute ledger 是

$$
\operatorname{Cost}
\approx
L\left(aNd^2+bN^2d\right),
$$

其中第一项概括 QKV/output projections 与 MLP，第二项是 attention interaction，$a,b$ 取决于实现细节。

DiT 在控制数据、objective 和训练设置的模型族中改变 depth、width、heads 与 patch size，观察到 model GFLOPs 与 FID 等指标的强相关。这支持：**在该受控家族里，扩大 backbone compute 是有效 scaling 方向。**

它不推出：

$$
\forall \mathcal S,
\quad
\operatorname{GFLOPs}\uparrow
\Longrightarrow
\operatorname{quality}\uparrow.
$$

至少还要区分：

1. parameter count；
2. per-forward model GFLOPs；
3. total training FLOPs；
4. sampling NFE 与 total sampling FLOPs；
5. hardware latency 与 memory。

DiT 论文还专门比较增加 model compute 与增加 sampling compute，后者没有在该实验中补偿较小模型。这个结论属于具体实验，不应推广为“更多采样步永远无效”。

***

## 23. PixArt：效率来自训练过程分解，不只来自 block

PixArt-alpha 面对的是 text-to-image DiT 训练成本。它把训练路线分解为 pixel dependency initialization、text--image alignment 与 aesthetic adaptation，并复用 pretrained components。backbone 使用 DiT-style blocks 与 text cross-attention。

这种分阶段设计的研究价值在于：大模型效率不仅取决于 forward architecture，还取决于初始化从哪里来、哪些能力在哪个阶段学习、数据如何排列。若将总 GPU-hours 的下降完全归因于某个 attention block，就忽略了 training curriculum 与 transfer。

因此 PixArt 在本章承担“系统训练分解”职责，不承担普适复杂度 theorem。

***

## 24. SiT：相同 backbone 下，objective 仍能改变结果

SiT 以 DiT backbone 为基础，在相同 model structure、parameter count 与 GFLOPs 下，系统研究 interpolant、prediction、continuous/discrete time 与 deterministic/stochastic sampling。它对本章最重要的意义不是再发明一个 Transformer，而是提供反事实边界：

$$
\mathcal B_{SiT}\approx\mathcal B_{DiT},
$$

但

$$
\mathcal O_{SiT}\neq\mathcal O_{DiT}
$$

仍可带来不同结果。因此“Transformer 更强”不能解释同 backbone 的全部差异。

SiT 的 interpolant、velocity/score 关系和 sampler 属于 D9。本章只保留 architecture control 这个结论，避免提前把 flow 路线塞入架构章节。

***

## 25. MMDiT：联合 attention，不共享全部权重

text-to-image 中，image latent 与 text embedding 是两种统计性质不同的 token。普通 cross-attention 常让 image query 读取固定 text K/V：

$$
Q_x=XW_Q^x,
\qquad
K_c=CW_K^c,
\qquad
V_c=CW_V^c,
$$

$$
Y_x
=\operatorname{softmax}(Q_xK_c^\top/\sqrt d)V_c.
$$

text token 不在该层被 image 更新。MMDiT 则先用 modality-specific weights 产生两套 Q/K/V：

$$
(Q_x,K_x,V_x)=T_x(X),
$$

$$
(Q_c,K_c,V_c)=T_c(C),
$$

再沿 token axis 联合：

$$
Q=[Q_x;Q_c],
\qquad
K=[K_x;K_c],
\qquad
V=[V_x;V_c].
$$

joint attention 为

$$
Y
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V,
\qquad
Y=[Y_x;Y_c].
$$

attention matrix 可分成四块：

$$
QK^\top
=
\begin{bmatrix}
Q_xK_x^\top & Q_xK_c^\top\\
Q_cK_x^\top & Q_cK_c^\top
\end{bmatrix}.
$$

它同时包含 image--image、image--text、text--image 和 text--text 交互。attention 后再 split，两种 modality 可以继续使用各自的 output projection、normalization 与 MLP 参数。

![Cross-attention、MMDiT joint attention、kernel linear attention 与复杂度边界。](/images/diffusion/d8_attention_interfaces.png)

### 25.1 计算代价

若 image/text token 数分别是 $N_x,N_c$，joint matrix 是

$$
(N_x+N_c)\times(N_x+N_c).
$$

相比只做 $N_x\times N_c$ cross-attention，它还显式包含两个 self-interaction block。是否值得取决于 condition length、image token budget、width 与任务。

### 25.2 SD3 的 objective 不属于本节

MMDiT 与 rectified-flow training 同时出现在 SD3 technical report。前者是 backbone，后者改变 path/objective。D9 会讨论 rectified flow；本章不能因为它们同属一个系统就把两者写成一个“架构公式”。

***

## 26. Linear attention：结合律省掉哪张矩阵

softmax attention 必须形成或等价处理 pairwise logits。kernel attention 选非负 feature map $\varphi$，定义

$$
y_i
=
\frac{
\sum_j
\varphi(q_i)^\top\varphi(k_j)v_j
}{
\sum_j
\varphi(q_i)^\top\varphi(k_j)
}.
$$

利用结合律，numerator 为

$$
\sum_j
\varphi(q_i)^\top\varphi(k_j)v_j
=
\varphi(q_i)^\top
\left(
\sum_j\varphi(k_j)v_j^\top
\right),
$$

denominator 为

$$
\sum_j
\varphi(q_i)^\top\varphi(k_j)
=
\varphi(q_i)^\top
\left(
\sum_j\varphi(k_j)
\right).
$$

因此可以先计算

$$
S_{KV}=\sum_j\varphi(k_j)v_j^\top,
\qquad
s_K=\sum_j\varphi(k_j),
$$

再对每个 query 计算

$$
y_i
=\frac{\varphi(q_i)^\top S_{KV}}
{\varphi(q_i)^\top s_K}.
$$

当 query/key/value width 都是 $O(d)$，leading cost 约为 $O(Nd^2)$，避免 $N\times N$ storage。

### 26.1 它不等于 softmax attention

以上两种计算顺序对**同一个 kernel attention**完全等价；一般并不满足

$$
\varphi(q)^\top\varphi(k)
=\exp(q^\top k/\sqrt d).
$$

所以不能写“linear attention 只是 softmax 的无损快速实现”。更换 kernel 改变了 function family。

### 26.2 渐近复杂度不等于实际速度

$O(Nd^2)$ 只有在 $N$ 相对 $d$ 足够大、kernel 实现高效且其他模块不占主导时才显示优势。SANA 还加入 local Mix-FFN/convolution、RoPE、normalization 与 Triton kernels，以补充 linear attention 的局部建模和硬件效率。其完整结果是这些组件与 DC-AE、text encoder、objective、sampler 的组合。

***

## 27. REPA：让 denoiser 借用现成视觉表征

Diffusion Transformer 不只学习生成 vector field，hidden states 还会形成视觉 representation。REPA 的问题是：如果这些 representation 相比 DINOv2 等 self-supervised encoder 更弱，denoiser 是否在重复花费容量学习语义？

令 clean image $x^*$ 经 frozen teacher encoder 得到 patch targets：

$$
y^*=F_{teacher}(x^*).
$$

noisy latent $z_t$ 经 diffusion transformer 的中间层得到

$$
h_t=f_\theta(z_t,t,c),
$$

再用 trainable projection head 映射：

$$
\tilde y_t=h_\phi(h_t).
$$

REPA 最大化 patch-wise similarity。使用 negative cosine 时，

$$
\mathcal L_{REPA}(\theta,\phi)
=-
\mathbb E_{x^*,\epsilon,t}
\left[
\frac{1}{N}\sum_{i=1}^N
\frac{(y_i^*)^\top\tilde y_{t,i}}
{\|y_i^*\|_2\|\tilde y_{t,i}\|_2}
\right].
$$

总 objective 为

$$
\mathcal L
=\mathcal L_{generation}
+\lambda\mathcal L_{REPA}.
$$

官方实现对 projected feature normalize 后计算 patch inner product，并把 projection loss 加到 generative loss。teacher 只在训练时提供 target，推理不必运行 teacher。

![REPA 的 clean-teacher/noisy-student 训练路径，以及 cosine loss 的 norm-scale 边界。](/images/diffusion/d8_repa_alignment.png)

***

## 28. REPA 证明了什么，没有证明什么

### 28.1 Cosine alignment 不约束 norm

对正标量 $a,b>0$，

$$
-\frac{(ay)^\top(b\tilde y)}
{\|ay\|_2\|b\tilde y\|_2}
=
-\frac{y^\top\tilde y}
{\|y\|_2\|\tilde y\|_2}.
$$

因此 negative cosine 对 feature norm scale 不敏感。图中 scale sweep 的理论 cosine deviation 恒为零，而 MSE 会显著变化。

### 28.2 Empirical improvement 不等于唯一因果识别

REPA 在论文实验中提高 alignment 并改善训练速度/生成指标。这支持“外部 visual feature 是有效 auxiliary signal”。但 auxiliary loss 同时改变：

- intermediate representation；
- gradient path；
- optimization conditioning；
- implicit regularization。

仅从共同改善不能推出

$$
\text{representation gap}
=\text{the unique bottleneck}.
$$

### 28.3 Teacher 带来自己的偏置

DINOv2、MAE、MoCo 等 teacher 学到不同 invariance。对齐会把这些偏置引入 generator。对 ImageNet classification 有利的 representation 未必对精确文字渲染、细粒度纹理、医学信号或物理场最优。

### 28.4 理论仍是开放问题

原论文把 theoretical analysis、time-varying alignment 和更多 domain 留作未来工作。因此本教程不补造“REPA 加速收敛”的普适 theorem。完整公式与边界见 [REPA source note](https://arxiv.org/abs/2410.06940 "官方论文页面")。

***

## 29. 统一比较：每条路线把什么交给了谁

| 路线               | denoiser state                      | 主要信息通路                                      | 计算优势                            | 主要风险                                 |
| ---------------- | ----------------------------------- | ------------------------------------------- | ------------------------------- | ------------------------------------ |
| pixel U-Net      | pixels                              | multiscale conv + lateral skip              | 强局部 inductive bias              | high-res activation                  |
| cascade          | multi-resolution pixels             | base -> conditional upsamplers              | 分摊 resolution                   | train-test mismatch、multi-model cost |
| latent U-Net     | AE latent                           | encoder -> U-Net -> decoder                 | 大幅减少 spatial state              | lossy reconstruction、decoder bias    |
| U-ViT            | latent/image patches                | constant-grid Transformer + long skip       | global token mixing             | dense pair cost                      |
| DiT/SiT          | latent patches                      | ViT + adaptive norm                         | regular scaling interface       | token budget、data/compute demand     |
| MMDiT            | image + text tokens                 | modality-specific weights + joint attention | bidirectional multimodal mixing | larger joint matrix/system coupling  |
| SANA-style       | high-compression latent             | linear DiT + local blocks                   | lower token/interaction cost    | kernel/AE/implementation trade-off   |
| REPA-trained DiT | noisy latent + clean teacher target | auxiliary feature alignment                 | training signal reuse           | teacher bias、causal ambiguity        |

这张表不是“模型选择排行榜”。一个研究项目至少应先回答：

1. 输出需要精确保留哪些信息？
2. 最大限制是 memory、training FLOPs、latency 还是数据？
3. condition 是短类别、长文本还是多模态序列？
4. first-stage reconstruction error 是否可接受？
5. 比较时哪些七元组坐标真的被控制？

架构选择是约束下的优化，不是沿时间线无条件替换旧方法。

***

## 30. 说明代码：把论文接口缩成可执行不变量

d8\_architecture\_representation.py（补充材料暂未公开） 不训练网络，也不下载权重。它用 NumPy 实现以下最小接口：

| code component                      | 对应结论                                | 检查                        |
| ----------------------------------- | ----------------------------------- | ------------------------- |
| `sinusoidal_embedding`              | frequency-pair time encoding        | $\|e(t)\|^2=d/2$          |
| `patchify/unpatchify`               | patch 是 entry rearrangement         | exact round trip          |
| `space_to_channel/channel_to_space` | DC-AE non-parametric shortcut       | exact round trip          |
| zero-gated residual                 | adaLN-Zero identity init            | output equals input       |
| `joint_attention`                   | separate projections + joint matrix | image/text shape、row sum  |
| kernel attention two orders         | linear reassociation                | direct = associative      |
| `negative_cosine`                   | direction not norm                  | positive-scale invariance |
| compression calculator              | $f,c,p$ arithmetic                  | token/pair/element ratios |

固定 seed 的核心结果为：

$$
\text{patch round-trip error}=0,
\qquad
\text{space/channel round-trip error}=0,
$$

$$
\text{adaLN-Zero identity error}=0,
$$

$$
\text{joint-attention row-sum error}
\approx2.22\times10^{-16},
$$

$$
\text{linear-attention associativity error}
\approx2.22\times10^{-16},
$$

$$
\text{cosine scale-invariance error}
\approx4.16\times10^{-17}.
$$

五张图由同一脚本生成，因此图中的 token 数、pair ratio 与 algebraic checks 共用一套实现。

***

## 31. 常见错误：架构章节最容易偷换哪些概念

### 错误 1：U-Net 首次提出了 diffusion model

U-Net 2015 是 segmentation architecture；DDPM 等后来工作把它改造成 time-conditioned denoiser。

### 错误 2：所有 skip connection 都等价

concat 与 add 有不同 channel contract；U-ViT token long skip 也不同于 spatial U-Net skip。

### 错误 3：time embedding 定义了 noise schedule

embedding 编码传入的 $t$；schedule 定义 $t$ 对应什么噪声状态。二者必须对齐，但不是同一个对象。

### 错误 4：attention 加进 U-Net 就等于 DiT

U-Net 仍有显式 multiscale spatial hierarchy；DiT 的基本 state 是 patch token sequence。

### 错误 5：cascade 只是把同一个模型跑三次

不同 stage 学不同 conditional distributions，通常有不同 resolution、architecture 与训练数据接口。

### 错误 6：conditioning augmentation 精确匹配上游误差

它扩大训练 condition support，并在实验中提高 robustness；一般不等于未知 $p_{base}$ 的 exact error law。

### 错误 7：latent diffusion 是可逆 change of variables

perceptual autoencoder 通常有损降维；生成是 decoder pushforward，不是普通 flow likelihood formula。

### 错误 8：$f=8$ 就表示压缩 $64$ 倍

$64$ 是 spatial area ratio；element ratio 还取决于 latent channels $c$，token ratio还取决于 patch $p$。

### 错误 9：rFID 低就保证 generation FID 低

rFID 只隔离 first-stage reconstruction；generation 还包含 latent model error。

### 错误 10：patchify 本身丢信息

non-overlapping patch rearrangement 可逆；learned projection、bottleneck 和后续网络才可能丢信息。

### 错误 11：adaLN-Zero 让训练后的 block 恒为 identity

它只保证初始化 residual gate 为零；训练后 gate 会改变。

### 错误 12：DiT scaling curve 是普适 scaling theorem

它是受控 model family 的 empirical correlation，不能覆盖所有数据、objective 和 compute regime。

### 错误 13：SiT 的提升证明 SiT 架构优于 DiT

SiT 刻意保持近似相同 backbone，主要改变 interpolant/objective/sampling axis。

### 错误 14：MMDiT 就是多加一层 cross-attention

MMDiT 构造 joint image/text attention matrix，同时保留 modality-specific parameters。

### 错误 15：linear attention 是 softmax attention 的精确快速版

结合律只保证同一 kernel attention 的两种计算顺序等价；kernel 一般不等于 softmax exponential kernel。

### 错误 16：$O(Nd^2)$ 必然比 $O(N^2d)$ 快

实际 crossover 依赖 $N,d$、常数、kernel、memory、hardware 和其他网络模块。

### 错误 17：REPA 证明 semantic representation 是唯一瓶颈

它提供有效 auxiliary signal 的经验结果，同时改变 optimization/regularization，未完成唯一因果识别。

### 错误 18：SANA/DC-AE 的速度只来自 compression ratio

系统还改变 channels、patching、linear/local blocks、text encoder、training、sampler 和 implementation。

***

## 32. 章节小结

1. Denoiser 的 input/time/condition/target/output contract 应先于 backbone 名字固定。
2. U-Net 用 multiscale hierarchy 与 lateral skip 在 global context 和 local detail 之间分工。
3. DDPM/ADM 加入 time embedding、attention 与 adaptive normalization，把 U-Net 变成多噪声 denoiser。
4. Cascade 把高分辨率分给多个 conditional models，但引入上游 condition shift；conditioning augmentation 是 robustness 方案，不是 exact error marginalization。
5. Latent diffusion 把感知压缩交给 autoencoder，学习 decoder pushforward 所需的 latent distribution。
6. spatial factor $f$、latent channels $c$、patch side $p$ 分别决定空间、元素和 token budget。
7. patchify 是可逆重排，learned projection 和 lossy autoencoder 不是。
8. U-ViT 保留 long skip information route；DiT 用 patch tokens 和 adaptive normalization 建立 regular scaling interface。
9. adaLN-Zero 的严格结论是 initialization identity，不是性能定理。
10. model GFLOPs、training FLOPs、sampling FLOPs、NFE 和 latency 必须分别报告。
11. MMDiT 用 modality-specific weights 形成 bidirectional joint attention；linear attention 用 kernel associativity 避免 $N^2$ matrix，但不等于 softmax。
12. REPA 说明 pretrained visual features 能成为有效训练信号；它没有证明唯一 bottleneck，也没有消除 teacher bias。
13. SDXL、PixArt、SANA 等现代结果来自完整系统，architecture、objective、data 与 compute 不应混为单一原因。

***

## 33. 研究式思考题

1. 若两个 U-Net 的 parameter count 相同，但 channel 分配到不同 resolution，如何构造比总 FLOPs 更有解释力的 compute ledger？
2. Conditioning augmentation 能否被写成 distributionally robust optimization？uncertainty set 应如何从上游 generator error 估计？
3. 对 cascade 的每一级分别测量 calibration 与 support coverage，能否预测 error compounding，而不仅依赖最终 FID？
4. 在有损 decoder 下，latent score error 如何通过 $D_\psi$ 的 Jacobian、曲率与非 injectivity 影响 image distribution？
5. 除 rFID 外，什么 first-stage metric更能预测 downstream diffusion 的生成难度？是否需要同时测 reconstruction 与 latent geometry？
6. 固定 effective stride $fp$，改变 $f$ 与 $p$ 会保持 token 数不变，却如何改变 autoencoder 与 denoiser 的职责？
7. 当 $N$ 很小而 $d$ 很大时，DiT block 的主要开销可能从 attention 转向 MLP。架构 scaling 应如何随 compression 调整 width/depth？
8. adaLN-Zero 的 identity initialization 如何影响 early optimization dynamics？能否建立不依赖 FID 的 conditioning/gradient 分析？
9. MMDiT 的四个 attention blocks 是否都必要？如何在控制参数和 compute 下消融 image--image、text--text 与双向 cross blocks？
10. 哪类 kernel feature map 能在 diffusion hidden-state distribution 上逼近 softmax attention，同时保持数值稳定和线性复杂度？
11. REPA 的最佳 teacher 是否应该与 noise level $t$ 变化？高噪声时对齐 global semantics、低噪声时对齐 local texture会发生什么？
12. 如何设计实验，区分 REPA 的收益来自 semantic target、额外 gradient、regularization 还是 optimization conditioning？
13. 对医学图像、遥感或物理场，高压缩 autoencoder 应保留“感知真实”还是任务守恒量？如何把守恒约束放进第一阶段？
14. 如果把 architecture、objective、data、training compute、sampler 和 guidance 都视为 design axes，什么样的 benchmark 才能支持可复现的因果结论，而不只是系统排名？
