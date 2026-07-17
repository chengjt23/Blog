---
title: 少步生成与 Flow 转向：从局部速度到一步映射
description: 拆解 distillation、flow matching、rectified flow、consistency、flow map 与一步生成的学习对象。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 9
slug: d9-few-step-flow
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
scope: 比较局部场、离散转移、端点映射和分布蒸馏接口，明确少步方法并非只是在原 solver 上减步。
---
Diffusion 最直接的速度瓶颈是采样：一个网络已经训练好，却要被调用几十次甚至上百次。D6 的自然回答是设计更好的数值求解器；本章追问更激进的问题：

> 如果只允许一次或少数几次网络调用，我们究竟应该继续逼近原来的局部微分方程，还是直接学习一个不同的数学对象？

Progressive Distillation、Flow Matching、Rectified Flow、Consistency Models、Flow Map Matching、Shortcut Models、MeanFlow、DMD 和 ADD 都曾被放进“加速 Diffusion”这个大口袋。但它们并不只是在同一算法上减少 `steps`：

- solver 仍消费一个**局部速度场**；
- progressive distillation 学习一个更大的**教师离散转移**；
- consistency model 学习沿轨迹不变的**端点映射**；
- flow-map 方法学习任意两时刻间的**有限映射**；
- Shortcut/MeanFlow 学习带区间条件的**平均速度**；
- DMD/ADD 直接约束**输出分布**，不要求复现唯一教师轨迹。

本章的目标不是给方法排榜，而是把这些对象、目标和理论边界彻底拆开。

***

## 1. 本章先固定两套时间方向

Diffusion 与 Flow Matching 文献常把下标 `0` 用成相反含义。若不先分开，后面的“正向”“反向”和 velocity 符号必然混乱。

### 1.1 Flow 路线

Flow Matching、Rectified Flow 与 stochastic interpolant 段落使用

$$
\tau\in[0,1],\qquad
Z_0\sim p_{\mathrm{base}},\qquad
Z_1\sim p_{\mathrm{data}}.
$$

生成方向是 $\tau:0\to1$。

### 1.2 Diffusion / consistency 路线

Consistency 与 distillation 段落保留 diffusion noise-time 约定

$$
t\in[\epsilon,T],\qquad
x_\epsilon\approx x_{\mathrm{data}},\qquad
x_T\approx x_{\mathrm{noise}}.
$$

生成方向是 $T\to\epsilon$。这里 $\epsilon>0$ 是 data-side cutoff，不是高斯噪声变量；随机噪声写作 $\xi$。

### 1.3 一般 flow map

当不关心哪一端是 data 时，统一写

$$
\Phi_{s,t}(x_s)=x_t.
$$

$\Phi_{s,t}$ 只表示“从时刻 $s$ 到 $t$”的映射，允许 $s<t$ 或 $s>t$。

***

## 2. 五种函数接口：一步生成争论的真正核心

考虑 ODE

$$
\frac{d x_t}{dt}=b_t(x_t).
$$

围绕它至少有五个不同对象。

| 对象             | 记号              | 输入          | 输出              | 是否天然一步        |
| -------------- | --------------- | ----------- | --------------- | ------------- |
| 局部速度           | $b_t(x)$        | 当前状态、单个时间   | 瞬时切线            | 否             |
| 数值一步           | $\Psi_h(x,t;b)$ | 当前状态、步长、局部场 | 近似下一状态          | 只是一阶/高阶近似     |
| 端点 consistency | $f(x_t,t)$      | 当前状态、单个时间   | 固定 data-side 端点 | 是，但只针对一个端点    |
| 两时刻 flow map   | $\Phi_{s,t}(x)$ | 状态、起止两时刻    | 精确/学习的有限转移      | 是             |
| 平均速度           | $u(x,s,t)$      | 状态、起止两时刻    | 单位时间有限位移        | 配合 $x+(t-s)u$ |

它们的关系是

$$
\Phi_{s,t}(x)
=x+(t-s)u(x,s,t),
$$

以及在足够光滑时

$$
\lim_{t\to s}u(x,s,t)=b_s(x).
$$

极限相等不表示有限区间相等。一步生成的许多争论，本质上都在问：网络应该输出左边的局部极限，还是右边的有限积分量？

***

## 3. 为什么一个更好的 solver 仍不等于一步映射

Euler 一步为

$$
\Psi_h^{\mathrm{Euler}}(x,t)
=x+h\,b_t(x).
$$

精确 flow map 则是

$$
\Phi_{t,t+h}(x)
=x+\int_t^{t+h}b_\tau(x_\tau)\,d\tau.
$$

两者之差来自整段轨迹上的场变化，而不是只来自时间标签：

$$
\Phi_{t,t+h}(x)-\Psi_h^{\mathrm{Euler}}(x,t)
=
\int_t^{t+h}
\left[b_\tau(x_\tau)-b_t(x)\right]d\tau.
$$

只有当场在该轨迹段上近似常量，或 $h$ 很小时，一步局部切线才可靠。

### 3.1 可解析旋转反例

令

$$
\dot x_t=\omega Jx_t,\qquad
J=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix}.
$$

精确映射是旋转

$$
\Phi_{s,t}(x)=R_{\omega(t-s)}x,
$$

而一步 Euler 为

$$
x+(t-s)\omega Jx.
$$

前者保持半径，后者把半径乘成

$$
\sqrt{1+\omega^2(t-s)^2}.
$$

步长再大，局部切线再准确，也不能自动变成圆弧端点。

![局部速度、数值积分与有限 flow map](/images/diffusion/d9_local_field_vs_flow_map.png)

**图 3.1：** 左图用旋转场展示“一步切线不等于有限映射”；右图把 solver error 与 exact-map oracle 分成两条轴。图中的 oracle 不是一个现实神经网络，只表示“若直接知道有限映射，单次调用可到达什么对象”。说明代码中 $\omega=1.2$ 时，一步 Euler endpoint error 为 $1.314$；增加到 128 步后 Euler error 为 $1.072\times10^{-2}$，Heun 以 256 NFE 达到 $3.340\times10^{-5}$。

### 3.2 D6 与 D9 的边界

[D6](/blog/diffusion/d6-sampling-solvers/)研究固定 learned field 下的离散误差、稳定性、NFE 和 grid。D9 研究的是：

$$
\text{learn a different finite-transition object}.
$$

高阶 solver 可以减少 exact field 下的 truncation error，却不会消除 field estimation error，也不会凭一次 evaluation 获得未查询轨迹上的全部信息。

***

## 4. 技术演进不是一条线，而是 learned object 的分叉

![少步生成方法按学习对象排列的时间线](/images/diffusion/d9_method_timeline_interfaces.png)

**图 4.1：** 横轴是首个公开版本附近的历史位置，纵轴是网络学习的数学接口。节点的上下位置不是质量排名。同一年出现的方法可能共享实现，却优化不同对象。

问题链可以压缩为：

1. **多步太慢**：progressive distillation 压缩教师离散轨迹；
2. **边缘速度难监督**：Flow Matching 用 conditional path target；
3. **路径太弯**：Rectified Flow 把 coupling 与 straight interpolation 提到中心；
4. **能否直接回端点**：Consistency Models 学习 trajectory-invariant endpoint map；
5. **能否跳任意区间**：flow-map/shortcut 方法显式输入两个时间或 step size；
6. **能否从局部信号学习平均速度**：MeanFlow 使用 total-derivative identity；
7. **是否必须匹配路径**：DMD/ADD 改为 distribution/adversarial objective。

这些方案解决不同问题，不能仅按“一步 FID”折成同一类别。

***

## 5. 从随机插值到 continuity equation

Flow Matching 的核心不是“把 SDE 改成 ODE”这句口号，而是一个条件期望恒等式。

令 $Z_\tau$ 是可微随机路径，其密度为 $p_\tau$。定义 Eulerian marginal velocity

$$
b_\tau(z)
:=
\mathbb E[\dot Z_\tau\mid Z_\tau=z].
$$

取任意光滑紧支撑测试函数 $\varphi$，则

$$
\frac{d}{d\tau}\mathbb E[\varphi(Z_\tau)]
=
\mathbb E[
\nabla\varphi(Z_\tau)^\top\dot Z_\tau
].
$$

对 $Z_\tau$ 条件化：

$$
\frac{d}{d\tau}
\int \varphi(z)p_\tau(z)\,dz
=
\int
\nabla\varphi(z)^\top b_\tau(z)p_\tau(z)\,dz.
$$

若边界项消失，分部积分给出

$$
\int \varphi(z)
\left[
\partial_\tau p_\tau(z)
+\nabla\cdot(p_\tau(z)b_\tau(z))
\right]dz
=0.
$$

因此在弱意义下

$$
\boxed{
\partial_\tau p_\tau
+\nabla\cdot(p_\tau b_\tau)=0.
}
$$

这说明：只要能学习 $b_\tau$，ODE

$$
\dot z_\tau=b_\tau(z_\tau)
$$

就能实现同一条 one-time marginal curve。这里还没有说路径 law、endpoint coupling 或 OT optimality。

***

## 6. Flow Matching 的直接目标为什么不可用

理想 marginal Flow Matching objective 是

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{\tau\sim q(\tau)}
\mathbb E_{Z_\tau\sim p_\tau}
\left[
w(\tau)
\|v_\theta(Z_\tau,\tau)-b_\tau(Z_\tau)\|^2
\right].
$$

困难在于 $b_\tau(z)$ 常是一个对 endpoints/latent variables 积分后的 mixture field。即使容易采样一条随机插值，也不一定能在给定 $z$ 处计算

$$
\mathbb E[\dot Z_\tau\mid Z_\tau=z].
$$

Flow Matching 的突破是：不先显式算这个条件期望，而是直接回归产生它的随机 target。

***

## 7. Conditional Flow Matching：完整 Pythagorean 推导

令 $C$ 表示 conditional path 所需的随机变量，例如 endpoint $Z_1$、source $Z_0$ 或 latent Gaussian。设

$$
Y_\tau:=\dot Z_\tau,
\qquad
b_\tau(Z_\tau)=\mathbb E[Y_\tau\mid Z_\tau].
$$

Conditional Flow Matching 使用

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E
\|Y_\tau-v_\theta(Z_\tau,\tau)\|^2.
$$

加减 marginal velocity：

$$
Y_\tau-v_\theta
=
(Y_\tau-b_\tau)
+(b_\tau-v_\theta).
$$

平方展开：

$$
\begin{aligned}
\mathcal L_{\mathrm{CFM}}
={}&
\mathbb E\|Y_\tau-b_\tau\|^2
+\mathbb E\|b_\tau-v_\theta\|^2\\
&+
2\mathbb E[
(Y_\tau-b_\tau)^\top
(b_\tau-v_\theta)
].
\end{aligned}
$$

交叉项为

$$
\begin{aligned}
&\mathbb E\left[
\mathbb E[
Y_\tau-b_\tau(Z_\tau)
\mid Z_\tau
]^\top
(b_\tau-v_\theta)
\right]\\
&=0.
\end{aligned}
$$

所以

$$
\boxed{
\mathcal L_{\mathrm{CFM}}
=
\underbrace{
\mathbb E\|Y_\tau-b_\tau(Z_\tau)\|^2
}_{\text{与 }\theta\text{ 无关}}
+
\mathcal L_{\mathrm{FM}}.
}
$$

在可交换梯度与期望的条件下，

$$
\nabla_\theta\mathcal L_{\mathrm{CFM}}
=
\nabla_\theta\mathcal L_{\mathrm{FM}}.
$$

这正是 [Flow Matching source note](https://arxiv.org/abs/2209.03003 "官方论文页面") 中 Theorem 2 的核心结构。

### 7.1 这个等价没有说什么

它没有保证：

- 两个 finite-batch estimator 的方差相同；
- 任意 $q(\tau)$ 或 $w(\tau)$ 都等价；
- finite network 达到 population optimum；
- 学到局部场后一步 Euler 就正确；
- conditional coupling 自动是 global OT coupling。

***

## 8. Affine interpolant：路径、target 与参数化

最常见的构造是

$$
Z_\tau=a(\tau)Z_1+b(\tau)Z_0.
$$

sample-wise velocity target 为

$$
\dot Z_\tau
=
\dot a(\tau)Z_1+\dot b(\tau)Z_0.
$$

当

$$
\Delta_\tau
=
a(\tau)\dot b(\tau)-b(\tau)\dot a(\tau)\ne0,
$$

$(Z_\tau,\dot Z_\tau)$ 与 $(Z_1,Z_0)$ 之间是可逆的二维线性变换：

$$
\begin{bmatrix}
Z_\tau\\
\dot Z_\tau
\end{bmatrix}
=
\begin{bmatrix}
a&b\\
\dot a&\dot b
\end{bmatrix}
\begin{bmatrix}
Z_1\\
Z_0
\end{bmatrix}.
$$

因此 data、noise 与 velocity prediction 在非奇异位置可互相换算。这与 [D5](/blog/diffusion/d5-parameterization-training-design/) 的 affine target algebra 完全兼容。

但要注意三个层次：

1. target 可逆不表示 MSE weighting 相同；
2. 坐标变换不表示网络 optimization conditioning 相同；
3. 一般 affine velocity 不等于 VP 文献中对 angle $\phi$ 的 $v$-prediction，除非时间参数也一致。

### 8.1 线性插值

取

$$
a(\tau)=\tau,\qquad b(\tau)=1-\tau,
$$

则

$$
Z_\tau=(1-\tau)Z_0+\tau Z_1,
\qquad
\dot Z_\tau=Z_1-Z_0.
$$

target 与 $\tau$ 无关，但 marginal velocity 仍依赖 $(z,\tau)$，因为条件分布

$$
p(Z_0,Z_1\mid Z_\tau=z)
$$

随位置和时间变化。

***

## 9. Gaussian mixture 例子：随机 target 如何平均成 marginal field

令

$$
Z_0\sim\mathcal N(0,\sigma_0^2I),
$$

而

$$
Z_1\mid K=k
\sim
\mathcal N(\mu_k,\sigma_1^2I),
\qquad
\Pr(K=k)=\pi_k.
$$

在线性插值下，

$$
Z_\tau\mid K=k
\sim
\mathcal N(
\tau\mu_k,
s_\tau^2I
),
$$

其中

$$
s_\tau^2
=(1-\tau)^2\sigma_0^2+\tau^2\sigma_1^2.
$$

定义 target $Y=Z_1-Z_0$。条件于 component $k$，联合 Gaussian conditioning 给出

$$
\mathbb E[Y\mid Z_\tau=z,K=k]
=
\mu_k
+
\frac{
\tau\sigma_1^2-(1-\tau)\sigma_0^2
}{
s_\tau^2
}
(z-\tau\mu_k).
$$

component posterior responsibility 为

$$
r_k(z,\tau)
=
\frac{
\pi_k\,
\mathcal N(z;\tau\mu_k,s_\tau^2I)
}{
\sum_j
\pi_j\,
\mathcal N(z;\tau\mu_j,s_\tau^2I)
}.
$$

最终 marginal velocity 是

$$
\boxed{
b_\tau(z)
=
\sum_k
r_k(z,\tau)
\mathbb E[Y\mid Z_\tau=z,K=k].
}
$$

![CFM 随机条件 target 与 marginal velocity](/images/diffusion/d9_conditional_flow_matching.png)

**图 9.1：** 左图每条 line/arrow 来自可直接采样的 endpoint target；右图是它们在固定 $Z_\tau=z$ 下的条件平均。两者不是两种互斥算法，而是 noisy regression target 与 population optimum 的关系。

说明代码用 50 万样本检查条件残差与任意 $Z_\tau$-可测函数的近似正交性，误差为 $1.000\times10^{-3}$。

***

## 10. Flow Matching 仍是局部场学习

一个最小训练循环是：

```python
for x_data in loader:
    z0 = sample_base(x_data.shape)
    tau = sample_time(x_data.shape[0])
    z_tau = a(tau) * x_data + b(tau) * z0
    target = da(tau) * x_data + db(tau) * z0
    loss = mean_square(model(z_tau, tau) - target)
    update(loss)
```

采样仍需：

```python
z = sample_base()
for tau_s, tau_t in time_grid:
    z = ode_step(model, z, tau_s, tau_t)
```

官方 Flow Matching code（补充材料暂未公开） 的 `AffineProbPath.sample` 返回 `x_t` 与 `dx_t`；ODESolver（补充材料暂未公开） 仍反复查询 local velocity。

所以“用 Flow Matching 训练”与“一步生成”是两件事。后者还需要路径足够直、显式 distillation，或直接学习 finite map。

***

## 11. Rectified Flow：关键不是公式线性，而是 coupling

Rectified Flow 从 endpoint coupling

$$
(Z_0,Z_1)\sim\pi
$$

出发，定义

$$
Z_\tau=(1-\tau)Z_0+\tau Z_1
$$

并最小化

$$
\mathcal L_{\mathrm{RF}}(v)
=
\int_0^1
\mathbb E_\pi
\left[
\|Z_1-Z_0-v(Z_\tau,\tau)\|^2
\right]d\tau.
$$

population minimizer 是

$$
\boxed{
v^*(z,\tau)
=
\mathbb E_\pi[
Z_1-Z_0\mid Z_\tau=z
].
}
$$

从公式看，它是 linear-path CFM；Rectified Flow 的研究重点则是：

- endpoint coupling 如何决定 line crossings；
- conditional expectation 如何把非 causal 直线集合变成 causal ODE；
- induced coupling 的 transport cost 如何变化；
- 如何用 reflow 让当前模型 coupling 的轨迹更直。

***

## 12. “边缘保持”证明与 sample-line 误区

随机直线 $Z_\tau$ 的 density 为 $p_\tau$。由第 5 节，

$$
v^*(z,\tau)
=
\mathbb E[\dot Z_\tau\mid Z_\tau=z]
$$

满足

$$
\partial_\tau p_\tau
+\nabla\cdot(p_\tau v^*)=0.
$$

因此在 regularity/uniqueness 条件下，ODE

$$
\dot Y_\tau=v^*(Y_\tau,\tau)
$$

与随机线性插值共享 one-time marginals。

但一般没有

$$
Y_\tau=(1-\tau)Z_0+\tau Z_1
$$

逐样本成立。原因是同一个 $(z,\tau)$ 可能被许多 endpoint lines 穿过；ODE 只能给出一个确定方向，于是取条件平均。

这可以叫“causalization”或“rewiring”：

$$
\text{non-causal endpoint line ensemble}
\longrightarrow
\text{single-valued Eulerian field}.
$$

***

## 13. Rectified Flow 与 OT：最容易被夸大的地方

样本级直线具有最短欧氏长度，不代表 marginal flow 已是 global optimal transport。

必须区分：

1. 每条 conditional interpolation 是直线；
2. endpoint coupling $\pi$ 是否是 OT coupling；
3. 条件平均后的 ODE induced coupling 是什么；
4. 选定 convex transport cost 下是否有单调改进；
5. 是否真的达到 Wasserstein optimum。

[Rectified Flow source note](https://arxiv.org/abs/2209.03003 "官方论文页面") 记录了论文的 convex-cost comparison；它不能被升级成

$$
\text{Rectified Flow always computes the OT map}.
$$

若初始 coupling 是 product coupling，直线集合可能高度交叉。直不等于好 coupling。

***

## 14. Reflow：先用当前模型生成 coupling，再重新拉直

设第一代 learned ODE flow map 为

$$
\widehat\Phi^{(1)}_{0,1}.
$$

采样

$$
Z_0\sim p_{\mathrm{base}},
\qquad
\widehat Z_1
=
\widehat\Phi^{(1)}_{0,1}(Z_0).
$$

这给出 model-induced coupling

$$
\pi^{(1)}
=
\operatorname{Law}(Z_0,\widehat Z_1).
$$

第二代模型不再用独立 endpoint pairing，而在

$$
Z_\tau^{(2)}
=(1-\tau)Z_0+\tau\widehat Z_1
$$

上回归

$$
\widehat Z_1-Z_0.
$$

![coupling、当前曲线与 reflow 训练对](/images/diffusion/d9_coupling_reflow.png)

**图 14.1：** 左图是任意 coupling 下相互交叉的直线；中图是当前 deterministic flow 的弯曲轨迹及其 induced endpoint pairs；右图是 reflow 用这些 pairs 重新构造的直线 target。示意例中随机 coupling 有 50 个 pairwise crossings，model-induced monotone coupling 为 0；当前曲线平均 bending energy 为 $1.574$，straight chord 为 0。

### 14.1 Reflow 不是免费的

它增加了：

- teacher ODE integration；
- endpoint-pair 生成与存储，或 online teacher compute；
- 新一轮训练；
- teacher coupling error；
- 再次回归的 approximation error。

官方 Rectified Flow code（补充材料暂未公开） 明确把 initial training、pair generation、reflow training 与 optional distillation 分成不同阶段。

因此“一步 Rectified Flow”通常不是“把原模型 Euler N 改成 1”。

***

## 15. Progressive Distillation：把两个教师步反解成一个学生 target

Progressive Distillation 在历史上早于 consistency/flow-map 热潮。它压缩的是确定性 diffusion sampler。

DDIM-style update 从 $t$ 到 $s<t$ 为

$$
z_s
=
\alpha_s\hat x(z_t,t)
+
\frac{\sigma_s}{\sigma_t}
\left[
z_t-\alpha_t\hat x(z_t,t)
\right].
$$

教师从 $t$ 做两个半步，得到

$$
z_{t''}^{\mathrm{teacher}},
\qquad
t''=t-\frac1N.
$$

学生要用一个大步到同一 endpoint。令它应预测的 clean target 为 $\widetilde x$，则

$$
z_{t''}^{\mathrm{teacher}}
=
\alpha_{t''}\widetilde x
+
\frac{\sigma_{t''}}{\sigma_t}
\left(
z_t-\alpha_t\widetilde x
\right).
$$

解出

$$
\boxed{
\widetilde x
=
\frac{
z_{t''}^{\mathrm{teacher}}
-(\sigma_{t''}/\sigma_t)z_t
}{
\alpha_{t''}
-(\sigma_{t''}/\sigma_t)\alpha_t
}.
}
$$

这不是两个 teacher predictions 的平均，而是**反演一次学生 DDIM update 所需的 target**。

训练完成后：

$$
\text{student}\to\text{new teacher},
\qquad
N\to N/2.
$$

不断重复，得到 $N,N/2,N/4,\ldots$。

### 15.1 它改变了什么

- solver 公式可不变；
- 学生 prediction function 已改变；
- 每一阶段的 teacher trajectory 也改变；
- 一步结果累计了多轮 teacher、target 和 optimization error。

[Progressive Distillation note](https://arxiv.org/abs/2202.00512 "官方论文页面") 与官方实现（补充材料暂未公开）给出完整 target/parameterization 映射。说明代码将反演 target 再代回一次 DDIM update，最大误差为 $6.661\times10^{-16}$。

***

## 16. Consistency Model：学习“沿轨迹不变的 data-side endpoint”

考虑 diffusion probability-flow ODE 的 exact flow map。定义

$$
f^*(x_t,t)
:=
\Phi_{t,\epsilon}(x_t).
$$

若 $x_s,x_t$ 属于同一条轨迹，则

$$
\Phi_{t,\epsilon}(x_t)
=
\Phi_{s,\epsilon}(x_s),
$$

所以

$$
\boxed{
f^*(x_t,t)=f^*(x_s,s).
}
$$

并且有 boundary condition

$$
\boxed{
f^*(x,\epsilon)=x.
}
$$

Consistency 的“一步”不是把局部 drift 积一次，而是让网络直接近似

$$
x_t\mapsto x_\epsilon.
$$

### 16.1 Boundary parameterization

常用结构是

$$
f_\theta(x,t)
=
c_{\mathrm{skip}}(t)x
+
c_{\mathrm{out}}(t)F_\theta(x,t),
$$

并要求

$$
c_{\mathrm{skip}}(\epsilon)=1,
\qquad
c_{\mathrm{out}}(\epsilon)=0.
$$

这样无论 $F_\theta$ 输出什么，

$$
f_\theta(x,\epsilon)=x.
$$

结构只保证边界，不保证任意 $s,t$ 的 global consistency。说明代码用简化

$$
c_{\mathrm{skip}}=1,\qquad
c_{\mathrm{out}}=t-\epsilon
$$

检查 boundary error 为 0。

***

## 17. Consistency Distillation：教师负责相邻轨迹对

设时间网格

$$
\epsilon=t_1<t_2<\cdots<t_N=T.
$$

从 $x_{t_{n+1}}$ 出发，teacher score/PF ODE solver 产生相邻较低噪声状态

$$
\widehat x_{t_n}^{\,\phi}
=
\Psi_{t_{n+1}\to t_n}
(x_{t_{n+1}};\phi).
$$

在线模型与 target/EMA 模型匹配：

$$
\mathcal L_{\mathrm{CD}}
=
\mathbb E
\left[
\lambda(t_n)
d\left(
f_\theta(x_{t_{n+1}},t_{n+1}),
f_{\theta^-}(
\widehat x_{t_n}^{\,\phi},
t_n
)
\right)
\right].
$$

`stopgrad` / EMA 阻止目标与在线网络同时追逐。若相邻一致性误差足够小，并能沿网格传播，就逼近全局 endpoint consistency。

### 17.1 Algorithm skeleton

```text
repeat
    sample data x and noise level index n
    sample x_{t_{n+1}}
    teacher_solver: x_{t_{n+1}} -> x_hat_{t_n}
    online  = f_theta(x_{t_{n+1}}, t_{n+1})
    target  = stopgrad(f_ema(x_hat_{t_n}, t_n))
    update theta using distance(online, target)
    update EMA target network
until convergence
```

[Consistency Models note](https://arxiv.org/abs/2202.00512 "官方论文页面") 对齐原文 p. 5 的 CD/CT algorithms 与 theorem boundary；PyTorch official code（补充材料暂未公开） 对齐 preconditioning、loss 和 sampler。

***

## 18. Consistency Training：没有 pretrained teacher，不等于没有教师结构

原始 CT 在 VE-style corruption 中可写成

$$
x_t=x+t\xi,
\qquad
\xi\sim\mathcal N(0,I).
$$

同一 $(x,\xi)$ 构造相邻 levels：

$$
x_{t_n}=x+t_n\xi,
\qquad
x_{t_{n+1}}=x+t_{n+1}\xi.
$$

训练

$$
d\left(
f_\theta(x_{t_{n+1}},t_{n+1}),
f_{\theta^-}(x_{t_n},t_n)
\right).
$$

这里没有 pretrained diffusion model，但仍有：

- chosen corruption path；
- shared clean/noise coupling；
- adjacent discretization；
- boundary parameterization；
- target/EMA network；
- $N\to\infty$ 的 population argument。

有限 $N$ 下，两个 noisy states 并不等于 exact PF ODE 的相邻状态。CT 的理论联系来自 denoising/score identity 与 infinitesimal limit，不能写成 finite-grid exact teacher-free theorem。

***

## 19. Improved CT、LCM 与 sCM：解决的是不同层次

### 19.1 Improved Consistency Training

2023 improved-training 工作重点调整：

- noise-level sampling；
- discretization schedule；
- target EMA；
- pseudo-Huber metric；
- loss weighting 与 curriculum。

这些改进处理 optimization/stability，不改变 consistency function 的定义。

### 19.2 Latent Consistency Models

LCM 把 CD 放进 pretrained latent diffusion 的 PF ODE。系统同时包含：

- latent autoencoder；
- teacher latent diffusion；
- text/guidance conditioning；
- solver-based adjacent target；
- target/EMA network；
- consistency boundary；
- 可选 LoRA adaptation。

LCM official training script（补充材料暂未公开） 显式包含 teacher、predicted-origin conversion、guidance-scale conditioning 与 boundary scalings。

因此“LCM 2--4 steps”是完整 latent system 的结果，不只由一个 consistency equation 决定。

### 19.3 Continuous-time consistency / sCM

若 exact consistency function 沿 ODE 轨迹恒定，则

$$
\frac{d}{dt}f(x_t,t)
=
\partial_t f(x_t,t)
+J_xf(x_t,t)b_t(x_t)
=0.
$$

sCM 把这类 continuous-time tangent condition 与 TrigFlow parameterization、稳定训练和 scaling 结合。

局部 PDE 加 boundary 可推出全轨迹 invariance；有限网络把局部 residual 训练到较小，并不自动获得 exact global endpoint map。

***

## 20. 从 endpoint consistency 到 two-time flow map

Exact flow map 定义为

$$
\Phi_{s,t}:\mathbb R^d\to\mathbb R^d,
\qquad
\Phi_{s,t}(x_s)=x_t.
$$

在 ODE 解存在且唯一时：

$$
\boxed{
\Phi_{s,s}=\operatorname{Id},
}
$$

$$
\boxed{
\Phi_{t,r}\circ\Phi_{s,t}
=
\Phi_{s,r},
}
$$

$$
\boxed{
\Phi_{s,t}^{-1}
=
\Phi_{t,s}.
}
$$

### 20.1 Composition proof

令

$$
y=\Phi_{s,t}(x).
$$

从 $y$ 在 $t$ 时刻继续解到 $r$，得到

$$
\Phi_{t,r}(y).
$$

这条拼接轨迹与“从 $x$ 在 $s$ 直接解到 $r$”满足同一 ODE 和同一初值。由唯一性，

$$
\Phi_{t,r}(\Phi_{s,t}(x))
=
\Phi_{s,r}(x).
$$

### 20.2 不要滥用 semigroup

对 time-homogeneous autonomous ODE，可以写

$$
T_{a+b}=T_b\circ T_a.
$$

一般 diffusion/flow 的 $b_t$ 显式依赖时间，正确对象是 two-parameter evolution family：

$$
\Phi_{s,t}\ne T_{t-s}
$$

通常成立。把所有 flow map 都叫 one-parameter semigroup 会丢失绝对时间条件。

***

## 21. Endpoint consistency 只是 flow map 的一个切片

Consistency endpoint 为

$$
f(x,t)=\Phi_{t,\epsilon}(x).
$$

它固定第二个时间为 $\epsilon$。完整 flow map 则允许

$$
\Phi_{s,t}(x)
\quad\text{for arbitrary }(s,t).
$$

两者能力不同：

| 查询               |    endpoint model $f(x,t)$ | two-time map $\Phi_{s,t}(x)$ |
| ---------------- | -------------------------: | ---------------------------: |
| noise $\to$ data |                         支持 |                           支持 |
| 任意中间 jump        |                      不直接支持 |                           支持 |
| map composition  |                   隐式依赖共同端点 |                           显式 |
| inverse          |                      需额外结构 |     exact map 下 $\Phi_{t,s}$ |
| 多种 step budget   | 可用 multistep sampling 间接实现 |                    由区间查询直接表达 |

若 $f_t=\Phi_{t,\epsilon}$ 可逆，理论上

$$
\Phi_{s,t}
=
f_t^{-1}\circ f_s.
$$

但神经 endpoint model 未必可逆，也未显式提供 $f_t^{-1}$。所以 endpoint consistency 不应直接改名为完整 flow map。

***

## 22. Average velocity：有限映射的一个稳定参数化

写

$$
\Phi_{s,t}(x)
=
x+(t-s)u(x,s,t).
$$

则

$$
u(x,s,t)
=
\frac{\Phi_{s,t}(x)-x}{t-s}.
$$

它是 finite displacement per unit time。

### 22.1 Local limit

若 flow 对时间可微，

$$
\begin{aligned}
\lim_{t\to s}u(x,s,t)
&=
\lim_{t\to s}
\frac{\Phi_{s,t}(x)-\Phi_{s,s}(x)}{t-s}\\
&=
\partial_t\Phi_{s,t}(x)\big|_{t=s}\\
&=
b_s(x).
\end{aligned}
$$

这给 local Flow Matching signal 与 finite map 之间的接口。

### 22.2 为什么要除以 $t-s$

直接预测 displacement

$$
\Delta(x,s,t)=\Phi_{s,t}(x)-x
$$

在 $t=s$ 时趋于 0。预测 $u=\Delta/(t-s)$ 则在 diagonal 上接到非退化 local velocity。代价是必须处理 $t\approx s$ 的数值稳定和 boundary parameterization。

***

## 23. Shortcut Models：把 composition 变成 bootstrap target

取 midpoint

$$
m=\frac{s+t}{2},
\qquad
x_m=\Phi_{s,m}(x).
$$

由 composition，

$$
\Phi_{s,t}(x)
=
\Phi_{m,t}(x_m).
$$

展开 average velocity：

$$
x+(t-s)u(x,s,t)
=
x_m+(t-m)u(x_m,m,t),
$$

而

$$
x_m=x+(m-s)u(x,s,m).
$$

所以

$$
\boxed{
u(x,s,t)
=
\frac{m-s}{t-s}u(x,s,m)
+
\frac{t-m}{t-s}u(x_m,m,t).
}
$$

在 midpoint 情况：

$$
\boxed{
u(x,s,t)
=
\frac12u(x,s,m)
+
\frac12u(x_m,m,t).
}
$$

Shortcut Models 用一个网络 $s_\theta(x,t,d)$：

- $d\approx0$ 时回归 local Flow Matching target；
- $2d$ 时用两个 $d$-shortcut 拼成 stop-gradient target。

```python
small_1 = shortcut(x, t, d)
x_mid = x + d * small_1
small_2 = shortcut(x_mid, t + d, d)
target_2d = stopgrad(0.5 * small_1 + 0.5 * small_2)
```

Shortcut official target（补充材料暂未公开） 对齐这一递归。

### 23.1 Bootstrap 的隐藏风险

第二次查询点

$$
\widehat x_m
=
x+d\,s_\theta(x,t,d)
$$

是模型生成的，不是 exact $x_m$。若第一半步有误差，第二半步 target 被 off-trajectory state 污染。误差会同时改变：

- target value；
- target input distribution；
- 后续更长 shortcut 的 bootstrap chain。

因此 composition identity 是 exact；用它训练 approximate neural maps 仍有 distribution shift。

***

## 24. Flow-map self-distillation：三种 off-diagonal 约束

[How to Build a Consistency Model](https://arxiv.org/abs/2406.07507 "官方论文页面") 把模型写成

$$
\widehat\Phi_{s,t}(x)
=
x+(t-s)\widehat u(x,s,t).
$$

训练分两部分：

1. diagonal/local loss：$s=t$ 附近与 known interpolant velocity 对齐；
2. off-diagonal self-distillation：让有限区间 map 满足 ODE 或 composition。

论文讨论三类目标。

### 24.1 Lagrangian self-distillation

沿 sample trajectory $I_s$ 要求 predicted endpoint 随起始时刻变化正确。示意 residual 为

$$
\partial_t\widehat\Phi_{s,t}(I_s)
-
b_t(
\widehat\Phi_{s,t}(I_s)
).
$$

### 24.2 Eulerian self-distillation

把 map 当作关于输入空间与时间的函数，约束 transport PDE。它需要 spatial derivatives/JVP，但避免显式小步 rollout。

### 24.3 Progressive self-distillation

直接匹配 composition：

$$
\widehat\Phi_{s,t}
\approx
\widehat\Phi_{m,t}\circ
\widehat\Phi_{s,m}.
$$

这与 shortcut bootstrap 最接近，也最容易积累 approximate-map distribution shift。

论文对部分 Lagrangian/Eulerian objectives 在 smoothness 与 small-loss 假设下给出 Wasserstein bounds；它没有证明 SGD 会达到这些 small-loss 条件。论文也报告 progressive 版本因 compounding error 更困难，这个负结果必须保留。

***

## 25. MeanFlow：从瞬时 velocity 推导平均 velocity identity

MeanFlow 使用 orientation

$$
z_r
=
z_t-(t-r)u(z_t,r,t).
$$

即

$$
z_t-z_r
=(t-r)u(z_t,r,t).
$$

假设终点状态沿 local field 运动：

$$
\frac{dz_t}{dt}=v(z_t,t).
$$

固定 $r$ 和 trajectory label，对 $t$ 求导：

$$
v(z_t,t)
=
u(z_t,r,t)
+
(t-r)
\frac{d}{dt}u(z_t,r,t).
$$

total derivative 为

$$
\frac{d}{dt}u(z_t,r,t)
=
\partial_tu(z_t,r,t)
+
J_zu(z_t,r,t)v(z_t,t).
$$

于是

$$
\boxed{
u(z_t,r,t)
=
v(z_t,t)
-
(t-r)
\left[
\partial_tu(z_t,r,t)
+
J_zu(z_t,r,t)v(z_t,t)
\right].
}
$$

这就是 MeanFlow identity。训练时可以用 JVP

$$
J_zu\,v
$$

而不构造完整 Jacobian。

### 25.1 为什么不能漏 spatial JVP

网络输入状态 $z_t$ 自身随 $t$ 变化。只算 explicit $\partial_tu$ 会把

$$
\frac{dz_t}{dt}
$$

引起的变化漏掉。

![flow-map composition、average velocity 与 MeanFlow JVP](/images/diffusion/d9_flowmap_shortcut_meanflow.png)

**图 25.1：** 左图展示 exact two-time composition；中图说明 average velocity 只在 interval 缩到 0 时回到 instantaneous velocity；右图显示完整 total derivative 的 residual 在机器精度，而漏掉 spatial JVP 时，区间 $h=1$ 的 residual 为 $0.4405$。

### 25.2 可解析 exponential flow

令

$$
\dot z_t=\rho z_t.
$$

则

$$
z_r=e^{-\rho(t-r)}z_t,
$$

所以

$$
u(z_t,r,t)
=
z_t
\frac{1-e^{-\rho(t-r)}}{t-r}.
$$

代入完整 identity 可精确成立。说明代码最大误差为 $4.441\times10^{-16}$。

### 25.3 MeanFlow 与普通 FM 的关系

在 $r=t$ diagonal：

$$
u(z_t,t,t)=v(z_t,t).
$$

off-diagonal 则通过 total-derivative identity传播 local signal。训练常用 conditional velocity 替代 inaccessible marginal $v$，再用 stop-gradient/JVP 构造 target。

MeanFlow 不是“一步 Euler Flow Matching”：它学习的是 integral average，并显式引入区间 $r,t$。

***

## 26. Distribution Matching Distillation：不要求唯一教师轨迹

Flow/consistency 主线约束 trajectory 或 finite map。DMD 改问：

> 能否只让一步 generator 的输出分布接近教师分布，而不指定每个 noise 应走哪条轨迹？

令

$$
x=G_\theta(z),
\qquad
z\sim p_z,
$$

诱导

$$
p_{\mathrm{fake},\theta}.
$$

目标是

$$
\mathcal K(\theta)
=
\operatorname{KL}
\left(
p_{\mathrm{fake},\theta}
\|p_{\mathrm{real}}
\right).
$$

对 generator-induced transport 做变分，可得到 score-difference 形式：

$$
\boxed{
\nabla_\theta\mathcal K
=
\mathbb E
\left[
\left(
s_{\mathrm{fake}}(x)
-s_{\mathrm{real}}(x)
\right)^\top
\frac{\partial G_\theta(z)}{\partial\theta}
\right],
}
$$

其中

$$
s_{\mathrm{fake}}=\nabla_x\log p_{\mathrm{fake}},
\qquad
s_{\mathrm{real}}=\nabla_x\log p_{\mathrm{real}}.
$$

实际在多个 Gaussian noise levels 上计算，以使 densities/scores 更规则，并复用 diffusion denoiser。

### 26.1 为什么需要 fake-score model

一步 generator 是 implicit distribution，不能直接算

$$
\nabla_x\log p_{\mathrm{fake}}(x).
$$

DMD 另训练一个 denoiser/score estimator 适应当前 generator samples。因此系统包含：

- frozen real/teacher score；
- changing fake-score model；
- one-step generator；
- 交替或耦合优化。

fake score error 会直接污染 generator gradient。

### 26.2 Gaussian sanity check

若

$$
p_{\mathrm{fake}}=\mathcal N(\theta,\sigma_f^2),
\qquad
p_{\mathrm{real}}=\mathcal N(\mu,\sigma_r^2),
$$

则

$$
\frac{d}{d\theta}
\operatorname{KL}
(p_{\mathrm{fake}}\|p_{\mathrm{real}})
=
\frac{\theta-\mu}{\sigma_r^2}.
$$

在 fake samples 上取 score difference 的 Monte Carlo 平均得到同一结果。说明代码误差为 $2.192\times10^{-3}$。

***

## 27. DMD、ADD 与 DMD2：旁支内部也不能混写

### 27.1 DMD

DMD 组合：

- distribution matching score gradient；
- fake-score regression；
- teacher-generated regression pairs；
- classifier-free guidance adaptation。

regression 项帮助保留 noise-to-image mapping 与覆盖率，所以原方法不是纯 KL-gradient algorithm。

### 27.2 ADD

Adversarial Diffusion Distillation 组合：

$$
\mathcal L
=
\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}
+
\lambda_{\mathrm{distill}}\mathcal L_{\mathrm{score-distill}}.
$$

adversarial branch 强化 perceptual realism，teacher score branch 提供 diffusion prior。视觉锐利或 FID 改善不证明 path consistency 或 exact distribution equality。

### 27.3 DMD2

DMD2 引入：

- online fake-score update，减少固定离线 regression dataset 依赖；
- generator/fake-score two-timescale schedule；
- GAN loss；
- 可选 denoising objective；
- one-step/few-step system changes。

DMD2 official code（补充材料暂未公开） 中 `compute_distribution_matching_loss` 与训练循环（补充材料暂未公开）明确分开 generator turn 和 guidance/fake-score turn。

### 27.4 与 consistency 的根本差异

| 方法族               | 直接匹配什么                                     | 是否要求同轨迹 | 主要辅助对象                |
| ----------------- | ------------------------------------------ | ------: | --------------------- |
| Progressive/CD    | teacher transition/endpoint                |       是 | teacher、EMA           |
| Shortcut/MeanFlow | finite map/average velocity identity       |   是或自生成 | self-target/JVP       |
| DMD/DMD2          | output marginal distribution               |       否 | fake score、GAN        |
| ADD               | adversarial realism + teacher score signal |       否 | discriminator、teacher |

[Distribution distillation note](https://arxiv.org/abs/2311.17042 "官方论文页面") 保留了这条分界。

***

## 28. Stochastic Interpolants：统一的是 marginal/field，不是 path law

一般 stochastic interpolant 可写为

$$
Z_\tau
=
I(\tau,Z_0,Z_1)
+\gamma(\tau)\xi,
\qquad
\xi\sim\mathcal N(0,I),
\qquad
\xi\perp(Z_0,Z_1).
$$

其 sample derivative 为

$$
\dot Z_\tau
=
\partial_\tau I(\tau,Z_0,Z_1)
+\dot\gamma(\tau)\xi.
$$

marginal velocity：

$$
\boxed{
b_\tau(z)
=
\mathbb E[
\partial_\tau I
+\dot\gamma\,\xi
\mid Z_\tau=z
].
}
$$

Gaussian latent 还给 score identity：

$$
s_\tau(z)
=
\nabla_z\log p_\tau(z)
=
-\frac1{\gamma(\tau)}
\mathbb E[
\xi\mid Z_\tau=z
]
$$

在相应 scalar-noise setup 与非零 $\gamma$ 下成立。
更精确地说，这里还要求 $p_\tau$ 可微，并允许把对 $z$ 的微分移入关于 endpoints 的积分；恒等式随后由 Gaussian integration by parts 得到。若噪声不是独立的标准 Gaussian，或者 $\gamma$ 是矩阵，右侧形式必须相应修改。

### 28.1 Same-marginal ODE/SDE family

若选择非负 $\varepsilon_\tau$，则可构造

$$
\text{ODE:}\qquad
dZ_\tau=b_\tau(Z_\tau)d\tau,
$$

$$
\text{forward SDE:}\qquad
dZ_\tau
=
\left[
b_\tau+\varepsilon_\tau s_\tau
\right]\!(Z_\tau)d\tau
+
\sqrt{2\varepsilon_\tau}\,dW_\tau.
$$

相应 original-time backward drift 为

$$
b_\tau-\varepsilon_\tau s_\tau.
$$

这些过程共享 $p_\tau$，但 transition kernels、temporal covariance、quadratic variation 与 path laws 不同。

这与 D4 的 probability-flow principle 一致：same marginals 不等于 same paths。

***

## 29. Diffusion、Flow Matching 与 Schrödinger Bridge 的边界

三者可以共享：

- continuity/Fokker--Planck equation；
- score 与 conditional expectation；
- probability-flow velocity；
- neural quadratic regression；
- ODE/SDE simulation。

但定义问题不同。

### 29.1 Score diffusion

先选择 forward noising law，再学习 reverse score/dynamics。

### 29.2 Flow Matching

先选择 endpoint coupling/interpolant，得到 marginal curve，再回归生成它的 velocity。

### 29.3 Schrödinger Bridge

给 reference path law $R$ 与两个 hard marginals，求

$$
P^*
=
\arg\min
\left\{
\operatorname{KL}(P\|R):
P_0=\mu_0,\,
P_T=\mu_T
\right\}.
$$

objective 同时选择 distinguished endpoint coupling 与 conditional path law。

所以

$$
\boxed{
\text{generic stochastic interpolant}
\ne
\text{Schrödinger Bridge}.
}
$$

共享 interpolation technique 不足以推出 path-space KL optimality。完整比较见 [Bridge B12](/blog/schrodinger-bridge/b12-diffusion-flow-matching-unification/)。

***

## 30. 一个统一但不偷换对象的算法视图

### 30.1 Local-field route

```text
choose coupling/path
sample conditional path state and tangent
regress local velocity
integrate learned ODE with a solver
```

代表：Flow Matching、基础 Rectified Flow、stochastic interpolant velocity。

### 30.2 Teacher-transition route

```text
run a pretrained teacher over one or more small steps
construct a larger student transition target
train student
optionally promote student to teacher and repeat
```

代表：progressive distillation、CD、LCM、InstaFlow 的 distillation stage。

### 30.3 Self-consistent map route

```text
anchor diagonal/small-step behavior
query model on subintervals or derivatives
construct stop-gradient finite-map target
train arbitrary-step map or average velocity
```

代表：flow-map self-distillation、Shortcut Models、MeanFlow。

### 30.4 Distribution route

```text
sample one-step generator outputs
estimate real and fake noisy scores
update generator using score difference
update fake-score model / discriminator
```

代表：DMD/DMD2/ADD。

***

## 31. 说明代码：读什么，不读什么

d9\_few\_step\_flow\.py（补充材料暂未公开） 不训练模型。它实现：

1. 解析旋转 ODE 的 local field、Euler/Heun 与 exact flow map；
2. 2D Gaussian-mixture CFM 的 conditional targets 与 exact marginal velocity；
3. arbitrary coupling、model-induced curved paths 与 reflow straight chords；
4. progressive DDIM target inversion；
5. consistency boundary parameterization；
6. exponential flow 的 composition、shortcut recursion 与 MeanFlow JVP；
7. Gaussian DMD score-difference gradient；
8. 五张章节图。

运行：

```powershell
# 本地验证脚本暂未公开
# 本地验证脚本暂未公开
```

固定 seed 检查结果：

| identity                            |                 error |
| ----------------------------------- | --------------------: |
| rotation inverse                    | $2.220\times10^{-16}$ |
| rotation norm preservation          | $4.441\times10^{-16}$ |
| CFM orthogonality                   |  $1.000\times10^{-3}$ |
| flow-map composition                | $1.776\times10^{-15}$ |
| shortcut recursion                  | $2.220\times10^{-15}$ |
| MeanFlow identity                   | $4.441\times10^{-16}$ |
| progressive target inversion        | $6.661\times10^{-16}$ |
| consistency boundary                |                   $0$ |
| Gaussian score-gradient Monte Carlo |  $2.192\times10^{-3}$ |

这些检查验证公式实现，不是 image benchmark，也不支持“某方法效果最好”的经验结论。

***

## 32. 误差必须按层分账

一步/few-step 结果至少包含以下误差：

| 层                       | 误差来源                                               | 典型方法                      |
| ----------------------- | -------------------------------------------------- | ------------------------- |
| conditional regression  | finite data/network/optimization                   | FM/RF                     |
| path/coupling           | 选择的 interpolant 与 endpoint pairing                 | RF/stochastic interpolant |
| solver                  | finite NFE truncation/stability                    | FM/diffusion sampler      |
| teacher                 | pretrained score 与 teacher integration             | progressive/CD/LCM        |
| distillation            | target inversion 与 stage compounding               | progressive/InstaFlow     |
| consistency propagation | adjacent loss 到 global endpoint                    | CT/CD/sCM                 |
| map bootstrap           | off-trajectory queries与composition error           | Shortcut/self-distill     |
| derivative target       | JVP、stop-gradient、average velocity regression      | MeanFlow                  |
| fake score              | changing generator distribution 的 score estimation | DMD/DMD2                  |
| adversarial             | discriminator bias、coverage/optimization           | ADD/DMD2                  |
| system                  | architecture、latent decoder、guidance、data          | 所有大模型                     |

低 NFE 下，solver error 可能下降，却暴露 model/map error；反过来，学习有限 map 也可能牺牲可逆性、likelihood 或多步灵活性。

***

## 33. 理论结论与经验结论的边界

### 33.1 可作为 exact identity 的内容

- continuity equation 的 weak derivation；
- conditional expectation projection；
- exact ODE flow-map composition；
- average-velocity midpoint identity；
- MeanFlow total derivative；
- Gaussian score-difference sanity check。

### 33.2 需要 assumptions 的 theorem

- ODE existence/uniqueness 与 invertibility；
- FM/CFM gradient interchange；
- RF transport-cost comparison；
- consistency local-to-global error；
- flow-map Wasserstein bounds；
- interpolant ODE/SDE same-marginal statements。

### 33.3 只能作为 empirical claim

- 一步 FID/视觉质量；
- reflow 在具体系统中把路径“拉直多少”；
- pseudo-Huber/lognormal schedule 的稳定收益；
- LCM/SDXL-Turbo 等系统 latency；
- Shortcut/MeanFlow 在特定 architecture/compute 下的 ranking；
- 2025--2026 新方法是否会成为长期主线。

“公式成立”与“神经训练能找到公式的解”之间，还有 approximation、optimization 与 finite-data 三层。

***

## 34. 常见错误

1. **把一步 Euler 叫一步 flow map。** 前者是局部近似，后者是有限积分对象。
2. **把 Flow Matching 等同一步生成。** 普通 FM 仍需 ODE solver。
3. **把 straight conditional paths 等同 global OT。** coupling 可能完全不是 OT coupling。
4. **认为 Rectified Flow 只是一条线性插值公式。** coupling 与 reflow 才是其低 NFE 叙事的关键。
5. **把 reflow 当 sampling trick。** 它需要生成新 pairs 并重新训练。
6. **把 progressive target 当两个 teacher outputs 的均值。** 它来自反演一次学生 DDIM update。
7. **把 Consistency Model 写成 score model 的新 sampler。** 它学习 endpoint map。
8. **认为 boundary parameterization 保证全局 consistency。** 它只固定 $t=\epsilon$。
9. **把 CT 的“from scratch”理解为无路径假设。** corruption coupling、grid 与 EMA 仍在。
10. **把 endpoint consistency 直接叫任意 two-time flow map。** 它只固定一个终点。
11. **对 nonautonomous flow 无条件使用 one-parameter semigroup。** 正确对象通常是 $\Phi_{s,t}$。
12. **把 average velocity 当 endpoint 或 local velocity。** 它是单位区间 displacement。
13. **Shortcut target 中第二次查询仍用真实 midpoint。** 实际使用 model-generated midpoint。
14. **MeanFlow 只计算 $\partial_tu$。** total derivative 还含 $J_zu\,v$。
15. **把 stop-gradient 当数学等号。** 它是 optimization implementation choice。
16. **把 DMD 写成 trajectory distillation。** 它直接匹配 output distribution。
17. **忽略 fake-score model。** 其误差决定 DMD generator gradient bias。
18. **把 adversarial realism 当 coverage 证明。** sharp samples 不等于 distribution equality。
19. **把 same marginals 当 same path law。** ODE 与 SDE 可共享全部 one-time densities 仍有不同 temporal law。
20. **看到 stochastic interpolant 就称 SB。** 没有 reference path KL 与 endpoint constraints 就不是该 variational problem。
21. **用新发布日期替代影响力证据。** 2025--2026 结果需等待正式发表、独立采用和复现。
22. **把 architecture 改进归给 objective。** D8 已说明 backbone、representation、data 与 compute 必须单独记账。

***

## 35. 问题—方案—局限—下一步

| 问题                       | 方案                    | 局限                            | 引出的下一问                         |
| ------------------------ | --------------------- | ----------------------------- | ------------------------------ |
| 固定 field 多步太慢            | 高阶 solver             | 一步信息不足                        | 能否改变 learned object            |
| marginal field 难监督       | CFM                   | 仍是 local field                | 路径能否更直                         |
| coupling 产生弯曲/交叉         | RF/reflow             | 多阶段 retraining                | 能否直接学 endpoint                 |
| teacher trajectory 想压缩   | progressive/CD        | teacher/stage cost            | 能否 from scratch                |
| 不要 pretrained teacher    | CT/sCM                | local-to-global propagation   | 能否学任意区间                        |
| endpoint map 太单一         | flow-map matching     | two-time training cost        | 能否 self-bootstrap              |
| arbitrary jump target 难得 | Shortcut/self-distill | compounding shift             | 能否从 local derivative 得 average |
| 平均 velocity 无 teacher    | MeanFlow JVP identity | derivative/optimization error | 是否必须匹配轨迹                       |
| trajectory 不唯一           | DMD/ADD               | auxiliary score/GAN risk      | 如何评估 quality/coverage/theory   |

这条路线没有终结 Diffusion。它把“生成过程”拆成更多可选择的接口。

***

## 36. 本章小结

1. Solver 加速与 learned one/few-step map 是两类问题。
2. 局部 velocity、numerical step、endpoint map、two-time map、average velocity 与 output distribution 必须分开。
3. Continuity equation 把随机路径的 conditional mean tangent 变成 marginal velocity。
4. CFM loss 等于 marginal FM loss 加一个与参数无关的 conditional variance。
5. Affine path 的 data/noise/velocity targets 可代数转换，但 weighting 与 optimization 不自动等价。
6. 普通 Flow Matching 学 local field，仍需 integration。
7. Rectified Flow 的核心杠杆是 coupling 与 reflow，不只是 linear interpolation。
8. RF ODE 保持 interpolation marginals，不逐样本保持原 endpoint line。
9. Straight conditional path 不等于 global OT。
10. Progressive Distillation 通过反演学生大步构造 sharp teacher endpoint target。
11. Consistency Model 学 $x_t\mapsto x_\epsilon$ 的 trajectory-invariant endpoint map。
12. CD、CT、LCM 与 sCM 的 teacher、coupling、grid 与 scaling 职责不同。
13. Exact flow map 满足 identity、composition 与 inverse；nonautonomous 情形是 two-parameter family。
14. Endpoint consistency 是完整 flow map 的一个固定终点切片。
15. Shortcut recursion来自 exact composition，但 neural bootstrap 会产生 off-trajectory distribution shift。
16. MeanFlow 学 average velocity，target 必须包含 total derivative 与 spatial JVP。
17. DMD/ADD 匹配 output distribution/realism，不要求复现唯一 flow。
18. fake-score、target network、teacher solver 与 discriminator 都是算法状态，不能从方法名中省略。
19. Stochastic interpolants 可统一 marginal velocity、score、ODE 与 SDE；same marginals 不是 same path law。
20. Generic interpolant 不等于 Schrödinger Bridge，后者由 reference path-space KL 定义。
21. 最新 one-step 结果仍是 architecture、data、latent representation、guidance、training 和 evaluation 的系统结论。

***

## 37. 研究式思考题

1. 对给定 learned local field，能否从曲率、Lipschitz constant 与 score error 推导“低于多少 NFE 后应该改学 finite map”的判据？
2. CFM conditional target variance 如何随 endpoint coupling 变化？能否把 coupling learning 直接设计成 variance reduction？
3. Rectified Flow induced coupling 的 convex-cost 改进与 trajectory curvature reduction 是否等价？构造反例需要什么结构？
4. Reflow 的 teacher integration error、pair finite-sample error 与第二代 regression error如何组合成 endpoint bound？
5. Progressive distillation repeated halving 的误差是线性、乘性还是依赖 contraction？哪些 sampler geometry 能抑制 stage compounding？
6. Consistency boundary 选 $\epsilon>0$ 会产生怎样的 truncation bias？若令 $\epsilon\to0$，网络 conditioning 与数据流形奇异性如何变化？
7. CT 的 shared-noise adjacent pairs 与 exact PF ODE pairs 的差异能否写成显式 local bias expansion？
8. 一个只学 endpoint map 的模型在什么条件下可恢复 likelihood 或 inverse？是否必须使用可逆 architecture？
9. 对 nonautonomous flow map，如何设计 time embedding 使 composition 误差可控，而不是只在训练网格上记忆 $(s,t)$？
10. Shortcut bootstrap 中，如何用 uncertainty 或 cycle consistency 检测 model-generated midpoint 已离开训练分布？
11. MeanFlow identity 的 JVP target 有哪些 stop-gradient 选择？它们对应同一 population fixed point，却会如何改变 optimization dynamics？
12. 能否把 MeanFlow 与 flow-map composition loss组合，分别约束 differential 与 integral consistency？
13. DMD 的 fake-score estimator 与 generator 是双层动态系统。什么 timescale separation 条件能使 score-gradient近似可信？
14. ADD/DMD2 中 adversarial loss改善 texture但可能改变 coverage；如何设计不依赖单一 FID 的诊断矩阵？
15. 若两个方法有相同 1-NFE FID，但一个近似 flow map、另一个只匹配 marginal，如何用 coupling、invertibility、editing 与 likelihood 实验区分？
16. Stochastic interpolant 中选择不同 $\varepsilon_\tau$ 得到 same marginals 的 ODE/SDE。哪些 downstream task真正依赖 path law，而不只依赖 endpoint samples？
17. 要让 stochastic interpolant 成为 Schrödinger Bridge solver，还必须学习/优化哪些 coupling 与 path-space quantities？
18. 2025--2026 的 flow-map/MeanFlow 后续研究应满足哪些证据，才值得从 frontier candidate 升级为教程历史里程碑？

***

下一章将离开连续 Gaussian 状态空间，进入[离散 Diffusion 与 Diffusion Language Models](/blog/diffusion/d10-discrete-diffusion-language-models/)：当状态是 token/category、forward process 是 Markov chain 或 CTMC 时，velocity、score、ratio 与并行解码必须重新定义。
