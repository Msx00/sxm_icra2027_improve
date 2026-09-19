# DistriSurg：面向双目腹腔镜新视角合成的分布式几何传输与可见性感知补全

## 1. 核心结论

当前工作的论文定位不应是“把 LaMa 用到腹腔镜图像补全”，也不应只是“给 LaMa 增加深度、置信度或时序信息”。这些方向已经分别被深度感知内镜视频补全、双目时空补全和置信度引导内镜补全覆盖。更有竞争力的论文问题应定义为：

> 给定一个腹腔镜源视图、稀疏或含错误的深度以及相机几何，如何把可观测组织可靠地传输到目标视图，并且只在真正不可见的区域生成内容，同时给出可校准的不确定性，避免把投影缺陷误当成需要生成的新组织？

建议论文主线命名为 **DistriSurg**：**Distributional Geometry Transport and Visibility-Constrained Completion for Stereo Endoscopic Novel-View Synthesis**。LaMa/UFFC 是其中的补全骨干，而不是论文的核心贡献。

最值得做的三个贡献是：

1. **分布式表面泼溅（Distributional Surface Splatting, DSS）**：用深度分布、表面方向和软可见性取代硬点投影与最近邻小孔填充，从源头减少黑点、碰撞和错误遮挡。
2. **传输—生成解耦（Transport-or-Synthesize）**：根据投影支持度、深度方差和碰撞熵，连续地选择“保留/修复已观测内容”或“生成不可见内容”；可信像素最终硬拷贝，禁止网络改写。
3. **几何闭环和风险校准**：联合预测目标 RGB、深度和不确定性，通过双向重投影、表面法向和选择性风险约束，让模型不仅“看起来像”，还要在几何上自洽，并能标出不应被信任的生成区域。

这条路线比当前 LCM-LoRA 更适合 ICRA/MICCAI；若想冲击 TMI/Medical Image Analysis，则还需多数据集、下游几何任务、外科医生盲评和严格统计分析。

## 2. 当前实验告诉我们的事实

在 dataset89 中按 `valid_mask / image_area > 0.5` 筛选后，共有 47 帧。当前同一批帧的初步结果为：

| 方法 | 全图 PSNR ↑ | 全图 SSIM ↑ | 缺失区 PSNR ↑ | 缺失区 SSIM ↑ |
|---|---:|---:|---:|---:|
| SD1.5 Inpainting | 15.520 | 0.5447 | 11.817 | 0.2683 |
| LaMa | **22.641** | **0.7271** | **20.265** | **0.6657** |
| 当前 LCM + Endoscopy LoRA | 22.532 | 0.7089 | 20.114 | 0.6338 |

此外，“只补小孔再生成真实大缺失区”的旧模式优于“把所有无效投影都交给扩散模型”：前者在相同 47 帧上约为 22.662 dB / 0.699 / 0.2555 LPIPS，后者约为 21.957 dB / 0.6299 / 0.3398 LPIPS。

这说明：

- 当前瓶颈首先是**几何传输质量和 mask 定义**，不是生成步数或提示词。
- 大量黑点混合了源深度无效、离散 forward warp 的采样孔、深度碰撞和真实 disocclusion；把它们全部标成“待生成”在物理意义上不正确。
- LaMa 已经是很强的基线。只做领域微调或修改 loss，通常很难形成足够强的论文创新。
- 47 帧适合做原型验证，不足以支持高质量论文的统计结论。

## 3. 相关研究版图与真正的空白

### 3.1 通用图像补全

Partial convolution 只对有效像素卷积并更新 mask，Gated Convolution 学习每个位置和通道的动态门控，它们奠定了不规则孔洞补全的基本范式。[^1][^2] LaMa 使用 Fast Fourier Convolution 获得全局感受野，对大孔洞和高分辨率具有较好泛化能力。[^3] 但后续 UFFC 工作指出，原始 FFC 存在频谱偏移、异常空间激活和频域感受野受限等问题，并提出位置编码、动态跳连和频域修正。[^4] MAT 则通过 mask-aware transformer 限制无效 token 参与注意力，擅长大孔洞补全。[^5]

因此，“把 LaMa 换成 UFFC/MAT”可以提高基线，但不是独立创新。它们都没有解决腹腔镜 RGB-D 重投影中“哪些孔洞来自采样，哪些来自遮挡，哪些深度本身不可信”的问题。

### 3.2 几何引导的新视角补全

SynSin 将源图像编码成特征点云，进行可微投影后用 refinement network 合成目标视图；Softmax Splatting 用软最大化解决 forward warp 中多个源像素映射到同一位置的碰撞问题。[^6][^7] GeoFill 联合优化深度尺度/偏移与相机姿态，再进行 3D 重投影和参考图像补全。[^8] LeftRefill 与 MultiDiff 展示了预训练扩散模型在参考引导补全和深度 warp 条件下生成一致新视图的能力。[^9][^10]

这些方法证明“几何传输 + 生成”是正确结构，但仍有三个空白：其一，常把每个深度值视为确定真值；其二，缺少针对稀疏、镜面、软组织深度的分布式可见性；其三，补全器通常无法区分“可靠传输像素”和“必须合成像素”。这正是本项目可以进入的空白。

### 3.3 内镜视频补全与直接竞争工作

DAEVI 已经使用时空深度估计、双模态通道融合和深度增强判别器做内镜视频补全。[^11] SSIFNet 进一步融合光流传播、双目时空 focal transformer、视差匹配和跨视图重投影，并采用自监督模拟遮挡训练。[^12] Endo-STTN 使用时序 GAN 去除镜面反射，并通过双目视差、光流和特征匹配验证下游收益。[^13] 2026 年的 ConFiT 已把高/低置信度区域分别交给时序传播和空间生成，因此“置信度三分支”本身也不再新颖。[^14]

视频补全方面，E2FGVI 联合训练光流补全、特征传播和内容合成；ProPainter 使用双域传播、flow-guided deformable alignment 和 mask-guided sparse transformer；DGDVI 已把深度补全和深度引导传播整合进视频补全。[^15][^16][^17]

与这些工作的关键区分应是：DistriSurg 研究的是**已标定相机之间的目标视图重建和 disocclusion**，核心不确定性来自 RGB-D 几何投影，而不是工具 mask 后的普通视频补洞；DSS 输出的是物理可解释的支持度、方差与碰撞熵，不是一个从图像中直接预测的模糊“confidence map”。

### 3.4 2025–2026 年的新竞争压力

2025 年已有点云引导扩散模型用于腹腔镜肝脏新视角视频合成。[^18] 2026 年又出现了“深度 splatting + diffusion inpainting”从单目手术视频恢复双目视图的临床会议摘要，使用 12,912 帧并直接报告 PSNR/SSIM/LPIPS。[^19] EndoDAV 开始强调内镜深度的空间精度和时间一致性。[^20] 更广义的动态 NVS 也明确采用“可共视像素由 3D 重建传输，不可见像素由视频扩散生成”的拆分。[^21]

这意味着论文不能只写成“depth-guided splatting + diffusion/LaMa”。真正可防御的新颖性必须落在：**不确定深度的分布式投影、可见性约束的专家路由、几何闭环和校准评估**。

## 4. 建议的网络：DistriSurg

### 4.1 输入与输出

输入为源视图 RGB $I_s$、稀疏深度 $D_s$、深度有效 mask $M_d$、相机内参 $K_s,K_t$ 和相对位姿 $T_{s\rightarrow t}$。网络输出目标视图 $\hat I_t$、目标深度 $\hat D_t$、投影可靠度 $R_t$ 和最终预测风险 $U_t$。

整体流程为：

```text
I_s, D_s, M_d, K, T
        │
        ├─ Depth Reliability Encoder → ΔD, log σ², normal, source confidence
        │
        ├─ Multi-scale Distributional Surface Splatting
        │       └→ warped RGB/features, expected depth, support, variance, collision entropy
        │
        ├─ Physics-derived Visibility Router
        │       ├→ Transport/Refinement Expert
        │       └→ UC-UFFC Synthesis Expert
        │
        ├─ Hard trusted-pixel composition → target RGB
        └─ RGB-D joint decoder → target depth + calibrated uncertainty
                              │
                              └─ backward reprojection / geometry cycle
```

### 4.2 模块一：深度可靠性编码器

不要直接把无效深度用邻居值填满后当作真值。用轻量 RGB-D encoder 对每个源像素预测：

$$
\mu_q=D_s(q)+\Delta D(q),\qquad
\sigma_q^2=\exp(s_q),\qquad
\mathbf n_q=N(q),\qquad
c_q=\operatorname{sigmoid}(C(q)).
$$

其中 $\mu_q$ 是校正后的深度均值，$\sigma_q^2$ 表示观测不确定性，$\mathbf n_q$ 是表面法向，$c_q$ 是由输入有效性、局部 RGB-D 一致性和镜面/工具区域共同决定的可靠度。原始有效深度使用残差校正；无效深度允许预测，但必须具有更高方差，不能与实测深度等价处理。

可用 heteroscedastic depth NLL 监督均值和方差：

$$
\mathcal L_{d\text{-nll}}=
\frac{|D^{gt}-\mu|}{\exp(s)}+s.
$$

若训练数据没有稠密深度，可在双目一致区域用视差/重投影自监督，并使用 EndoDAV 一类时序深度模型作为 teacher，但 teacher 结果只能是软先验，不能冒充真值。[^20]

### 4.3 模块二：分布式表面泼溅 DSS

对源像素 $q$ 不再只投影一个 3D 点，而把深度表示为 $z_q\sim\mathcal N(\mu_q,\sigma_q^2)$。工程上可使用确定性的三点或五点 sigma samples，避免昂贵 Monte Carlo。每个样本投影到目标像素 $p$ 后，权重定义为：

$$
w_{qk\rightarrow p}=c_q\,a_{qk}\,
\mathcal K_{\Sigma_q}\!\left(p-\pi\left(T_{s\rightarrow t}
\Pi^{-1}(q,z_{qk})\right)\right)
\exp(-\beta z_{qk}),
$$

其中 $a_{qk}$ 是深度分布样本权重，$\mathcal K_{\Sigma_q}$ 是由深度方差、投影 Jacobian 和表面法向共同决定的各向异性 footprint，$\exp(-\beta z)$ 让近表面在碰撞处获得更高可见性。

在每个尺度上聚合图像/特征：

$$
F^w_t(p)=\frac{\sum_{q,k}w_{qk\rightarrow p}F_s(q)}
{\sum_{q,k}w_{qk\rightarrow p}+\epsilon}.
$$

同时输出四个重要的物理量：

- 支持质量 $S(p)=\sum w$：目标位置有多少可信源观测；
- 期望深度 $\bar z(p)$；
- 条件深度方差 $V(p)$：几何是否稳定；
- 碰撞熵 $H(p)=-\sum\tilde w\log\tilde w$：是否存在多个竞争表面。

这比二值 `known_mask` 信息更丰富。小型采样孔会具有邻域支持，可以由连续 footprint 自然覆盖；真实 disocclusion 的支持度接近零；遮挡边界通常表现为高方差/高碰撞熵。

### 4.4 模块三：物理驱动的传输—生成路由

构造多尺度路由器：

$$
g(p)=G\big(S(p),V(p),H(p),M_d^w(p),\nabla\bar z(p),
\text{mask topology}\big),
$$

其中 $g\in[0,1]$ 表示需要“合成”的程度。低风险共视区域进入 Transport Expert，仅做颜色/照明残差修正；零支持的 disocclusion 进入 Synthesis Expert；高熵遮挡边界采用软混合。

建议 Synthesis Expert 以 UFFC 为骨干，因为其修正了原始 FFC 的若干频域问题。[^4] 但要增加 **uncertainty-conditioned UFFC（UC-UFFC）**：在每个编码/解码尺度，用 $S,V,H$ 产生 FiLM/SPADE 参数，调节局部与全局频域分支。可靠传输区域抑制生成分支，不可见区域开放全局组织纹理合成。

最终不是让网络重绘整张图，而是：

$$
I^{mix}_t=(1-g)I^{trans}_t+gI^{syn}_t,
$$

$$
\hat I_t=M_{trust}I^w_t+(1-M_{trust})I^{mix}_t.
$$

$M_{trust}$ 由严格的投影可靠性阈值产生，并在输出端硬合成。因此可信区域像素误差理论上为零，网络无法为了提高感知质量而偷偷改写已观测组织。

### 4.5 模块四：RGB-D 联合解码与双向几何闭环

补全器同时预测目标 RGB 和目标深度，而不是只优化外观。用 $\hat D_t$ 将目标结果反投影回源视图，并只在双向可见像素上计算：

$$
\mathcal L_{cycle}=
\rho\!\left(I_s-W(\hat I_t,\hat D_t,T_{t\rightarrow s})\right)
+\lambda_z\rho\!\left(D_s-W(\hat D_t,T_{t\rightarrow s})\right).
$$

再加入目标深度、法向和边缘一致性：

$$
\mathcal L_{geo}=\lambda_d\mathcal L_{depth}
+\lambda_n(1-\mathbf n^{gt}\cdot\hat{\mathbf n})
+\lambda_e\|\nabla\hat D_t-\nabla D_t^{gt}\|_1.
$$

这一模块针对生成模型最危险的问题：输出纹理可能逼真，但双目视差或表面形状错误。Endo-STTN 已经说明内镜补全应通过光流、视差和特征匹配等下游几何指标检验，而不应只看 PSNR。[^13]

### 4.6 总损失

建议第一版使用：

$$
\begin{aligned}
\mathcal L=
\lambda_h\mathcal L_{hole-charb}
+\lambda_s\mathcal L_{seam}
+\lambda_p\mathcal L_{LPIPS/DISTS}
+\lambda_f\mathcal L_{freq/wavelet}
+\lambda_g\mathcal L_{geo}
+\lambda_c\mathcal L_{cycle}
+\lambda_u\mathcal L_{uncertainty}
+\lambda_r\mathcal L_{router}.
\end{aligned}
$$

其中：

- `hole-charb` 只监督真实缺失区域，并按孔洞面积归一化；
- `seam` 在 3/7/15 像素边界带上监督 RGB 与梯度；
- `freq/wavelet` 约束组织纹理频谱，但权重不宜过高；
- `router` 用由几何可见性生成的伪标签预训练路由器，再联合微调；
- `uncertainty` 同时做误差 NLL 与排序/校准约束，使高风险区域真正对应高误差。

不建议第一版依赖大权重 GAN loss；对医疗图像，GAN 容易提高锐度却制造不存在的组织细节。若使用判别器，应只作为低权重感知项，并报告去除它的消融。

## 5. 为什么这套贡献比几个直观方案更强

| 方案 | 能否改善指标 | 能否构成主要创新 | 主要问题 |
|---|---|---|---|
| LaMa 在内镜数据上微调 | 是 | 弱 | 属于领域适配，审稿人会要求更强网络贡献 |
| LaMa + depth channel | 可能 | 弱 | DAEVI/DGDVI 已有深度引导补全 |
| LaMa + confidence mask | 可能 | 弱 | ConFiT 已采用置信度分区补全 |
| LaMa + optical flow | 是 | 中等偏弱 | E2FGVI、ProPainter、SSIFNet 已覆盖 |
| UFFC 替换 FFC | 是 | 弱 | 直接采用已有模块 |
| 软 splatting + LaMa | 是 | 中等 | Softmax Splatting/SynSin 已建立相似范式 |
| **分布式深度 + 各向异性 DSS + 可见性路由 + RGB-D 闭环** | **预期是** | **强** | 需要严格证明每一模块不可由普通 soft splat 替代 |

与直接竞争方法的差异建议在论文中用下表明确表达：

| 方法 | 任务 | 显式相机几何 | 投影不确定性 | 双目/时序 | 传输与生成解耦 | 输出风险校准 |
|---|---|---:|---:|---:|---:|---:|
| LaMa/UFFC | 单图补全 | 否 | 否 | 否 | 否 | 否 |
| DAEVI | 内镜视频补全 | 部分（深度） | 否 | 时序 | 否 | 否 |
| SSIFNet | 工具遮挡补全 | 视差/光流 | 否 | 双目+时序 | 部分 | 否 |
| ConFiT | 工具遮挡补全 | 否 | 图像置信度 | 时序 | 高/低置信分支 | 否 |
| GeoFill | 参考图像补全 | 是 | 否 | 双视图 | 部分 | 否 |
| 2026 mono-to-stereo 摘要 | 手术双目恢复 | 深度 splat | 未报告 | 双目目标 | warp+diffusion | 否 |
| **DistriSurg** | 双目内镜 NVS | **是** | **深度分布、支持、方差、碰撞熵** | 双目，可扩展时序 | **显式连续路由** | **是** |

“第一种”或“首个”这样的表述必须在投稿前再进行题名、摘要和专利检索。目前更稳妥的 claim 是“我们提出一种……”，而不是未经证实地声称“首次”。

## 6. 数据与训练设计

### 6.1 数据切分

SCARED 原始挑战由 7 个训练数据集和 2 个测试数据集组成，包含结构光深度、双目相机及机器人运动信息；内镜镜面反射、弱纹理、非刚性形变和位姿误差都会影响深度与重投影。[^22] 2026 年发布的 SCARED-C 针对原始运动学位姿误差进行 COLMAP 与尺度校正，提供约 17,135 个可靠 RGB-D 对，可作为更稳健的几何训练来源。[^23]

建议：

1. 训练集使用 SCARED 的非测试病例，优先采用 SCARED-C 中校正后的可靠序列。
2. dataset8/9 保持完全零样本，只用于最终测试，不用于阈值、mask 参数或 checkpoint 选择。
3. 按 dataset / patient / keyframe sequence 切分，禁止随机按帧切分，避免相邻帧泄漏。
4. 增加至少一个跨域测试集，如 Hamlyn、SERV-CT 或 StereoMIS；若无完整深度，也可用于无参考时序/立体一致性和人工评价。
5. 当前 47 帧作为 overlap > 0.5 的 stress subset 保留，但同时报告全测试集和不同 overlap 分箱，避免选择性报告。

需要特别检查 SCARED-C 实际包含的 dataset/keyframe 列表，不能默认它与原始 SCARED 的 8/9 测试结构一一对应。

### 6.2 训练样本构造

每个训练样本使用真实双目对 $(I_s,D_s,I_t,D_t,K,T)$。为了让网络学会真实投影缺陷，mask 不应主要来自随机矩形或自由笔刷，而应由训练时的几何 forward warp 自动生成。

建立“投影退化课程”：

- 深度随机失效：按真实数据统计模拟点状、条纹、反光相关的无效深度；
- 深度噪声：随深度、梯度和镜面强度变化的 heteroscedastic noise；
- 标定/位姿扰动：小幅旋转、平移、焦距和主点扰动；
- 遮挡边界扰动：对前后景碰撞区域重点采样；
- overlap curriculum：由高重叠逐步扩展至 0.5–0.6 的困难样本；
- 真实 disocclusion 与人工 free-form mask 分开标记，后者只作辅助增强。

### 6.3 分阶段训练

| 阶段 | 训练内容 | 建议冻结项 | 目标 |
|---|---|---|---|
| A | 深度可靠性编码器 | 补全器 | 学到均值、方差、法向和有效性 |
| B | DSS renderer | 补全器 | 对齐 target depth/RGB，稳定可见性排序 |
| C | UC-UFFC 补全 | reliability encoder 可部分冻结 | 学会真实投影 mask 的区域补全 |
| D | Router + RGB-D joint head | 无 | 端到端优化传输/生成和几何闭环 |
| E | 可选 3 帧时序扩展 | 单帧主干先冻结 | 加入姿态/流引导的特征记忆 |

先完成 A–D，再决定是否加入 E。时序分支虽然可能提高结果，但相关工作拥挤，会增加工程量并模糊 DSS 的核心贡献。对于 ICRA，若 A–D 能实时运行且立体几何更准，单帧模型已有清晰故事；对于 MedIA/TMI，时序与临床评价更重要。

## 7. 实验协议：从“图像好看”升级为可信几何评价

### 7.1 必须报告的指标

| 维度 | 指标 | 说明 |
|---|---|---|
| 全图 | PSNR、SSIM、LPIPS、DISTS | 与已有工作兼容，但不能独立支撑结论 |
| 缺失区 | hole-PSNR/SSIM/LPIPS/DISTS | 真正衡量生成区域 |
| 边界 | 3/7/15 px seam-band 指标、gradient error | 定位接缝和黑点问题 |
| 已知区 | L1、最大绝对误差 | 硬合成后应接近严格为零 |
| 深度 | AbsRel、RMSE、δ1、normal angular error | 检验解剖几何 |
| 立体 | disparity EPE、left-right reprojection error | 检验左右眼一致性 |
| 时序 | warp error、tLPIPS、flicker、VFID | 若输出视频必须报告 |
| 校准 | ECE、NLL、AURC、risk-coverage curve | 检验风险图是否可信 |
| 效率 | FPS、参数量、FLOPs、显存、延迟 | ICRA/实时应用的重要指标 |

LPIPS 和 DISTS 值得保留：既有手术视图重建研究通过 10 名外科医生的读片实验发现，这类感知指标与专家判断更相关。[^24]

### 7.2 分层评价

至少按以下变量分层：

- overlap：0.5–0.6、0.6–0.7、0.7–0.8、>0.8；
- hole ratio 与最大连通孔洞面积；
- 源深度有效率；
- 镜面反射强度；
- 工具是否出现；
- dataset/keyframe，而非只给所有帧的总体均值。

统计检验应以序列或病例为单位做 paired bootstrap 95% CI，并使用 Wilcoxon signed-rank 或配对 permutation test。不能把相邻帧当成相互独立的数千个样本来夸大显著性。

### 7.3 下游任务与读片实验

高质量医学论文至少增加两类证据：

1. **下游几何任务**：用补全前后图像执行 stereo depth/disparity、光流、特征匹配和相机位姿估计，报告误差是否真正下降。
2. **盲法人工评价**：外科医生或有经验的内镜研究人员比较结构合理性、立体舒适度、组织边界连续性和潜在误导风险；必须隐藏方法名并随机顺序。

若资源允许，可加入工具/组织分割作为下游任务。MICCAI 2025 的内镜伪影补全研究已通过下游分割评价修复价值，同时也因数据规模和相对领域基线的增量有限而暴露出审稿风险。[^25]

## 8. 基线和消融矩阵

### 8.1 公平基线

优先级从高到低：

1. 几何基线：hard z-buffer、最近邻小孔填充、bilinear splat、Softmax Splatting；
2. 单图补全：OpenCV Telea、LaMa/big-LaMa、UFFC、MAT；
3. 当前生成基线：SD1.5 Inpainting、当前 LCM + Endoscopy LoRA；
4. 视频补全：E2FGVI、ProPainter；
5. 医学直接基线：DAEVI、DGDVI、SSIFNet；代码不可得时应明确说明，并至少复现其可实现的核心设定。

所有方法使用完全相同的 target mask、分辨率、测试帧和 known-pixel hard composition。对需要训练的方法，训练数据量和预训练信息要列清楚，避免“自己的方法看过内镜训练集，而 LaMa 完全 zero-shot”的不公平比较。

### 8.2 主消融

| ID | 组成 | 要回答的问题 |
|---|---|---|
| A0 | 原始 LaMa | 基础补全能力有多强？ |
| A1 | A0 + 内镜领域训练 | 收益是否仅来自数据？ |
| A2 | A1 + UFFC | 更好的频域骨干贡献多少？ |
| A3 | A2 + hard point warp | 普通几何条件是否足够？ |
| A4 | A2 + deterministic soft splat | 收益是否只是 soft splatting？ |
| A5 | A2 + DSS | 深度分布和各向异性 footprint 的净收益？ |
| A6 | A5 + physics router | 传输/生成解耦是否有效？ |
| A7 | A6 + RGB-D cycle | 几何自洽是否提高？ |
| A8 | A7 + uncertainty calibration | 是否能识别失败区域？ |
| A9 | A8 + temporal memory（可选） | 视频信息的额外收益？ |

DSS 内部还要消融：单点/三点/五点深度样本，各向同性/各向异性 footprint，是否使用法向，$S/V/H$ 的不同组合，硬/软可见性，以及不同 overlap 区间。

## 9. 审稿人最可能攻击的地方

1. **“只是拼模块”**：必须把 DSS 写成明确的可微概率渲染模型，并证明其在孔洞率、遮挡排序和深度误差上优于 ordinary splat。
2. **“与 DAEVI、SSIFNet、ConFiT 重复”**：正文首图就区分 instrument removal 与 calibrated stereo NVS，强调投影分布和可见性物理量。
3. **“数据太小”**：47 帧不能作为主结果；扩大训练/测试并以病例为统计单位。
4. **“SCARED 位姿不准”**：采用或对照 SCARED-C，报告标定扰动鲁棒性。
5. **“生成了不存在的解剖结构”**：输出风险图、保留合成区域标识、做 risk-coverage 和人工误导性评价。
6. **“只赢 0.1 dB”**：当前方法与 LaMa 差距很小。目标不应只定为 PSNR；应争取至少稳定的约 0.5 dB 级改善，同时在 LPIPS/DISTS、seam、depth/disparity 和校准上形成一致优势。
7. **“baseline 不公平”**：统一 mask、训练数据、hard composition、图像尺寸，并报告 checkpoint 与推理时间。
8. **“视频闪烁”**：若论文宣称视频补全，就必须加入时序模型和指标；否则明确限定为 per-frame stereo NVS，并把视频扩展列为未来工作。

临床表述上，应将输出定位为研究级视图重建、回顾性可视化或训练辅助，而不是替代真实内镜视野。不可见区域本质上没有唯一真值，任何生成模型都可能产生合理但错误的组织；风险图和显式合成区域标记是方法设计的一部分，而不是免责声明。

## 10. 最新 NVS 工作对 novelty claim 的限制

UC-NeRF 已经针对稀疏内镜视图使用多视图一致性学习器估计几何 correspondence、特征先验和 uncertainty。[^26] 2026 年的 ExtraGS 也使用 uncertainty-guided virtual camera sampling、扩散修复和 confidence-weighted fine-tuning改善内镜视角外推。[^27] EndoSparse 则把 diffusion appearance prior 与 depth geometric prior 引入稀疏视图内镜 Gaussian Splatting。[^28]

所以不能声称：

- “首次把 uncertainty 用于内镜 NVS”；
- “首次把 diffusion/inpainting 与内镜几何重建结合”；
- “首次进行 depth-guided surgical stereo synthesis”。

可以重点论证的技术主张是：

- 将**单个不可靠深度点建模为分布**，并通过投影 Jacobian 与表面方向形成目标平面的各向异性概率 footprint；
- 从该分布式渲染过程中显式导出 support、conditional depth variance 与 collision entropy；
- 使用这些物理统计量约束 transport/synthesis 路由和可信像素硬保持；
- 在成对双目目标真值下联合校准外观误差与几何误差。

最终是否能使用“首个分布式表面 splatting 用于手术视图补全”仍需在投稿前进行一次专门 novelty search。

## 11. 实施路线与停止条件

### Phase 0：建立不可争议的评测基础（约 1–2 周）

- 固化 dataset8/9 的 47 帧列表与所有 mask；保存 manifest，确保所有方法完全同帧。
- 增加 LPIPS/DISTS、seam band、known-region、overlap bins 和 sequence-level bootstrap。
- 复现 hard splat、最近邻小孔、Softmax Splatting、LaMa、UFFC 和当前 LCM。
- 统计黑点来源：source-depth invalid、sampling hole、collision、out-of-FOV、true disocclusion。

**通过条件**：能够用互斥标签解释至少 95% 的 target invalid pixels，并生成按类型分层的基线表。

### Phase 1：先做 DSS，不接生成网络（约 2–4 周）

- 实现多尺度 differentiable DSS；先用固定深度噪声模型，再加入 learned $\sigma$。
- 与 hard/bilinear/softmax splat 比较 raw warp coverage、visible-region PSNR、depth EPE 和 occlusion boundary error。
- 可视化 $S,V,H$，检查它们是否与实际重投影误差单调相关。

**通过条件**：DSS 在不使用生成网络时就显著减少 sampling holes，并提高 visible-region RGB/depth；否则不要继续堆补全模块，应先修正 renderer。

### Phase 2：UC-UFFC 与路由（约 3–5 周）

- 用真实几何 mask 训练 UFFC；加入多尺度可靠性调制。
- 实现 Transport Expert、Synthesis Expert 和 hard trusted composition。
- 对比 binary mask、learned image confidence、physics statistics 三种路由输入。

**通过条件**：相对领域微调 UFFC，在 hole LPIPS/DISTS、seam error 和 known-region preservation 上有一致收益，而不仅是单个 PSNR 提升。

### Phase 3：RGB-D 闭环与校准（约 3–5 周）

- 添加目标深度/法向 head 和 backward geometry cycle。
- 添加 uncertainty NLL、risk ranking 和 calibration evaluation。
- 做深度/视差/特征匹配下游实验。

**通过条件**：图像质量提高的同时 disparity/depth error 不恶化；risk-coverage 曲线明显优于无校准模型。

### Phase 4：扩数据、临床评价与写作（约 4–8 周）

- 完成 SCARED-C/原始 SCARED 严格切分和至少一个跨域集。
- 如目标为 MedIA/TMI，增加时序扩展、外科医生盲评和下游任务。
- 完成所有消融、效率、统计显著性和失败案例。

## 12. 投稿定位

| 目标 | 最适合的论文故事 | 最低证据要求 |
|---|---|---|
| ICRA/IROS | 实时、几何可靠的手术双目新视图生成 | 新 renderer、FPS、机器人/内镜应用、跨序列测试 |
| MICCAI | 面向不可靠 RGB-D 的可信内镜视图补全 | 医学直接基线、多数据集、几何/下游指标、消融 |
| Medical Image Analysis / TMI | 可信手术视图重建系统 | 更大数据、时序、临床读片、统计、泛化与风险分析 |
| CVPR/ICCV | 通用的 distributional view completion 原理 | 除内镜外还需多个通用 RGB-D/NVS 数据集和更强通用 SOTA |

以现有数据和代码基础，最现实的第一目标是 **MICCAI 或 ICRA**。如果 DSS 在几何指标上非常强并能接近实时，ICRA 故事更自然；如果多数据集和临床/下游评价更完整，MICCAI 更自然。

## 13. 推荐题目、贡献写法与论文结构

推荐题目：

> **DistriSurg: Distributional Geometry Transport and Visibility-Constrained Completion for Stereo Endoscopic Novel-View Synthesis**

更偏 ICRA 的题目：

> **Reliability-Calibrated Stereo Endoscopic View Synthesis via Distributional Surface Splatting**

摘要中的三条贡献可写为：

1. We formulate stereo endoscopic view synthesis as reliability-aware transport of co-visible tissue and constrained generation of disoccluded content, instead of treating every projection hole as an inpainting mask.
2. We introduce a distributional surface splatting renderer that propagates uncertain depth observations into anisotropic target-plane footprints and exposes support, depth variance, and collision entropy for visibility-aware routing.
3. We develop a geometry-conditioned Fourier completion network with hard preservation of trustworthy observations, joint RGB-D cycle consistency, and calibrated failure prediction, and evaluate it under cross-dataset zero-shot settings using appearance, geometry, temporal, downstream, and risk metrics.

论文结构建议：

1. Introduction：黑点不是同一种缺失；“transport what is observed, synthesize only what is unseen”。
2. Related Work：image/video inpainting、geometry-guided NVS、surgical/endoscopic reconstruction。
3. Method：problem formulation、DSS、visibility router、UC-UFFC、RGB-D cycle、uncertainty calibration。
4. Experiments：datasets/splits、baseline fairness、main results、geometry/downstream、calibration、efficiency。
5. Ablations and failure analysis：projection cause taxonomy、模块消融、幻觉风险。
6. Discussion：临床边界、不可观测区域的多解性、未来时序/3D 扩展。

## 14. 最终建议

立即投入工程实现的顺序应是：**评测协议 → DSS renderer → reliability maps → UC-UFFC/router → RGB-D cycle**。不要先花大量时间继续调 prompt、LCM steps 或把更多黑点交给扩散模型；现有实验已经表明那条路不会形成明显优势。

论文成败的关键不是把 LaMa 的 22.641 dB 提高到 22.8 dB，而是证明以下完整因果链：

> 深度不确定性被正确建模 → 投影孔洞与碰撞减少 → 可见/不可见区域被正确分流 → 可信组织不被改写 → 补全后的左右眼在 RGB 与几何上同时更一致 → 模型能够识别自身可能失败的位置。

只要这条链能被主实验、消融、校准曲线和下游几何任务共同支持，就有机会形成一篇方法贡献明确、医学意义可信、审稿人较难用“简单应用 LaMa”否定的论文。

## Sources

[^1]: Liu et al., “Image Inpainting for Irregular Holes Using Partial Convolutions,” ECCV 2018. [CVF paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html)
[^2]: Yu et al., “Free-Form Image Inpainting With Gated Convolution,” ICCV 2019. [CVF paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Free-Form_Image_Inpainting_With_Gated_Convolution_ICCV_2019_paper.html)
[^3]: Suvorov et al., “Resolution-Robust Large Mask Inpainting With Fourier Convolutions,” WACV 2022. [CVF paper](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)
[^4]: Chu et al., “Rethinking Fast Fourier Convolution in Image Inpainting,” ICCV 2023. [CVF paper](https://openaccess.thecvf.com/content/ICCV2023/html/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.html)
[^5]: Li et al., “MAT: Mask-Aware Transformer for Large Hole Image Inpainting,” CVPR 2022. [CVF paper](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MAT_Mask-Aware_Transformer_for_Large_Hole_Image_Inpainting_CVPR_2022_paper.html)
[^6]: Wiles et al., “SynSin: End-to-End View Synthesis From a Single Image,” CVPR 2020. [CVF paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wiles_SynSin_End-to-End_View_Synthesis_From_a_Single_Image_CVPR_2020_paper.html)
[^7]: Niklaus and Liu, “Softmax Splatting for Video Frame Interpolation,” CVPR 2020. [CVF paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.html)
[^8]: Zhao et al., “GeoFill: Reference-Based Image Inpainting With Better Geometric Understanding,” WACV 2023. [CVF paper](https://openaccess.thecvf.com/content/WACV2023/html/Zhao_GeoFill_Reference-Based_Image_Inpainting_With_Better_Geometric_Understanding_WACV_2023_paper.html)
[^9]: Cao et al., “LeftRefill: Filling Right Canvas Based on Left Reference Through Generalized Text-to-Image Diffusion Model,” CVPR 2024. [CVF paper](https://openaccess.thecvf.com/content/CVPR2024/html/Cao_LeftRefill_Filling_Right_Canvas_based_on_Left_Reference_through_Generalized_CVPR_2024_paper.html)
[^10]: Müller et al., “MultiDiff: Consistent Novel View Synthesis From a Single Image,” CVPR 2024. [CVF paper](https://openaccess.thecvf.com/content/CVPR2024/html/Muller_MultiDiff_Consistent_Novel_View_Synthesis_from_a_Single_Image_CVPR_2024_paper.html)
[^11]: “Depth-Aware Endoscopic Video Inpainting,” MICCAI 2024. [MICCAI paper](https://papers.miccai.org/miccai-2024/209-Paper0179.html)
[^12]: “SSIFNet: Spatial–Temporal Stereo Information Fusion Network for Self-Supervised Surgical Video Inpainting,” Computerized Medical Imaging and Graphics, 2025. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/40865420/)
[^13]: Daher et al., “A Temporal Learning Approach to Inpainting Endoscopic Specularities and Its Effect on Image Correspondence,” Medical Image Analysis, 2023. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/37812856/)
[^14]: Li et al., “Enhancing Spinal Endoscopy Visualization: A Confidence-Guided Framework for Instrument Occlusion Removal,” The Visual Computer, 2026. [DOI](https://doi.org/10.1007/s00371-026-04519-6)
[^15]: Li et al., “Towards an End-to-End Framework for Flow-Guided Video Inpainting,” CVPR 2022. [CVF paper](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Towards_an_End-to-End_Framework_for_Flow-Guided_Video_Inpainting_CVPR_2022_paper.html)
[^16]: Zhou et al., “ProPainter: Improving Propagation and Transformer for Video Inpainting,” ICCV 2023. [CVF paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_ProPainter_Improving_Propagation_and_Transformer_for_Video_Inpainting_ICCV_2023_paper.html)
[^17]: “Depth-Guided Deep Video Inpainting,” IEEE Transactions on Multimedia, 2024. [University of Glasgow repository](https://eprints.gla.ac.uk/310208/)
[^18]: Tang et al., “Point-Guided Latent Diffusion Model for Novel View Synthesis in Laparoscopic Liver Surgery,” Healthcare Technology Letters, 2025. [Publisher page](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/htl2.70032)
[^19]: “Bringing Depth Perception Back to Robotic Surgery: AI Diffusion-Based Stereo Reconstruction From Monoscopic Surgical Video,” Journal of Urology abstract, 2026. [Publisher page](https://www.auajournals.org/doi/10.1097/01.JU.0001191284.35857.1d.14)
[^20]: Zhou et al., “EndoDAV: Depth Any Video in Endoscopy With Spatiotemporal Accuracy,” MICCAI 2025. [MICCAI paper](https://papers.miccai.org/miccai-2025/0288-Paper1355.html)
[^21]: Chen et al., “Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis From Monocular Videos,” NeurIPS 2025. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4459c3c143db74ee52afebdf56836375-Abstract-Conference.html)
[^22]: Allan et al., “Stereo Correspondence and Reconstruction of Endoscopic Data Challenge,” 2021. [arXiv](https://arxiv.org/abs/2101.01133)
[^23]: Han et al., “SCARED-C: Corrected RGB-D Endoscopic Dataset,” 2026. [Dataset card](https://huggingface.co/datasets/juseonghan/SCARED-C)
[^24]: Brundyn et al., “Stereo Video Reconstruction Without Explicit Depth Maps for Endoscopic Surgery,” 2021. [arXiv](https://arxiv.org/abs/2109.08227)
[^25]: “Endoscopic Artifact Inpainting Using a Phong-Inspired Reflection Model,” MICCAI 2025. [MICCAI paper and reviews](https://papers.miccai.org/miccai-2025/0296-Paper1575.html)
[^26]: “UC-NeRF: Uncertainty-Aware Conditional Neural Radiance Fields From Endoscopic Sparse Views,” IEEE Transactions on Medical Imaging, 2025. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/39531569/)
[^27]: Hsieh et al., “ExtraGS: Enhancing Endoscopic View Extrapolation via Diffusion-Guided 3D Gaussian Splatting,” 2026. [arXiv](https://arxiv.org/abs/2607.12785)
[^28]: “EndoSparse: Real-Time Sparse View Synthesis of Endoscopic Scenes Using Gaussian Splatting,” MICCAI 2024. [MICCAI paper](https://papers.miccai.org/miccai-2024/277-Paper0791.html)
