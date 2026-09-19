# DistriSurg 方法定义

## 研究问题

输入源视图 RGB、稀疏深度、深度有效 mask、源/目标内参和源到目标位姿，输出目标视图 RGB、目标深度与像素级风险。模型用 DSS 可靠性统计约束几何传输与内容生成的软融合，不在输出端执行可信像素硬复制。

## 模块

1. `DepthReliabilityEncoder` 对观测深度做小残差校正，对无效深度做上下文补全，并预测深度标准差、表面法向、传输置信度和源特征。
2. `DistributionalSurfaceSplat` 使用确定性 sigma samples 表示每个深度分布。投影 Jacobian 将深度方差传播为目标平面的各向异性 footprint；soft nearest-surface term 处理遮挡竞争。
3. DSS 同时输出 warped features、期望深度、support、coverage、depth variance、collision entropy 和目标置信度。
4. `VisibilityRouter` 用物理统计量生成 synthesis prior，学习分支只能预测有界残差；通过可靠性测试的 DSS 像素将 synthesis gate 置零，其余区域进行软融合。
5. `TransportExpert` 对 warp 做有界颜色校正；`UncertaintyConditionedUFFC` 预测合成 RGB、深度与风险。最终 RGB-D 由可靠性约束的 gate 加权融合，不再 hard-compose 原始 warp。
6. 训练时将预测目标 RGB-D 反向 DSS 到源视图，形成几何闭环。

最终输出为

$$
\widehat I_t=(1-g)I_t^{\mathrm{trans}}+gI_t^{\mathrm{syn}},\qquad
\widehat D_t=(1-g)\bar D_t+gD_t^{\mathrm{syn}},\qquad
U_t=U_t^{\mathrm{syn}}.
$$

可靠区域满足 $g=0$，但输出仍来自可学习的有界 transport expert，而不是对 DSS warp 的逐像素硬复制。清理后的可靠性 mask 只用于区域损失和诊断，不改写最终预测。

主设置中的路由可靠区域为

$$
M^{\mathrm{rel}}=M^{\mathrm{dss}}\land[S\geq0.35]\land[V\leq9],
$$

因为 collision entropy 的硬阈值为 1，在 $\mathcal H\in[0,1]$ 上不产生额外筛除；$\mathcal H$ 仍参与目标置信度和 soft prior。区域损失使用
$M^{\mathrm{reg}}=M^{\mathrm{dss}}\land\neg\operatorname{Open}_7(1-M^{\mathrm{rel}})$，该清理不改变 gate 或最终输出。

## DSS 方程

对源像素 $q$ 的深度分布取样 $z_{qk}=\mu_q+a_k\sigma_q$，投影权重为

$$
w_{qk\to p}=c_q\alpha_k
\mathcal K_{\Sigma_q}\left(p-\pi(T\Pi^{-1}(q,z_{qk}))\right)
\exp\left(-\frac{z_{qk}-z_{min}(p)}{\tau}\right).
$$

目标特征为

$$
F_t(p)=\frac{\sum_{q,k}w_{qk\to p}F_s(q)}
{\sum_{q,k}w_{qk\to p}+\epsilon}.
$$

$\Sigma_q$ 由基础 footprint、投影 Jacobian 传播的深度方差和表面倾角组成。代码额外计算加权深度方差和与深度离散度耦合的 collision entropy，避免把普通亚像素插值错误识别为多表面冲突。

## 数据协议

- 训练：`task2-icra/train_scenes.txt` 中的 13 个 iMED 场景，根目录为 `iMed/datasets/task2-nvs`。
- 零样本测试：EndoVis/SCARED dataset8/9，代码禁止训练 scene 与评测 scene 交叉。
- 筛选：用未经过学习式深度补全的 raw DSS mask，严格保留 `raw_overlap > 0.5` 的帧。
- 指标：全图、raw hole、raw visible、7-pixel seam 的 PSNR/SSIM，以及 known drift、risk ECE/AURC 和投影缺陷类型占比。

## 投影诊断标签

`projection_taxonomy` 仅是可解释诊断，不宣称为遮挡真值：

- 灰：可靠或未分类；
- 黄：局部 sampling gap；
- 绿：学习式深度补全可恢复；
- 蓝：仍无支持或可能 disocclusion；
- 红：高方差/高碰撞风险。

## 主方法配置

训练与推理统一使用 `configs/train.yaml`。主方法直接设置
`hard_composition: false`，不再依赖额外的消融配置覆盖。
