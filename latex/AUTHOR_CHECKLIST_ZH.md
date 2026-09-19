# DistriSurg 作者核对与实验 TODO

## 一句话论文论证

> In calibrated stereo endoscopic view synthesis, we propose to propagate per-pixel depth distributions through geometry and use the resulting support, variance, and collision statistics to constrain transport, synthesis, and risk; the claim must ultimately be supported by renderer, completion, geometry, calibration, and zero-shot evidence, and is bounded to known camera geometry and research-grade visualization.

中文：在已标定双目内镜新视角合成中，将每像素深度分布传入几何渲染，并用投影产生的支持度、方差和碰撞统计约束传输、生成与风险；最终结论必须由 renderer、补全、几何、校准和零样本证据共同支撑，适用边界是相机几何已知的研究级可视化。

## 术语表（已锁定）

| Canonical term | 首次定义 | 不采用的变体/备注 |
|---|---|---|
| DistriSurg | Distributional Geometry Transport and Visibility-Constrained Completion for Stereo Endoscopic Novel-View Synthesis | 全文不改成 LaMa-based method |
| DSS | distributional surface splatting | 不写 probabilistic Gaussian splatting；这里不是 3DGS 场景表示 |
| depth reliability encoder | 输出 depth mean/scale、normal、confidence、source features | 不与 target risk 混称 uncertainty encoder |
| support $S$ | DSS 权重质量和 | 区分 coverage $C=1-e^{-S}$ |
| depth variance $V$ | DSS 条件深度方差 | 单位 mm$^2$ |
| collision entropy $H$ | 归一化权重熵乘深度色散因子 | 不是语义 entropy |
| synthesis gate $g$ | 需要生成的连续程度 | $g=0$ 为 transport，$g=1$ 为 synthesis |
| trusted mask $M^{\rm trust}$ | 满足 support/variance/entropy 阈值的 DSS 像素 | 不等于 raw valid mask |
| Fourier synthesis expert | uncertainty-conditioned Fourier U-Net | 受 LaMa/UFFC 启发，但没有加载其预训练权重，勿称 UFFC 复现 |
| risk $U$ | 预测图像误差尺度的归一化变量 | 不宣称概率校准，除非 ECE/AURC 实验支持 |
| raw overlap | raw-depth renderer-only valid ratio | 必须严格 `> 0.5`，不能写 `>= 0.5` |

## Claim--evidence 对照

| Claim | 所需证据 | 当前状态 |
|---|---|---|
| DSS 比 hard/ordinary soft splat 更好地处理不确定深度 | raw coverage、visible PSNR/depth、occlusion ordering、分层 CI | 需要实验 |
| $S,V,H$ 能表达投影可靠性 | 与真实重投影误差的单调性、分箱曲线、AUC/相关系数 | 需要实验 |
| physics router 的收益不是网络容量带来的 | binary mask、image confidence、$S/V/H$、参数量匹配消融 | 需要实验 |
| hard composition 保留观测证据 | known drift（均值、最大值）应为零或浮点误差量级 | 代码保证，仍需报告 |
| RGB-D backward cycle 提高几何一致性 | RGB、depth/disparity、cycle off/on 消融 | 需要实验 |
| risk 能识别失败区域 | ECE、NLL、AURC、risk--coverage 曲线、OOD 失效案例 | 需要实验 |
| 模型具有跨数据零样本泛化 | dataset8/9 完全不参与训练、阈值、checkpoint 选择；sequence bootstrap | 协议已实现，结果待跑 |
| 适合 ICRA 的效率主张 | end-to-end FPS、DSS-only latency、显存、参数量、硬件/精度 | 需要实验 |

## 必做实验 TODO

1. 冻结训练 manifest：13 个训练场景、总 pair 数、checkpoint hash、随机种子和最终 config。
2. 冻结 dataset8/9 renderer-only `raw_overlap > 0.5` 帧列表；报告总帧数及各 sequence 数量。
3. 公平基线：hard point、邻近小孔、soft/softmax splat、LaMa、UFFC、MAT、SD1.5 Inpainting、LCM+LoRA。
4. 主指标：full/hole/visible/seam PSNR、SSIM、LPIPS、DISTS；known drift。
5. 几何指标：target depth AbsRel/RMSE、normal angular error、disparity EPE、left-right reprojection error。
6. 风险指标：ECE、NLL、AURC、risk--coverage；按 overlap、hole ratio、depth validity、tool 分层。
7. DSS 消融：1/3/5 sigma points，各向同性/各向异性，去 normal，hard/soft visibility，radius 1/2。
8. Router 消融：binary mask、learned image confidence、physics prior、physics + bounded residual、逐一去掉 $S/V/H$。
9. 训练消融：去 hard composition、去 target depth、去 backward cycle、去 risk loss。
10. 统计：sequence/patient 为单位的 paired bootstrap 95% CI 和 paired permutation 或 Wilcoxon；不能把相邻帧当独立样本。
11. 效率：同一 GPU、同一分辨率、统一 warm-up；报告端到端和 renderer-only。
12. 失败案例：specularity、thin tool、large disocclusion、calibration perturbation、deformation、OOD anatomy。
13. 如能组织人工评价：盲法、随机顺序，评价结构合理性、立体舒适度、边界连续性和潜在误导。

## 写作决策

- Introduction 采用“应用/任务 → 立即暴露技术挑战 → 现有方法边界 → 观察驱动方案”的结构。
- Related Work 按机制分为 inpainting、geometry-guided transport/completion、endoscopic NVS/uncertainty，而不是按年份罗列。
- 摘要、Results、Discussion 和 Conclusion 中所有需要真实数据的句子保留红色 TODO；没有沿用旧 LaMa/LCM 先导值。
- 不使用“first”“state of the art”“clinically safe”或“real-time”等未证实表述。
- 生成区域必须定位为研究级推断，不能替代真实内镜观测。

