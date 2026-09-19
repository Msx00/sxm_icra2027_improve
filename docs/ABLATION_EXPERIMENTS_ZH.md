# DistriSurg 消融实验设计与运行规范

## 1. 实验原则

主消融采用“完整模型减去一个因素”的设计，除被研究因素外，训练场景、输入分辨率、优化步数、随机种子、数据增强、损失权重和 zero-shot 帧列表保持一致。这样每一行都能回答一个明确的因果问题。`hard_point`、`soft_splat` 和 `dss_no_router` 是机制递增的 renderer 基线，不应与单因素消融混为一类。

所有 dataset89 推理继续使用严格的 renderer-only `raw_overlap > 0.5` 条件。不得根据 dataset8/9 的结果挑 checkpoint、调阈值或修改训练超参数。

所有消融必须使用同一种数值精度。默认配置固定为 RTX 4090 支持的 BF16；不要让部分实验使用不稳定的 FP16、部分实验使用 FP32，否则运行稳定性和速度均不可公平比较。

## 2. 推荐主表：core suite

| 运行名 | 唯一变化 | 要回答的问题 | 重点指标 |
|---|---|---|---|
| `full` | 无 | 完整 DistriSurg | 全部指标 |
| `no_depth_completion` | 不补全 source depth | 深度补全是否改善投影支持 | coverage、hole PSNR、depth error |
| `deterministic_depth` | 三个 sigma 点改为单个均值点 | 分布式深度是否必要 | collision、seam、hole PSNR |
| `isotropic_footprint` | 去掉法向相关 footprint | 各向异性 footprint 是否有效 | boundary/seam、coverage |
| `no_physics_router` | physics router 改为 binary valid gate | transport/synthesis 分流是否有效 | hole/visible PSNR、known drift |
| `no_hard_composition` | 去掉可信像素硬覆盖 | 硬约束是否防止已知区域漂移 | known drift、visible PSNR |
| `no_rgbd_cycle` | 去掉反向 RGB-D cycle | 几何闭环是否提高一致性 | disparity/depth、seam、PSNR |
| `no_uncertainty_loss` | 风险损失权重置零 | 风险头是否真正得到校准 | risk ECE、AURC |
| `no_trust_cleanup` | 关闭空间一致可信掩码 | 盐椒路由清理是否改善视觉连续性 | isolated islands、seam、PSNR |

建议先以 seed 6666 完成全部 core 实验；只有在完整模型与关键消融差异接近噪声时，再对 `full`、`deterministic_depth`、`no_physics_router`、`no_hard_composition`、`no_rgbd_cycle` 和 `no_trust_cleanup` 增加 7777/8888 两个种子。不要为了让表格更漂亮而只给部分不利结果增加种子。

空间一致清理核固定为 `2*ceil(max_footprint_px)+1=7`，这是由 DSS footprint 尺度预先确定的规则。dataset89 上观察到的粉色盐椒点只能作为失败诊断，不能用于继续搜索核大小；如需改变规则，必须在训练域验证集决定并重新冻结协议。

## 3. Renderer 机制链：renderer suite

| 运行名 | 组成 | 用途 |
|---|---|---|
| `hard_point` | observed depth、单点、窄 footprint、binary gate | 最弱几何基线 |
| `soft_splat` | observed depth、单点、固定 Gaussian footprint | 验证收益是否仅来自 soft splatting |
| `dss_no_router` | depth completion、distributional anisotropic DSS、无 physics router/cycle | 隔离 DSS 本身的收益 |

这组三行应额外报告 renderer-only coverage、可见区域 RGB error、target-depth error、遮挡边界排序、DSS latency 和显存。只比较最终生成 PSNR 无法证明新 renderer 的贡献。

## 4. DSS 聚焦实验：dss suite

`dss_1_sample`、`full`（3 samples）、`dss_5_samples` 比较 sigma 点数量；`isotropic_footprint` 检验法向相关 footprint。应同时报告质量和效率，并检查 5 点采样是否只增加耗时而没有稳定收益。

## 5. 运行命令

先查看实验名称和 dry-run：

```bash
bash run_ablation.sh --list
SUITE=core DRY_RUN=1 bash run_ablation.sh
```

训练推荐主消融：

```bash
SUITE=core GPU_IDS=0,1,3 PHASE=train bash run_ablation.sh
```

训练后统一 zero-shot 推理并生成比较表：

```bash
SUITE=core GPU_IDS=0,1,3 EVAL_GPU_ID=0 PHASE=eval bash run_ablation.sh
```

指定少数实验或多随机种子：

```bash
EXPERIMENTS=full,no_rgbd_cycle SEEDS=6666,7777,8888 \
GPU_IDS=0,1,3 PHASE=train bash run_ablation.sh
```

短流程只用于验证代码，不能进入论文表格：

```bash
SUITE=core STEPS=20 SAVE_EVERY=20 PHASE=train bash run_ablation.sh
```

脚本默认跳过 manifest 中 `completed=true` 的运行，并在存在 `latest.pt` 时自动恢复。输出分别位于 `checkpoints/ablations/<name>/seed_<seed>` 和 `outputs/ablations/<name>/seed_<seed>`，不会互相覆盖。

## 6. 统计与报告

1. 主表至少报告 full/hole/visible/seam PSNR、SSIM、known drift、risk ECE/AURC、几何误差、参数量、峰值显存和端到端延迟。
2. 置信区间以 sequence 为抽样单位做 paired bootstrap；不能把相邻视频帧当独立样本。
3. 同一比较必须使用完全相同的合格帧 manifest。建议保存首次 `full` 运行的 sample IDs，并在所有消融中进行集合一致性检查。
4. 多 seed 先对每个 seed 做 sequence-macro mean，再汇总 seed 均值与标准差；不要把 seed 和 frame 混在一起 bootstrap。
5. `no_uncertainty_loss` 主要解释风险校准，不应仅凭其 PSNR 判断该模块是否有效。
6. `hard_point`、`soft_splat` 属于复合机制基线；论文中只有 core suite 的行可以表述为单因素净贡献。
