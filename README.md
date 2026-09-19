# DistriSurg

DistriSurg 是面向 ICRA/MICCAI 论文的研究实现：**Distributional Geometry Transport and Visibility-Constrained Completion for Stereo Endoscopic Novel-View Synthesis**。

它不是简单的 LaMa wrapper。代码实现了深度分布建模、各向异性可微 splatting、soft visibility、投影可靠性统计、Transport/Synthesis 双专家路由、UC-UFFC、可信像素硬保持、RGB-D 几何闭环和风险校准。

可信像素硬保持前会对像素级路由执行与 DSS footprint 尺度绑定的空间一致性清理，消除 isolated synthesis islands；原始与清理后的 mask/gate 会同时保存，便于审计和消融。

## 当前状态

- 核心网络和损失已实现，可在 CPU/GPU 前向、反向训练。
- 同时支持 iMED 固定 `pose.txt` 和 dataset89 逐帧 `pose_pairs.txt`。
- 正式训练划分使用 `/home/data/mashixing/dataset_8tb/iMed/comparison/task2-icra/train_scenes.txt` 中的 13 个场景。
- dataset8/9 被锁定为 zero-shot evaluation；训练脚本默认进行 scene 泄漏检查。
- 已实现严格 `raw DSS overlap > 0.5` 筛选、逐帧指标、sequence-level bootstrap CI 和投影缺陷诊断图。
- 仓库中的模型尚未进行完整 10,000-step 训练，因此当前代码完成不等于已经获得论文数值。

## 目录

```text
distrisurg/
  data/scared.py          两种位姿格式和严格帧配对
  geometry/dss.py         分布式表面 splatting
  geometry/diagnostics.py 投影缺陷诊断
  models/reliability.py   深度分布、法向、置信度和源特征
  models/uffc.py          uncertainty-conditioned Fourier U-Net
  models/router.py        物理先验双专家路由
  models/distrisurg.py    完整网络与反向几何闭环
  losses.py               RGB、边界、深度、cycle、router、风险损失
  metrics.py              mask-aware 指标和 scene bootstrap
scripts/train.py
scripts/infer_dataset89.py
configs/ablations/
tests/test_core.py
```

## 环境

默认复用已经安装好的环境：

```bash
/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11
```

仅依赖 PyTorch、NumPy、Pillow 和 PyYAML，不需要联网下载权重。也可安装为 editable package：

```bash
python -m pip install -e .
```

## 测试

```bash
bash run_tests.sh
```

测试覆盖：配置校验、真实 iMED 固定位姿读取、真实 dataset89 逐帧位姿读取、DSS identity/translation、renderer 梯度、刚体逆变换、完整网络、hard composition、loss backward。

## 正式训练

```bash
GPU_ID=0 bash run_train.sh
```

默认配置为 512×640、10,000 optimizer steps、batch 1、gradient accumulation 4。输出到 `checkpoints/distrisurg_main`，包含：

默认混合精度为 BF16。RTX 4090 支持 BF16，且它比 FP16 具有更大的动态范围；本项目的 DSS、FFT、深度 NLL 与多损失联合训练不应改回 FP16。如需关闭混合精度，使用 `--set train.mixed_precision=false`。

- `training_manifest.json`：训练根目录、冻结 scene list、禁止使用的评测目录和完整配置；
- `train_log.jsonl`；
- `step_*.pt` 与 `latest.pt`。

恢复训练：

```bash
RESUME=/path/to/latest.pt GPU_ID=0 bash run_train.sh
```

若上一次运行在第一个 checkpoint 前失败，重新运行时旧的 manifest 和
`train_log.jsonl` 会移动到输出目录下的 `previous_runs/failed_<timestamp>/`，
避免新旧曲线混合。若已经存在 checkpoint，脚本会要求显式设置 `RESUME` 或换一个
输出目录，不会静默覆盖。

## dataset89 零样本推理

完整模型：

```bash
CHECKPOINT=checkpoints/distrisurg_main/latest.pt \
GPU_ID=0 OVERWRITE=1 bash run_dataset89.sh
```

`run_dataset89.sh` 默认对面积不超过整图 `0.02` 的孤立未信任连通域做
Telea 局部修复，大区域仍由网络补全；每帧修复位置保存在
`small_hole_repair_mask/`，参数也会写入 `run_manifest.json`。关闭该可视化后处理：

```bash
SMALL_HOLE_MAX_AREA_RATIO=0 GPU_ID=0 OVERWRITE=1 bash run_dataset89.sh
```

论文主表应使用关闭后处理的结果，或把该步骤明确列为单独消融，不能把观察
dataset89 后选定的参数用于无偏零样本主结果。

只运行未训练的 raw DSS 几何基线：

```bash
MODE=renderer GPU_ID=0 OVERWRITE=1 bash run_dataset89.sh
```

## iMED 验证集评测

验证集评测与单元测试分开运行。默认读取冻结的 7 个验证场景、`latest.pt`，
评测所有 `raw_overlap > 0` 的有效帧，并采用与可视化推理一致的 2% 小孔修复：

```bash
GPU_ID=0 bash run_validation.sh
```

iMED 验证帧的原始深度支持率通常约为 0.42，不能照搬 dataset89 的 `>0.5`
筛选，否则会丢掉正常验证帧。如确需统一筛选，可显式设置
`MIN_OVERLAP=0.5`。

指定 checkpoint 和输出目录：

```bash
CHECKPOINT=checkpoints/distrisurg_main/step_0010000.pt \
OUTPUT=outputs/validation_step10000 \
GPU_ID=0 bash run_validation.sh
```

论文模型选择建议同时保留无后处理版本：

```bash
SMALL_HOLE_MAX_AREA_RATIO=0 \
OUTPUT=outputs/validation_step10000_no_post \
GPU_ID=0 bash run_validation.sh
```

## 统一对比方法推理

以下命令依次运行 DistriSurg、SD1.5 Inpainting、LaMa 和监督 LoRA，并将
iMED validation 与 EndoVis all 分别写入 `results/validation` 和
`results/zeroshot`：

```bash
GPU_ID=0 bash run_benchmark_inference.sh
```

可用 `SPLITS=validation`、`METHODS=distrisurg,lama` 或 `MAX_FRAMES=1`
选择子集和执行冒烟测试。MAT 仅在权重安装后通过 `METHODS=...,mat` 启用。

输出包括 renders、warps、targets、raw/trusted/hole masks、support、variance、collision entropy、risk、synthesis gate、projection taxonomy、逐帧 CSV、bootstrap JSON 和完整运行 manifest。

## 消融

推荐使用统一脚本运行消融，完整设计见
`docs/ABLATION_EXPERIMENTS_ZH.md`：

```bash
bash run_ablation.sh --list
SUITE=core DRY_RUN=1 bash run_ablation.sh
SUITE=core GPU_IDS=0,1,3 PHASE=train bash run_ablation.sh
```

例如 deterministic soft splat：

```bash
ABLATION_CONFIG=configs/ablations/a4_soft_splat.yaml \
TRAIN_OUTPUT=checkpoints/a4_soft_splat GPU_ID=0 bash run_train.sh

ABLATION_CONFIG=configs/ablations/a4_soft_splat.yaml \
CHECKPOINT=checkpoints/a4_soft_splat/latest.pt \
OUTPUT=outputs/a4_soft_splat OVERWRITE=1 GPU_ID=0 bash run_dataset89.sh
```

训练和推理必须加载相同 ablation config，否则 checkpoint 的方法定义与评测 manifest 不一致。

## 关键实验注意事项

1. `raw_overlap` 由原始有效深度 DSS 得到，不使用目标左目 GT，也不使用学习式 depth completion。
2. dataset89 只能用于最终零样本测试，不能用于调 threshold 或选择 checkpoint。
3. 论文表格还应接入 LPIPS/DISTS、stereo disparity/depth、光流/特征匹配和外科医生盲评；当前仓库先提供不依赖额外模型的核心指标。
4. `projection_taxonomy` 是诊断标签，不应在论文中称为 ground-truth occlusion taxonomy。
5. 默认 DSS `radius=1` 平衡显存和覆盖率；`radius=2` 需要单独报告速度/显存并保证所有 baseline 公平。

更完整的研究论证、相关工作、审稿风险和实验矩阵见 `docs/PUBLICATION_RESEARCH_PLAN_ZH.md`。
