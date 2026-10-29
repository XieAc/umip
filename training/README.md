# 训练（PRO6000 / RTX 5090）

这个目录是 XAC 策略训练的**当前代码**，训练不需要连接 UR5、D435i 或 Dynamixel 夹爪。

**数据不在仓库里。** 当前数据是 `dataset_v3/`（2026-10-25 用修正后的 T_I_G 从原始数据重建），把 `replay.zarr.zip`、`replay_manifest.csv`、`replay_manifest.json` 放到本目录的 `dataset_v3/` 下即可；`train_v2.sh`、`check_dataset.sh`、`eval_offline.py` 的默认数据集都是 `dataset_v3`。旧的 `dataset_v2` 用的是有 10° 误差的旧标定，不要再用。

数据包：18 个 episode，1636 帧，10 Hz Zarr。按原始 session 分组，16 条用于训练（1193 个窗口），2 条用于验证（173 个窗口）。当前用于部署的模型是 `xac_v3_h1img1_800ep_20261025`：

```bash
bash train_v2.sh data/outputs/xac_v3_h1img1_800ep training.num_epochs=800 task.low_dim_obs_horizon=1 task.img_obs_horizon=1
```

环境：Python 3.10 + torch 2.11.0+cu128（支持 Blackwell）。`requirements-lock.txt` 是 RTX 5090 上与 PRO6000 完全一致的 `pip freeze`（不含 torch/torchvision，先用 `--index-url https://download.pytorch.org/whl/cu128` 装这两个）。ResNet18 预训练权重需放在 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。下文的 `~/Desktop/umip/train` 指本目录。

夹爪标签是两指尖总开口宽度（米），不是电机角度。模型输出宽度，部署时再用 `calibration/gripper_width.json` 的实测映射转换为两个电机目标。

## 快速开始（推荐：v2 配置）

```bash
cd ~/Desktop/umip/train
conda activate umip
bash check_dataset.sh                                   # 数据检查
bash train_v2.sh data/outputs/xac_v2_smoke training.debug=True training.action_eval_every=1   # 冒烟
tmux new -s xac_v2
bash train_v2.sh                                        # 正式训练，输出 data/outputs/xac_v2_<时间>
```

`train_v2.sh` 的第一个参数是输出目录，后面的参数原样传给 Hydra，例如：

```bash
bash train_v2.sh data/outputs/xac_v2_800ep training.num_epochs=800
```

输出目录必须是新的；已存在时脚本会拒绝运行。ResNet18 预训练权重默认读取
`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`，可用 `PRETRAINED_WEIGHTS` 环境变量指定其他路径。

## v2 训练策略

配置文件：`diffusion_policy/config/train_xac_v2.yaml`（唯一的训练配置）。

| 项目 | 设置 |
|---|---|
| 视觉编码器 | ResNet18 ImageNet 预训练，**微调**，学习率 `training.obs_encoder_lr=5e-5`；BN 统计量固定（`frozen_bn`） |
| 输入归一化 | `input_norm: imagenet`，真正执行 ImageNet 均值/方差归一化 |
| 特征聚合 | `spatial_embedding`，参与训练 |
| 图像增强（仅训练时） | 随机裁剪 95% 后缩放、±5° 旋转、ColorJitter（亮度 0.3，对比度 0.4，饱和度 0.5，色调 0.08）；推理时用中心裁剪 |
| 扩散网络 | U-Net `down_dims=[128,256,512]`，DDIM 训练 50 步、推理 16 步，预测噪声 |
| 优化器 | AdamW，网络 lr 3e-4，betas (0.95, 0.999)，weight decay 1e-6，梯度裁剪 1.0，BF16 |
| 学习率 | 前 500 步线性预热，之后 cosine 衰减到 0 |
| 轮数 / batch | 400 轮（可覆盖），batch 32 |
| EMA | power 0.75，上限 0.9999；验证、选模型、部署都用 EMA 权重 |
| 观测 / 动作 | 2 帧 RGB + 相对位姿 + 宽度；预测 16 步，部署执行前 8 步 |

### 验证与模型选择

- **每轮**：计算验证集扩散噪声 loss（`val_loss`），只记录，不用于任何决策。
  这个指标与动作质量相关性很差：实验中它在第 130 轮左右开始上升，而实际动作误差一直下降到最后一轮。
- **每 10 轮**（`training.action_eval_every`）：用 EMA 模型在验证集全部窗口上真正采样动作，
  计算前 8 步的位置误差（mm）、旋转误差（度）、夹爪误差（mm），并与"原地不动"基线对比。
- **评分** `val_action_score` = 三项误差分别除以"原地不动"误差后相加。3.0 等于原地不动，越低越好。
- `best.ckpt`：评分最低的一次（`training.best_metric=val_action_score`）。
- `last.ckpt`：每轮覆盖。`epoch_00100.ckpt` 等：每 100 轮另存（`training.save_every`）。

### 早停

v2 **不早停**，跑满设定轮数。原因：噪声 loss 会误导；cosine 学习率要跑到末尾才能收敛；
实测 400 轮的最佳结果出现在最后一轮。过拟合由 `best.ckpt` 的选择来规避。

`best.ckpt` 是按同样 2 条验证演示挑选的，报告的验证误差会略偏乐观。
最终效果以实机测试或从未参与训练/挑选的新演示为准。

## 离线评估

```bash
python eval_offline.py data/outputs/<run>/checkpoints/best.ckpt
```

输出验证集和部分训练集上的 mm/度误差、"原地不动"基线和评分。任何旧 checkpoint 也能用它评估。
训练集与验证集误差差距很大，说明模型在记忆数据，需要更多演示。

2026-10-22 结果（验证集，执行的前 8 步）：

| 模型 | 位置 | 旋转 | 夹爪 | 评分 |
|---|---|---|---|---|
| 原地不动 | 24.4 mm | 1.38° | 6.3 mm | 3.00 |
| 旧方案最好（plateau，已不可加载） | 18.8 mm | 3.45° | 6.1 mm | — |
| 旧部署默认（10k，已不可加载） | 38.7 mm | 7.24° | 20.7 mm | — |
| v2 冻结 backbone，400 轮 | 10.6 mm | 1.22° | 3.7 mm | 1.90 |
| **v2 微调，400 轮** | **9.0 mm** | **1.18°** | **3.5 mm** | **1.78** |

## 断点续训

使用原输出目录、原配置直接调用 `train.py`：

```bash
python train.py --config-dir=data/outputs/<run>/.hydra --config-name=config \
  training.resume=True hydra.run.dir=data/outputs/<run>
```

数据、标定、shape 或策略配置改变时会被拒绝，需要新建实验。
cosine 调度的长度由总轮数决定，想多训练应新开实验，不要续训时改 `num_epochs`。

## 已删除的旧训练方案（2026-10-22）

旧方案有以下错误，相关代码已删除：

1. `frozen: True` 冻结整个编码器，`spatial_embedding` 一直停留在随机初始值。现在 `frozen` 只冻结 ResNet，pooling 层始终训练。
2. `imagenet_norm: True` 从未生效。现在输入始终做 ImageNet 归一化，该选项已删除。
3. 无图像增强且编码特征只缓存一次。特征缓存代码已删除。
4. plateau 调度、早停和 `best.ckpt` 都依据 `val_loss`，选中了训练不足的模型。plateau 和早停已删除，`best.ckpt` 固定按动作误差选择。

旧方案训练出的 checkpoint 加载时会被拒绝（`ObsoleteCheckpointError`），不能再用于评估或部署，需要用 v2 重新训练。
删除前的代码备份在 `backups/pre_cleanup_20261022/` 和 `backups/pre_refactor_v2_*`。

## 主要文件

| 文件 | 作用 |
|---|---|
| `train.py` | Hydra 入口 |
| `train_v2.sh` | 启动 v2 新实验 |
| `eval_offline.py` | checkpoint 离线动作误差评估 |
| `check_dataset.sh` | 数据与标定检查 |
| `diffusion_policy/config/train_xac_v2.yaml` | v2 配置 |
| `diffusion_policy/config/task/xac.yaml` | 任务/数据定义 |
| `diffusion_policy/workspace/train_xac_workspace.py` | 训练循环、验证、选模型、checkpoint |
| `diffusion_policy/common/action_metrics.py` | 动作误差和评分 |
| `diffusion_policy/common/checkpoint_compat.py` | 加载旧 checkpoint 时的配置检查 |
| `test_obs_encoder.py` | 编码器冻结范围、归一化、增强和加载规则测试 |
| `diffusion_policy/model/vision/timm_obs_encoder.py` | 视觉编码器 |
| `diffusion_policy/dataset/xac_dataset.py` | XAC 数据加载与检查 |

## 新环境安装

```bash
bash install_pro6000.sh xac_train      # 默认 CUDA 12.8 PyTorch
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash install_pro6000.sh xac_train
```

安装结束必须看到 `cuda_available: True` 和正确的 GPU 名称。当前机器使用的是 `~/miniconda3/envs/umip`。

## 夹爪标签

动作标签为 `[TCP位置3(米), TCP轴角3(弧度), 两指尖总开口1(米)]`，网络内部转换为相对当前 TCP 的
`[位置3, 6D旋转, 宽度1]` 共 10 维。宽度增大表示张开。夹爪从闭合到张开时，ID 1 编码器值减小，ID 2 增大。
代码使用 `m1-m2` 的实测标定关系，不能对两个电机 tick 求平均。

## 部署

部署在 .53（`~/Desktop/umi_core`）上运行 `test/test_model_inference.py`，它读取 checkpoint 中的配置和 EMA 权重。
将 `best.ckpt` 复制到 .53 后用 `--checkpoint` 指定；默认只预览预测，不发送运动指令。v2 checkpoint 已验证能在 .53 上加载，单次推理约 87 ms（RTX 3060）。

请保留 `dataset_v3/` 的 Zarr 与 manifest、`calibration/` 和整个输出目录。
