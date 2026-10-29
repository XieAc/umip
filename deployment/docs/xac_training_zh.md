# XAC 数据采集后在 PRO6000 上训练

> **当前训练代码在 PRO6000 的 `~/Desktop/umip/train`，推荐使用其中的 v2 配置（`train_v2.sh`）。**
> 完整说明以该目录的 `README.md` 为准。本目录（umi_core）只保留部署推理代码，旧训练代码已移到 `archive/removed_training_20261022/`；
> 视觉编码器、`checkpoint_compat.py` 和 `train_xac_v2.yaml` 与训练端同步。

原源码许可保存在 `LICENSE.training`。旧官方配置仍有其硬件假设，不要直接使用。

## 数据和夹爪定义

默认模型为 ResNet18 视觉编码器 + 条件一维 U-Net Diffusion Policy。v2 微调 ResNet18（ImageNet 归一化、图像增强）。
输入为 D435i 的 RGB 224×224 居中 letterbox，以及最近两帧 G 系 TCP 位姿和夹爪实测宽度。
深度、IMU 用于原始记录/同步，本配置没有把它们作为策略输入。
采集和部署应保持相机安装、RGB 通道、预处理及 G 系定义一致。

默认数据频率为 10 Hz，不再做官方 3 倍降采样；位姿和宽度已对齐到图像时间轴，训练时延设为 0。
预测 16 帧动作，首末跨度 1.5 秒。Zarr 中 action 是每帧绝对
`[位置3(米), 轴角3(弧度), 开口宽度1(米)]`，加载器转换成相对当前 TCP 的位置3、旋转6D和宽度1，合计10维。
不要将 episode 中的未来相对 `action.npy` 当成 Zarr 标签。

**夹爪标签为两指尖总距离，不是电机角度、角度平均值或二值开关。**
当前 calibration/gripper_width.json 的实测关系：

| 状态 | 总开口 | ID 1 tick | ID 2 tick |
|---|---:|---:|---:|
| 闭合 | 0 mm | 3926 | 133 |
| 张开 | 100 mm | 3785 | 289 |

闭合到张开：ID 1 减小141 tick（约−12.39°），ID 2 增加156 tick（约+13.71°）；
一个编码器 tick 对应360/4096°。这是电机轴角度，不能直接称为连杆/指尖夹角。
没有连杆几何尺寸时，角度与开口使用实测分段插值，不假设线性。
采集使用 `m1-m2` 到宽度映射，两侧角度平均会抵消。执行用各侧独立的
宽度到tick插值，现有 `XC330Controller.width_to_targets()` 已实现这个反向映射。
训练无需再给夹爪维度取反；0米为闭合，数值增加为张开。
不能仅从 tick 大小给出顺/逆时针，因为它取决于观察方向和安装。

采集保存 `gripper_width_calibration.json` 快照。编码器完整转数会对齐到标定分支；
超出标定差值范围会记录 NaN，构建时排除，避免静默饱和标签。
paired_difference 模型返回的 d_left/d_right 是总宽度的一半，不能当独立测量的两指同步误差。
若换执行夹爪，必须使用执行端独立标定，不能把采集端 tick 直接发送给另一套机构。

## 采集与导出

继续使用 README 中 collect → prepare → build → check 流程。
至少需要两个原始 session 才能做分组训练/验证；实际任务应采集更多完整成功演示，
并单独保留不进入训练包的完整 session 作为最终测试。
确认 shake 已裁掉、抓取与释放的图像/宽度对齐、Stray 无跟踪跳变。

```bash
python deploy.py export --episodes-root dataset/replay \
  --output dataset/replay.zarr.zip --manifest dataset/replay_manifest.csv \
  --min-segment-length 32
```

本版导出器增加 `data/timestamps` 和 `meta/source_session`。
旧 Zarr 必须重新导出；已有输出不会被覆盖，需用新文件名。
训练按原始 session 分组划分验证集，避免同一演示的片段跨两个集合。
每个片段帧间隔必须在配置频率的±25%内；持续丢帧、混合帧率应先修复/重采样或排除。
训练校验不会自动重采样。用另一个已确认频率时设置 `task.dataset_frequeny`，
不要仅为了通过检查修改频率。采集原始文件、episode、标定快照也应归档。

## PRO6000 环境

训练机：`ssh pro6000@192.168.30.185`，代码目录 `~/Desktop/umip/train`，conda 环境 `~/miniconda3/envs/umip`
（PyTorch 2.11 + CUDA 12.8，RTX PRO 6000 Blackwell）。训练不需要 RealSense、Dynamixel SDK、UR RTDE。

新数据导出后，把 `replay.zarr.zip`、`replay_manifest.csv/.json` 和 `calibration/gripper_width.json`
复制到训练机的新数据目录（当前为 `dataset_v2/`）和 `calibration/`，然后运行 `DATASET_DIR=<目录> bash check_dataset.sh`，训练时用 `DATASET=<目录>/replay.zarr.zip bash train_v2.sh`。

新机器安装环境：

```bash
bash install_pro6000.sh xac_train      # 默认 CUDA 12.8 PyTorch
```

ResNet18 预训练权重需提前缓存到 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。

## 训练（v2）

在训练机执行：

```bash
cd ~/Desktop/umip/train
conda activate umip
bash train_v2.sh data/outputs/xac_v2_smoke training.debug=True training.action_eval_every=1   # 冒烟
tmux new -s xac_v2
bash train_v2.sh                                    # 正式训练，默认 400 轮
bash train_v2.sh data/outputs/xac_v2_800ep training.num_epochs=800   # 指定目录和轮数
```

要点（详见训练机 `README.md`）：

- 微调 ResNet18（学习率 5e-5），网络其余部分 3e-4；500 步预热后 cosine 衰减；batch 32；BF16；EMA。
- 训练时做随机裁剪、旋转和颜色增强，推理时中心裁剪。
- 每 10 轮用 EMA 模型在验证 session 上采样动作，计算位置 mm、旋转度、夹爪 mm 误差，
  按相对"原地不动"的评分 `val_action_score` 保存 `best.ckpt`。扩散噪声 `val_loss` 只记录，不用于决策。
- 不早停，跑满设定轮数；每 100 轮另存 `epoch_XXXXX.ckpt`。

输出目录包含 `.hydra/`、`console.log`、`metrics.jsonl`、`data_contract.json`、`normalizer.pt`、
`checkpoints/best.ckpt`、`last.ckpt` 和每 100 轮快照。checkpoint 包含模型、EMA、optimizer、学习率调度器和随机数状态。

离线评估任意 checkpoint：

```bash
python eval_offline.py data/outputs/<run>/checkpoints/best.ckpt
```

断点续训使用原输出目录和保存的配置：

```bash
python train.py --config-dir=data/outputs/<run>/.hydra --config-name=config \
  training.resume=True hydra.run.dir=data/outputs/<run>
```

修改数据、模型或标定后使用新实验目录。旧的 plateau 训练方案已删除；它训练出的 checkpoint 加载时会被拒绝，需要用 v2 重新训练。

## 部署与验证边界

部署端使用 `test/test_model_inference.py`（本机 .53）加载训练 checkpoint 的 EMA 权重。
将 checkpoint 从训练机复制到 `data_model/` 下，用 `--checkpoint` 指定；脚本中的默认路径已不存在。
默认只读取相机和机器人并打印预测，不发送运动；加 `--enable-motion` 并输入 `RUN` 才会运动。
v2 checkpoint 已验证可在本机加载，单次推理约 87 ms（RTX 3060）。

离线回归检查：

```bash
python -m unittest test_policy_loading test_xac_training test_collection_deployment test_replay_deployment test_replay_xac_ur5 -v
```

离线误差只说明模型在已有演示上的拟合程度。当前只有 18 条演示，验证集只有 2 条，
`best.ckpt` 又是按这 2 条挑选的，验证误差偏乐观。必须先用预览模式检查预测，
再低速短段实机测试，并用未参与训练的新演示评估。
