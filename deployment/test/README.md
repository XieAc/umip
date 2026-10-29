# XAC 真机模型测试

`test_model_inference.py` 将 XAC Diffusion Policy 接到 D435i、UR5 和 XC330。模型输入是最近两帧 `camera0_rgb`、末端位置、末端 6D 旋转和夹爪总开口；模型输出为 16 步、每步 10 维的相对动作。预览默认显示预测序列的第 1 步；实际运动使用带时间戳的未来动作轨迹，并固定扩散采样种子以减少重规划抖动。采样和推理各在独立线程中运行，主循环持续以 125 Hz 发送位置插值、姿态 Slerp 插值后的 `servoL` 指令。

默认运行模式是预览：读取相机和 UR5，执行模型推理并打印目标，不发送运动命令。只有同时加入 `--enable-motion` 并输入 `RUN`，程序才会调用 UR5 `servoL`。程序还会检查 G 点距离、Base Z、跟随误差和 UR5 安全停止。

## 1. 前置条件

在连接硬件的 Linux 电脑上执行：

```bash
cd /home/eirl3060/Desktop/umi_core
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

实时测试建议使用 CUDA。程序默认使用 `cuda:0`；CUDA 不可用时会回退 CPU，但 CPU 推理通常达不到 10 Hz。

确认以下文件存在，并且与当前夹爪安装和 TCP 定义匹配：

```text
data_model/outputs/xac_bf16_cached_lr1e-3_10k_20261011/checkpoints/best.ckpt
calibration/tcp_gripper_01_gx_down_candidate.json
calibration/gripper_width.json
```

当前 BF16 缓存训练的小模型 `best.ckpt` 约 221 MB，包含完整 EMA 模型和训练状态。加载时保留训练使用的 BatchNorm/GroupNorm 结构，直接使用 checkpoint 内部权重，不下载 ImageNet 权重。

训练和部署都使用 RGB 224x224 中心 letterbox、10 Hz 数据和最近两帧观测。相机安装、G 坐标、夹爪宽度单位必须与采集时相同。拆装法兰、夹爪、手机或相机支架后，必须重新检查 TCP 和宽度标定。

## 2. 只读硬件检查

先确认设备和项目文件：

```bash
python deploy.py check --project
python deploy.py gripper-check \
  --port /dev/ttyUSB0 --baudrate 57600 --ids 1 2
```

启动 D435i 检查彩色和深度流：

```bash
python deploy.py camera \
  --width 640 --height 480 --fps 30 --no-imu
```

按 `Ctrl-C` 停止相机。模型会把 640x480 彩色图像转换为 224x224 输入。

检查 UR5 当前状态和 TCP 变换：

```bash
python deploy.py verify-tcp \
  --robot-ip 192.168.1.101 \
  --calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --test translation --axes x
```

该命令默认只读，不会移动机器人。运行前 UR5 必须上电、静止、没有运行中的程序，也不能处于保护停止或急停状态。

当前候选 TCP 文件要求独立验证轴方向，文件中仍标记 `orientation_verified: false`。在实际运动前，应先在空旷区域用小步长验证 G 轴方向：

```bash
python deploy.py verify-tcp \
  --robot-ip 192.168.1.101 \
  --calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --test translation --axes x \
  --step-mm 3 --speed-mm-s 1 \
  --report calibration/tcp_axis_x_report_01.json
```

输入 `RUN` 后，程序还会要求确认每个测试段。首次只验证一个轴；方向确认后再验证其他轴。

## 3. 模型预览，不运动

先运行预览，不要加 `--enable-motion`：

```bash
python test/test_model_inference.py \
  --robot-ip 192.168.1.101 \
  --checkpoint data_model/outputs/xac_bf16_cached_lr1e-3_10k_20261011/checkpoints/best.ckpt \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --d435i-width 640 --d435i-height 480 --d435i-fps 30 \
  --d435i-no-imu \
  --minimum-base-z-m <现场安全下限> \
  --maximum-base-z-m <现场安全上限> \
  --preview-steps 5 \
  --trace logs/replay/policy_preview.json
```

UR5 不动是正常的，因为没有 `--enable-motion`。预览阶段不会调用 `servoL`，也不会给夹爪发送运动目标。输出中的 `step=0 target=[...] width=...` 表示推理成功。

如果要读取真实夹爪宽度，增加：

```bash
--enable-gripper --gripper-port /dev/ttyUSB0
```

预览阶段仍不会打开夹爪扭矩。

`--minimum-base-z-m` 和 `--maximum-base-z-m` 必须按现场测量填写。将它们设为 `-1` 和 `1` 基本等于关闭 Base-Z 保护，不适合正式运动测试。

测试入口还支持：

```bash
--action-index 1       # 预览显示的动作行；运动时允许使用的最早动作行，默认 1
--inference-seed 0     # 固定扩散随机种子，默认 0；-1 表示每次重新随机采样
```

训练数据的动作窗口从当前观测时刻开始，因此第 0 步通常接近当前位姿。运动时以 checkpoint 中的 10 Hz 动作频率为每一行赋予时间，跳过已经过时或不足以完成平滑衔接的行；实际首行通常为第 2～4 行，取决于推理耗时。`--policy-hz` 只控制推理更新频率，不改变训练动作的时间间隔。新轨迹从当前伺服指令衔接，推理期间继续执行上一条轨迹。固定种子只能减少采样噪声，不能保证模型完成特定方向的任务。

默认平滑和限速参数如下（速度、加速度限制施加于控制器的活动 TCP）：

```text
--trajectory-blend-s 0.1
--max-linear-speed-m-s 0.05
--max-angular-speed-rad-s 0.3
--max-linear-acceleration-m-s2 0.1
--max-angular-acceleration-rad-s2 0.6
--prediction-timeout-s 1.0
```

程序在运动确认前先热身模型。输入 `RUN` 后，先连接 UR5 控制接口并初始化夹爪，再等待初始化后采集的两帧新观测，最后启动连续推理和伺服；每次等待新观测最多 5 秒，连接或初始化期间留下的旧帧不会直接触发运行超时。相机读取使用容量为 2 的队列，并丢弃旧帧；独立采样线程以训练频率读取最新图像及邻近时刻的机器人状态，夹爪串口读取和写入也在此线程中进行，不阻塞伺服循环。图像采集与机器人状态读取并非硬件同步。运动期间预测、观测或 RTDE 状态超时，以及跟随误差、距离或 Base-Z 超限时，程序停止伺服。

每次推理 trace 还会记录坐标变换诊断字段：

```text
relative_action_pose10d              当前预测首次选用的动作行，原始相对 G 动作
raw_target_g_m_rad                   时间插值后、限速前的 G 位姿
target_g_m_rad                       限速后的实际指令 G 位姿（UR Base 中）
controller_target_active_tcp_m_rad   TCP 补偿后实际发送给 UR 控制器的目标
actual_g_m_rad                       读取回来的 G 实际位姿
actual_active_tcp_m_rad              读取回来的 UR 激活 TCP 位姿
control_interval_s                  相邻伺服周期的实际间隔
inference_time_s                    当前预测的推理耗时
policy_age_s                        当前预测距其观测时刻的时间
source_action_index                 当前预测首次采用的动作行
observation_gap_s                   两帧观测的主机采样时间间隔
camera_timestamp_ms                 相机原始帧时间戳
```

判断方法：如果 `target_g_m_rad` 的变化方向错误，问题在模型输出或输入/动作表示；如果 `target_g_m_rad` 正确但 `controller_target_active_tcp_m_rad` 方向错误，问题在 `T_F_G`；如果两者都正确而 `actual_g_m_rad` 不跟随，问题在 UR5 执行或安全限速。

## 4. 首次低速测试：只测试 UR5

预览目标和 TCP 方向确认后，先不加 `--enable-gripper`：

```bash
python test/test_model_inference.py \
  --robot-ip 192.168.1.101 \
  --checkpoint data_model/outputs/xac_bf16_cached_lr1e-3_10k_20261011/checkpoints/best.ckpt \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion \
  --d435i-width 640 --d435i-height 480 --d435i-fps 30 \
  --d435i-no-imu \
  --minimum-base-z-m <现场安全下限> \
  --maximum-base-z-m <现场安全上限> \
  --max-duration-s 5 \
  --max-distance-from-start-m 0.02 \
  --max-following-error-m 0.01 \
  --max-rotation-following-error-rad 0.15 \
  --policy-hz 10 \
  --trace logs/replay/policy_arm_only.json
```

出现 `Type exactly "RUN" to enable UR5 motion:` 后，只有确认工作区无人、示教器速度低、急停可触达、起始位姿安全、TCP 方向已验证后，才输入 `RUN`。第一次将距离和持续时间限制得更小。程序不会自动回到起始位姿。

## 5. 测试夹爪

机械臂测试稳定后，再启用 XC330：

```bash
python test/test_model_inference.py \
  --robot-ip 192.168.1.101 \
  --checkpoint data_model/outputs/xac_bf16_cached_lr1e-3_10k_20261011/checkpoints/best.ckpt \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --enable-gripper \
  --gripper-port /dev/ttyUSB0 --gripper-baudrate 57600 \
  --d435i-width 640 --d435i-height 480 --d435i-fps 30 \
  --d435i-no-imu \
  --minimum-base-z-m <现场安全下限> \
  --maximum-base-z-m <现场安全上限> \
  --max-duration-s 5 \
  --max-distance-from-start-m 0.02 \
  --max-following-error-m 0.01 \
  --trace logs/replay/policy_arm_gripper.json
```

第一次夹爪测试不要夹工件，确认两个电机方向、宽度映射和电流都正常。夹爪宽度必须落在 `gripper_width.json` 的标定范围内。

## 6. 坐标和参数说明

也可以用旧模式直接提供当前活动 TCP 偏置：

```bash
--tcp-offset X Y Z RX RY RZ
```

单位是米和轴角弧度。`--tcp-calibration` 与 `--tcp-offset` 不能同时使用；优先使用电脑端 TCP 标定文件，因为它会检查预览到运动连接期间活动 TCP 是否改变。

`--d435i-serial 939622070870` 可用于指定相机序列号。若机器上有多个 RealSense，建议显式指定。

## 7. 日志和故障处理

预览时 `--trace` 保存每次推理结果，运动时保存每个伺服周期的指令、反馈和耗时：

```bash
cat logs/replay/policy_preview.json
cat logs/replay/policy_arm_only.json
cat logs/replay/policy_arm_gripper.json
```

出现以下情况时，程序会停止伺服并退出；处理原因后应重新从预览开始：

```text
Active UR TCP differs
Distance protection
Base-Z protection
Following error too large
UR5 safety stop
D435i RGB-D frame did not arrive
Policy prediction is stale
Camera/robot observations are stale
RTDE robot state is stale
夹爪宽度超出标定范围
```

常见问题：

- `Checkpoint ... missing cfg`：checkpoint 不是 XAC 训练输出，或精简时删除了 `cfg`。
- `epoch=?`、`Contract frequency_hz=?`：精简部署 checkpoint 正常删除了训练元数据，不影响推理。
- `frame_queue has no attribute wait_for_frames`：确认使用已修复的 `driver/d435i.py`，其队列读取必须调用 `wait_for_frame()`。
- `align.process ... composite_frame`：确认使用已修复的驱动，它会先调用 `as_frameset()` 再进行对齐。
- `Active UR TCP differs`：示教器实际 TCP 与标定文件不一致，停止并核对 TCP。
- `D435i did not return a color frame`：检查 USB、RealSense 权限、分辨率、帧率和序列号。
- 夹爪宽度异常：检查电机 ID、波特率、零位、标定文件和指尖安装状态。

该入口用于受限范围的闭环策略验证，不替代 TCP 方向验收、碰撞规划或工业安全评审。

离线验证控制逻辑（不连接硬件）：

```bash
python -m unittest -v test_policy_control test_policy_loading
```

## 实时相机画面

部署默认显示 D435i 彩色原图，热身、等待 RUN、预览推理和运动时都会更新（约 10 Hz）。窗口由独立进程显示，使用容量为 1 的图像队列，显示延迟时丢帧。画面不改变模型的 RGB 224x224 输入。

窗口中按 Q、Esc 或关闭窗口只关闭图像，机械臂仍按原任务运行；停止任务请在终端按 Ctrl+C。无桌面显示环境时自动禁用窗口；也可添加 `--no-camera-preview`。程序结束时自动清理显示进程。

运动开始前等待有效的首个预测，最多 5 秒。默认预测观测年龄上限为 1 秒，适应约 300～400 ms 的推理及相邻预测间隔；实时相机观测年龄上限仍为 0.5 秒。超时错误打印观测年龄和该预测的推理耗时。
