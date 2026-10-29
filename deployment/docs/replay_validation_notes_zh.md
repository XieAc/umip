# UMI → UR5 轨迹重播验证注意事项

本文记录 `replaydata1` 这条自制 UMI/夹爪数据在左臂 UR5 上能够大致重播的原因、
已经完成的验证，以及继续做夹爪联动测试时必须保持的条件。

## 当前结论

当前已经完成了**机械臂的初步轨迹重播**：

- 输入是 `replaydata1/dataset/replay/001` 中的 UMI 处理后 episode，不是
  `logs/pendant_demo_001.csv`。CSV 只是示教器手动采集的粗略参考路径。
- 这条 episode 有 81 帧，源时长约 8.006 s，最大相对位移约 0.216 m。
- 原始 TCP 方向会使运动方向与预期不一致。根据“`Wg -Z = G +X`”建立了
  `calibration/tcp_gripper_01_gx_down_candidate.json`，并保留原始
  `calibration/tcp_gripper_01.json` 不变。
- 候选 TCP 的 `+X_G` 已用 10 mm、1 mm/s 的实机平移测试验证。去程 G 位移约为
  `[-7.034, -0.088, -7.049] mm`（UR Base），换算到 `Wg` 约为
  `[-0.088, -0.011, -9.958] mm`，即沿 `Wg -Z`。
- 使用候选 TCP 的 arm-only replay 已正常结束，轨迹方向和形状与 UMI 采集时大致一致，
  没有保护停止；终端观察到的最大跟随误差约为 4 mm。该次运行的
  `gripper_enabled` 为 `false`，所以还没有验收机械臂和夹爪同步动作。

“大致重播成功”在这里表示方向、相对形状、起点锚定和运动安全检查已经通过初步现场验证。
它不表示指尖接触点误差、整套夹爪同步、避障余量或最终任务成功率已经验收。
episode 的最后一帧距离起点仍约 48 mm，源数据本身没有完全回到起点，不能以“没有回到
原点”判定本次 replay 失败。

## 本次形成的配置和文件

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `calibration/tcp_gripper_01.json` | 原始电脑端 `T_F_G` | 保留作对照；方向没有按本次结论替换 |
| `calibration/tcp_gripper_01_gx_down_candidate.json` | 按 `Wg -Z = G +X` 生成的候选 TCP | 当前左臂 replay 使用 |
| `calibration/tcp_axis_x_gx_down.json` | 10 mm `+X_G` 去程/回程实机报告 | 已完成，端点误差约 0.055/0.041 mm |
| `go_home.py` | 左臂 IP、斜 45° 安装和 `Wg` Home 姿态配置 | 用于理解方向和回 Home，不作为 replay 的全局轨迹变换 |
| `replaydata1/dataset/replay/001` | UMI 处理后的重播输入 | 当前使用的 81 帧 episode |
| `logs/replay/replay_gx_down_8s.json` | 候选 TCP 的 arm-only replay trace | `completed: true`，`gripper_enabled: false` |
| `logs/replay/replay_arm_gripper.json` | 候选 TCP 的机械臂+XC330 replay trace | `completed: true`，`gripper_enabled: true` |
| `logs/pendant_demo_001.csv` | 示教器粗略参考轨迹 | 不会被 `deploy.py replay` 直接读取 |

候选文件只验证了 `+X_G` 的方向。更换夹爪、探针、法兰安装方向或开口状态后，仍需重新
复核 TCP 原点、`+Y_G`、`+Z_G`、工具负载和工作空间。
报告中的 `physical_geometry_verified: false` 是程序的静态标记，表示软件报告不能替代
操作者对物理方向和避障的观察；它不否定本次已经完成的 `+X_G` 现场方向确认。

## 坐标系关系

统一用 `T_A_B` 表示“把 B 系坐标变换到 A 系”。相关坐标如下：

| 符号 | 含义 |
|---|---|
| `W` | UMI/Stray 记录使用的世界坐标 |
| `I` | 手机坐标 |
| `Wg` | `go_home.py` 中用于解释左臂安装方向的逻辑世界坐标 |
| `B` | UR5 Base 坐标 |
| `F` | UR5 法兰坐标 |
| `G` | 采集和重播共用的夹爪坐标 |
| `H` | UR 控制器当前激活的 TCP |

采集端外参关系是：

```text
T_W_G = T_W_I @ T_I_G
```

机器人端电脑侧 TCP 关系是：

```text
T_B_G = T_B_H @ inverse(T_F_H) @ T_F_G
T_B_H_target = T_B_G_target @ inverse(T_F_G) @ T_F_H
```

这里的 `H` 可以是法兰零偏置，也可以是控制器中已有的临时 TCP。程序读取实际的
`T_F_H`，所以不会把已有偏置重复加一次；电脑侧方案也不会调用 `setTcp`。

### `go_home.py` 的 Wg 到 UR Base

左臂配置为 `LEFT_MOUNT_Y_DEG = -45°`。代码对安装旋转取逆，因此位置关系为：

```text
R_B_Wg = R_y(+45°) @ R_z(+90°)
p_B    = R_B_Wg @ p_Wg
```

Home 姿态的方向还包括 `R_y(180°)`：

```text
R_B_TCP = R_y(+45°) @ R_z(+90°) @ R_y(180°)
```

这个关系用于理解斜 45° 安装下 `go_home.py` 和示教器方向。当前 replay 不需要先把
整条 UMI 轨迹乘上这个全局变换：`deploy.py replay` 先求 episode 的相对 G 运动，再把
相对运动锚定到机器人当前的 G 起点，最终命令仍然在 `B` 中发送。
所以，曾经出现的 `[+131, +60, +160] mm` 三轴数值不能单独作为 `Wg` 到 `B` 的映射；
它既没有包含旋转，也不是当前 replay 的固定平移补偿。

### 当前候选 TCP 的方向

候选文件的旋转部分 `R_F_G` 为：

```text
R_F_G = [[0, 0, 1],
         [0,-1, 0],
         [1, 0, 0]]
```

矩阵的列是 G 轴在 F 中的方向，所以：

```text
+X_G = +Z_F
+Y_G = -Y_F
+Z_G = +X_F
```

这与实机确认的 `Wg -Z = G +X` 一致。原始 TCP 的 `+X_G = -Z_F`，因此本次没有把
原始文件直接覆盖，而是使用候选文件进行验证。

### Replay 的起点锚定

重播时使用的实际关系为：

```text
T_B_G_start = T_B_H_actual @ inverse(T_F_H) @ T_F_G
T_B_G(t)     = T_B_G_start @ inverse(T_W_G_start) @ T_W_G(t)
T_B_H_target = T_B_G(t) @ inverse(T_F_G) @ T_F_H
```

第二行只使用 episode 的相对运动。它会把采集时的起点搬到当前 UR5 G 起点，
不会自动找工件，也不会自动把工作台或 UMI 世界坐标对齐到机器人。

## 已验证命令

先固定本次使用的文件：

```bash
cd /home/eirl3060/Desktop/umi_core
conda activate umip
EPISODE="replaydata1/dataset/replay/001"
ROBOT_IP="192.168.1.101"
TCP_CAL="calibration/tcp_gripper_01_gx_down_candidate.json"
```

方向验证使用 `verify-tcp` 的 `--calibration` 参数：

```bash
python deploy.py verify-tcp \
  --robot-ip "$ROBOT_IP" \
  --calibration "$TCP_CAL" \
  --test translation --axes x \
  --step-mm 10 --speed-mm-s 1 \
  --enable-motion \
  --report calibration/tcp_axis_x_gx_down.json
```

机械臂空载重播使用 `replay` 的 `--tcp-calibration` 参数。每次先把 G 起点摆回相同位置，
并用现场测量的 UR Base Z 上下限替换变量：

```bash
read -r -p "G原点允许的UR基座最小Z（米）: " SAFE_MIN_Z
read -r -p "G原点允许的UR基座最大Z（米）: " SAFE_MAX_Z
python deploy.py replay \
  --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" \
  --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" \
  --maximum-base-z-m "$SAFE_MAX_Z" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.3 \
  --max-duration-s 30 \
  --enable-motion \
  --trace logs/replay/replay_gx_down_8s.json
```

命令显示 `Safety checks: PASS` 后才输入精确的 `RUN`。`replay` 的这个提示后不需要
再额外按回车；`verify-tcp` 的每个运动段则还需要回车确认。

## 下一步：先做夹爪联动短测试

本条 episode 的夹爪宽度从约 `0.0976 m` 开始，源时间约 3.2 s 后才明显闭合。因而 3 秒、
低速 `replay` 往往还没有播到闭合动作。确认执行夹爪、电机 ID、串口和宽度标定后，建议先运行：

```bash
python deploy.py replay \
  --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" \
  --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" \
  --maximum-base-z-m "$SAFE_MAX_Z" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.1 \
  --max-duration-s 45 \
  --enable-motion --enable-gripper \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baudrate 57600 \
  --gripper-calibration calibration/gripper_width.json \
  --trace logs/replay/replay_gx_down_gripper_test.json
```

短测试通过后，再去掉 `--max-duration-s 45` 做完整机械臂+夹爪重播。执行夹爪的初始开口
应接近 `gripper_width_start_m`，且必须在宽度标定范围内。

当前 `--enable-gripper` 已接入 `driver/xc330.py`：使用本机 Linux 的 FTDI 串口（默认先找
`/dev/serial/by-id`，再回退 `/dev/ttyUSB0`）、57600 baud、Dynamixel Protocol 2.0、
ID 1/2 和 `paired_difference_tick` 宽度标定。初始化会先读取两侧位置，再切换到
current-based position mode；任一通信或电流保护异常都会停止 replay，并在清理阶段关闭两侧扭矩。
默认目标电流为 600、保持电流为 100、profile velocity 为 30、acceleration 为 20；
这些参数可用 replay 命令行选项覆盖。

D435i 已提供独立的 `driver/d435i.py` 接口和 `python deploy.py camera` 检查命令。
若要在机械臂 replay 期间只启动相机并后台排空帧，可增加 `--enable-d435i`；这不会把实时
图像接入当前 open-loop 位姿控制，也不会替换 episode 中已经录制的相机数据。

## 每次运行前的检查

1. 夹爪、相机、支架和探针（若已取下）安装状态与标定时一致。
2. 使用候选 TCP；不要同时再加一个把方向翻转 180° 的参数，也不要把原始 TCP 和候选 TCP 混用。
3. `SAFE_MIN_Z`/`SAFE_MAX_Z` 针对的是 G 原点在 UR Base 的 Z，不是 `Wg` 的 Z；还要检查整个工具、
   指尖和相机的实际避障空间。
4. 先只读预检，再低速 arm-only，最后才启用夹爪；示教器速度滑块保持低位，急停保持可触及。
5. 每次短测或中止后重新摆回起点；程序不会自动回 Home 或回到源轨迹起点。
6. 保存不同的 trace 文件名，并检查 `completed`、`error`、目标/实际 G 位姿和跟随误差。



完整流程如下，在 `~/Desktop/umi_core` 中执行。

```
cd ~/Desktop/umi_core

# 1. 激活环境
conda activate umip

# 2. 设置重播参数
export ROBOT_IP="192.168.1.101"
export EPISODE="$PWD/replaydata1/dataset/replay/001"
export TCP_CAL="$PWD/calibration/tcp_gripper_01_gx_down_candidate.json"
export GRIPPER_CAL="$PWD/calibration/gripper_width.json"

# 根据机器人实际安全工作空间设置；下面只是当前左臂重播示例值
export SAFE_MIN_Z="-0.30"
export SAFE_MAX_Z="0.30"

# 确认 episode 文件存在
ls "$EPISODE/data"/d435i_timestamps.npy \
   "$EPISODE/data"/robot0_eef_pos.npy \
   "$EPISODE/data"/robot0_eef_rot_axis_angle.npy \
   "$EPISODE/data"/robot0_gripper_width.npy
```

先做离线检查，不会连接 UR5，也不会驱动夹爪：

```
python deploy.py replay \
  --episode "$EPISODE" \
  --tcp-calibration "$TCP_CAL" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.3 \
  --report logs/replay/replay_preflight.json
```

接入 XC330 后，先执行只读检查：

```
python deploy.py gripper-check \
  --port /dev/ttyUSB0 \
  --baudrate 57600 \
  --ids 1 2
```

确认两个电机都能读取到型号、位置、电流、电压和温度后，再运行机械臂和夹爪同步重播。首次建议低速：

```
python deploy.py replay \
  --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" \
  --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" \
  --maximum-base-z-m "$SAFE_MAX_Z" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.1 \
  --max-duration-s 90 \
  --enable-motion \
  --enable-gripper \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baudrate 57600 \
  --gripper-ids 1 2 \
  --gripper-calibration "$GRIPPER_CAL" \
  --gripper-current-limit-raw 600 \
  --gripper-goal-current-raw 600 \
  --gripper-hold-current-raw 100 \
  --gripper-profile-velocity 30 \
  --gripper-profile-acceleration 20 \
  --trace logs/replay/replay_arm_gripper.json
```

这条轨迹源时长约 8 秒，`--speed-scale 0.1` 时实际约 80 秒，所以设置 90 秒上限。运行前，夹爪当前开口必须接近 episode 首帧的 `gripper_width_start_m`，否则程序会拒绝初始化，防止突然动作。

只重播机械臂、不连接或控制 XC330：

```
python deploy.py replay \
  --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" \
  --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" \
  --maximum-base-z-m "$SAFE_MAX_Z" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.3 \
  --max-duration-s 30 \
  --enable-motion \
  --trace logs/replay/replay_arm_only_latest.json
```

这条命令没有 `--enable-gripper`，因此不会打开 XC330 扭矩，也不会发送夹爪目标。

如果还要同时启动 D435i，可以在机械臂重播命令中加入：

```
--enable-d435i --d435i-no-imu
```

例如：

```
python deploy.py replay \
  --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" \
  --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" \
  --maximum-base-z-m "$SAFE_MAX_Z" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.1 \
  --max-duration-s 90 \
  --enable-motion \
  --enable-gripper \
  --enable-d435i \
  --d435i-no-imu \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baudrate 57600 \
  --gripper-calibration "$GRIPPER_CAL" \
  --trace logs/replay/replay_arm_gripper_d435i.json
```

每次真实运行前确认工作区清空、示教器速度滑块较低、急停可触及，并在程序提示后输入大写 `RUN`。
