# UR5 轨迹重播与问题定位

`replay/replay_xac_ur5.py` 先读取已处理的 episode，计算相对 G 位姿，再把轨迹锚定到当前 UR5 的 G 起点。它不执行模型推理，也不根据相机重新规划。

排查运动方向时按这个顺序：离线确认数据轨迹方向；读取真实 UR5 起点但不运动；低速、短范围只重播机械臂；对照 trace 判断问题在数据、TCP 补偿还是 UR5 执行；最后才启用夹爪和 D435i。

## 输入与坐标

episode 至少需要：

```text
<episode>/data/d435i_timestamps.npy
<episode>/data/robot0_eef_pos.npy
<episode>/data/robot0_eef_rot_axis_angle.npy
<episode>/data/robot0_gripper_width.npy
<episode>/data/valid_mask.npy
```

如果存在 `<episode>/auxiliary/T_W_G.npy`，程序会自动校验它与位姿数组一致；不一致时停止。

坐标关系为：

```text
T_B_G(t) = T_B_G(start) @ inverse(T_W_G(0)) @ T_W_G(t)
```

相对锚定只改变起点，不改变源轨迹的 XYZ 位移。使用 `--tcp-calibration` 时，程序读取电脑端 `T_F_G`，把 G 目标转换成当前 UR 控制器 TCP H 的目标，不调用 `setTcp`。

## 1. 离线诊断

不连接机器人即可检查数据和相对轨迹：

```bash
cd /home/eirl3060/Desktop/umi_core
conda activate umip
python deploy.py replay \
  --episode replay/replaydata1/dataset/replay/001 \
  --diagnose \
  --dump-targets logs/replay/replay_001_targets.json \
  --speed-scale 0.1 \
  --max-distance-from-start-m 1.0 \
  --report logs/replay/replay_001_offline.json
```

输出包括源坐标和锚定后 Base 坐标的 `delta xyz`、范围及最大速度。两者的 `delta xyz` 应相同。

当前 `replay/001` 的基线约为：

```text
总位移 XYZ = [-20.7, +32.0, -29.3] mm
运动范围 XYZ = [117.3, 218.5, 61.8] mm
```

正常重播应能看到约 29 mm 的 G-Z 变化。若真机只有几毫米 Z 变化，应停止检查 TCP/目标转换。

## 2. 读取起点但不运动

```bash
python deploy.py replay \
  --episode replay/replaydata1/dataset/replay/001 \
  --robot-ip 192.168.1.101 \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --max-distance-from-start-m 0.25 --speed-scale 0.1 \
  --minimum-base-z-m -1 --maximum-base-z-m 1 \
  --report logs/replay/replay_001_live_preview.json
```

没有 `--enable-motion` 时只读取 UR5、计算目标和执行安全检查，不会发送运动命令。

## 3. 低速 arm-only 重播

```bash
python deploy.py replay \
  --episode replay/replaydata1/dataset/replay/001 \
  --robot-ip 192.168.1.101 \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --speed-scale 0.1 --max-duration-s 180 \
  --max-distance-from-start-m 0.25 \
  --minimum-base-z-m -1 --maximum-base-z-m 1\
  --max-linear-speed-m-s 0.03 --max-angular-speed-rad-s 0.3 \
  --max-following-error-m 0.01 --max-rotation-following-error-rad 0.15 \
  --trace logs/replay/replay_001_arm_only.json
```

确认工作区清空、示教器速度低、急停可触达后，在提示处输入大写 `RUN`。源时长约 8 秒，`speed-scale=0.05` 时实际约 160 秒。按 `Ctrl-C` 会停止伺服。

## 4. 用 trace 定位

```bash
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('logs/replay/replay_001_arm_only.json').read_text())
print('completed:', d.get('completed'), 'error:', d.get('error'))
for row in d.get('samples', [])[:3] + d.get('samples', [])[-3:]:
    print('source', row['source_index'], 'target G', [round(x, 4) for x in row['target_pose'][:3]], 'controller', [round(x, 4) for x in row['controller_target_pose'][:3]], 'actual G', [round(x, 4) for x in row['actual_pose'][:3]])
PY
```

- `target_pose` 和 `actual_pose` 都有约 29 mm Z 变化：数据、TCP 和执行基本一致。
- `target_pose` 有约 29 mm Z 变化，`actual_pose` 没跟随：检查 UR5 伺服、速度滑块、控制器状态。
- 离线诊断有 Z 变化，但 `target_pose` 没变化：检查当前起点读取、TCP 补偿和目标生成。
- `target_pose` 与 `controller_target_pose` 方向异常：检查 `T_F_G` 候选标定。
- 仅启用夹爪后异常：停用夹爪，检查串口、电机 ID 和电流限制。

## 5. 夹爪同步

先做只读检查：

```bash
python deploy.py gripper-check --port /dev/ttyUSB0 --baudrate 57600 --ids 1 2
```

机械臂重播确认后再运行：

```bash
python deploy.py replay \
  --episode replay/replaydata1/dataset/replay/001 \
  --robot-ip 192.168.1.101 \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --enable-gripper \
  --gripper-port /dev/ttyUSB0 --gripper-baudrate 57600 \
  --speed-scale 0.15 --max-duration-s 180 \
  --max-distance-from-start-m 0.25 \
  --minimum-base-z-m -1 --maximum-base-z-m 1\
  --trace logs/replay/replay_001_arm_gripper.json
```

首次不要夹工件。当前开口必须接近 episode 首帧宽度，宽度必须在标定范围内。

## 6. D435i

重播控制不使用实时图像。`--enable-d435i` 只启动相机并后台排空帧：

```bash
python deploy.py replay ... --enable-d435i --d435i-no-imu
```

相机异常时先单独运行 `python deploy.py camera --width 640 --height 480 --fps 30 --no-imu`，不要把相机错误和轨迹方向问题放在同一次运动测试中排查。

## 安全限制

`--minimum-base-z-m -1 --maximum-base-z-m 1` 几乎关闭 Base-Z 保护，不用于正式测试。`--max-following-error-m 0.2` 也过大，定位阶段建议使用 10 mm 或更小。限位不能替代现场避障、低速设置和急停监护。
