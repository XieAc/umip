# 电脑端 TCP 坐标变换的慢速验证

入口：`python deploy.py verify-tcp`。命令通过 `--calibration` 指定电脑端 TCP；若省略该参数，
程序默认读取原始 `calibration/tcp_gripper_01.json`，当前左臂不要省略它。
当前左臂已验证 `+X_G` 方向的文件是
`calibration/tcp_gripper_01_gx_down_candidate.json`；原始
`calibration/tcp_gripper_01.json` 保留作对照。
依赖 `numpy scipy ur-rtde`。默认只读；实际运动必须显式加 `--enable-motion` 并输入 `RUN`。

## 坐标关系

`B` 是 UR 基座，`F` 是法兰，`G` 是夹爪，`H` 是控制器当前激活 TCP。
脚本从只读状态接口获取真实 `T_F_H`，用以下公式处理反馈和命令：

```text
T_B_G = T_B_H @ inverse(T_F_H) @ T_F_G
T_B_H_target = T_B_G_target @ inverse(T_F_G) @ T_F_H
```

因此控制器激活 TCP 为六个零或已有夹爪偏置均可使用，不会重复补偿。
脚本不调用 `setTcp`，夹爪标定保存在电脑上。这里全部使用 UR 基座坐标；
不使用 `ur5.py` / `go_home.py` 的世界坐标及 45 度安装变换。

当前候选 TCP 的约定是：`G+X=F+Z`、`G+Y=F-Y`、`G+Z=F+X`，探针尖在 G 的 `+X`
方向 80 mm。它对应用户确认的 `Wg -Z = G +X`。原始 TCP 文件的 `G+X=F-Z`，
不要在未重新验证的情况下用于本次左臂 replay。

## 1. 只读预览

```bash
python deploy.py verify-tcp \
  --calibration calibration/tcp_gripper_01_gx_down_candidate.json
```

显示真实激活 TCP、软件换算的夹爪位姿、探针尖位置，以及各 G 正轴在基座系中的方向。
此模式不会建立 RTDEControl 或上传控制脚本，也不会移动机器人。

无机器人时也可检查变换和计划：

```bash
python deploy.py verify-tcp \
  --calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --offline-pose 0.3 -0.5 0.3 0 0 0 --test both --pivot probe
```

`--offline-pose` 是假设的当前 H 位姿；非零 H 偏置可由 `--offline-active-tcp` 指定。

## 2. 平移轴方向验证

工具和探针脱离固定点，给整个工具留出空间，然后运行：

```bash
python deploy.py verify-tcp --calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --test translation --axes x \
  --report calibration/tcp_axis_x_report_01.json
```

输入 `RUN` 后仍需回车才开始每个测试。默认沿 `+X_G` 平移 3 mm，速度目标 1 mm/s。
平移参考 `go_home.py` 使用 `moveL`，加速度为 10 mm/s²；异步执行以便监控实际位移和响应 Ctrl+C。
到达后停止，实际观察方向，然后回车返回原位；`q` 可原地退出。
将 `--axes x` 换为 `--axes y` 或 `--axes z` 分别检查其他轴；省略 `--axes` 可依次检查三个轴。
每次使用新的报告路径，程序拒绝覆盖已有报告。

## 3. 旋转及偏移补偿验证

保持探针尖固定的测试：

```bash
python deploy.py verify-tcp --calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --test rotation --pivot probe --axes y \
  --report calibration/tcp_probe_pivot_y_report_01.json
```

默认绕探针尖的 G `+Y` 方向旋转 2 度，再经确认返回。
实际观察探针尖应接近原位置；不要让探针戳在工装上承受旋转产生的接触力。
`--axes z` 可检查另一个倾斜方向。绕探针自身 `X` 轴旋转不能充分验证沿 X 的 80 mm 偏移。

夹爪 G 原点固定的测试：

```bash
python deploy.py verify-tcp --calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --enable-motion --test rotation --pivot gripper --axes y \
  --report calibration/tcp_gripper_pivot_y_report_01.json
```

此时应观察 G 原点保持固定；探针尖会绕 G 走圆弧。80 mm 探针旋转 2 度时，
针尖位移约 2.79 mm，属于预期，不能把它当作变换错误。

## 速度、停止与结果解释

- `--step-mm` 默认 3，最多 10；`--angle-deg` 默认 2，最多 5。
- `--speed-mm-s` 默认 1，最多 2；`--angular-speed-deg-s` 默认 0.5，最多 1。
- 平移使用低速 `moveL`。旋转使用 125 Hz 的平滑位姿插值发送 `servoL`。旋转速度由插值控制；`servoL` 的 speed/acceleration
  参数在当前 ur-rtde 中不生效。旋转时根据工具偏移杠杆臂自动延长测试时间。
- 启动前要求机器人静止、没有运行或暂停中的程序；检查路径采样点的可达性及控制器安全限位。
- Ctrl+C 停止伺服并退出，停止后不会自动返回。报告记录已完成的动作和停止原因。
- 启动时的 `speed_scaling=0` 可能来自脚本启动/同步阶段，不直接等同于示教器速度滑块为零。
  程序根据实际位移与终点反馈判断运动结果；平移持续未到达目标时超时停止并报告原因。
- 报告中的误差是用同一标定变换重建的跟随误差，不能独立证明物理 TCP 精度。
  物理轴方向和参考点是否保持固定，必须实际观察或用外部测量验证。
- 脚本不会永久保存或改变激活 TCP，不会改变工具负载，也不会改动重播程序。
