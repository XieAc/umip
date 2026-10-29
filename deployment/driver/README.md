# 本机硬件驱动接口

## XC330

`xc330.py` 是重播使用的接口，不要直接把交互式的 `gripper_torque_control.py`
嵌入 replay。默认配置来自 `xac_config.py`：Linux 优先选择 FTDI
`/dev/serial/by-id` 链接，找不到时使用 `/dev/ttyUSB0`；57600 baud、Protocol 2.0、
ID 1/2。宽度由 `calibration/gripper_width.json` 的
`paired_difference_tick` 模型转换为两个连续位置目标。

首次接线只做只读检查：

```bash
python deploy.py gripper-check
```

实时显示两个编码器角度和标定后的夹爪总开口 width（mm）：

```bash
python driver/read_width.py
python driver/read_width.py --port /dev/ttyUSB0 --once
python driver/read_width.py --angles 338.466796875 16.171875
```

`--angles` 按 ID 1、2 的顺序输入角度（度），无需连接硬件。脚本沿用
`xac_config.py` 的串口配置及宽度换算，默认每 0.1 秒刷新，可用 `--interval`
调整。只读取编码器，不修改扭矩或控制模式；超出标定范围时饱和到端点，width 始终限制在 0–100 mm。

该命令只读取 ID、型号、扭矩状态、位置、电流、电压和温度，不切换模式、不写目标位置，
也不打开扭矩。

启用 replay 夹爪：

```bash
python deploy.py replay ... \
  --enable-motion --enable-gripper \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baudrate 57600 \
  --gripper-calibration calibration/gripper_width.json
```

初始化时会检查当前开口是否接近 episode 首帧，再切换到 current-based position
mode。通信失败、轨迹宽度超出标定范围或周期电流超过阈值都会中止 replay，并在清理
阶段关闭两侧扭矩。

## D435i

`d435i.py` 提供 `D435iStream`，默认使用 `640x480` RGB-D，请求 30 Hz，设备不支持
时选择可用共同帧率，并默认打开 accel/gyro。独立启动检查：

```bash
python deploy.py camera --no-preview
```

重播期间如只需要启动并排空相机帧，可加入 `--enable-d435i`。当前 open-loop replay
不读取这些实时图像；要做视觉策略，还需要另外实现观测同步、预处理和推理接口。
