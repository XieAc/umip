# UR5 安装 UMI 后的 TCP 对点标定

入口：`python deploy.py calibrate-tcp`。依赖 `numpy scipy ur-rtde`。
程序只建立 RTDEReceive 连接；不会上传控制脚本、移动机器人、开启 freedrive、
设置 TCP 或改变夹爪扭矩。手动移动使用示教器或已验证的控制方式。

## 1. 准备

- 固定最终装配，明确与手持采集一致的 G 原点和轴。记录夹爪开口及保持方法。
- 在工作区固定尖点。用夹爪上的同一个刚性参考点接触它，避免挤压软指尖。
- 核实控制器当前 TCP 偏置。只有确认当前 TCP 为法兰、偏置为零时，下面六个零才适用。
  `ur5.py` 不设置 TCP，不能仅凭使用它就认定偏置为零。
- 采样期间不得改变激活 TCP；偏置单位 m/rad，旋转为旋转向量。
- 若用探针，结果对应探针尖，必须依据探针到 G 的已知偏置换算，不能直接当 G 使用。

## 2. 采样

在项目目录运行，替换 `ROBOT_IP` 以及参考点说明：

```bash
python deploy.py calibrate-tcp collect \
  --robot-ip ROBOT_IP \
  --active-tcp 0 0 0 0 0 0 \
  --reference "填写实际G参考点、开口和保持方法" \
  --output calibration/tcp_samples_01.json
```

每次手动对准固定点、停稳、在终端按回车。绕不同轴改变姿态后再次对点。
建议 6～10 个姿态，不能只平移、只绕一个轴旋转或姿态几乎不变。
`u` 撤销最后一点；`q` 保存退出。Ctrl+C 也保存已有点；文件存在时拒绝覆盖。
每点读取约 0.5 秒，检测时间戳及静止程度。此检测不能判断你是否真的对准尖点。

## 3. 离线求位置

```bash
python deploy.py calibrate-tcp solve \
  --samples calibration/tcp_samples_01.json \
  --output calibration/tcp_position_01.json
```

`translation_m` 是法兰坐标系下参考点的位置。查看 `fit.rms_mm`、`fit.max_mm`。
默认以最大误差 2 mm 作初筛（超过时退出码为 2，结果仍保存），按任务精度选择
`--tolerance-mm`；这不是精度保证。程序拒绝秩不足或条件数大于 100 的样本集。

## 4. 独立验证

再次执行采样命令，输出改为 `calibration/tcp_validation_01.json`。
使用另一组 6～10 个姿态；本组内保持同一固定点。若重新连接时 TCP 已变化，
必须更新 `--active-tcp`，不能继续填旧值。

```bash
python deploy.py calibrate-tcp validate \
  --samples calibration/tcp_validation_01.json \
  --calibration calibration/tcp_position_01.json \
  --output calibration/tcp_validation_report_01.json
```

检查 `position_within_tolerance` 及最大误差；验证不重新拟合位置偏置。
重复解除/恢复夹紧状态进行复核。若点随开口或受力移动，固定 TCP 不会补偿它。

## 5. 方向及设置

固定点只能求位置，不能求方向。依据 CAD、安装测量或方向工装，独立确定
G 相对法兰的旋转，转换为旋转向量后提供 `--g-rotvec RX RY RZ`，重新运行 solve，
使用新的输出路径。不能用未知方向的三个零占位，也不能把欧拉角填进去。
结果会包含 `tcp_offset_m_rad` 和 `T_F_G`。

在示教器保存、激活完整 TCP，并填写当前工具负载和重心；或由已有控制程序
在合适时机调用 `setTcp(tcp_offset_m_rad)`。本标定工具不自动应用结果。
在空旷区域低速验证工具轴方向，然后按 README 的只读检查、低速短段重播顺序验收。
方向未确定时，位置标定文件仍可用于第 4 步独立验证，但不是完整 TCP。
