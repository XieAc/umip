# XAC / UR5 标定、采集、重播与训练流程

本目录按 `collect_xac` 适配：D435i RGB-D/IMU、iPhone Stray、双 XC330
铰链夹爪（ID 1/2，57600 baud），通过 ur-rtde 控制 UR5。
将整个 `umi_core` 复制到连接 UR5 的电脑即可使用，不依赖父目录下的源码。
Python 建议 3.10 或更新版本。Linux 和 Windows 均提供终端采集入口。

这里的“部署”包括采集、轨迹重播和策略推理。训练在 PRO6000 的 `~/Desktop/umip/train` 上进行；
策略部署入口是 `test/test_model_inference.py`（`eval_real.py` 仍为空）。
本次没有把官方 WSG、鱼眼相机和固定 59.94 Hz 的假设带入 XAC。

目录约定：`calibration/` 是唯一有效的标定数据目录；`replay/` 集中保存 UR5
重播和 TCP 验证实现；`driver/` 保存 XC330、D435i 等硬件接口；`replaydata1/`
只保存 episode 数据；`logs/replay/` 保存少量最终重播验证记录。旧版重复代码和
标定副本放在 `archive/replaydata1_legacy/`，不作为运行入口。

当前重播实现和入口：

```text
replay/replay_xac_ur5.py       episode 检查、起点锚定、TCP 补偿和重播
replay/verify_tcp_transform.py TCP 方向、平移和旋转验证
driver/xc330.py                XC330 双电机接口
driver/d435i.py                D435i 启动和帧读取接口
deploy.py                      统一命令入口
```

日常使用根目录的统一入口：

```bash
python deploy.py replay --help
python deploy.py verify-tcp --help
```

完整顺序：**确定 TCP 定义 → 手持端标定 → 采集与构建 → UR5 端标定 →
低速轨迹重播验证 → 批量采集与导出 → 离线训练 → 适配在线策略接口 → 实机评估**。
训练在带 GPU 的电脑上运行，UR5 控制器负责执行运动；不能把 Python 训练代码直接装到示教器上运行。

## 安装与标定

```bash
cd umi_core
python -m pip install -r requirements.txt
python deploy.py check --project
```

仅重播需安装 `numpy scipy ur-rtde dynamixel-sdk`。夹爪 replay 使用
`driver/xc330.py` 中的 XC330-M288-T 接口：ID 1/2、Dynamixel Protocol 2.0、
Linux 57600 baud，启动时切换到限流位置模式并在退出时关闭两侧扭矩。
Linux 默认优先使用 `/dev/serial/by-id` 下的 FTDI 串口，找不到时回退到
`/dev/ttyUSB0`；Windows 为 `COM6`。可用环境变量
`XAC_DXL_PORT`、`XAC_DXL_BAUDRATE` 覆盖。Linux 用户需有串口和 RealSense USB 访问权限。

接线后可先做不写寄存器的 XC330 检查：

```bash
python deploy.py gripper-check
```

它只读取两个电机的连通性、型号、扭矩状态、位置、电流、电压和温度，不切换工作模式、
不写目标位置，也不会打开扭矩。检查失败时先处理串口权限、端口占用、ID 或波特率，
不要直接用 replay 反复试通信。

D435i 的可复用启动接口位于 `driver/d435i.py`，默认使用本机采集一致的
`640x480` RGB-D、请求 30 Hz（设备不支持时自动选择 15 Hz 等可用共同帧率），
并可选打开 accel/gyro。`python deploy.py camera --no-preview` 可做无窗口启动检查；
按 `Ctrl+C` 停止。相机不参与当前 open-loop replay 的位姿控制。

硬件连接关系如下：

```text
电脑 USB ── U2D2 ── Dynamixel 总线 ── XC330 ID 1、ID 2
电脑 USB ── D435i（独立 USB 设备，不占用 /dev/ttyUSB0）
```

U2D2 和两个 XC330 共用一个串口，默认是 `/dev/ttyUSB0`、57600 baud，两个电机靠
Dynamixel ID 区分。D435i 不通过 U2D2；硬件 RGB-D 流默认请求 30 Hz，数据集再按时间戳
保存为约 10 Hz。重播时 XC330 目标更新约 10 Hz，UR5 姿态插值仍以 125 Hz 执行。

`calibration/` 包含从当前 XAC 复制的 `gripper_width.json`、`T_I_G.npy` 和
`T_I_G.json`。复制不代表标定已通过独立实验验收；更换安装关系后应重新标定。
TIG 双轴求解和独立验证仍在原 `collect_xac` 工作区执行，验收后同步正式三份文件。
`python deploy.py calibrate-width` 可重新测量夹爪宽度；不要使用 collection_umip 的 PCHIP 模型。

### 哪些东西需要重新标定

**UR5 上也需要标定，但不是再把手持端的 `T_I_G` 原样用一遍。**
通常需要做的是末端工具标定，不是修改 UR5 出厂关节运动学参数。

| 项目 | 含义与用途 | 何时需要做 |
|---|---|---|
| 手持端 `T_I_G` | 夹爪 G 系在手机 I 系中的位姿，将 Stray 轨迹换算为夹爪轨迹 | 首次使用，手机/支架/夹爪相对位置或 G 定义变化后 |
| 机器人端 `T_F_G` | 夹爪 G 系在 UR 法兰 F 系中的位姿，可保存在电脑并由软件换算；也可配置成示教器 TCP | 首次安装、拆装法兰/夹爪、改变工具原点或方向后 |
| 夹爪宽度映射 | 双电机 tick 到指尖总开口，单位米 | 首次使用、更换指尖、零位变化；另一套执行夹爪必须单独测量 |
| 工具负载和重心 | 当前法兰所承载夹爪、相机、支架等的质量及重心 | 安装部件或质量分布变化后，在示教器更新 |
| 工作台及运动范围 | UR 基座 B 系下的桌面、障碍物、允许高度和起点 | 首次布置或移动机器人/工作台/夹具后 |
| 相机安装与时延 | 相机相对夹爪的位置、图像处理方式、观察和执行时间关系 | 更换安装、相机参数或进入在线策略部署时验证 |

统一采用 `T_A_B` 把 B 系坐标转换到 A 系：

```text
手持采集：T_W_G = T_W_I @ T_I_G
机器人：  T_B_G = T_B_F @ T_F_G
```

W 是 Stray 世界坐标，I 是手机坐标，B 是 UR 基座坐标，F 是法兰坐标，G 是夹爪 TCP，
H 是控制器当前激活的 TCP。两端必须使用相同的 G 原点及轴方向。

### 当前左臂的坐标系和已验证方向

本次左臂安装还使用了 `go_home.py` 中的逻辑世界坐标 `Wg`。左臂配置为
`LEFT_MOUNT_Y_DEG = -45°`，程序取其逆变换，所以位置从 `Wg` 到 UR 基座的关系是：

```text
R_B_Wg = R_y(+45°) @ R_z(+90°)
p_B    = R_B_Wg @ p_Wg
```

`go_home.py` 的 Home 姿态另加 `R_y(180°)`：
`R_B_TCP = R_y(+45°) @ R_z(+90°) @ R_y(180°)`。这个变换只说明斜 45° 安装下
示教器/`go_home` 方向与 `Wg` 的关系；当前 `deploy.py replay` 采用起点锚定，
直接在 UR 基座 `B` 中执行相对 G 轨迹，不把 `Wg` 到 `B` 的全局变换重复套到轨迹上。
因此，单独写一个 `[+131, +60, +160] mm` 的三轴位移不能代替这个旋转变换，
也不是当前 replay 使用的固定偏置；当前位置由机器人实时 G 位姿锚定。

用户确认的物理关系是 `Wg -Z = G +X`。因此本次实际重播使用的
`calibration/tcp_gripper_01_gx_down_candidate.json` 为：

```text
R_F_G = [[0, 0, 1],
         [0,-1, 0],
         [1, 0, 0]]  （平移仍为 [0.0010135, 0.0248291, 0.3313082] m）

+X_G = +Z_F,    +Y_G = -Y_F,    +Z_G = +X_F
```

原始 `calibration/tcp_gripper_01.json` 的 `+X_G = -Z_F` 与这次确认的方向相反，
所以没有修改原文件，而是生成候选文件并先做低速实机验证。10 mm 的 `+X_G` 去程实测
G 在 UR 基座中约移动 `[-7.034, -0.088, -7.049] mm`，换到 `Wg` 后约为
`[−0.088, −0.011, −9.958] mm`，与 `Wg -Z` 一致。完整轴向仍应在拆装后重新复核。

对于铰链和软指尖，应选定可复现的刚性参考点，记录标定时的开口和接触状态。
固定 TCP 不会自动补偿开合、受力导致的指尖接触点变化。完整的重播验证记录见
[重播验证注意事项](docs/replay_validation_notes_zh.md)。

### 手持端标定顺序

1. 固定手机、D435i、夹爪和支架，标记 G 原点和三个坐标轴。
2. 用卡尺采集覆盖工作范围的宽度点，运行 `python deploy.py calibrate-width`。
   再用没有参与拟合的几个开口验证；电机 ID、零位、指尖都要与采集时一致。
3. 按原 `collect_xac` 工作区中的 TIG 标定说明进行双轴滑移求解，
   用独立 session 验证旋转和平移；通过后将正式 `T_I_G.npy` 及来源 JSON 同步到本目录。
4. 运行 `python deploy.py check --project`。这里的 PASS 只表示文件、依赖及数值检查通过，
   不等于针尖固定点误差或现场精度已通过验收。当前附带矩阵来源仍为实验候选。

### UR5 端 TCP 标定与验收

也可使用本目录新增的只读对点采样工具：`python deploy.py calibrate-tcp`。
具体采样、离线求解及独立验证命令见 [UR5 TCP 标定操作指南](docs/ur5_tcp_calibration.md)。
它不发送运动指令；固定点法求位置，轴方向需独立确定，结果不会自动应用到机器人。

电脑端 `T_F_G` 的轴向及旋转补偿验证可运行 `python deploy.py verify-tcp`（默认只读）。
实际小距离慢速测试见 [TCP 坐标变换验证指南](docs/ur5_tcp_transform_verification.md)。

#### 你当前“夹紧时定义 TCP、无外力时自动张开”的执行器

这里必须先区分三个概念：

1. **TCP 几何参考点**：法兰到夹爪坐标系 `G` 原点和轴方向的刚体关系，写成 `T_F_G`。
2. **夹爪开口状态**：两个执行指尖之间的距离，以及对应的电机 tick/宽度映射。
3. **机械自由状态**：无外力、扭矩关闭或释放机构后，执行器因弹性/连杆结构自动张开的状态。

TCP 应定义在一个刚性、可重复的参考点上。若你把 TCP 定义为“夹紧状态下两指接触的几何中心”，
必须确认这个中心相对于法兰在不同开口和姿态下不会明显移动。对于铰链夹爪，指尖接触点通常会
随开口变化，因此它只能作为任务参考状态，不能默认当作所有开口状态都成立的固定 TCP。
如果接触点会移动，应改用夹爪本体上的刚性参考点（例如固定工装孔、指尖安装座中心或已知探针尖），
并在采集端和 UR5 端使用同一个 `G` 定义；夹紧状态的指尖中心只作为任务几何偏置记录。

按下面顺序操作：

1. **固定装配状态**：将执行夹爪、相机、支架和指尖全部安装到最终位置。记录夹爪型号、指尖版本、
   安装方向、螺钉状态和装配照片。后续任何拆装都应重新复核 TCP。
2. **定义 G 原点和轴**：明确 `G` 原点到底是夹紧状态接触中心，还是夹爪上的刚性参考点；明确
   `+X_G/+Y_G/+Z_G`。它必须和手持 UMI 采集时的定义一致，不能只保证位置相同而忽略轴方向。
3. **固定夹爪机械状态**：在标定 TCP 时，先将执行器置于你定义的“夹紧状态”，但不要让电机或
   工件受力。若无外力会自动张开，应使用机械限位、低风险保持工装或按照执行器规定的保持方式
   固定到该状态；不要用手强行顶住并把弹性变形当成 TCP 几何。
4. **标定 `T_F_G` 平移**：用示教器 TCP 位置向导，让同一个刚性参考点在至少 4 个不同法兰姿态下
   对准同一固定点。若 G 点不可直接触达，可安装已知长度和方向的刚性探针。
   **向导得到探针尖 TCP 时，必须依据已知刚体偏置换算回 G 点，不能直接把探针尖当作夹爪 TCP。**
5. **标定 `T_F_G` 旋转**：依据安装几何、工装或示教器工具方向功能，使 TCP 三个轴与采集端
   `G` 系一致。仅靠多个姿态对准同一点只能求位置，不能完整确定旋转。
6. **验证夹紧状态重复性**：解除并重新进入夹紧状态至少 3 次，每次用同一固定点复核 TCP 位置。
   如果重复进入夹紧状态的误差已经接近任务容差，说明“夹紧状态 TCP”不够稳定，应改用刚性参考点，
   或增加机械定位结构后再标定。
7. **验证开口对 TCP 的影响**：分别记录自由张开、半开和夹紧状态，在不接触工件的情况下读取/测量
   G 点相对法兰的位置。如果 G 点随开口发生不可忽略的位移，重播时仍只能把 `G` 当固定 TCP，
   不能声称指尖接触点在所有状态下都精确跟随；需要在任务设计中保留接触偏差，或建立随开口变化的
   工具几何模型。当前 `replay/replay_xac_ur5.py` 使用固定 TCP，不会自动补偿该变化。
8. **保存 TCP 标定**：当前左臂重播使用电脑端
   `calibration/tcp_gripper_01_gx_down_candidate.json`，通过 `--tcp-calibration` 加载，
   不需要在示教器永久保存夹爪 TCP。工具负载与重心仍在控制器正确配置；其他装配必须使用
   经过方向验证的对应文件。
   旧方式可在示教器激活夹爪 TCP，再用 `--tcp-offset TX TY TZ RX RY RZ` 核对；这两个选项不能同时使用。
   参数为米和**轴角旋转向量、弧度**，不是欧拉角或角度。重播程序不调用 `setTcp`。
9. **单独标定宽度映射**：TCP 标定完成后，再标定执行器的电机 tick 到指尖开口宽度的映射。
   至少覆盖自由张开、任务工作区中间开口和夹紧附近的多个点；记录无外力自动张开的实际宽度，
   但不要把它误写成零宽度。若执行夹爪与手持夹爪不同，保存独立的
   `robot_gripper_width.json`，重播用 `--gripper-calibration` 指定它。
10. **做方向和空间验收**：空旷区域低速沿 TCP X/Y/Z 各移动一小段，确认物理方向与采集端标签一致；
    再测量工作台、指尖、相机和支架的最低安全高度，设置 `SAFE_MIN_Z/SAFE_MAX_Z` 时留出余量。

保存以下标定记录，之后才开始轨迹重播：

```text
T_F_G / 电脑端 TCP 标定文件及六维参数
TCP 定义时的夹紧状态和保持方法
自由张开、半开、夹紧时的宽度与电机 tick
夹爪/相机/支架负载和重心
TCP 固定点复测误差、轴方向验证结果
工作台高度、SAFE_MIN_Z、SAFE_MAX_Z
```

拆装后至少重新做固定点、轴方向、夹紧状态重复性和宽度范围复核。相同工具整体转移到 UR5 后，
只要手机到 G 的刚体关系未改变，`T_I_G` 可以经复核复用；但 `T_F_G`、执行器宽度映射以及
自由张开状态必须在 UR5 执行端独立确认。

### 是否需要相机手眼标定或世界坐标对齐

当前相对轨迹重播只用 episode 中的夹爪位姿，不读取实时相机，所以不要求先解出 Stray 世界到
UR 基座的固定变换，也不要求为重播新增 D435i 手眼求解。起点锚定只是搬移整条轨迹，
**不会自动定位物体或对齐工作台**。物体位置、朝向和初始抓取关系必须人工布置一致。

后续纯 RGB 策略首先需要训练与部署的相机视角、G 定义和预处理一致；如果采用深度反投影、
外部相机定位或视觉伺服，把相机坐标中的点转换到机器人坐标，则需要对应的相机内参和
相机到工具/基座外参，并做独立验证。不要把 `T_I_G` 当成 D435i 到机器人的外参。

## 采集与构建

```bash
python deploy.py collect
```

按 `s` 开始/停止，`q` 退出。夹爪保持扭矩关闭，由人手操作。
程序确认第一帧 RGB-D 已保存后才提示开始。随后启动 Stray，静止约 1 秒建立基线，
做一次清晰的短 shake；shake 结束后等待约 1.5 秒并确认有效数据阶段，再进行演示。
这是手持 XAC 采集，不会切换 UR5 freedrive 或录制机械臂示教。
采集后将 Stray 导出文件放入对应 session：

```bash
python deploy.py prepare data/001 /path/to/odometry.csv /path/to/imu.csv
python deploy.py check --session data/001
python deploy.py build --stray-root data --dataset-root dataset --dataset-group replay --timestamp 001 --yes
python deploy.py check --episode dataset/replay/001
```

先查看 episode 的图像、TCP 轨迹、宽度曲线和有效区间，确认抓取闭合、搬运和释放在时间上对应。
检查没有跟踪跳变、连续丢帧、手机导出错配或夹爪宽度饱和。`check --episode` 是自动检查的一部分，
不能代替回看。修改 `T_I_G` 后需要从原始 session 重建 episode，再重新导出，不能只换部署文件。

对 `002`、`003` 等 session 重复 prepare、build 和 check，保持同一 `--dataset-group replay`。
保留原始 Stray、D435i、宽度日志、标定快照及失败原因，避免只有 Zarr 无法追溯。
训练数据应覆盖任务中预期出现的物体初始位置、朝向、光照和操作变化；先用少量完整演示跑通全流程。

默认原始数据写入本目录 `data/`，可通过 `XAC_DATA_ROOT`、`XAC_DATASET_ROOT` 覆盖。
命令行显式路径相对于当前终端目录。episode 中位置/宽度为米，轴角为弧度，时间戳为秒。
UR5 重播读取 episode 的绝对 TCP 位姿和有效区间，使用实际时间戳插值；
不将 `action.npy` 的未来相对动作直接发给机器人，也不将 Zarr 当成重播输入。

## UR5 重播

当前实现位于 `replay/replay_xac_ur5.py`，推荐通过 `python deploy.py replay` 调用；
`replaydata1/` 只保存 episode 数据，旧版重复脚本已移入
`archive/replaydata1_legacy/`。重播验证记录保存在 `logs/replay/`，其中当前已完成的
两次验证是 `replay_gx_down_8s.json`（机械臂）和 `replay_arm_gripper.json`（机械臂+XC330）。

当前左臂已验证的重播使用**电脑端 TCP 标定**：
`--tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json`。
原始 `calibration/tcp_gripper_01.json` 保留作对照，不应在未重新验证方向时替代候选文件。
不用在 UR5 安装文件保存夹爪 TCP，程序也不会调用 `setTcp`。
机器人当前激活 TCP 可以是法兰零偏置，也可以是之前临时设置的夹爪 TCP；程序自动读取实际偏置，避免重复补偿。

### 1. 准备一条可重播的 episode

输入是 `build` 生成的 episode 目录，不是原始视频、`action.npy`、模型 checkpoint 或 `.zarr.zip`。
例如 `dataset/replay/001/data/` 至少包含：

```text
d435i_timestamps.npy
robot0_eef_pos.npy
robot0_eef_rot_axis_angle.npy
robot0_gripper_width.npy
valid_mask.npy
```

当前已提供采集包 `replaydata1/dataset/replay/001`，包含 81 帧有效数据（约 10 Hz、8.01 秒）。
这条轨迹最大相对位移约 0.216 m，因此预检必须显式设置至少 0.25 m 的距离上限。
应确认手持采集和机器人标定使用同一 G 原点及轴方向；`T_I_G` 是手机外参，不能代替 `T_F_G`。

在同一个终端设置本次重播参数：

```bash
cd /home/eirl3090/Desktop/umi_core
conda activate umip
EPISODE="replaydata1/dataset/replay/001"
ROBOT_IP="192.168.1.101"
TCP_CAL="calibration/tcp_gripper_01_gx_down_candidate.json"
python deploy.py check --episode "$EPISODE"
```

程序只选取最长的连续有效片段，不保证把一个 episode 中所有分段都重播；报告会显示选择的帧区间。

### 2. 离线检查轨迹（不连接 UR5）

```bash
python deploy.py replay --episode "$EPISODE" \
  --tcp-calibration "$TCP_CAL" --max-distance-from-start-m 0.25 --speed-scale 0.1 \
  --report logs/replay/replay_offline_01.json
```

查看 `frames`、源帧区间、总时长、最大位移、线速度、角速度和夹爪宽度范围。
离线没有真实机器人起点，XYZ 范围不是实际工作区，不能据此确定现场 Z 限位。
离线计算控制器命令速度时假设激活 TCP 为零；下一步只读检查会使用真实偏置重新计算。

### 3. 摆放起点并执行只读工作区检查

若探针仅用于标定，重播前取下探针并复核夹爪装配未松动。
用示教器将夹爪 G 原点摆到演示起始位置对应的实际位置和朝向，周围留出运动空间。
程序从当前位置直接开始相对重播，不会先去 Home，也不会自动找工件。
每次重播都会重新锚定起点；执行过 3 秒测试后，需要重新摆回同一个起点才能比较后续测试。

轨迹锚定和电脑端补偿为：

```text
T_B_G_start  = T_B_H_actual @ inverse(T_F_H) @ T_F_G
T_B_G(t)     = T_B_G_start @ inverse(T_W_G_start) @ T_W_G(t)
T_B_H_target = T_B_G(t) @ inverse(T_F_G) @ T_F_H
```

`B` 是 UR 基座、`H` 是控制器当前激活 TCP。G 的插值在每个控制周期完成后才换算成 H 命令，
因此旋转引起的法兰位置补偿也会执行。这里不用 `go_home.py` 的世界坐标或 45 度安装变换。

Z 上下限必须来自现场测量，单位米，针对**夹爪 G 原点的 UR 基座 Z**，不是世界 Z。
倾斜安装时基座 Z 可能为负，不能直接填写 `0` 作为下限。
G 原点限位不等于整个工具避障，须结合指尖、相机和工作台的实际空间确认轨迹。
下面两条命令会提示你输入测得的上下限：

```bash
read -r -p "G原点允许的UR基座最小Z（米）: " SAFE_MIN_Z
read -r -p "G原点允许的UR基座最大Z（米）: " SAFE_MAX_Z
python deploy.py replay --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" --maximum-base-z-m "$SAFE_MAX_Z" \
  --speed-scale 0.1 --report logs/replay/replay_robot_check_01.json
```

没有 `--enable-motion` 就只读，不上传控制脚本。查看 `anchor_source`、起点、XYZ 范围，
确认 `tcp_mode` 是 `computer_side`，同时查看软件补偿后的控制器命令速度。
预览后不要更换激活 TCP；若改变，程序会拒绝执行，需重新检查。

### 4. 首次实机：只动机械臂，低速运行 3 秒

```bash
python deploy.py replay --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" --maximum-base-z-m "$SAFE_MAX_Z" \
  --enable-motion --speed-scale 0.1 --max-duration-s 3 \
  --trace logs/replay/replay_arm_3s_01.json
```

检查输出后输入 `RUN`，重播就开始，**这里不需要像 verify-tcp 一样再按回车**。
终端每秒显示源时间、G 位移和跟随误差。暂时不加 `--enable-gripper`，夹爪不被程序控制。
`0.1` 表示十分之一速度，3 秒墙钟时间约对应原演示前 0.3 秒；若演示开始本来静止，运动可能很小。
`max-duration-s` 只限制运行时间，预检查仍审查整条有效轨迹。

### 5. 完整空载机械臂重播

重新摆回相同起点，确认短段方向和误差正常后，去掉 3 秒限制：

```bash
python deploy.py replay --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" --maximum-base-z-m "$SAFE_MAX_Z" \
  --enable-motion --speed-scale 0.1 \
  --trace logs/replay/replay_arm_full_01.json
```

本步骤仍不控制夹爪。实际执行时长约为原有效轨迹时长的 10 倍。

### 6. 加入夹爪，先短段再完整

确认执行夹爪的宽度标定有效、电机 ID 1/2、串口和波特率正确。
默认 Linux 串口 `/dev/ttyUSB0`、波特率 57600；另一套执行夹爪应使用单独标定文件。
将夹爪实际开口预先调到报告中 `gripper_width_start_m` 对应的值附近，
程序开启扭矩前会检查起始电机位置，轨迹全部开口必须在标定范围内。

```bash
python deploy.py replay --episode "$EPISODE" \
  --robot-ip "$ROBOT_IP" --tcp-calibration "$TCP_CAL" \
  --minimum-base-z-m "$SAFE_MIN_Z" --maximum-base-z-m "$SAFE_MAX_Z" \
  --enable-motion --enable-gripper \
  --gripper-port /dev/ttyUSB0 --gripper-baudrate 57600 \
  --gripper-calibration calibration/gripper_width.json \
  --speed-scale 0.1 --max-duration-s 3 \
  --trace logs/replay/replay_with_gripper_3s_01.json
```

完整机械臂轨迹和夹爪短段通过后，再运行上面的命令，移除 `--max-duration-s 3`，
并将 trace 改成 `logs/replay/replay_with_gripper_full_01.json`。最后再布置实际任务工件。

XC330 replay 默认使用本机驱动的限流位置模式：目标电流 600、保持电流 100、
profile velocity 30、acceleration 20（均为 Dynamixel 原始单位）。若现场夹爪负载、
电源或限流参数不同，应先在空载短段中调整并记录以下参数：

```bash
--gripper-current-limit-raw 600 \
--gripper-goal-current-raw 600 \
--gripper-hold-current-raw 100 \
--gripper-profile-velocity 30 \
--gripper-profile-acceleration 20
```

`--enable-d435i` 是可选的相机启动开关。它只启动并后台排空 D435i 帧，不把实时图像
送入当前重播控制器；相机串号或无 IMU 设备可以这样指定：

```bash
python deploy.py replay ... --enable-d435i --d435i-serial SERIAL --d435i-no-imu
```

如果只是检查相机而不运行机器人，使用 `python deploy.py camera --no-preview`。

### 停止、报告和限位

- `Ctrl+C` 会停止伺服、退出控制脚本，并关闭程序开启的夹爪扭矩；不会自动回到起点。
- 正常结束、限时结束、异常和 Ctrl+C 都保存已采集的 trace；查看 `completed` / `error`。
  `completed=true` 表示正常结束，也可能是 3 秒限时结束，不代表整个源轨迹已播完。
- Trace 中 `target_pose` / `actual_pose` 是 G 在 UR 基座系的位姿；
  `controller_target_pose` / `controller_actual_pose` 是真实发送和读回的 H 位姿。
- 默认最大位移为起点周围 0.12 m，G 和软件补偿后控制器命令的线速度上限为 0.05 m/s，
  角速度上限 0.5 rad/s。超限时先核查轨迹和现场空间；速度问题优先降低 `--speed-scale`。
  经确认需要更大位移范围时显式提供 `--max-distance-from-start-m`，不要为了运行而盲目放宽。
- 重跑时更换报告/trace 名称，保留之前的实验记录。控制器启动瞬间速度缩放为零不直接作为拒绝理由。
- UR5/CB 系列默认 125 Hz。相对重播不需要策略模型；125 Hz 是插值控制频率，不是采集频率。
- 旧方式仍可用 `--tcp-offset TX TY TZ RX RY RZ`，要求示教器已激活相同的 G TCP；
  与 `--tcp-calibration` 互斥，当前电脑端方案使用后者即可。

## 后续训练

### 导出训练集并检查时间语义

以下命令在 `umi_core` 执行：

```bash
python deploy.py export --episodes-root dataset/replay --output dataset/replay.zarr.zip --manifest dataset/replay_manifest.csv --min-segment-length 32 --exclude-source-sessions
```

末尾不带编号的 `--exclude-source-sessions` 表示空排除名单，与当前默认值相同，可以省略。
若本批确有坏 session，则填入本批的编号，例如 `--exclude-source-sessions 004 009`。
导出摘要中的 `exclusion_reasons` 仍有旧批次 003/005 的说明，不能将它们当作本批自动排除依据；
实际排除名单以 `excluded_source_sessions` 和 CSV 中导出的片段为准。
即使清空名单，程序仍会按有效掩码过滤并切分连续区间。32 帧只是本例的初筛长度，不是成功判据。

检查 `replay_manifest.csv` 与同名 `.json` 中的来源、有效片段数、帧数及
`frequency_hz_median` / `frequency_hz_range`。当前数据集通常保存为 10 Hz，
但必须以每批数据的实际时间戳统计为准。
导出器不会把混合帧率、非均匀时间戳自动重采样为均匀频率；训练加载器按帧索引采样，
所以持续丢帧、大间隔或混合采样率的数据应先修复/重采样或排除，再导出。

训练使用 RGB `224×224` 居中 letterbox、无预设夹爪遮罩；深度/IMU 留作原始记录和同步，
下面的默认策略不把它们作为网络输入。

| 数据位置 | 动作含义 |
|---|---|
| episode 的 `action.npy` | 构建器生成的未来相对动作，不能直接作为下面训练命令的输入 |
| 导出 Zarr 的 `data/action` | 每帧 `[位置3, 轴角3, 宽度1]`，7 维绝对位姿/宽度 |
| UmiDataset 提供给网络 | 按配置转成相对位姿及 6D 旋转表示，`3+6+1=10` 维 |

不要为匹配网络的 10 维而手工改写 Zarr；使用对应加载器完成转换。
数据划分应按原始演示/采集批次隔离，同一 session 切出的相似片段不能同时落到训练和最终测试集。
XAC 加载器已按 source_session 分组划分验证集；还应单独保留未进入训练包的完整 session 作为最终测试。

### 训练（在 PRO6000 上进行）

当前训练代码位于 PRO6000（`ssh pro6000@192.168.30.185`）的 `~/Desktop/umip/train`，推荐 v2 配置：

```bash
cd ~/Desktop/umip/train
conda activate umip
tmux new -s xac_v2
bash train_v2.sh                                    # 默认 400 轮，输出 data/outputs/xac_v2_<时间>
bash train_v2.sh data/outputs/xac_v2_800ep training.num_epochs=800
python eval_offline.py data/outputs/<run>/checkpoints/best.ckpt   # 离线 mm/度误差
```

v2 微调 ResNet18 并做图像增强，每 10 轮按验证集上实际采样的动作误差选 `best.ckpt`，不早停。
本目录只保留部署推理所需的代码；旧训练代码已移到 `archive/removed_training_20261022/`，旧方案训练的 checkpoint 加载时会被拒绝。环境、数据要求、训练策略和断点续训见
[XAC 训练指南](docs/xac_training_zh.md) 和训练机上的 `README.md`。

## 训练完成后的策略部署（真机）

部署入口是 `test/test_model_inference.py`。它从 checkpoint 读取策略配置和 EMA 权重，
实时读取 D435i RGB、UR5 TCP 和夹爪宽度，构造与训练一致的观测，推理后以 125 Hz servoL 插值执行。
`eval_real.py` 仍为空，`deploy.py replay` 不加载 checkpoint。
用旧训练方案得到的 checkpoint 会被拒绝加载（`ObsoleteCheckpointError`）。

### 当前模型

```text
data_model/xac_v2_h1img1_800ep_20261022/checkpoints/best.ckpt
sha256 b3fef38a518da6138f4068ab613ed1e8fbb27fa92dde080fd44cbc4226b909ab
```

v2 方案，800 轮，2026-10-22，训练数据 `dataset_v2`，**位姿和图像都只用当前 1 帧**（训练参数
`task.low_dim_obs_horizon=1 task.img_obs_horizon=1`）。部署脚本从 checkpoint 自动读取帧数，启动时打印
`Observation horizon: image=1, pose/width=1`。

选它的原因：2 帧模型会从"上一帧到当前帧的位移"判断速度。机械臂限速或在方块附近放慢时，它以为几乎没动，
每步往后（-Z_G）偏约 4 mm，表现为停在方块前、往后退、不闭合（2026-10-22 实测）。1 帧模型在静止输入下偏差约 0.1 mm/步。

| 模型 | 离线评分 | 静止时每步后偏 |
|---|---|---|
| `xac_v2_h1img1_800ep_20261022`（当前） | 1.74（位置 10.4 mm、旋转 1.10°、夹爪 3.3 mm） | -0.1 mm |
| `xac_v2_dataset2_800ep_20261022`（2 帧，备选） | 1.63（7.9 mm / 1.06° / 3.4 mm） | -4.3 mm |

"原地不动"评分为 3.00（24.4 mm / 1.38° / 6.3 mm）。训练机原文件位于
`pro6000:~/Desktop/umip/train/data/outputs/xac_v2_h1img1_800ep_20261022/`。在本机（RTX 3060）单次推理约 91 ms。

**测试中：`xac_v3_h1img1_800ep_20261025`**（修正 T_I_G 后的 `dataset_v3` 训练，配置同上）。
**v3 的完整上机流程、参数和 10-29 全部故障记录以 `data_model/xac_v3_h1img1_800ep_20261025/README.md` 为准。**
和 v2 不同，**必须加 `--no-auto-tilt`**，直接从竖直的 go_home 起步。2026-10-29 状态：软地板（按较低指尖，急停线 0.363）、
夹爪开关模式和"夹爪动作时机械臂停住"都已加入；能下到地板并触发夹紧，夹紧后的抬起和放碗还在验证。

新模型可在本机直接从训练机复制（比经其他电脑中转快）：

```bash
mkdir -p data_model/<run>/checkpoints
scp pro6000@192.168.30.185:~/Desktop/umip/train/data/outputs/<run>/checkpoints/best.ckpt data_model/<run>/checkpoints/
```

### 上机前检查

1. **起始姿态：`RUN` 后脚本自动绕方块前倾（`--auto-tilt`，默认开启）**：采集时手持夹爪和 iPhone 整体是斜着拿的，
   18 条演示开始时指尖朝碗的方向前倾，总倾角 9.3°~13.5°（平均前倾 10.4°、侧倾 3.6°），每次略有不同。
   go_home 后夹爪是竖直的，模型从没见过；竖直起步会停在方块前、往后退或压到桌面（2026-10-22 实测）。
   输入 `RUN` 后，脚本先绕方块（指尖方向前方 10.6 cm 的点）旋转约 11° 到平均姿态，等相机拿到新画面后才开始推理。
   这一步两指之间只移动约 19 mm，法兰 / 示教器 TCP 会移动约 80 mm，看起来动作较大是正常的；
   加 `--tilt-duration-s 6` 可让它更慢（默认 2 s）。"竖直向下"按 go_home 时的指尖方向计算（底座斜装 45°）。
   `--frame-correction`（默认关闭）是另一种假设：把这 10° 当成 `T_I_G` 标定误差在软件里补偿；
   只有确认标定有误时才用 `--frame-correction --no-auto-tilt`。
   以后补采数据时，建议有意识地变化倾斜角（接近竖直、5°、10°、15°、左右略斜）、起始高度和方块位置。
2. **起始位置与演示一致**：两指之间正对方块、距离约 10.6 cm，完全张开（95~99 mm），方块和碗摆在接近采集时的位置。
   **以画面为准**：演示开头的 224×224 画面中，方块边长约 17 像素，中心约在 (115, 97)，即水平居中、竖直方向在中上部。
   2026-10-22 实测起始时方块只有约 13 像素、中心 y≈85（偏远偏前），结果夹爪下降到抓取高度时方块仍在指尖前上方，
   夹爪在方块后方来回试探、不闭合。此时应把起始位置再往下放约 3~4 cm。
   可先跑第 1 步预览并加 `--save-images`，打开图像目录中最早的一张，检查方块的大小和位置。
   当前只有 18 条演示，物体位置差别大时成功率会明显下降。
3. **安全范围**：任务需要向前约 25~29 cm、向下约 8~12 cm。默认离起点 0.12 m 的限制会在中途停止，
   因此完整任务要放宽到 0.35 m。
   本机 UR5 底座斜装 45°，`--minimum/maximum-base-z-m` 限制的是**底座坐标系的 Z**，不是真实的竖直高度，
   设定方法与轨迹重播相同。
   **软地板（2026-10-29 新增，按竖直方向、按较低的指尖）**：`--floor-depth-m` 让较低那个指尖沿竖直向下（`--world-down-in-base`，默认 go_home 时的指尖方向）不超过该深度，超过时竖直抬回，
   侧向移动、姿态和夹爪命令照常执行，到达地板后模型仍能继续闭合；`--max-depth-m` 是同一深度的急停线，必须比地板至少深 0.5 mm。
   限制的是**两个指尖中更低的那个**（指尖在 G 原点两侧各 `--fingertip-half-span-m`，默认 0.05 m = 全开），不是 G 原点：夹爪运行中会倾斜 4~13°，靠下的指尖会比 G 原点（两指尖中点）低几毫米。10-29 15:58 G 原点停在地板上，倾斜约 6.7° 的那个指尖仍深 5.8 mm。
   不能用底座 Z 做地板：底座斜装 45°，夹爪侧向移动也会改变底座 Z（10-29 15:37 底座 Z 停在旧地板上，指尖仍深 4.5 mm，碰桌）。
   用户 10-29 指定急停线 0.363（最低指尖深度），软地板 0.3625（高 0.5 mm）。桌面、方块或起点高度变了要重新测：把夹爪手动移到允许的最低位置，运行只读脚本 `python test/read_g_depth.py`，它会打印这两个值。
4. **速度**：默认限速 0.05 m/s，演示峰值是 0.2~0.4 m/s，动作会慢几倍。第一次保持默认，
   确认动作正确后再逐步提高 `--max-linear-speed-m-s`（例如 0.1）。
5. 示教器速度调低，急停按钮放在手边。运动模式需要输入 `RUN` 才会开始。

### 步骤

```bash
cd ~/Desktop/umi_core
conda activate umip
CKPT=data_model/xac_v2_h1img1_800ep_20261022/checkpoints/best.ckpt
SAFE_MAX_DEPTH=0.363    # 急停线：较低那个指尖沿竖直向下的深度（米），用户 10-29 指定
SAFE_FLOOR_DEPTH=0.3625 # 软地板：较低那个指尖到这里不再往下但继续执行夹爪，比急停线高 0.5 mm
SAFE_MIN_Z=-0.20        # 底座 Z 粗略下限，只作后备
SAFE_MAX_Z=...       # 安全上限（米）
COMMON="--checkpoint $CKPT --robot-ip 192.168.1.101 \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --minimum-base-z-m $SAFE_MIN_Z --maximum-base-z-m $SAFE_MAX_Z \
  --floor-depth-m $SAFE_FLOOR_DEPTH --max-depth-m $SAFE_MAX_DEPTH \
  --d435i-no-imu"
# 测 v3 模型时：CKPT 换成 data_model/xac_v3_h1img1_800ep_20261025/checkpoints/best.ckpt，并在 COMMON 末尾加 --no-auto-tilt

# 1. 预览：读取相机和机械臂，打印预测，不发送运动
python test/test_model_inference.py $COMMON --preview-steps 30 --trace logs/policy_preview.json \
  --save-images logs/img_preview

# 2. 只动机械臂（不动夹爪），限制 12 cm，验证下降方向
python test/test_model_inference.py $COMMON --enable-motion \
  --max-distance-from-start-m 0.12 --trace logs/policy_arm_only.json \
  --save-images logs/img_arm_only

# 3. 完整任务（测试能否抓取并放入碗中）：加夹爪，放宽距离和时长
python test/test_model_inference.py $COMMON --enable-motion --enable-gripper \
  --gripper-mode switch --grip-close-below-at-floor-m 0.093 \
  --max-distance-from-start-m 0.35 --max-duration-s 60 \
  --max-linear-speed-m-s 0.15 --max-linear-acceleration-m-s2 0.5 \
  --max-angular-speed-rad-s 0.5 --max-angular-acceleration-rad-s2 1.5 \
  --trace logs/policy_full.json --save-images logs/img_full
```

- **第 3 步才是完整的抓取测试**；第 1、2 步只用于检查。
- 第 1 步应看到：目标位置朝方块向下移动；宽度先保持张开，靠近方块后再减小。
  不符合时不要进入第 2 步。
- 第 1、2 步不连接夹爪，模型看到的宽度是 `--default-width-m`（默认 0.098 m，即张开，与演示开头一致）。
  不要把它设成 0.05 这类半闭合值，否则模型会以为已经夹住方块，直接朝碗移动。
- 第 2 步只验证下降和接近，触发距离保护而停止是正常的。
- 每次换 trace 文件名和 `--save-images` 目录，保留记录。`--save-images` 保存模型每一步实际看到的 224×224 图像，
  文件名与 trace 中的 `observation_monotonic_s` 对应。失败时把 trace 和图像目录发给分析的人。
- trace 中 `observation_width_m` 是模型看到的实测夹爪宽度（不接夹爪时为 `--default-width-m`），
  `width_m` 是发给夹爪的命令宽度，`floor_active` 表示这一步是否被软地板截断。
- 用了软地板后，第 2 步应看到机械臂停在地板高度（trace 中 `floor_active` 为 true，`actual_tip_depth_m` 不超过地板），而不是触发 Depth 急停。
- 第 3 步终端应依次打印 `[gripper] closing` → `[arm] holding ...` → `[gripper] closed` → `[arm] resuming ...`，`closing`/`closed` 只出现一次；反复夹了又松说明部署脚本不是最新版。
- `192.168.1.101` 是左臂，与轨迹重播时一致；换臂或换 TCP 时同步修改 `--robot-ip` 和 `--tcp-calibration`。
- 夹爪串口默认读取 `xac_config.py` 的 `DXL_PORT`，需要时用 `--gripper-port` 指定（例如 `--gripper-port /dev/ttyUSB0`）。
- 修改了 `CKPT`、`SAFE_MIN_Z` 等变量或 README 里的 `COMMON` 后，要重新执行 `COMMON=...` 那几行，新参数才会生效。

### 故障排查（2026-10-22、10-29 实测）

**报 `Device or resource busy`**：有别的程序占用相机，最常见的是 `realsense-viewer`。
viewer 能看到画面只说明相机正常；运行部署脚本前必须关闭 viewer 和其他相机程序。

**启动阶段偶尔丢帧**：部署脚本启动相机后会先预热（最多 5 s 等到 3 帧），运行中单帧超时会自动重试，
连续 1 s 没有画面才停止。预热失败时会直接提示下面的处理方法。

**相机报"读不到数据"/"RGB-D frame did not arrive"**，有两个已确认的原因：

1. **深度流卡死**：USB 集线器或夹爪串口重新插拔后，相机重连时彩色画面正常、深度一帧都不出。
   部署脚本要同时拿到彩色和深度才算读到一帧，所以一直报读不到数据。处理方法：先关闭所有用到相机的程序，再运行

   ```bash
   python driver/reset_d435i.py
   ```

   看到 `color 30/30, depth 30/30` 和 `OK` 就恢复了。复位后仍不稳定时，把相机直接插到电脑的 USB3 口，不经过集线器。
2. **没有带 `--d435i-no-imu`**：开启 IMU 时每秒只能成功读到 1~3 帧，关闭后为 16~19 帧。
   策略不使用 IMU，部署时必须带这个参数（已写在 `COMMON` 里）。

**机械臂斜着下降**（2026-10-22 第 2 步实测，夹爪坐标系下走了 X +57 mm、Z +82 mm）：
演示中"接近方块"是沿 +X（指尖方向，约 106 mm），"运往碗"是沿 +Z（约 274 mm）。
当时不连接夹爪，模型看到的宽度被固定为 0.05 m，而演示中接近方块时宽度是 95~99 mm，
模型因此判断"已经夹住"，往下的同时混入了往碗的 +Z 运动。默认值已改为 0.098 m（张开）。
这与底座斜装无关：观测和动作都是相对夹爪自身坐标系的，底座朝向不影响模型，无需重新训练。

**边闭合边下降，指尖碰到桌面后被撑开**（2026-10-29 v3 第 3 步实测，trace `logs/v3_full.json`）：
沿指尖下降到约 78 mm 时预测宽度开始减小，最低到 76 mm；机械臂继续下降到约 94 mm，比允许的最低位置深约 14 mm，
指尖碰桌被撑开，模型看到宽度变大，又预测张开。演示中闭合时夹爪几乎不动（平均移动约 3 mm），0.5~1 s 内闭合到 40 mm 以下。
处理：设置软地板 `--floor-depth-m` / `--max-depth-m`（见"上机前检查"第 3 条），让它停在地板高度继续闭合。
**设了底座 Z 软地板仍碰到桌面**（10-29 15:37，`logs/v3_arm_only_floor.json`）：底座斜装 45°，底座 Z 不是竖直高度：夹爪侧向移动也会改变底座 Z。10-29 15:37 底座 Z 停在 -0.168 的旧地板上，但侧向偏了约 8 mm，指尖在竖直方向比实测最低位置深 4.5 mm，碰到桌面。现在地板按竖直深度限制，侧向怎么动都不会更深。
根本原因是画面不一致：同一台 D435i（序列号 939622070870，内参相同），夹爪张开时两指在 640×480 画面中的内侧边缘
演示约为 x=189/546，真机约为 x=83/586，真机指尖低约 55 像素，色调也偏青绿。模型用 1 帧相对位姿时位姿输入恒为单位阵，
是否闭合只能靠图像和夹爪宽度判断，所以需要让执行端夹爪的相机或手指安装、宽度标定与采集端一致。

**夹爪开关阈值（2026-10-29 由 `dataset_v3` 统计）**：

| 统计项 | 数值 |
|---|---|
| 靠近方块阶段（闭合前）的最小宽度 | 92.6~97.0 mm |
| 从 <90 mm 到 <40 mm 的时间 | 0.3~1.0 s（平均 0.6 s） |
| 夹住后的宽度（每条演示的中位数） | 0.7~30 mm，取决于人手捏紧程度 |
| 闭合后在碗上方重新张开的演示 | 18/18 |

据此在部署脚本中加入了 `--gripper-mode switch`（第 3 步命令已带上）：预测宽度 < 88 mm（`--grip-close-below-m`）时夹紧并锁住；
机械臂被软地板挡住时阈值放宽到 93 mm（`--grip-close-below-at-floor-m 0.093`，地板上预测 80~93 mm，下降途中 ≥ 96 mm）；
已夹紧且预测宽度 > 60 mm（`--grip-open-above-m`）时松开。夹紧和松开直接复用 `driver/gripper_torque_control.py`
（速度模式，两电机都停转满 1 s 后锁住并保持扭矩）。不能闭合后永久锁死，否则无法放下方块。
夹紧/松开期间机械臂原地停住并暂停"预测过期"检查，夹爪只把动作做完、不根据夹紧前的旧宽度做决定；拿到动作之后的新预测才继续（10-29 15:51 推理被串口读写卡住、16:46/16:49 夹了又松循环，都由此修复）。
88 mm 比靠近阶段最小宽度低约 4.6 mm，不会在靠近时误触发。用 0/1 标签重训解决不了"何时闭合"靠图像判断不准的问题，暂不建议。

**模型预测闭合，但夹爪卡在约 50 mm 没夹住**（2026-10-29 15:09，`logs/v3_full_floor.json`，默认位置模式）：
预测宽度降到约 20 mm，命令 17~24 mm，实测停在 50 mm，手指没碰到方块。这个夹爪从全开到全闭电机只转约 141 tick（1 tick ≈ 0.7 mm），
位置模式误差小时出力很小，顶不过软手指和连杆的弹力。没夹住后模型以为已夹住，开始往碗走（+Z_G），后续动作也就都错了。
处理：加 `--gripper-mode switch`。状态切换会打印 `[gripper] ...`，并另存为 trace 同名的 `*_gripper.json`。
旧版脚本备份在 `archive/pre_switch_20261029/`。

**加载模型报 `ObsoleteCheckpointError`**：用的是旧训练方案的模型，例如
`data_model/xac_lr1e-3_plateau_20261021_214253/`。旧方案有错误（图像 pooling 层未训练、输入未归一化、选模型方式错误），
这些模型已被禁止加载。请改用上面"当前模型"的路径。

闭环策略会随图像反馈重算动作，与轨迹重播分别验收。离线误差不代表真机成功率，
应在多种物体初始位置下统计成功次数。

## 离线验证

```bash
python -m unittest discover -s . -p "test_replay*.py" -v
python -m unittest discover -s . -p "test_collection_deployment.py" -v
python -m unittest test_policy_loading -v      # checkpoint 加载
```

这些测试验证数据和控制逻辑，不证明机器人现场可达性、避障或安装参数正确。
