# xac_v3_h1img1_800ep_20261025 使用说明

用**修正后的 T_I_G 标定**重建的数据（`dataset_v3`）训练的策略。v2 在真机上"靠近方块但不夹、抖动后移开"，v3 用来验证这是不是标定造成的。

## 模型信息

| 项目 | 内容 |
|---|---|
| checkpoint | `checkpoints/best.ckpt`（第 799 轮，与最后一轮相同） |
| sha256 | `fdc29855197809a4a809e1e2df39a96d3fe17de9c8f941fae9c08f2b4df9d627` |
| 训练机 | PRO6000，`~/Desktop/umip/train/data/outputs/xac_v3_h1img1_800ep_20261025/`，2026-10-25 |
| 训练数据 | `dataset_v3/replay.zarr.zip`（sha256 `bca6807c…`），18 条演示，1636 帧，10 Hz |
| 训练配置 | `train_xac_v2.yaml`，800 轮，`task.low_dim_obs_horizon=1 task.img_obs_horizon=1` |
| 观测 | 当前 1 帧 D435i RGB 224×224 + 当前 1 帧夹爪系位姿和夹爪宽度 |
| 输出 | 未来 16 步动作（相对位姿 + 夹爪总开口宽度），执行前 8 步 |

**和 v2 的唯一区别**：iPhone→夹爪外参改用 `calibration_tig_level/T_I_G.npy`（旋转修正 10°）。
旧标定让所有演示看起来朝同一方向前倾约 10°；修正后演示开始时指尖相对竖直平均只倾斜 2.6°。

离线验证（2 条验证演示，执行的前 8 步）：

| | 位置 | 旋转 | 夹爪 | 评分 |
|---|---:|---:|---:|---:|
| 本模型 | 10.7 mm | 1.11° | 3.0 mm | 1.72 |
| 原地不动 | 24.4 mm | 1.38° | 6.3 mm | 3.00 |

评分低于 3 表示比原地不动好。只有 2 条验证演示，结果偏乐观，以真机为准。

注意：用 1 帧相对位姿时，位姿输入恒为单位阵，模型**只靠图像和夹爪实测宽度**判断何时夹取。

## 真机测试指令（2026-10-29 最新）

按 0 → 1 → 2 → 3 → 4 的顺序做，每一步通过再做下一步。

### 0. 准备

- 关掉 `realsense-viewer` 等占用相机的程序（否则报 `Device or resource busy`）。
- 示教器速度调低，急停放在手边。
- 上一次运行如果被 Ctrl+Z 暂停过，先在那个终端输入 `fg` 再按 Ctrl+C，让它正常退出（`pgrep -af test_model_inference` 应为空）。
- 桌面、方块、碗与 10-29 测试时一致（地板深度按那时的桌面定；变了要重测，见"软地板"）。

```bash
cd ~/Desktop/umi_core
conda activate umip
python go_home.py        # 夹爪竖直向下
```

go_home 后把方块放在两指之间正前方，夹爪完全张开。起始画面应接近演示：方块边长约 17 像素，中心约在 (115, 97)（第 1 步检查）。

### 设置变量（每次新开终端、或改了任何变量后都要执行）

```bash
cd ~/Desktop/umi_core
conda activate umip
CKPT=data_model/xac_v3_h1img1_800ep_20261025/checkpoints/best.ckpt
SAFE_MAX_DEPTH=0.363    # 急停线：较低那个指尖沿竖直向下的深度（用户 10-29 指定），越过即报错停止
SAFE_FLOOR_DEPTH=0.3625 # 软地板：较低那个指尖到这里不再往下，夹爪继续工作；比急停线高 0.5 mm
SAFE_MIN_Z=-0.20        # 底座 Z 粗略下限，只作后备（底座斜装 45°，竖直高度由上面两项控制）
SAFE_MAX_Z=1          # 底座 Z 上限（米），按工作空间填；要高于起点（约 -0.113）
COMMON="--checkpoint $CKPT --robot-ip 192.168.1.101 \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --minimum-base-z-m $SAFE_MIN_Z --maximum-base-z-m $SAFE_MAX_Z \
  --floor-depth-m $SAFE_FLOOR_DEPTH --max-depth-m $SAFE_MAX_DEPTH \
  --gripper-mode switch --grip-close-below-at-floor-m 0.093 \
  --d435i-no-imu --no-auto-tilt"
T=$(date +%m%d_%H%M)    # 本次记录的文件名后缀，避免覆盖旧记录
```

- `--no-auto-tilt` 必须带，不要加 `--frame-correction`。
- 不要用终端历史（上箭头）里的旧命令：旧命令的地板参数或夹爪模式已经过时。

### 1. 预览（不运动）

```bash
python test/test_model_inference.py $COMMON --preview-steps 30 \
  --trace logs/v3_preview_$T.json --save-images logs/v3_img_preview_$T
```

通过条件：

- 启动时打印 `Observation horizon: image=1, pose/width=1`；
- 打印 `Soft floor (lower fingertip, world vertical): depth <= 0.3625 m, hard stop 0.3630 m; start depth ...`（起点约 0.288，离地板约 75 mm）；
- 打开 `logs/v3_img_preview_$T/` 中最早的一张：方块约 17 像素、中心约 (115, 97)。方块偏小或偏上，起点太远，往下放；偏大偏下，起点太近，往上抬；
- 打印的 `target` 朝方块方向（夹爪系 +X）变化，`width` 约 0.097 m（张开）。

### 2. 只动机械臂（不接夹爪）：验证软地板

```bash
python test/test_model_inference.py $COMMON --enable-motion \
  --max-distance-from-start-m 0.12 --max-duration-s 20 \
  --trace logs/v3_arm_$T.json --save-images logs/v3_img_arm_$T
```

输入 `RUN` 后才会运动。通过条件：

- 机械臂下降到地板高度后停住不再往下，**指尖不碰桌面**；
- 不出现 `Depth protection` 报错（出现说明越过急停线，先停下检查）；
- 到时长或距离保护后停止都是正常的。

不接夹爪时模型看到的宽度固定为 0.098 m（张开），所以这一步**本来就不会闭合**，只用于验证下降深度。

### 3. 单独验证夹爪开关模式（不接机械臂）

夹爪完全张开，两指之间放上方块：

```bash
python -c "
import time; from pathlib import Path
from driver.xc330_switch import XC330SwitchGripper
from xac_config import DXL_PORT
g=XC330SwitchGripper(DXL_PORT,57600,Path('calibration/gripper_width.json'),(1,2),600,80,close_below_m=0.088,open_above_m=0.06)
try:
    g.initialize(g.read_width_m()); t=time.time()
    while time.time()-t<4: g.command_width(0.02); print(round(g.read_width_m()*1000,1),g.state); time.sleep(0.1)
    input('已夹紧，回车松开'); t=time.time()
    while time.time()-t<2: g.command_width(0.095); time.sleep(0.1)
finally: g.close()
"
```

通过条件：打印 `[gripper] closing`，1~2 s 后 `[gripper] closed`，方块被夹紧，效果和 `python driver/gripper_torque_control.py` 里按 `g` 一样；回车后松开。这里夹不住，问题在夹爪本身，先不要做第 4 步。

### 4. 完整抓取（接夹爪）

```bash
python test/test_model_inference.py $COMMON --enable-motion --enable-gripper \
  --max-distance-from-start-m 0.35 --max-duration-s 60 \
  --trace logs/v3_full_$T.json --save-images logs/v3_img_full_$T
```

启动时**必须同时打印**：

- `Soft floor (lower fingertip, world vertical): depth <= 0.3625 m, hard stop 0.3630 m ...`
- `Gripper switch mode: grip below 88 mm (93 mm on the floor), release above 60 mm`

缺任何一行，按 Ctrl+C，重新执行"设置变量"。

正常过程中终端依次打印：

1. `[gripper] closing (..., at floor, threshold 93 mm)` 或 `[gripper] closing`（预测宽度低于阈值，开始夹紧）；
2. `[arm] holding while the gripper moves`（夹紧期间机械臂原地不动）；
3. `[gripper] closed (..., both motors stalled)`（夹紧并锁住）；
4. `[arm] resuming with a prediction observed after the gripper action`（拿到夹紧后的新预测，开始抬起、运往碗）；
5. 到碗上方：`[gripper] releasing` → `release_settling` → `open`。

`closing`/`closed` 只应出现一次。如果反复夹了又松，把记录发来分析。

结束后会生成 `logs/v3_full_$T.json`（每步记录）和 `logs/v3_full_${T}_gripper.json`（夹爪每次状态切换）。
第一次保持默认限速 0.05 m/s；动作正确后再在命令末尾加：

```bash
  --max-linear-speed-m-s 0.15 --max-linear-acceleration-m-s2 0.5 \
  --max-angular-speed-rad-s 0.5 --max-angular-acceleration-rad-s2 1.5
```

夹爪串口默认取 `xac_config.py` 的 `DXL_PORT`，需要时加 `--gripper-port /dev/ttyUSB0`。

## 部署脚本的相关机制

### 软地板（按竖直方向、按较低的指尖）

- `--floor-depth-m`：较低那个指尖沿竖直向下（`--world-down-in-base`，默认 go_home 时的指尖方向）超过该深度时，把目标竖直抬回；侧向移动、姿态和夹爪照常执行，所以到地板后模型仍能夹取。
- `--max-depth-m`：同一深度的急停线，越过就报错停止，必须比地板至少深 0.5 mm；起始位置已低于地板时脚本拒绝启动。
- 指尖位置：G 原点（两指尖中点）两侧各 `--fingertip-half-span-m`（默认 0.05 m = 全开）。夹爪运行中会倾斜 4~13°，靠下的指尖比中点低几毫米，所以按较低的指尖限制。
- 不用底座 Z：底座斜装 45°，夹爪侧向移动也会改变底座 Z。10-29 桌面扫描（`logs/table_sweep_20261029.json`）确认桌面法线与 go_home 指尖方向只差 0.01°。
- **重测**：桌面、方块或起点高度变了，把夹爪手动移到允许的最低位置，运行只读脚本 `python test/read_g_depth.py`，它打印较低指尖的深度和建议的 `SAFE_MAX_DEPTH` / `SAFE_FLOOR_DEPTH`（默认高 0.5 mm），不会让机械臂运动。

### 夹爪开关模式（`--gripper-mode switch`）

- 夹紧和松开直接复用 `driver/gripper_torque_control.py`：速度模式、速度 30；两电机都停转满 1 s 后锁住并保持扭矩，最长 5 s；松开时反转 1 s 后关扭矩。
- 模型的预测宽度只决定**什么时候**切换：
  - 预测宽度 < 88 mm（`--grip-close-below-m`）→ 夹紧；
  - 机械臂被软地板挡住时（最近 0.5 s 内），阈值放宽到 93 mm（`--grip-close-below-at-floor-m`）；
  - 已夹紧且预测宽度 > 60 mm（`--grip-open-above-m`）→ 松开。
- 阈值依据：`dataset_v3` 靠近阶段宽度最小 92.6 mm，开始闭合后平均 0.6 s 降到 40 mm 以下，夹住后 0.7~30 mm，18 条演示都会在碗上方张开；真机上下降途中预测宽度 ≥ 96 mm，在地板上 80~93 mm。
- 夹紧/松开期间机械臂原地停住（演示闭合时也只移动约 3 mm），并暂停"预测过期"检查；这段时间夹爪只把正在进行的动作做完，**不根据夹紧前的旧宽度做任何决定**。动作结束后，等到一帧动作之后采到的新预测才继续（1 s 内拿不到就报错停止），宽度从实测值开始过渡。
- 不加 `--gripper-mode switch` 时是原来的位置模式：电机跟随预测宽度，误差小时出力不够，会卡在约 50 mm 夹不住，不要再用。

### trace 字段

| 字段 | 含义 |
|---|---|
| `relative_action_pose10d` | 本步执行的预测动作（相对位姿 + 第 10 维宽度） |
| `observation_width_m` | 模型看到的实测夹爪宽度（不接夹爪时为 `--default-width-m`） |
| `width_m` | 当前轨迹的宽度（机械臂等夹爪时不会发给夹爪） |
| `floor_active` | 这一步是否被软地板截断 |
| `actual_tip_depth_m` / `actual_world_depth_m` | 较低指尖 / G 原点的竖直深度 |
| `gripper_hold` | 这一步机械臂是否在等夹爪 |

## 结果判断

- **到地板后夹紧、抬起并放进碗里**：流程打通，下一步在多种方块位置下统计成功率，再决定是否补采演示。
- **到地板后迟迟不夹、在方块附近侧向来回**：模型认为"还没到位"。先检查起始画面是否接近演示（方块约 17 像素、中心约 (115, 97)）。仍不行，问题在执行端画面与采集端不一致（见下方"已知问题"），需要调整硬件或补采数据。
- **夹住后不往碗走或乱走**：模型夹住后的阶段同样只靠画面和宽度判断，记下现象并把记录发来分析。

分析时需要提供：`logs/v3_full_$T.json`、`logs/v3_full_${T}_gripper.json`、`logs/v3_img_full_$T/`，以及现场现象（有没有夹住、有没有碰桌、往哪个方向偏）。

## 已知问题

- **执行端画面和采集端不一致**：同一台 D435i（序列号 939622070870，内参相同），但执行端夹爪上相机与手指的相对几何和采集端不同（10-29 早些时候：张开时两指内侧边缘演示约 x=189/546，真机约 x=83/586）。模型在真机上想下到比桌面允许的更深处才夹，所以需要软地板 + 地板放宽阈值。根本解决是让执行端相机/手指安装与采集端一致，或用执行端补采演示。
- 只有 18 条演示，方块位置差别大时成功率会明显下降。

## 故障记录（2026-10-29，新的在前）

| 时间 | 现象 | 原因 | 处理 |
|---|---|---|---|
| 16:49 | 夹住又松开，循环到结束 | 16:46 的修复不完整：夹爪仍保留夹紧前的旧宽度（82.7 mm）做判断 | 等夹爪期间改发"只推进、不做决定"命令（备份 `archive/pre_nodecision_20261029/`） |
| 16:46 | 夹住又松开，循环 60 s，不抬起 | 等夹爪期间仍把夹紧前的预测宽度（84.3 mm > 60）发给夹爪，夹紧即松开、松开又夹紧 | 同上。模型本身正确：夹紧后预测约 20 mm，并要抬起约 20~60 mm |
| 16:26 / 16:37 | 不闭合 | 那两次是第 2 步（不接夹爪），本来就不会闭合；地板上预测宽度 80~93 mm | 加地板放宽阈值 93 mm（备份 `archive/pre_floorgrip_20261029/`） |
| 16:12 | 碰桌 | 运行的是旧参数（地板 0.3618 按 G 原点），夹爪倾斜时较低指尖更深 | 改为按较低指尖限制，用户指定急停线 0.363（备份 `archive/pre_tipfloor_20261029/`） |
| 15:51 / 15:58 | 到方块上方不夹、侧向偏移 | 地板 0.3618 比模型会闭合的高度高约 3 mm | 地板按用户指定的 0.363 重设 |
| 15:51 | 夹完马上松开、不抬起 | 夹紧时串口读写占用 Python 全局锁，推理卡住触发"预测过期"停止，退出时卸力 | 夹爪动作期间机械臂停住并暂停过期检查，停转判定只读速度（备份 `archive/pre_gripperhold_20261029/`） |
| 15:37 | 碰桌 | 底座 Z 地板：侧向偏 8 mm，竖直方向深 4.5 mm | 改为按竖直深度（备份 `archive/pre_depthfloor_20261029/`） |
| 15:24 | 没夹住 | 用了旧命令（位置模式），夹爪卡在约 50 mm | 开关模式写进 `COMMON` |
| 15:09 | 预测闭合但没夹住 | 位置模式出力不够，卡在约 50 mm | 新增开关模式（备份 `archive/pre_switch_20261029/`） |
| 14:32 | 边闭合边下降、碰桌后被撑开 | 没有地板，下降比允许的最低位置深约 14 mm | 新增软地板（备份 `archive/pre_floor_20261029/`） |

## 其他故障排查

- **相机读不到数据 / RGB-D frame did not arrive**：关掉所有相机程序后运行 `python driver/reset_d435i.py`，看到 `color 30/30, depth 30/30` 和 `OK` 即恢复；仍不稳定就把相机直接插到电脑的 USB3 口。
- **加载报 `ObsoleteCheckpointError`**：路径指向了旧方案的模型，检查 `CKPT`。
- 其他问题见 `~/Desktop/umi_core/README.md` 中"训练完成后的策略部署（真机）"一节。
