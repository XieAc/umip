# UMIP：XAC 采集、UR5 重播与训练

本目录将 `collect_xac` 的硬件标定和 TIG 流程、`umi_core` 的采集/UR5 重播改进、
以及完整的官方 UMI 训练源码整合在一起。操作与修改清单见 [MERGE_AND_TRAINING.md](MERGE_AND_TRAINING.md)。
当前标定统一存放在 [`calibration/`](calibration/README.md)，原 TIG 标定目录已归入 `calibration/tig/`。正式标定来自 `collect_xac`。
**当前训练代码在 PRO6000（`ssh pro6000@192.168.30.185`）的 `~/Desktop/umip/train`，推荐用其中的 `train_v2.sh`；**
本目录的 `training/` 是较早的官方 UMI 副本，不含 v2 修复。训练与部署步骤见 [MERGE_AND_TRAINING.md](MERGE_AND_TRAINING.md#训练pro6000)。
训练配置假定导出数据为均匀 10 Hz，正式训练前应核对 manifest 中的实际帧率。

**已经完成标定和轨迹重播，准备采集训练数据：直接从[第 10 节：正式任务演示采集](#10-正式任务演示采集stray--d435i)开始，再按[第 11 节](#11-检查构建和导出-umi-数据集)检查与导出。** 标定用 TIG session 与任务演示分开保存；正式任务采集不需要运行 `tig_verify_capture.py`。

## XAC 手持端标定与采集

本目录用于双 XC330 铰链夹爪、Intel RealSense D435i 和 iPhone/Stray Scanner 的同步采集，并将结果构建为 UMI 数据集和 Zarr 文件。

本文是本目录的统一操作说明，包含：环境准备、夹爪宽度标定、`T_I_G` 探针外参标定、Stray/iPhone 录制与导出、数据检查、数据集构建和故障排查。

## 1. 坐标系和标定目标

项目使用以下齐次变换约定：

```text
T_W_G = T_W_I @ T_I_G
```

- `W`：Stray/ARKit 世界坐标系。
- `I`：iPhone 坐标系。
- `G`：夹爪 TCP 坐标系。
- `T_I_G`：夹爪 TCP 在 iPhone 坐标系中的位姿；平移单位是米。

建议在夹爪和支架上贴上轴标签，并始终使用同一套定义：`+X_G` 沿夹爪中心线朝指尖，`+Y_G` 为侧向，`+Z_G = +X_G × +Y_G`。如果机械结构采用其他定义也可以，但正负号必须在工装、`tig_meta.json` 和后续计算中保持一致。

## 2. 环境准备

所有命令在本目录执行：

```powershell
cd D:\zc\umi\umip
pip install -r requirements.txt

$env:XAC_DXL_PORT = "COM6"       # 改成实际 U2D2/串口号
$env:XAC_DXL_BAUDRATE = "57600"
# 可选：把 TIG 数据放到其他磁盘
$env:XAC_TIG_ROOT = ".\calibration\tig"

python check_pipeline.py --project
```

首次使用或更换夹爪、探针、iPhone 支架、TCP、零位后，都应重新检查并按本文重新标定。标定期间不要让多个程序同时占用串口或 D435i。

## 3. 夹爪宽度标定（与 T_I_G 分开）

宽度标定得到的是指尖总开口，不是 TCP 位姿，不能代替 `T_I_G`。

1. 确认两侧指尖之间没有物体，保持电机 Torque 关闭。
2. 用卡尺测量两侧指尖接触面的总距离，不要测外壳间距。
3. 覆盖接近闭合、中间和最大工作开口，建议采集 5～10 个点；每个点等夹爪完全静止后再读取编码器。
4. 运行：

   ```powershell
   python calibrate_gripper_width.py
   ```

5. 检查 `calibration/gripper_width.json` 已生成，并重新运行 `python check_pipeline.py --project`。

两台电机反向运动时，当前模型使用 `m1 - m2`，不要误用电机平均 tick。更换指尖、改变电机零位或发生碰撞后必须重新标定。

标定后运行 `python monitor_gripper_width.py`，以 1 Hz 实时显示两台电机的编码器角度、角度差和指尖总开口 `width`（mm）。脚本将现有标定转换为角度差插值；只读取编码器，不改变扭矩或工作模式。超出标定范围时显示限幅提示，按 `Ctrl+C` 退出。串口和波特率可通过 `--port COM6 --baudrate 57600` 指定；运行前退出占用同一串口的 driver 或采集程序。

## 4. T_I_G 标定前的机械工装（纯粹的机械结构测试）

这一节只搭建和检查工装，不录制 Stray，也不计算矩阵。目标是让探针尖端在不同姿态下都能重复回到同一个空间固定点。

### 4.1 固定 iPhone、TCP 和孔板

1. 将 iPhone 锁紧在最终支架上。标定开始后不能手持 iPhone，也不能移动或重新调整支架。
2. 将探针夹具安装在最终 TCP 位置。标定期间不能拆下探针后凭感觉重新安装。
3. 优先使用钢针或刚性标定杆；探针应笔直、无毛刺，夹紧处不能晃动。
4. 使用带锥孔的钢制或铝制固定板。用螺钉、压板或台钳把孔板刚性固定在工作台上，不能放在纸板、泡沫或会滑动的薄片上。
5. 锥孔应允许探针从多个姿态接近并自动导向到孔中心。孔不能太浅，避免针尖只停在孔边。

牙签会弯曲，只适合初步检查流程，不能作为正式标定工具。正式结果应使用钢针或其他刚性探针。

### 4.2 Torque 和手动移动

全程保持两个电机 `Torque disabled`。探针实验只允许手动移动夹爪，不要运行自动开合或抓取程序。出现明显反力、卡边、夹具滑动或碰撞风险时，立即停止并断电检查。

### 4.3 重复性预检

在不打开 Stray 录制的情况下做 8～10 次：

1. 将针尖轻轻放入锥孔，不要用力侧推。
2. 保持片刻后把针尖完全抬离孔。
3. 改变夹爪的俯仰、滚转或偏航姿态。
4. 再次轻放入同一个孔。

合格表现是针尖每次都能自然进入孔，几乎不需要横向推力，并且不会卡边、弯曲或因夹具松动改变位置。若不合格，先检查针是否弯曲、孔板是否松动、锥孔是否过浅、探针夹具是否有回差，以及探针是否与滑移导向同轴；不要用软件矩阵掩盖机械问题。

### 4.4 设置滑移零位

旋转标定需要沿至少两条不平行方向采集。推荐：

- `-Z_G`：探针朝下，沿针自身轴线逐级向下滑移，`direction = -1`。
- `+X_G`：探针朝前，沿针自身轴线逐级向前滑移，`direction = +1`。

每个方向都要建立清楚的机械零位。使用直线导轨、套筒或带刻度夹具记录从零位开始的实测累计滑移，推荐 `0/10/20/30 mm`。不要通过“重新夹一次”来估计滑移量。每条轴总行程建议至少 25～30 mm。

## 5. T_I_G 初值和程序自测

首次使用、尚无 `umip/calibration/T_I_G.npy` 时，必须在第 8 节旋转求解前设置初值。当前求解器依赖该文件检测稳定接触段；只有已存在且适用于当前手机、支架和 TCP 的初值时，才可跳过重新设置。初值只用于接触检测和比较，不代表最终标定结果。

以下命令在 `umip` 目录下运行。根据当前设备的实际安装关系填写初值，平移输入毫米，旋转输入 XYZ 欧拉角度数：

```powershell
python calibrate_tig.py --translation-mm x,y,z --rotation-deg rx,ry,rz
python calibrate_tig.py --check
python calibrate_tig_rotation.py --synthetic
```

命令格式示例（数值不代表你的设备外参，请替换为实际估计值）：

```powershell
python calibrate_tig.py --translation-mm 45,179,280 --rotation-deg 0,180,0
```

`--synthetic` 必须显示 `PASS`。正式矩阵文件为 `calibration/T_I_G.npy`，同目录的 JSON 保存来源、单位和坐标约定。写入正式文件前程序会要求输入大写 `YES`。

注意：第 5 节先将初值写入这个路径，文件存在不代表标定已经完成；第 9 节通过独立验证后，再将最终结果写入同一路径。如果已经完成第 6～7 节的数据采集，但缺少此文件，可现在补设初值，然后继续第 8 节，不需要仅因缺少初值而重新采集。初值应对应采集时的安装关系。

### 5.1 什么时候启动 Stray Scanner

本节的命令属于初值设置和程序自测，运行时不需要启动 iPhone 上的 Stray Scanner。`calibrate_tig.py` 写入的只是接触检测用初值，`--check` 检查文件和坐标约定，`calibrate_tig_rotation.py --synthetic` 使用内置模拟数据验证求解器，必须输出 `PASS`。

只有进入正式旋转标定、需要真实的 iPhone 位姿时，才启动 Stray Scanner。推荐顺序如下：

1. 固定 iPhone、夹爪、探针和孔板，确认机械重复性预检通过；标定过程中不要移动支架或拆装探针。
2. 在电脑运行 `python tig_verify_capture.py`，为每个滑移级别创建独立 session，并填写实际轴、方向和滑移量。
3. 在 iPhone 上打开 **Stray Scanner** 开始录制当前 session。让针尖反复轻触同一个固定孔，每个 session 至少 8 次，且姿态要有明显变化；每次接触后完全移出再改变姿态。
4. 停止录制并从 Stray 导出完整文件夹，至少包含 `odometry.csv`（建议同时导出 `imu.csv` 和 `rgb.mp4`），然后放入对应 session 目录。
5. 对另一条不平行轴重复上述采集。推荐分别采集 `-Z_G` 和 `+X_G`，每条轴使用 `0/10/20/30 mm` 四个滑移级别。
6. 数据齐备后运行 `python calibrate_tig_rotation.py`，选择“计算旋转”模式，检查拟合质量并生成候选矩阵。

因此，`--synthetic` 自测阶段不启动 Stray；创建 session 后、开始真实录制时启动 Stray。候选矩阵通过独立 session 验证后，才可写入正式文件。

### 5.2 两类探针操作的区别

实际操作中会看到两类“用探针接触”的步骤，它们属于同一套标定流程：

- **固定点接触**：探针从不同姿态反复进入同一个固定孔，用来检查接触重复性、建立固定点并验证初值。第 4.3 节的 8～10 次预检不打开 Stray；正式 session 中则要打开 Stray 并记录位姿。
- **滑移级别采集**：探针装在导向套筒或直线导向中，在同一轴线上移动到 `0/10/20/30 mm` 等位置。每个位置都要新建一个独立 session，并在该位置从至少 8 个不同姿态接触同一个固定孔，然后再换下一个位置或另一条轴。

这里的距离单位是毫米（mm），不是厘米（cm）。

## 6. 使用 Stray/iPhone 采集 TIG session

### 6.1 在电脑上创建 session

运行：

```powershell
# 默认使用 .\calibration\tig，所有 TIG session 与结果归入统一标定目录
python tig_verify_capture.py
```

交互菜单中选择：

- `1`：新建 `z` 轴 session。
- `2`：新建 `x` 轴 session。
- `3`：新建 `y` 轴 session（只有实际使用第三轴时才需要）。
- `4`：列出已有 session。
- `0`：退出。
根据探针实际滑移方向选择：

  - 选 1（z 轴）：探针沿夹爪坐标系 -Z_G 方向滑移，通常是探针朝下、沿自身轴线向下移动。
  - 选 2（x 轴）：探针沿夹爪坐标系 +X_G 方向滑移，通常是探针朝前、沿自身轴线向前移动。

创建 session 时填写：

- `direction`：沿 `-Z_G` 填 `-1`，沿 `+X_G` 填 `+1`。
- `slide_mm`：该级相对第一组零位的实测累计滑移，单位毫米。
- 探针长度：从 TCP 基准到针尖的实测长度，单位毫米；默认值只作提示。

程序会创建类似下面的目录，并写入 `tig_meta.json`：

```text
.\calibration\tig\
└── data_0908_143000\
    └── tig_meta.json
```

### 6.2 在 iPhone 的 Stray Scanner 中录制

每个滑移级别都新建一个电脑 session；每条 session 对同一个固定孔录制至少 8 次接触。

电脑端创建 session 只是登记元数据，不会自动连接 iPhone。完成 session 设置后，再按下面步骤在 iPhone 上启动 Stray 录制；录制、导出并放置数据后，才进入下一个 session 或运行旋转求解。

1. 确认 iPhone 仍固定在最终支架上，打开 Stray Scanner。
2. 在 Stray 中开始录制并保持画面稳定。不要手持手机绕着夹爪走动。
3. 让探针尖端轻轻进入同一个固定孔，保持约 1.5～2 秒。
4. 把探针完全移出孔，再改变夹爪姿态；俯仰、滚转、偏航都应有明显变化。
5. 重复至少 8 次。姿态越分散越好，但每次接触必须仍是同一个孔。
6. 严禁针尖留在孔内强行扭转夹爪，这会弯针、扩大孔或产生横向误差。
7. 结束 Stray 录制并导出完整文件夹。

每个 session 的导出文件夹应包含至少 `odometry.csv`；建议保留 Stray 同时导出的 `imu.csv` 和 `rgb.mp4`，便于时间同步检查和人工回看。`odometry.csv` 必须含有以下八列：

```text
timestamp,x,y,z,qx,qy,qz,qw
```

### 6.3 放置导出数据

把每个 Stray 导出的完整文件夹放进对应的 TIG session 中，不要只复制一个 CSV，也不要把它放进 `d435i`：

```text
.\calibration\tig\data_0908_143000\
├── tig_meta.json
└── stray_export_xxxxx\
    ├── odometry.csv
    ├── imu.csv
    └── rgb.mp4
```

程序会递归搜索 `odometry.csv`，因此导出文件夹名称可以不同，但一个 session 最好只保留一份候选 odometry 文件。

## 7. 沿两个不平行方向采集

推荐采集以下 8 条 session：

| 轴 | 探针方向 | `direction` | `slide_mm` |
|---|---|---:|---:|
| Z | 沿 `-Z_G` 向下 | -1 | 0, 10, 20, 30 |
| X | 沿 `+X_G` 向前 | +1 | 0, 10, 20, 30 |

每一级都应是独立 session，每条 session 至少 8 次明显不同姿态的接触。不要把多个滑移级别混录在一个 session，也不要把两条轴的数据混在一起。`tig_meta.json` 中的轴、方向和滑移量必须与实际机械操作一致。

## 8. 计算旋转候选

### 接触筛选规则（2026-09-14 更新）

数据筛选与相邻滑移对比允许每条 session 不使用全部接触，但**累计最多舍弃 2 段**，
不是每轮可舍弃 2 段；不能先套用旧规则删 1 段后再额外删 2 段。没有明显离群段可以不删。
本规则使用专用计算入口：

```powershell
conda activate umip
python analyze_tig_adjacent.py
```

- 从原始 odometry 和当前初值重新检测全部稳定候选段，速度阈值 0.02 m/s、
  最短持续 0.6 s、中心窗口 0.5 s、段间隔 0.35 s，不限制为前 8 段。
- “偏差”定义为固定点模型下每个接触平均位姿的三维欧氏残差，单位 mm，
  不等于段内方差。只使用本 session 选择接触，不能为了贴近标注滑移距离或照片尺寸选段。
- 枚举舍弃 0、1、2 段，至少保留 6 段；剔除候选中的每一段残差必须大于
  `max(3 mm, 2 × 保留段残差中位数)`，且保留段 RMS 相比全段拟合至少下降 20%。
  保留姿态的求解秩须为 6、条件数不超过 100。
- 在满足上述条件的方案中取保留段 RMS 最小者；完全并列时优先少删，再按接触编号排序。
  无符合条件的剔除方案则保留全段。以上阈值是本次可复算的诊断规则，不是独立验收标准。
- 保存全部候选段、保留/删除编号、时间范围、剔除前后误差及留一检查。
  相邻组按元数据的实际针长/滑移级别排序，报告 I 系针尖位移向量、距离、实测差值及
  对拟合轴的轴向/横向分量。距离接近标注值仍不代表滑移方向正确。
- 默认输出 `calibration/tig/adjacent_max2_20260914/report.json` 和
  `adjacent_displacements.csv`；可用 `--output` 指定其他报告目录。
  原始数据和正式外参不改动，结果仍需第 9 节独立验证。自动稳定段不保证是真正触孔，
  应结合视频/接触时间记录核对，不能把空中静止当作接触。

**兼容说明：下面的旧交互式旋转、平移验证和 audit_tig_data.py 尚未接入上述专用筛选入口，
仍采用旧的固定删 1 段规则。不能将它们的输出误认为“最多删 2 段”的新结果。**

旧 XAC 求解默认从检测到的接触片段中剔除一次段内方差最大的接触，再拟合。
方差指标为检测窗口内针尖世界坐标的 `Var(X)+Var(Y)+Var(Z)`，单位 mm²；
它不同于接触点之间的拟合残差。并列时剔除时间靠前的一段。
例如要求检测 8 段时，实际保留 7 段；希望保留 8 段请采集并指定 9 段。
至少需要检测 4 段，剔除后仍需满足姿态秩条件。
旋转求解的最近一次筛选记录保存到 session 下 `tig_contact_selection/discarded_contact.json`；
检查缓存同时保留每组方差与被剔除片段。平移验证在输出目录保存 `discarded_contact.json`。
合成自测保持原始样本数。此次代码变化会使旧检查缓存自动失效。

开始前，在 `umip` 目录运行 `python calibrate_tig.py --check`，确认 `calibration/T_I_G.npy` 存在且可读取。若提示缺少本设备外参，先完成第 5 节的初值设置；第 7 节只负责沿不同方向采集，不会生成该矩阵文件。

### 8.1 交互式方式（推荐）

```powershell
python calibrate_tig_rotation.py
```

选择“计算旋转”模式。程序会优先读取每个 session 的 `tig_meta.json`，按轴分组并显示直线拟合质量；先确认 Z 轴质量，再处理 X 轴，最后组合完整旋转矩阵。

### 8.2 命令行方式

如果没有 `tig_meta.json`，可以手动提供 session 和滑移量：

```powershell
python calibrate_tig_rotation.py --compute-rotation `
  --z-sessions "z0,z10,z20,z30" --z-lengths-mm "0,10,20,30" `
  --x-sessions "x0,x10,x20,x30" --x-lengths-mm "0,10,20,30"
```

候选矩阵会保存为某个 session 下的 `tig_rotation_calibration/T_I_G_candidate.json`，不会覆盖正式的 `calibration/T_I_G.npy`。

检查输出时应满足：

- 每条接触 session 的接触残差均值最好不超过 5 mm。
- 直线拟合的横向残差没有明显离群点。
- X、Z 两条轴不能接近平行。
- 旋转矩阵正交，行列式为 `+1`。
- 滑移尺度误差接近 0，说明实测滑移量和 ARKit 位移尺度一致。

如果只有一条轴，程序最多只能给出一个旋转列，完整 `T_I_G_candidate` 会为空；必须补采不平行的第二条轴。

### 8.3 可选的管线自检和 Kabsch

如果想先验证“Stray 位姿 + 接触检测 + 固定点求解”本身是否可靠，可以在纸上画一条直线，在直线上选 3 个以上点。每个点分别录制一个 session，并用卡尺测量相邻点间距，然后运行：

```powershell
python calibrate_tig_rotation.py --line-check `
  --line-sessions "p1,p2,p3" --line-spacings-mm "50,50"
```

这项检查不依赖当前 `T_I_G`，只验证重建的固定点是否共线、间距是否与卡尺一致。若固定点不共线，应先检查 Stray 跟踪、孔板和接触动作。

如果能够用卡尺准确测出至少 3 个非共线的针尖相对 TCP 偏移，可以用 Kabsch/Umeyama 一次性拟合旋转和平移：

```powershell
python calibrate_tig_rotation.py --kabsch `
  --kabsch-sessions "a,b,c,d" `
  --tip-g "0,0,0.03193" `
  --tip-g "0.020,0,0.03193" `
  --tip-g "0,0.015,0.045" `
  --tip-g "0.010,0.005,0.012"
```

`--tip-g` 的单位是米，且必须与 session 顺序一一对应。只有在这些 TCP 坐标中的针尖偏移确实量准时才使用 Kabsch；否则优先采用双轴滑移法，再用固定点 session 精修平移。

## 9. 独立验证和平移精修

候选矩阵不能直接安装。保持手机、支架、夹爪、探针和固定孔不变，重新录制一条没有参与拟合的新 session，至少 8 次接触。

先验证候选旋转：

```powershell
python calibrate_tig_rotation.py --verify-only `
  --demo-dir ".\calibration\tig\verification_session" `
  --tig-json ".\calibration\tig\data_0908_143000\tig_rotation_calibration\T_I_G_candidate.json"
```

建议验收：

- G 系 XY 残差不超过 3 mm。
- G 系轴向残差不超过 5 mm。
- 接触段内部平均残差不超过 5 mm。

若接触残差大于 5 mm，先重录或修复工装，不要靠修改矩阵掩盖问题。轴向残差主要可能来自针长/零位假设，XY 残差主要反映旋转方向；但接触质量差时这些判断都不可信。

通过独立验证后，可用新 session 精修平移：

```powershell
python validate_tig_fixed_point.py --fit-translation `
  --demo-dir ".\calibration\tig\verification_session" `
  --tig-json ".\calibration\tig\data_0908_143000\tig_rotation_calibration\T_I_G_candidate.json"
```

平移精修后，必须再使用另一条全新的 session 复核，而不是重复使用拟合数据。确认无误后写入正式矩阵：

```powershell
python calibrate_tig.py --matrix ".\calibration\tig\verification_session\tig_fixed_point_validation\T_I_G_candidate.json"
python calibrate_tig.py --check
python check_pipeline.py --project
```

如果精修输出文件名不同，以程序实际打印的 JSON 路径为准。安装前必须在提示处输入大写 `YES`。正式文件是：

```text
umip/calibration/T_I_G.npy
umip/calibration/T_I_G.json
```

## 10. 正式任务演示采集（Stray + D435i）

本节采集用于策略训练的完整任务演示，例如“抓起物体 → 搬运 → 放到目标区域 → 松开”。由人手移动整套 XAC 装置并手动开合夹爪；程序记录 D435i RGB-D/IMU 和夹爪编码器，iPhone 上的 Stray 独立记录位姿。此入口不会控制 UR5，也不是机械臂 freedrive 示教。

### 10.1 采集前准备

1. 确认 `calibration/T_I_G.npy`、`T_I_G.json` 和 `gripper_width.json` 是当前装配已验证的采集端标定。完成重播后若在其他目录更新过标定，应核对并同步到本目录；不要用另一只执行夹爪的宽度标定覆盖采集端文件。
2. 固定 iPhone、D435i 和夹爪的相对安装关系，移除仅用于标定的探针。若拆装改变了已标定的安装关系或 TCP 定义，先复核标定。保持训练采集与未来机器人部署的相机安装和视角一致。
3. 连接 D435i、U2D2 与双 XC330，确认实际串口、电机 ID 1/2 和 57600 波特率。关闭其他占用相机或串口的程序。采集程序会关闭电机 Torque，开合由人手完成。
4. 选定一种任务及成功标准，布置起始物体与目标区域，保证关键操作在相机画面内。先采 5～10 条成功演示检查全流程，再扩大数量。
5. 保留本批标定文件副本，并记录任务名称、装配版本及每条演示是否成功。失败演示保留原始记录并标记原因，导出时明确排除。

### 10.2 启动电脑端采集

在 PowerShell 中执行（`COM6` 改成实际串口）：

```powershell
cd D:\zc\umi\umip
$env:XAC_DXL_PORT = "COM6"
$env:XAC_DXL_BAUDRATE = "57600"
python deploy.py check --project
```

检查没有 `[FAIL]` 后启动：

```powershell
python deploy.py collect
```

也可以双击本目录的 `start_collection.cmd`，它会先检查再启动采集；若需要设置环境变量，建议在同一个 PowerShell 窗口运行上述命令。项目检查只检查依赖、文件与数值，不代替真实标定验收；相机首次保存 RGB-D 帧是在按 `s` 后确认。

默认写入 `D:\zc\umi\umip\data`。若已设置 `XAC_DATA_ROOT`，以终端打印的实际目录为准，后续 prepare/check/build 也要使用该路径。

按 `s` 开始采集后，会自动打开 `D435i RGB Preview` 窗口。程序选择当前分辨率下 RGB 和深度共同支持的最高硬件帧率，每个 RGB 帧都刷新预览；数据仍按 10 Hz 保存。画面中的 `LIVE` 表示配置的硬件帧率（实际显示速度受 USB、曝光和电脑负载影响），`SAVE` 表示保存频率，`Frames` 表示本条已保存帧数。只对需保存的帧进行深度对齐，以减少预览开销。窗口可拖动缩放，画面上的文字只用于预览，不写入保存的图像。停止采集时先点击终端恢复焦点，再按 `s`；预览窗口会自动关闭。点击窗口右上角关闭按钮仅关闭本条预览，录制继续，下一条采集会重新打开预览。若只单独录制相机，可使用 `python record_435i.py --output-dir <输出目录> --no-preview` 关闭预览。

### 10.3 录制一条完整任务演示

1. 把装置放到本条演示的起始位置，布置好物体，终端保持焦点；按一次 `s` 新建并开始电脑端 session。
2. **等待终端打印“开始采集：……（现在启动 iPhone 记录并执行一次 shake）”**。该提示表示 D435i 已保存首帧，记下实际 session 编号。编号按原始数据目录内最大数字编号加一生成，已有数据时不一定从 `001` 开始。
3. 在固定于装置上的 iPhone 打开 Stray Scanner，开始一条新的录制。电脑不会自动启动或停止手机录制。
4. 保持整套装置静止约 1 秒，并确认终端出现 `shake_armed` 事件（已完成静止基线、等待 shake）；随后让手机与 D435i 随装置一起做一次明显、短促的旋转摇动。不要单独晃动手机或改变支架安装关系。
5. shake 后回到任务起点并稳定，等待约 1.5 秒，**以终端出现 `valid_data_started` 事件（进入有效数据阶段）为准**，再开始任务。没有进入有效阶段时不要继续演示，停止并重录。
6. 连贯完成接近、抓取闭合、抬起、搬运、放置、松开和撤离，关键步骤不要发生在 shake 或等待阶段。夹爪宽度保持在已标定范围内。
7. 完成后短暂停留，按 `s` 停止电脑端记录，等待“采集结束”提示，再停止手机 Stray 录制。两端都停止后再整理或重置场景。
8. 导出手机本条录制并标注相同 session 编号；下一条再次按 `s`，程序会创建新编号。每条演示都要独立启动 Stray、建立基线和做 shake。全部录完按 `q` 退出。

### 10.4 手机导出与电脑 session 对应

下面以实际编号 `001` 为例；电脑端录制完成后包含：

```text
data/001/
├── d435i/
│   ├── rgb/
│   ├── depth/
│   ├── timestamps.csv
│   ├── imu.csv
│   ├── intrinsics.json
│   └── meta.json
├── gripper_width_log.csv
└── session.json
```

从 Stray 导出本条录制的完整文件夹，保留 `odometry.csv`、`imu.csv`，以及可用的 `rgb.mp4` 等原文件。建议按编号归档，例如 `D:\Stray导出\001\`，避免把手机的第二条录制配给电脑的第一条。

使用以下命令将两份手机 CSV 校验并复制到对应电脑 session 根目录（路径包含空格时也保留双引号）：

```powershell
python deploy.py prepare data\001 "D:\Stray导出\001\odometry.csv" "D:\Stray导出\001\imu.csv"
```

完成后应存在 `data/001/odometry.csv` 与 `data/001/imu.csv`；`data/001/d435i/imu.csv` 是相机 IMU，不能用手机 IMU 覆盖。该命令拒绝覆盖已有文件，已成功 prepare 的 session 不需要再运行。若发现配错，先核对并备份错误文件，再纠正对应关系。

### 10.5 批量采集安排

每条重复“布置 → 电脑开始 → 手机开始 → 静止基线 → shake → 等有效阶段 → 完整任务 → 两端停止 → 手机导出对应编号”。同一种任务保持动作目标一致，逐步变化物体初始位置、朝向和预期光照，不要只重复完全相同的起点。

先用小批数据完成第 11 节导出，并按 `MERGE_AND_TRAINING.md` 跑小规模训练，确认输入、loss 和 checkpoint 正常，再扩大采集。当前只有 18 条演示，模型明显在记忆训练集（训练误差约 3 mm，验证约 9 mm），**增加演示数量（建议 50 条以上、覆盖不同起始位置）是目前提升效果最有效的办法**。额外保留若干完整 session 作为最终测试，不将它们放进训练包；同一 session 切出的片段不能同时用于训练和最终测试。

## 11. 检查、构建和导出 UMI 数据集

### 11.1 每条演示检查与构建

以下命令在 `D:\zc\umi\umip` 执行，以 session `001` 为例。统一使用 `replay` 作为数据组名，可直接匹配现有 XAC 训练配置的默认路径；本批其他编号替换 `001` 即可。

```powershell
python deploy.py check --session data\001
python estimate_sync_by_shake.py --demo-dir data\001
python deploy.py build --stray-root data --dataset-root dataset --dataset-group replay --timestamp 001 --yes
python deploy.py check --episode dataset\replay\001
```

`build_dataset.py` 已包含 shake 时间同步、时间裁剪、iPhone 位姿插值、`T_I_G` 外参变换、夹爪宽度对齐和未来动作生成。动作 7 维为 `[dx,dy,dz,rx,ry,rz,future_gripper_width_m]`，位移/宽度用米，旋转用弧度。

2026-10-20 起，离线构建默认使用 `--start-source shake-pause --sync-method burst-correlation --start-delay-seconds 0.4`：用整段两端 gyro 细化同步，覆盖完整 shake 和回弹后，从两端低运动状态持续 0.4 秒的末端开始保留，恢复张开接近与闭合起始。录制时 `valid_data_started` 提示不再直接决定训练起点；未完成录制时 shake 流程的记录仍拒绝自动打包。每条 `metadata/sync.json` 记录实际选点与同步校正。若要复现旧规则，显式指定 `--start-source realtime-valid --sync-method single-peak`；该模式读取原事件时间，等待参数不改写历史事件。本批原始数据现位于 `data_1020_raw/`，重构时需将 `--stray-root` 指向该目录。当前训练数据为 `dataset_v2/`（2026-10-22 重建，用 `--sync-method burst-task-correlation` 复核同步，说明见 `dataset_v2/README.md`）；旧的 `dataset/` 已删除。10-08 批原始数据保存在 `dataset_old/data_raw/`（另有完整打包 `dataset_old.zip`）。

检查时应没有 `[FAIL]`，RGB/Depth 数量一致，`valid_mask` 是连续有效区间，TCP 运动方向与实际操作一致。更新 `T_I_G` 后，旧 episode 必须重新构建并重新导出 Zarr。

同步估计命令用于查看对齐结果，构建器也会执行同步。自动检查通过后仍需人工回看 RGB、TCP 轨迹、夹爪宽度与有效区间：闭合应对应抓取、松开应对应放置；排查 Stray 跳变、持续丢帧、数据错配、宽度饱和以及有效区间缺少任务开头或结尾。先验收第一条，再按相同方式处理其余 session。

### 11.2 导出通过验收的训练数据

本批演示均已构建后执行：

```powershell
python deploy.py export --episodes-root dataset\replay --output dataset\replay.zarr.zip --manifest dataset\replay_manifest.csv --min-segment-length 32
```

如果本批 `004`、`009` 是失败、同步错误或预留测试 session，显式排除（示例编号必须换成本批真实编号）：

```powershell
python deploy.py export --episodes-root dataset\replay --output dataset\replay.zarr.zip --manifest dataset\replay_manifest.csv --min-segment-length 32 --exclude-source-sessions 004 009
```

导出按有效掩码切分连续片段；32 帧只是最短片段筛选条件，不能判断任务成功。查看 `replay_manifest.csv` 与同名 JSON，确认来源 session、排除名单、有效片段数、帧数及 `frequency_hz_median` / `frequency_hz_range`。默认训练配置要求均匀约 10 Hz，导出器不会自动重采样；混合帧率、长间隔或持续丢帧应先处理或排除。

训练使用导出的 `replay.zarr.zip`，不要直接使用单条 episode 的未来相对 `action.npy`。Zarr 内动作是 7 维绝对位姿/宽度，训练加载器转换为网络使用的相对位姿与 6D 旋转表示。若改用其他数据组名，训练时也要修改 `task.dataset_path`。不同任务建议使用独立原始数据目录、数据组和训练输出目录，避免混入之前的演示。

完成后把 `replay.zarr.zip`、`replay_manifest.csv/.json` 和 `calibration/gripper_width.json` 复制到 PRO6000 的 `~/Desktop/umip/train/<新数据目录>/` 和 `calibration/`，训练时用 `DATASET=<新数据目录>/replay.zarr.zip` 指定，再按 [MERGE_AND_TRAINING.md](MERGE_AND_TRAINING.md#训练pro6000) 训练。保留原始 session、手机导出、标定快照、manifest 与数据包，便于后续定位问题或重新构建。

## 12. 目录和环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `XAC_DATA_ROOT` | `umip/data` | 普通采集原始 session |
| `XAC_DATASET_ROOT` | `umip/dataset` | episode 和 Zarr 中间结果 |
| `XAC_CALIBRATION_ROOT` | `umip/calibration` | 所有正式标定与默认 TIG 工作目录的统一根目录 |
| `XAC_TIG_ROOT` | `<标定根目录>/tig` | TIG 标定 session、候选与报告；通常无需单独设置 |
| `XAC_DXL_PORT` | `COM6` | U2D2 串口 |
| `XAC_DXL_BAUDRATE` | `57600` | Dynamixel 波特率 |

环境变量只对当前 PowerShell 窗口有效；新开窗口后需要重新设置。

## 13. 常见问题

- **TIG 接触残差大**：检查孔板是否移动、探针是否弯曲、夹具是否松动、姿态是否真的离开孔后再改变，以及 Stray 是否有跳变。
- **TIG 横向残差大**：检查滑移是否沿探针自身轴线、是否发生侧移、轴标签和 `direction` 符号是否正确。
- **候选矩阵为空**：通常是只采了一条轴，或两条轴接近平行；补采不平行的第二条轴。
- **验证时轴向残差大、XY 正常**：优先检查针尖长度、TCP 零位和 `probe-offset` 假设，不要立即改旋转矩阵。
- **找不到 `odometry.csv`**：确认 Stray 完整导出文件夹放在正确 session 下，且文件名确实为 `odometry.csv`。
- **`check_pipeline.py --session` 失败**：确认 session 中有 `d435i/timestamps.csv`、`d435i/imu.csv`、`gripper_width_log.csv`、Stray 的 `odometry.csv` 和 `imu.csv`。
- **没有进入有效数据阶段**：重新录制，先静止约 1 秒，再做明显 shake，等待终端提示后再开始任务。
- **串口打不开**：关闭 Dynamixel Wizard、串口助手和其他 Python 进程，检查实际 COM 号；不要使用 Bluetooth 虚拟串口。
- **D435i 启动失败**：退出 RealSense Viewer，换 USB 3.x 端口和线缆，确认没有其他程序占用相机。

安全原则：手动 TIG 实验和正式采集都必须保持 Torque 关闭或按当前程序的安全状态运行；出现反力立即停止。不要让多个程序同时占用串口或 D435i。

## 14. 在 UR5 上重播 XAC episode

UR5 轨迹重播使用本目录的 `replay_xac_ur5.py`，并在连接 UR5 的控制电脑上运行。
它读取 `build_dataset.py` 生成的 episode，以 UR5 当前 TCP 为起点，按照下式重放相对轨迹：

```text
T_B_G(t) = T_B_G_start @ inverse(T_W_G_start) @ T_W_G(t)
```

程序默认只检查数据；必须显式加入 `--enable-motion` 并在终端输入大写 `RUN` 才会发送运动。
首次只控制机械臂，验证方向后再加入 `--enable-gripper`。完整复制清单、依赖、dry-run 和
首次三秒低速命令见 `UR5_REPLAY.md`。不要对 XAC 数据使用官方旧版
`scripts_real/eval_replay_real_robot.py`，该脚本的夹爪类型、源频率和坐标假设均不匹配。
