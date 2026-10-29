"""
T_I_G 旋转矩阵验证与实验标定脚本。

用途：
    1. 用固定点接触实验验证当前 T_I_G 旋转矩阵是否正确（判断误差来自
       旋转、还是来自牙签长度假设）。
    2. 用"牙签沿自身轴线多级滑移 + 固定点接触"实验，直接从数据算出
       T_I_G 的旋转列（不需要测量牙签尖端的绝对偏移）。
    3. 可选：用卡尺量出牙签尖端在夹爪系里的绝对偏移后，用 Kabsch/Umeyama
       一次性整体求解 R_I_G 与 t_I_G。
    4. 合成数据自测（--synthetic）：无需硬件即可验证本脚本全部计算正确。

核心等式
--------
牙签尖端在 iPhone 坐标系 I 中的位置 q_I 只由"接触实验 + ARKit 位姿"决定，
与当前 T_I_G 完全无关（复用 validate_tig_fixed_point.py 的
solve_probe_point_and_fixed_point，解线性系统
    [R_W_I_i, -I] [q_I; c_W]^T = -t_W_I_i
其中 c_W 是纸面固定点的世界坐标）。

而 q_I 与待求外参的关系是：
    q_I = R_I_G @ tip_G + t_I_G          （tip_G = 牙签尖端在夹爪系 G 中的偏移）
所以：
    * 验证：把实验解出的 q_I 与"当前矩阵预测的 R_I_G @ tip_G + t_I_G"对比。
      残差沿 G 系 Z 方向 = 牙签长度/零点假设错；沿 G 系 X/Y 方向 = 旋转错。
    * 算旋转（多滑移直线拟合）：同一装法下把牙签沿自身轴线滑出 3~5 级，
      每级做一次固定点接触 session 得到 q_I(L_i)。因为
          q_I(L) = (R_I_G @ tip_base_G + t_I_G) + L * (R_I_G @ u)
      其中 u 是牙签实际轴线在 G 系中的单位方向（可以是倾斜的、与标称轴
      有夹角），所有点共线，拟合 3D 直线的方向就是 R_I_G @ u。恒定倾斜 /
      恒定 XY 偏差只是整体平移整条直线，不影响直线方向；逐点残差直接量化
      每次滑移的 XY 偏差。滑移方向与标称轴一致时（沿 +Z 推出），
      R_I_G @ u 就是 R_I_G 的 Z 列。
    * Kabsch：给出 3+ 个非共线 tip_G 绝对值后，Umeyama 一次性求 R_I_G+t_I_G。

验证判据（--verify-only）
-------------------------
    |z_dir| > 5 mm          牙签长度/零点假设与实验不符（旋转方向另看 X/Y）
    hypot(x_dir, y_dir) > 3 mm   旋转矩阵可疑
    hypot(x_dir, y_dir) <= 3 mm  旋转方向与实验一致
    接触质量差（接触点内残差 mean > 5 mm）时，以上判据不可信。

采集协议
--------
坐标约定（与 tig_verify_capture.py 采集程序一致）：
    x = 顺着夹爪向前方向；z = 与向前方向垂直、竖直向上；y = 侧向（右手系）。
    z 维牙签沿 -Z 向下伸出并逐组向下推；x 维牙签沿 +X 向前伸出并逐组向前推。
    tig_meta.json 记录每个 session 的滑移方向符号，模式 2 自动据此恢复旋转列，
    因此"沿 -Z 而不是 +Z"不会导致矩阵错误。

采集（推荐用 tig_verify_capture.py，自动建文件夹并写 tig_meta.json）：
    1. z 维度：牙签沿 -Z 向下装，做 4 组（滑移量 0/10/20/30mm 左右），
       每组一条 8 次固定点接触 session。
    2. 换 x 维度：牙签沿 +X 向前装，同样 4 组（0/10/20/30mm）。
    3. 每条 session：碰固定点保持 1.5~2 秒，明显转手腕（俯仰/滚转/偏航），
       共 8 次，姿态越分散越好；固定点可换。

验证模式（1）：
    牙签固定沿 -Z 向下，纸上画一个点，6~8 个差异明显的姿态碰该点各保持 1 秒。

算旋转模式（2）：
    程序按维度分步：先做 z（自动读 meta、展示质量指标），确认后收 x，
    两维齐全后组装完整旋转矩阵。单维也能输出该轴旋转列供判断方向。

自检模式（可选，检验实验管线本身）：
    1. 纸上画一条直线，沿直线触碰 3+ 个点（每个点一次 session，姿态可变）。
    2. 用尺子量相邻点间距。
    3. 运行 --line-check：验证重建的固定点世界坐标共线、间距与尺子一致。
       这一步不需要任何 T_I_G，只检验"ARKit + 接触 + 求解"整条管线是否可靠。

验证流程（推荐顺序）
--------------------
    1. 先跑 --synthetic 确认本脚本计算正确（无需硬件）。
    2. 采集一条验证 session，跑 --verify-only，看当前矩阵判级。
    3. 若旋转可疑，按"算旋转模式"采集，跑 --compute-rotation 得到候选矩阵。
    4. 用一条全新 session 对候选矩阵跑 --verify-only 独立验证。
    5. 通过后，人工决定是否把候选旋转写入 stray_shift_position.py；
       之后可用现有 validate_tig_fixed_point.py --fit-translation 精修平移。

运行方式
--------
交互式菜单（默认，推荐）：
    python calibrate_tig_rotation.py
    启动后在 VSCode 终端里输入数字选择模式（1 验证 / 2 算旋转 / 3 Kabsch /
    4 画线自检 / 5 合成自测 / 0 退出），session 文件夹通过编号选择，不需要
    手动粘贴路径。每个模式执行完会返回菜单，可连续操作（验证 -> 算旋转 ->
    独立验证）。

命令行（高级用法）：
    python calibrate_tig_rotation.py --synthetic
    python calibrate_tig_rotation.py --verify-only \
        --demo-dir "E:\\机器人\\umi\\Stray数据\\data_0801_1810"
    python calibrate_tig_rotation.py --compute-rotation \
        --z-sessions "dir0,dir1,dir2,dir3" \
        --z-lengths-mm "0,8,16,25"
    # 同时提供第二根轴（例如 +X_G）可组合出完整旋转矩阵：
    python calibrate_tig_rotation.py --compute-rotation \
        --z-sessions "z0,z1,z2,z3" --z-lengths-mm "0,8,16,25" \
        --x-sessions "x0,x1,x2" --x-lengths-mm "0,10,20"
    python calibrate_tig_rotation.py --kabsch \
        --kabsch-sessions "a,b,c,d" \
        --tip-g "0,0,0.03193" --tip-g "0.02,0,0.03193" --tip-g "0,0.015,0.045" --tip-g "0.01,0.005,0.012"
    python calibrate_tig_rotation.py --line-check \
        --line-sessions "p1,p2,p3" --line-spacings-mm "50,50"

说明：
    * 本脚本不修改任何正式外参；候选矩阵只写入 JSON，由你人工决定是否采用。
    * 计算出的 t_I_G 使用标称 tip_G=(0,0,0.03193) 作为初值，后续建议用
      validate_tig_fixed_point.py --fit-translation 精修平移。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# 允许从任意工作目录运行：把本文件所在目录加入模块搜索路径，
# 与项目内其他脚本的"同一目录互相 import"约定保持一致。
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import stray_shift_position as stray  # noqa: E402
import validate_tig_fixed_point as tig  # noqa: E402


# ============================================================
# 1. 默认参数
# ============================================================

# 本脚本的输出目录名，创建在使用的 session 文件夹下。
DEFAULT_OUTPUT_DIR_NAME = "tig_rotation_calibration"

# 交互式菜单模式的默认数据根目录：与 tig_verify_capture.py 采集程序一致。
# 采集程序会在 E:\机器人\umi\TIG验证 下创建 data_MMDD_HHMM 文件夹并写入
# tig_meta.json（轴/滑移量/方向/牙签长度），验证程序模式 2 直接读取。
# 启动后按"最新"定位 session，选 session 时直接回车即可使用最新一条。
DEFAULT_INTERACTIVE_ROOT = Path(r"E:\机器人\umi\TIG验证")

# 验证/标定共用的接触检测默认值（与 validate_tig_fixed_point.py 一致的思路）。
DEFAULT_CONTACT_COUNT = 8
DEFAULT_MAX_STABLE_SPEED_M_PER_S = 0.02
DEFAULT_MIN_CONTACT_DURATION_S = 0.6
DEFAULT_MIN_GAP_BETWEEN_CONTACTS_S = 0.35
DEFAULT_CENTER_WINDOW_S = 0.5

# 牙签接触端相对"原夹爪 TCP"的标称偏移，单位 m，夹爪坐标系 G 表达。
DEFAULT_PROBE_OFFSET_G_M = (0.0, 0.0, 0.03193)

# 验证判据阈值（mm）：
#   z_dir：残差沿 G 系 Z 轴的分量，主要反映牙签长度/零点假设。
#   x/y_dir：残差沿 G 系 X/Y 轴的分量，主要反映旋转矩阵。
DEFAULT_VERIFY_Z_THRESHOLD_MM = 5.0
DEFAULT_VERIFY_XY_THRESHOLD_MM = 3.0

# 接触质量阈值：单次接触内残差均值超过该值时，判定"接触/姿态质量差"，
# 验证与标定结论均不可信。
DEFAULT_CONTACT_QUALITY_THRESHOLD_MM = 5.0

# 算旋转模式至少需要的滑移级数。两级可确定方向，三段以上更适合量化离群和垂直残差。
DEFAULT_MIN_SLIDE_SESSIONS = 2

# 每个轴的默认滑移方向符号（与 tig_verify_capture.py 采集程序一致）：
#   direction = +1 表示牙签沿该轴正方向延伸；-1 表示沿负方向延伸。
#   z 维：牙签沿 -Z 向下伸出（direction=-1）；x 维：沿 +X 向前伸出（direction=+1）。
# 拟合方向 d = R·(direction·e_axis)，因此旋转列 = direction × d_unit。
DEFAULT_DIRECTION = {"z": -1, "x": 1, "y": 1}

# Kabsch 整体配准至少需要的非共线尖端点数。
DEFAULT_MIN_KABSCH_TIPS = 3

# 合成自测使用的真值外参（与当前正式外参一致，便于人工对照）。
SYNTHETIC_TRUE_T_I_G = np.array(
    [
        [1.0, 0.0, 0.0, 0.045522922],
        [0.0, -1.0, 0.0, 0.178812911],
        [0.0, 0.0, -1.0, 0.280028061],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# 合成自测的合成误差阈值。
SYNTHETIC_NOISELESS_TOL_M = 1e-6
SYNTHETIC_NOISY_TOL_M = 2e-3
SYNTHETIC_NOISY_ROT_TOL_DEG = 1.0


# ============================================================
# 2. 数据类
# ============================================================


@dataclass(frozen=True)
class QISolveResult:
    """一个 session 的 q_I 求解结果（q_I 与当前 T_I_G 无关）。"""

    demo_dir: Path
    odometry_csv_path: Path
    q_I_m: np.ndarray
    fixed_point_world_m: np.ndarray
    linear_rank: int
    condition_number: float
    contact_errors_mm: np.ndarray
    contact_quality_warning: bool
    timestamps_s: np.ndarray
    T_W_I: np.ndarray
    tip_positions_w_m: np.ndarray
    segments: list
    mean_rotations: np.ndarray
    mean_positions: np.ndarray
    discarded_contact: object = None
    contact_variances_mm2: object = None


@dataclass(frozen=True)
class VerifyResult:
    """验证模式的结果。"""

    demo_dir: Path
    q_I_m: np.ndarray
    predicted_q_I_m: np.ndarray
    residual_m: np.ndarray
    residual_z_dir_mm: float
    residual_x_dir_mm: float
    residual_y_dir_mm: float
    residual_xy_mm: float
    contact_quality_warning: bool
    judgement: str
    condition_number: float


@dataclass(frozen=True)
class SlideFitResult:
    """单根轴的多级滑移直线拟合结果。"""

    session_dirs: tuple[Path, ...]
    lengths_m: np.ndarray
    q_I_points_m: np.ndarray
    direction_I: np.ndarray
    anchor_I_m: np.ndarray
    scale_error: float
    per_point_3d_mm: np.ndarray
    per_point_along_mm: np.ndarray
    per_point_perp_mm: np.ndarray


@dataclass(frozen=True)
class RotationCandidate:
    """算旋转模式输出的候选外参。"""

    T_I_G_candidate: np.ndarray
    columns_I: dict[str, np.ndarray]
    column_source_axes: dict[str, str]
    column_scale_errors: dict[str, float]
    column_tilt_degrees: dict[str, float]
    axis_pair_angle_degrees: float | None
    used_slide_lengths_mm: dict[str, list[float]]
    per_slide_perp_mm: dict[str, np.ndarray]
    translation_source: str


# ============================================================
# 3. session 加载与 q_I 求解
# ============================================================


def _resolve_odometry_csv(demo_dir: Path) -> Path:
    """在 session 文件夹内找到 odometry.csv（复用现有搜索逻辑）。"""

    demo_dir = Path(demo_dir).resolve()
    if not demo_dir.is_dir():
        raise NotADirectoryError(f"session 路径不是文件夹：{demo_dir}")
    paths = tig.build_process_paths_without_output_guard(
        stray_root=stray.DEFAULT_STRAY_ROOT,
        explicit_demo_dir=demo_dir,
    )
    return paths.odometry_csv_path


DROP_HIGHEST_VARIANCE_CONTACT = False


def solve_q_I_from_demo(
    demo_dir: Path,
    detection_T_I_G: np.ndarray,
    detection_probe_offset_G_m: np.ndarray,
    contact_count: int = DEFAULT_CONTACT_COUNT,
    max_stable_speed_m_per_s: float = DEFAULT_MAX_STABLE_SPEED_M_PER_S,
    min_contact_duration_s: float = DEFAULT_MIN_CONTACT_DURATION_S,
    min_gap_between_contacts_s: float = DEFAULT_MIN_GAP_BETWEEN_CONTACTS_S,
    center_window_seconds: float = DEFAULT_CENTER_WINDOW_S,
) -> QISolveResult:
    """
    从一个固定点接触 session 解出牙签尖端在 iPhone 系中的位置 q_I。

    接触段检测需要"探针尖端世界轨迹"来识别静止片段，因此要用一个
    T_I_G 计算尖端位置场；但这里只影响"哪里算静止"，不影响 q_I 本身
    （静止片段内的平均 iPhone 位姿与 T_I_G 无关，q_I 只由 ARKit 位姿决定）。
    默认用当前正式外参做检测矩阵，与验证目标一致，旋转误差小时检测可靠。
    """

    demo_dir = Path(demo_dir).resolve()
    odometry_csv_path = _resolve_odometry_csv(demo_dir)

    odometry_data = stray.load_odometry_csv(odometry_csv_path)
    T_W_I, timestamps_s = stray.build_iphone_pose_sequence(odometry_data)

    probe_offset = np.asarray(detection_probe_offset_G_m, dtype=np.float64)
    probe_point_I = (
        detection_T_I_G[:3, :3] @ probe_offset + detection_T_I_G[:3, 3]
    )
    tip_positions_w_m = (
        T_W_I[:, :3, 3]
        + np.einsum("nij,j->ni", T_W_I[:, :3, :3], probe_point_I)
    )

    segments, _ = tig.detect_contact_segments(
        timestamps_s=timestamps_s,
        positions_m=tip_positions_w_m,
        contact_count=contact_count,
        max_stable_speed_m_per_s=max_stable_speed_m_per_s,
        min_contact_duration_s=min_contact_duration_s,
        min_gap_between_contacts_s=min_gap_between_contacts_s,
        center_window_seconds=center_window_seconds,
    )

    discarded_contact = None
    contact_variances_mm2 = None
    if DROP_HIGHEST_VARIANCE_CONTACT:
        original_segments = segments
        segments, discarded_contact, variances = tig.discard_highest_variance_contact(
            segments, tip_positions_w_m)
        contact_variances_mm2 = {
            segment.contact_index: variance for segment, variance in zip(original_segments, variances)
        }
        print(f"Discard contact #{discarded_contact.contact_index}: "
              f"variance={contact_variances_mm2[discarded_contact.contact_index]:.6f} mm^2; "
              f"retain {len(segments)} contacts")
        selection_dir = demo_dir / 'tig_contact_selection'
        selection_dir.mkdir(exist_ok=True)
        (selection_dir / 'discarded_contact.json').write_text(json.dumps({
            'metric': 'sum_xyz_population_variance_mm2',
            'discarded_contact_index': discarded_contact.contact_index,
            'start_index': discarded_contact.start_index,
            'end_index_exclusive': discarded_contact.end_index_exclusive,
            'variances_mm2': contact_variances_mm2,
            'retained_contact_indices': [s.contact_index for s in segments],
            'detection_T_I_G': detection_T_I_G.tolist(),
            'detection_probe_offset_G_m': probe_offset.tolist(),
        }, indent=2), encoding='utf-8')

    mean_rotations, mean_positions = tig.calculate_contact_mean_iphone_poses(
        T_W_I=T_W_I,
        segments=segments,
    )
    q_I_m, fixed_point_world_m, rank, condition_number = (
        tig.solve_probe_point_and_fixed_point(mean_rotations, mean_positions)
    )

    predicted_world_points = (
        mean_positions
        + np.einsum("nij,j->ni", mean_rotations, q_I_m)
    )
    contact_errors_mm = np.linalg.norm(
        predicted_world_points - fixed_point_world_m,
        axis=1,
    ) * 1000.0

    return QISolveResult(
        demo_dir=demo_dir,
        odometry_csv_path=odometry_csv_path,
        q_I_m=q_I_m,
        fixed_point_world_m=fixed_point_world_m,
        linear_rank=int(rank),
        condition_number=float(condition_number),
        contact_errors_mm=contact_errors_mm,
        contact_quality_warning=bool(np.mean(contact_errors_mm) > DEFAULT_CONTACT_QUALITY_THRESHOLD_MM),
        timestamps_s=timestamps_s,
        T_W_I=T_W_I,
        tip_positions_w_m=tip_positions_w_m,
        segments=segments,
        mean_rotations=mean_rotations,
        mean_positions=mean_positions,
        discarded_contact=discarded_contact,
        contact_variances_mm2=contact_variances_mm2,
    )


# ============================================================
# 4. 验证模式：残差沿 G 轴分解 + 判级
# ============================================================


def run_verify(
    demo_dir: Path,
    T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
    contact_count: int = DEFAULT_CONTACT_COUNT,
) -> tuple[VerifyResult, QISolveResult]:
    """
    验证 T_I_G 旋转：实验解出 q_I，与矩阵预测的 q_I 对比。

    残差沿 G 系三轴分解：
        residual·R_I_G[:, k] = 残差在 G 系第 k 轴上的分量。
    沿 Z = 牙签长度/零点假设错；沿 X/Y = 旋转错。

    返回 (VerifyResult, QISolveResult)：第二个返回值保留接触段等中间结果，
    供调用方保存 CSV，避免重复求解。
    """

    solve_result = solve_q_I_from_demo(
        demo_dir=demo_dir,
        detection_T_I_G=T_I_G,
        detection_probe_offset_G_m=probe_offset_G_m,
        contact_count=contact_count,
    )

    rotation_I_G = T_I_G[:3, :3]
    probe_offset = np.asarray(probe_offset_G_m, dtype=np.float64)
    predicted_q_I = rotation_I_G @ probe_offset + T_I_G[:3, 3]

    residual = solve_result.q_I_m - predicted_q_I
    residual_z_dir_mm = float(residual @ rotation_I_G[:, 2]) * 1000.0
    residual_x_dir_mm = float(residual @ rotation_I_G[:, 0]) * 1000.0
    residual_y_dir_mm = float(residual @ rotation_I_G[:, 1]) * 1000.0
    residual_xy_mm = float(np.hypot(residual_x_dir_mm, residual_y_dir_mm))

    if solve_result.contact_quality_warning:
        judgement = (
            "接触/姿态质量差：接触点内残差超过阈值，本次验证结论不可信。"
            "请检查是否真的点准了同一个点、姿态变化是否足够大。"
        )
    elif abs(residual_z_dir_mm) > DEFAULT_VERIFY_Z_THRESHOLD_MM:
        judgement = (
            f"牙签长度/零点假设与实验不符：残差沿 Z_G 为 "
            f"{residual_z_dir_mm:.1f} mm（阈值 {DEFAULT_VERIFY_Z_THRESHOLD_MM:.0f} mm）。"
            f"旋转是否可信请看 X/Y 平面分量：{residual_xy_mm:.1f} mm。"
        )
    elif residual_xy_mm > DEFAULT_VERIFY_XY_THRESHOLD_MM:
        judgement = (
            f"旋转矩阵可疑：残差沿 G 系 X/Y 平面为 {residual_xy_mm:.1f} mm"
            f"（阈值 {DEFAULT_VERIFY_XY_THRESHOLD_MM:.0f} mm），主要分量 "
            f"[x_dir={residual_x_dir_mm:.1f}, y_dir={residual_y_dir_mm:.1f}] mm。"
            "建议用 --compute-rotation 从实验重算旋转。"
        )
    else:
        judgement = (
            f"旋转方向与实验一致：残差沿 G 系 X/Y 平面为 {residual_xy_mm:.1f} mm"
            f"（阈值 {DEFAULT_VERIFY_XY_THRESHOLD_MM:.0f} mm），"
            f"沿 Z_G 为 {residual_z_dir_mm:.1f} mm。"
        )

    return (
        VerifyResult(
            demo_dir=demo_dir,
            q_I_m=solve_result.q_I_m,
            predicted_q_I_m=predicted_q_I,
            residual_m=residual,
            residual_z_dir_mm=residual_z_dir_mm,
            residual_x_dir_mm=residual_x_dir_mm,
            residual_y_dir_mm=residual_y_dir_mm,
            residual_xy_mm=residual_xy_mm,
            contact_quality_warning=solve_result.contact_quality_warning,
            judgement=judgement,
            condition_number=solve_result.condition_number,
        ),
        solve_result,
    )


def save_verify_outputs(
    output_dir: Path,
    verify_result: VerifyResult,
    T_I_G_used: np.ndarray,
    tig_source_description: str,
    probe_offset_G_m: np.ndarray,
    solve_result: QISolveResult,
) -> Path:
    """保存验证摘要与接触点数据。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_results, _ = tig.calculate_contact_point_results(
        segments=solve_result.segments,
        positions_m=solve_result.tip_positions_w_m,
    )
    tig.save_contact_points_csv(output_dir, contact_results)

    residual_mm = verify_result.residual_m * 1000.0
    lines = [
        "T_I_G rotation verification summary",
        "",
        f"demo_dir: {verify_result.demo_dir}",
        f"odometry_csv_path: {solve_result.odometry_csv_path}",
        f"output_dir: {output_dir}",
        "",
        f"T_I_G_source: {tig_source_description}",
        "T_I_G_used:",
        *[
            "  " + np.array2string(row, precision=6, separator=", ")
            for row in T_I_G_used
        ],
        "",
        "toothpick_probe:",
        f"  offset_G_m: {np.asarray(probe_offset_G_m, dtype=np.float64).tolist()}",
        "",
        "experimental_q_I_m:",
        f"  {np.array2string(verify_result.q_I_m, precision=6, separator=', ')}",
        "predicted_q_I_m (R_I_G @ tip_G + t_I_G):",
        f"  {np.array2string(verify_result.predicted_q_I_m, precision=6, separator=', ')}",
        "residual_I_mm:",
        f"  {np.array2string(residual_mm, precision=3, separator=', ')}",
        "",
        "residual decomposition along G axes (mm):",
        f"  z_dir: {verify_result.residual_z_dir_mm:.3f}   (牙签长度/零点假设方向)",
        f"  x_dir: {verify_result.residual_x_dir_mm:.3f}",
        f"  y_dir: {verify_result.residual_y_dir_mm:.3f}",
        f"  xy_planar: {verify_result.residual_xy_mm:.3f}   (旋转矩阵方向)",
        "",
        f"linear_system_condition_number: {verify_result.condition_number:.3g}",
        f"contact_quality_warning: {verify_result.contact_quality_warning}",
        f"contact_errors_mm mean: {float(np.mean(solve_result.contact_errors_mm)):.3f}",
        f"contact_errors_mm max: {float(np.max(solve_result.contact_errors_mm)):.3f}",
        "",
        f"judgement: {verify_result.judgement}",
        "",
    ]
    summary_path = output_dir / "verify_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


# ============================================================
# 5. 算旋转模式：多滑移直线拟合
# ============================================================


def fit_line_direction(
    points_m: np.ndarray,
    lengths_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    对"滑移量 L -> q_I(L)"拟合 3D 直线。

    模型：q_I(L) = anchor + L * direction。对每个坐标分量做一元线性回归：
        direction_k = cov(L, q_k) / var(L)，anchor = mean(q) - mean(L) * direction。

    返回 (direction, anchor, per_point_3d_mm, per_point_along_mm,
          per_point_perp_mm, scale_error)：
        direction     直线方向，单位 m/m（滑移 1 m 尖端移动的向量）。
        per_point_3d_mm   每点三维残差（含沿轴与垂直方向）。
        per_point_along_mm 残差沿直线方向的分量（主要来自尺子读数误差/非轴向滑移）。
        per_point_perp_mm  残差垂直直线方向的分量（主要来自滑移时 XY 偏差）。
        scale_error    norm(direction) - 1，衡量"滑移量读数 vs 实际移动"的尺度一致性。
    """

    points = np.asarray(points_m, dtype=np.float64)
    lengths = np.asarray(lengths_m, dtype=np.float64)
    if points.shape != (len(lengths), 3):
        raise ValueError("points_m 与 lengths_m 的行数不一致。")
    if len(lengths) < 2:
        raise ValueError("至少需要 2 个滑移点才能拟合直线。")

    length_mean = float(np.mean(lengths))
    length_variance = float(np.sum((lengths - length_mean) ** 2))
    if length_variance <= 0.0:
        raise ValueError("滑移量没有变化，无法拟合直线。请检查 --delta-lengths-mm。")

    direction = np.zeros(3, dtype=np.float64)
    for axis_index in range(3):
        direction[axis_index] = float(
            np.sum((lengths - length_mean) * (points[:, axis_index] - points[:, axis_index].mean()))
            / length_variance
        )
    anchor = points.mean(axis=0) - length_mean * direction

    model_points = anchor + np.outer(lengths, direction)
    residuals = points - model_points
    direction_unit = direction / (np.linalg.norm(direction) + 1e-12)
    along = residuals @ direction_unit
    perp = residuals - np.outer(along, direction_unit)
    scale_error = float(np.linalg.norm(direction) - 1.0)

    return (
        direction,
        anchor,
        np.linalg.norm(residuals, axis=1) * 1000.0,
        along * 1000.0,
        np.linalg.norm(perp, axis=1) * 1000.0,
        scale_error,
    )


def fit_slide_axis(
    session_dirs: list[Path],
    lengths_mm: list[float],
    detection_T_I_G: np.ndarray,
    detection_probe_offset_G_m: np.ndarray,
    contact_count: int,
) -> SlideFitResult:
    """对一组滑移 session 解 q_I 并拟合直线。"""

    if len(session_dirs) != len(lengths_mm):
        raise ValueError(
            f"滑移 session 数量（{len(session_dirs)}）与滑移量数量"
            f"（{len(lengths_mm)}）不一致。"
        )
    if len(session_dirs) < DEFAULT_MIN_SLIDE_SESSIONS:
        raise ValueError(
            f"至少需要 {DEFAULT_MIN_SLIDE_SESSIONS} 级滑移，当前只有"
            f"{len(session_dirs)} 级。"
        )

    lengths_m = np.asarray(lengths_mm, dtype=np.float64) / 1000.0
    q_I_points = np.empty((len(session_dirs), 3), dtype=np.float64)

    for index, session_dir in enumerate(session_dirs):
        solve_result = solve_q_I_from_demo(
            demo_dir=session_dir,
            detection_T_I_G=detection_T_I_G,
            detection_probe_offset_G_m=detection_probe_offset_G_m,
            contact_count=contact_count,
        )
        if solve_result.contact_quality_warning:
            print(
                f"  警告：{Path(session_dir).name} 接触质量差"
                f"（mean {np.mean(solve_result.contact_errors_mm):.1f} mm），"
                "建议检查接触是否点准。"
            )
        q_I_points[index] = solve_result.q_I_m
        print(
            f"  滑移 {index}（L={lengths_mm[index]:.0f} mm）："
            f"q_I={np.array2string(solve_result.q_I_m, precision=4, separator=', ')} m"
        )

    (
        direction,
        anchor,
        per_point_3d_mm,
        per_point_along_mm,
        per_point_perp_mm,
        scale_error,
    ) = fit_line_direction(q_I_points, lengths_m)

    return SlideFitResult(
        session_dirs=tuple(Path(d) for d in session_dirs),
        lengths_m=lengths_m,
        q_I_points_m=q_I_points,
        direction_I=direction,
        anchor_I_m=anchor,
        scale_error=scale_error,
        per_point_3d_mm=per_point_3d_mm,
        per_point_along_mm=per_point_along_mm,
        per_point_perp_mm=per_point_perp_mm,
    )


def assemble_rotation_from_axis_columns(columns_I: dict[str, np.ndarray]) -> np.ndarray:
    """
    把若干实测列向量按轴槽位组装成 SO(3) 旋转矩阵。

    columns_I 形如 {"z": 方向, "x": 方向}，键是标称轴名，值是拟合出的
    列向量（等于 R_I_G @ u）。槽位顺序按 R 的标准列顺序 x/y/z：
        轴 x -> 第 0 列，轴 y -> 第 1 列，轴 z -> 第 2 列。
    缺失的槽位用两两叉积按右手系补齐：
        e_x = e_y × e_z，e_y = e_z × e_x，e_z = e_x × e_y
    若提供的列都精确等于 R 的对应列，组装结果精确等于 R；
    测量误差带来的轻微非正交最后由 SVD 投影修回 SO(3)。
    """

    slot_index = {"x": 0, "y": 1, "z": 2}
    columns: dict[int, np.ndarray] = {}
    for axis_name, column in columns_I.items():
        if axis_name not in slot_index:
            raise ValueError(f"未知轴名：{axis_name}")
        norm = np.linalg.norm(column)
        if norm <= 1e-12:
            raise ValueError(f"轴 {axis_name} 的拟合列向量模长为零。")
        columns[slot_index[axis_name]] = column / norm

    for _ in range(3):
        for missing in (0, 1, 2):
            if missing in columns:
                continue
            if missing == 0 and 1 in columns and 2 in columns:
                columns[0] = np.cross(columns[1], columns[2])
            elif missing == 1 and 2 in columns and 0 in columns:
                columns[1] = np.cross(columns[2], columns[0])
            elif missing == 2 and 0 in columns and 1 in columns:
                columns[2] = np.cross(columns[0], columns[1])

    if not all(slot in columns for slot in (0, 1, 2)):
        raise ValueError(
            "提供的轴数据不足以组装完整旋转矩阵：需要至少两根不同轴，"
            "且它们能按右手系补齐第三列。"
        )

    matrix = np.column_stack([columns[0], columns[1], columns[2]])
    left, _, right_t = np.linalg.svd(matrix)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def rotation_angle_degrees(rotation_1: np.ndarray, rotation_2: np.ndarray) -> float:
    """两个旋转矩阵之间的夹角（度）。"""

    trace = float(np.trace(rotation_1.T @ rotation_2))
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def run_compute_rotation_with_two_axes(
    axis_sessions: dict[str, list[str]],
    axis_lengths_mm: dict[str, list[float]],
    current_T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
    contact_count: int,
    axis_directions: dict[str, int] | None = None,
    translation_tip_G_m: np.ndarray | None = None,
) -> RotationCandidate:
    """
    双轴版本：每组实验数据（滑移 session 列表 + 滑移量）对应一根标称轴。

    axis_sessions 形如 {"z": [dir0, dir1, ...], "x": [...]}，
    axis_lengths_mm 形如 {"z": [0, 8, 16, ...], "x": [...]}。
    每个轴独立做多滑移直线拟合得到列，再用两列组合成完整 SO(3)。

    axis_directions 是每个轴的滑移方向符号（+1/-1）。牙签沿 -e_axis 延伸时
    拟合方向 d = R·(-e_axis)，旋转列 = -d 才是 R·e_axis；因此列 = direction × d_unit。
    缺省时按默认约定（z 沿 -Z、x/y 沿 +）。

    translation_tip_G_m 是 z 维牙签尖端在夹爪系中的偏移，用于 t_I_G 占位
    初值（anchor - R @ tip_G）；缺省用标称 (0,0,0.03193)。平移最终由
    validate_tig_fixed_point.py --fit-translation 精修，这里只做占位。
    """

    for axis_name in ("z", "x", "y"):
        if axis_name not in axis_sessions:
            continue
        if axis_name not in axis_lengths_mm:
            raise ValueError(f"缺少 --{axis_name}-lengths-mm 参数。")
        if len(axis_sessions[axis_name]) != len(axis_lengths_mm[axis_name]):
            raise ValueError(
                f"轴 {axis_name} 的 session 数与滑移量数不一致。"
            )

    nominal_axis_map = {"z": 2, "x": 0, "y": 1}
    if axis_directions is None:
        axis_directions = {
            axis_name: DEFAULT_DIRECTION[axis_name]
            for axis_name in axis_sessions
        }
    if translation_tip_G_m is None:
        translation_tip_G_m = np.asarray(DEFAULT_PROBE_OFFSET_G_M, dtype=np.float64)
    else:
        translation_tip_G_m = np.asarray(translation_tip_G_m, dtype=np.float64)

    columns_I: dict[str, np.ndarray] = {}
    column_scale_errors: dict[str, float] = {}
    column_tilt_degrees: dict[str, float] = {}
    per_slide_perp_mm: dict[str, np.ndarray] = {}
    used_slide_lengths_mm: dict[str, list[float]] = {}
    anchors_I: dict[str, np.ndarray] = {}

    for axis_name in axis_sessions:
        fit_result = fit_slide_axis(
            session_dirs=[Path(path) for path in axis_sessions[axis_name]],
            lengths_mm=axis_lengths_mm[axis_name],
            detection_T_I_G=current_T_I_G,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        direction = axis_directions.get(axis_name, DEFAULT_DIRECTION.get(axis_name, 1))
        direction_unit = fit_result.direction_I / (
            np.linalg.norm(fit_result.direction_I) + 1e-12
        )
        # 旋转列 = direction × d_unit：拟合方向是 R·(direction·e_axis)，
        # 乘回 direction 才得到 R·e_axis（即旋转矩阵的对应列）。
        columns_I[axis_name] = direction * direction_unit
        column_scale_errors[axis_name] = fit_result.scale_error
        per_slide_perp_mm[axis_name] = fit_result.per_point_perp_mm
        used_slide_lengths_mm[axis_name] = axis_lengths_mm[axis_name]
        anchors_I[axis_name] = fit_result.anchor_I_m

        nominal_unit = np.zeros(3)
        nominal_unit[nominal_axis_map[axis_name]] = 1.0
        # 倾斜量用 |dot| 计算：拟合方向与标称轴方向相反时（例如 R 把 e_z 映成
        # -e_z）不算"装反"，只算与标称轴的偏角；只有真正的安装倾斜才增加该角度。
        tilt_degrees = float(
            np.degrees(
                np.arccos(np.clip(abs(direction_unit @ nominal_unit), 0.0, 1.0))
            )
        )
        column_tilt_degrees[axis_name] = tilt_degrees

        print(f"\n轴 {axis_name} 滑移直线拟合：")
        print(
            f"  方向（I 系，单位向量）：{np.array2string(direction_unit, precision=5, separator=', ')}"
        )
        print(f"  尺度误差：{fit_result.scale_error:.5f}")
        print(
            f"  与标称轴夹角（牙签倾斜量）：{tilt_degrees:.2f} deg；"
            "perp 残差最大 "
            f"{float(np.max(fit_result.per_point_perp_mm)):.3f} mm"
        )
        for index in range(len(fit_result.lengths_m)):
            print(
                f"    L={axis_lengths_mm[axis_name][index]:.0f} mm: "
                f"perp={fit_result.per_point_perp_mm[index]:.3f} mm, "
                f"along={fit_result.per_point_along_mm[index]:.3f} mm"
            )

    axes_fitted = list(columns_I.keys())
    if len(axes_fitted) < 1:
        raise ValueError("没有提供任何滑移轴数据。")
    # 单轴数据允许继续：只输出该轴旋转方向列，T_I_G_candidate 为 null
    #（JSON 的 notes 与 CLI 提示会说明需要第二根轴才能组合完整矩阵）。

    # 任意两列按轴槽位组装成 SO(3)（缺失列用右手系叉积补齐）。
    axis_pair_angle = None
    if len(axes_fitted) >= 2:
        first_axis, second_axis = axes_fitted[0], axes_fitted[1]
        dot_product = float(np.clip(columns_I[first_axis] @ columns_I[second_axis], -1.0, 1.0))
        axis_pair_angle = float(np.degrees(np.arccos(dot_product)))
        print(
            f"\n两轴列夹角：{axis_pair_angle:.2f} deg"
            "（应接近 90°，偏差来自牙签倾斜或实验误差）"
        )
        candidate_rotation = assemble_rotation_from_axis_columns(columns_I)
    else:
        candidate_rotation = None

    candidate_matrix = None
    if candidate_rotation is not None:
        candidate_matrix = np.eye(4, dtype=np.float64)
        candidate_matrix[:3, :3] = candidate_rotation
        probe_offset = np.asarray(probe_offset_G_m, dtype=np.float64)
        anchor_I = anchors_I[first_axis]
        candidate_matrix[:3, 3] = anchor_I - candidate_rotation @ translation_tip_G_m
        print(f"\n候选旋转矩阵（与当前矩阵夹角）：")
        current_rotation = current_T_I_G[:3, :3]
        print(
            f"  {rotation_angle_degrees(candidate_rotation, current_rotation):.2f} deg"
        )
        print("  候选 T_I_G：")
        for row in candidate_matrix:
            print("   " + np.array2string(row, precision=6, separator=", "))

    return RotationCandidate(
        T_I_G_candidate=candidate_matrix,
        columns_I=columns_I,
        column_source_axes={axis: axis for axis in axes_fitted},
        column_scale_errors=column_scale_errors,
        column_tilt_degrees=column_tilt_degrees,
        axis_pair_angle_degrees=axis_pair_angle,
        used_slide_lengths_mm=used_slide_lengths_mm,
        per_slide_perp_mm=per_slide_perp_mm,
        translation_source=(
            "t_I_G 初值 = anchor - R_I_G @ tip_G_nominal，"
            "仅用于占位；请用 validate_tig_fixed_point.py --fit-translation 精修平移。"
        ),
    )


def save_rotation_candidate_json(
    output_dir: Path,
    candidate: RotationCandidate,
    source_T_I_G: np.ndarray,
    source_description: str,
    probe_offset_G_m: np.ndarray,
) -> Path:
    """保存候选外参 JSON（与 validate_tig_fixed_point.py 的格式兼容）。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "T_I_G_candidate.json"

    column_data = {}
    for axis_name, column in candidate.columns_I.items():
        column_data[axis_name] = {
            "column_I": column.tolist(),
            "scale_error": candidate.column_scale_errors[axis_name],
            "tilt_from_nominal_deg": candidate.column_tilt_degrees[axis_name],
            "per_slide_perp_mm": (
                candidate.per_slide_perp_mm[axis_name].round(4).tolist()
            ),
            "slide_lengths_mm": candidate.used_slide_lengths_mm[axis_name],
        }

    candidate_data = {
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "matrix_semantics": "T_W_G = T_W_I @ T_I_G; probe point = TCP + R_I_G @ probe_offset_G_m",
        "source_T_I_G": source_description,
        "probe_offset_G_m": np.asarray(probe_offset_G_m, dtype=np.float64).tolist(),
        "rotation_calibration_method": "multi_slide_line_fit",
        "columns": column_data,
        "axis_pair_angle_deg": candidate.axis_pair_angle_degrees,
        "T_I_G_current": np.asarray(source_T_I_G, dtype=np.float64).tolist(),
        "T_I_G_candidate": (
            np.asarray(candidate.T_I_G_candidate, dtype=np.float64).tolist()
            if candidate.T_I_G_candidate is not None
            else None
        ),
        "translation_source": candidate.translation_source,
        "requires_independent_validation": True,
        "notes": [
            "旋转列来自多滑移直线拟合；若只有单轴数据，T_I_G_candidate 为 null，"
            "仅当提供第二轴后才能组合成完整矩阵。",
            "t_I_G 为占位初值，请用 validate_tig_fixed_point.py --fit-translation 精修。",
            "用全新 session 运行 --verify-only --tig-json 本文件路径做独立验证。",
        ],
    }
    candidate_path.write_text(
        json.dumps(candidate_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidate_path


# ============================================================
# 6. Kabsch / Umeyama 整体配准
# ============================================================


def umeyama_rigid_fit(points_p: np.ndarray, points_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Umeyama 刚体配准：q = R @ p + t。

    points_p 是夹爪系 G 中的已知尖端偏移（tip_G），points_q 是实验解出的
    iPhone 系位置（q_I）。返回 (R, t, per_point_residual_m)。

    注意两处容易出错的地方：
    1. 对 H = P'ᵀ Q'（P、Q 为 [N,3]）的 SVD H = U S Vᵀ，正确旋转是
       R = V Uᵀ（= right_tᵀ @ leftᵀ），而不是 U Vᵀ（两者互为转置；
       若真值 R 不对称，用错方向会得到 Rᵀ 类错误解）。
    2. 3 个点（协方差矩阵奇异，rank 2）时，SVD 的零奇异值方向 U_3/V_3
       配对是任意的，R = V Uᵀ 可能给出"部分行符号错误"的反射解；因此
       生成两个候选解（V Uᵀ 与翻转 U_3 后的 V U'ᵀ），选择逐点残差更小的
       那个——两个候选之一必然是真解，且带噪声数据也会选到残差更小的正确解。
    """

    points_p = np.asarray(points_p, dtype=np.float64)
    points_q = np.asarray(points_q, dtype=np.float64)
    if points_p.shape != points_q.shape or points_p.shape[1] != 3:
        raise ValueError("points_p 与 points_q 必须都是 [N, 3]。")
    if len(points_p) < DEFAULT_MIN_KABSCH_TIPS:
        raise ValueError(
            f"Kabsch 至少需要 {DEFAULT_MIN_KABSCH_TIPS} 个非共线尖端点，"
            f"当前只有 {len(points_p)} 个。"
        )

    centroid_p = points_p.mean(axis=0)
    centroid_q = points_q.mean(axis=0)
    covariance = (points_p - centroid_p).T @ (points_q - centroid_q)
    left, _, right_t = np.linalg.svd(covariance)

    # R = V U^T（right_t = V^T，所以 V = right_t.T）。
    rotation_candidates = [right_t.T @ left.T]
    left_flipped = left.copy()
    left_flipped[:, -1] *= -1.0
    rotation_candidates.append(right_t.T @ left_flipped.T)

    best_rms = float("inf")
    best_fit = None
    for rotation in rotation_candidates:
        translation = centroid_q - rotation @ centroid_p
        residuals = points_q - (points_p @ rotation.T + translation)
        rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
        if rms < best_rms:
            best_rms = rms
            best_fit = (rotation, translation, residuals)

    if best_fit is None:
        raise RuntimeError("Umeyama 求解失败。")
    return best_fit


def run_kabsch(
    kabsch_sessions: list[str],
    tip_g_list: list[list[float]],
    current_T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
    contact_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    多 session + 已知绝对尖端偏移 -> Umeyama 求解 R_I_G 与 t_I_G。

    返回 (R_I_G, t_I_G, q_I_points, per_point_residual_m,
          rotation_diff_deg, fit_rms_mm)。
    """

    if len(kabsch_sessions) != len(tip_g_list):
        raise ValueError(
            f"--kabsch-sessions（{len(kabsch_sessions)}）与 --tip-g"
            f"（{len(tip_g_list)}）数量不一致。"
        )
    if len(kabsch_sessions) < DEFAULT_MIN_KABSCH_TIPS:
        raise ValueError(
            f"Kabsch 至少需要 {DEFAULT_MIN_KABSCH_TIPS} 个非共线尖端点。"
        )

    tip_g_array = np.asarray(tip_g_list, dtype=np.float64)
    q_I_points = np.empty((len(kabsch_sessions), 3), dtype=np.float64)
    for index, session_dir in enumerate(kabsch_sessions):
        solve_result = solve_q_I_from_demo(
            demo_dir=session_dir,
            detection_T_I_G=current_T_I_G,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        q_I_points[index] = solve_result.q_I_m
        print(
            f"  session {index}：q_I={np.array2string(solve_result.q_I_m, precision=4, separator=', ')} m"
        )

    # 构型健康度检查：尖端点在夹爪系中的散布必须足够非退化。
    # 3 个点（协方差 rank 2）或接近共面/共线的配置在噪声下会给出不稳定解，
    # 因此建议至少 4 个非共面点。条件数太大时仅警告，不阻止计算。
    tip_centroid = tip_g_array.mean(axis=0)
    tip_covariance = (tip_g_array - tip_centroid).T @ (tip_g_array - tip_centroid)
    singular_values = np.linalg.svd(tip_covariance, compute_uv=False)
    tip_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 1e-12
        else float("inf")
    )
    if tip_condition > 50.0:
        print(
            f"  警告：尖端点构型接近退化（条件数 {tip_condition:.1f}），"
            "标定结果对噪声敏感。建议使用 4 个以上明显非共面的尖端位置。"
        )

    rotation, translation, residuals = umeyama_rigid_fit(tip_g_array, q_I_points)
    rotation_diff_deg = rotation_angle_degrees(rotation, current_T_I_G[:3, :3])
    fit_rms_mm = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))) * 1000.0)

    return rotation, translation, q_I_points, residuals, rotation_diff_deg, fit_rms_mm


# ============================================================
# 7. 自检模式：画线 + 尺子间距
# ============================================================


def run_line_check(
    line_sessions: list[str],
    line_spacings_mm: list[float],
    current_T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
    contact_count: int,
) -> dict:
    """
    沿纸上直线触碰 3+ 个点（每个点一个 session），检验重建的固定点世界坐标
    是否共线、相邻间距是否与尺子读数一致。此模式不依赖 T_I_G，只检验
    "ARKit + 接触 + 求解"实验管线本身。
    """

    if len(line_sessions) < 3:
        raise ValueError("--line-check 至少需要 3 个点（3 个 session）。")
    if len(line_spacings_mm) != len(line_sessions) - 1:
        raise ValueError(
            f"--line-spacings-mm 需要 {len(line_sessions) - 1} 个相邻间距值"
            f"（当前给了 {len(line_spacings_mm)} 个）。"
        )

    fixed_points_world = np.empty((len(line_sessions), 3), dtype=np.float64)
    for index, session_dir in enumerate(line_sessions):
        solve_result = solve_q_I_from_demo(
            demo_dir=session_dir,
            detection_T_I_G=current_T_I_G,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        fixed_points_world[index] = solve_result.fixed_point_world_m
        print(
            f"  点 {index}：c_W={np.array2string(solve_result.fixed_point_world_m, precision=5, separator=', ')} m"
        )

    centroid = fixed_points_world.mean(axis=0)
    centered = fixed_points_world - centroid
    # 直线方向在 XYZ 空间里求：用点坐标协方差矩阵的特征向量（PCA 主方向）。
    # 注意不能用 np.linalg.svd(centered) 的左奇异向量——那是"点空间"的
    # 向量（N 维），不是 XYZ 三维空间的方向，语义完全不同。
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    line_direction = eigenvectors[:, -1]
    projections = centered @ line_direction
    collinearity_errors_mm = np.linalg.norm(
        centered - np.outer(projections, line_direction),
        axis=1,
    ) * 1000.0

    measured_spacings_mm = np.asarray(line_spacings_mm, dtype=np.float64)
    reconstructed_spacings_mm = (
        np.linalg.norm(np.diff(fixed_points_world, axis=0), axis=1) * 1000.0
    )
    spacing_errors_mm = reconstructed_spacings_mm - measured_spacings_mm

    result = {
        "fixed_points_world_m": fixed_points_world,
        "line_direction_w": line_direction,
        "collinearity_errors_mm": collinearity_errors_mm,
        "reconstructed_spacings_mm": reconstructed_spacings_mm,
        "measured_spacings_mm": measured_spacings_mm,
        "spacing_errors_mm": spacing_errors_mm,
        "max_collinearity_mm": float(np.max(collinearity_errors_mm)),
        "max_spacing_error_mm": float(np.max(np.abs(spacing_errors_mm))),
    }
    print("\n画线自检结果：")
    print(f"  共线性残差（mm）：{np.array2string(collinearity_errors_mm, precision=2, separator=', ')}")
    print(f"  重建间距（mm）：{np.array2string(reconstructed_spacings_mm, precision=2, separator=', ')}")
    print(f"  尺子间距（mm）：{np.array2string(measured_spacings_mm, precision=2, separator=', ')}")
    print(f"  间距误差（mm）：{np.array2string(spacing_errors_mm, precision=2, separator=', ')}")
    if result["max_collinearity_mm"] < 2.0 and result["max_spacing_error_mm"] < 3.0:
        print("  结论：实验管线可靠（点共线、间距与尺子一致）。")
    else:
        print("  结论：实验管线存在明显误差，先排查接触/ARKit 质量再继续标定。")
    return result


def save_line_check_txt(output_dir: Path, result: dict) -> Path:
    """保存画线自检结果。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "T_I_G pipeline line-check summary",
        "",
        "fixed_points_world_m:",
        *[
            "  " + np.array2string(point, precision=6, separator=", ")
            for point in result["fixed_points_world_m"]
        ],
        f"line_direction_w: {np.array2string(result['line_direction_w'], precision=6, separator=', ')}",
        f"collinearity_errors_mm: {np.array2string(result['collinearity_errors_mm'], precision=3, separator=', ')}",
        f"reconstructed_spacings_mm: {np.array2string(result['reconstructed_spacings_mm'], precision=3, separator=', ')}",
        f"measured_spacings_mm: {np.array2string(result['measured_spacings_mm'], precision=3, separator=', ')}",
        f"spacing_errors_mm: {np.array2string(result['spacing_errors_mm'], precision=3, separator=', ')}",
        f"max_collinearity_mm: {result['max_collinearity_mm']:.3f}",
        f"max_spacing_error_mm: {result['max_spacing_error_mm']:.3f}",
        "",
    ]
    path = output_dir / "line_check_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ============================================================
# 8. 合成数据自测
# ============================================================


def make_random_pose_set(
    count: int,
    seed: int,
    position_center_m: np.ndarray,
    position_spread_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 count 个差异明显的随机位姿（固定种子，保证可复现）。"""

    rng = np.random.default_rng(seed)
    positions = position_center_m + rng.uniform(
        -position_spread_m, position_spread_m, size=(count, 3)
    )
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    return positions, rotations


def synthesize_session_directory(
    root_dir: Path,
    session_name: str,
    rotations: np.ndarray,
    T_I_G_true: np.ndarray,
    tip_G_m: np.ndarray,
    fixed_point_world_m: np.ndarray,
    seed: int,
    fps: float = 100.0,
    contact_duration_s: float = 1.5,
    motion_duration_s: float = 0.5,
    lift_m: float = 0.06,
    add_noise: bool = False,
) -> Path:
    """
    生成一个包含固定点接触的合成 odometry session。

    物理模型与真实实验一致：牙签尖端始终压在固定点 c_W 上，iPhone 随夹爪
    绕尖端转到不同姿态。因此每个接触姿态的 iPhone 位置由约束决定：
        tip_I = T_I_G_true @ [tip_G; 1]（恒定，I 系）
        p_W_I_i = c_W - R_W_I_i @ tip_I
    接触段保持 1.5 s（尖端不动、iPhone 静止）；姿态之间用"旋转 slerp +
    位置线性插值 + 抛物线抬升 lift"过渡，使尖端暂时离开固定点，产生足够
    的速度信号供真实 detect_contact_segments 区分接触段与移动段。
    """

    tip_offset_I = T_I_G_true[:3, :3] @ np.asarray(tip_G_m, dtype=np.float64) + T_I_G_true[:3, 3]
    fixed_point = np.asarray(fixed_point_world_m, dtype=np.float64)

    frame_count_hold = max(1, int(round(contact_duration_s * fps)))
    frame_count_motion = max(1, int(round(motion_duration_s * fps)))
    dt = 1.0 / fps

    timestamps: list[float] = []
    all_positions: list[np.ndarray] = []
    all_quats_xyzw: list[np.ndarray] = []

    time_index = 0.0
    pose_count = len(rotations)
    for pose_index in range(pose_count):
        rotation_now = rotations[pose_index]
        position_now = fixed_point - rotation_now @ tip_offset_I
        quat_now = Rotation.from_matrix(rotation_now).as_quat()  # (x, y, z, w)

        for _ in range(frame_count_hold):
            timestamps.append(time_index)
            all_positions.append(position_now)
            all_quats_xyzw.append(quat_now)
            time_index += dt

        if pose_index < pose_count - 1:
            rotation_next = rotations[pose_index + 1]
            position_next = fixed_point - rotation_next @ tip_offset_I
            quat_next = Rotation.from_matrix(rotation_next).as_quat()
            for step in range(1, frame_count_motion + 1):
                alpha = step / (frame_count_motion + 1)
                interp_rotation = Rotation.from_matrix(
                    Rotation.from_quat(
                        stray.slerp_quaternion_xyzw(quat_now, quat_next, alpha)
                    ).as_matrix()
                ).as_matrix()
                # 抛物线抬升：中间时刻抬到 lift_m，两端归零，保证尖端离开固定点。
                h = 0.5 - 0.5 * np.cos(np.pi * alpha)
                interp_position = (
                    (1.0 - alpha) * position_now
                    + alpha * position_next
                    + np.array([0.0, 0.0, 4.0 * h * (1.0 - h) * lift_m])
                )
                interp_quat = Rotation.from_matrix(interp_rotation).as_quat()
                timestamps.append(time_index)
                all_positions.append(interp_position)
                all_quats_xyzw.append(interp_quat)
                time_index += dt

    positions_array = np.asarray(all_positions, dtype=np.float64)
    quats_xyzw_array = np.asarray(all_quats_xyzw, dtype=np.float64)
    timestamps_array = np.asarray(timestamps, dtype=np.float64)

    if add_noise:
        rng = np.random.default_rng(seed + 1000)
        positions_array = positions_array + rng.normal(0.0, 3e-5, size=positions_array.shape)
        small_rotations = Rotation.from_rotvec(
            rng.normal(0.0, 1e-4, size=(len(quats_xyzw_array), 3))
        ).as_matrix()
        rotation_matrices = np.asarray(
            [Rotation.from_quat(q).as_matrix() for q in quats_xyzw_array]
        )
        rotation_matrices = np.einsum("nij,njk->nik", small_rotations, rotation_matrices)
        quats_xyzw_array = np.asarray(
            [Rotation.from_matrix(rotation_matrices[index]).as_quat() for index in range(len(rotation_matrices))]
        )

    session_dir = root_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    odometry_path = session_dir / "odometry.csv"
    _write_odometry_csv(odometry_path, timestamps_array, positions_array, quats_xyzw_array)

    # 一致性校验：接触保持段的牙签尖端世界位置应集中在一个固定点上。
    rotation_matrices = np.asarray(
        [Rotation.from_quat(q).as_matrix() for q in quats_xyzw_array[:frame_count_hold]]
    )
    tip_world_hold = (
        positions_array[:frame_count_hold]
        + np.einsum("nij,j->ni", rotation_matrices, tip_offset_I)
    )
    fixed_point_spread_mm = float(
        np.max(np.linalg.norm(tip_world_hold - fixed_point, axis=1)) * 1000.0
    )
    if fixed_point_spread_mm > 5.0:
        raise RuntimeError(
            f"合成 session 接触段尖端世界位置不集中（spread {fixed_point_spread_mm:.2f} mm），"
            "请检查生成逻辑。"
        )

    return odometry_path


def _write_odometry_csv(
    path: Path,
    timestamps: np.ndarray,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
) -> None:
    """按 Stray odometry.csv 格式写入（8 列，utf-8-sig 表头）。"""

    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for index in range(len(timestamps)):
            writer.writerow(
                [
                    f"{timestamps[index]:.9f}",
                    f"{positions[index, 0]:.9f}",
                    f"{positions[index, 1]:.9f}",
                    f"{positions[index, 2]:.9f}",
                    f"{quats_xyzw[index, 0]:.9f}",
                    f"{quats_xyzw[index, 1]:.9f}",
                    f"{quats_xyzw[index, 2]:.9f}",
                    f"{quats_xyzw[index, 3]:.9f}",
                ]
            )


def run_synthetic(ground_truth_json: Path | None) -> int:
    """
    合成数据全流程自测：验证 verify / compute_rotation / kabsch 在已知真值下
    能否精确恢复。两遍：
        1. 无噪声：断言恢复误差 < 1e-6 m（证明计算正确）。
        2. 加噪声：断言 < 2 mm / 1 deg（证明端到端鲁棒性）。
    """

    if ground_truth_json is not None:
        T_I_G_true, description = tig.load_tig_matrix(ground_truth_json)
        print(f"使用指定真值外参：{description}")
    else:
        T_I_G_true = SYNTHETIC_TRUE_T_I_G
        print("使用默认真值外参（当前正式外参）。")

    probe_offset_G_m = np.asarray(DEFAULT_PROBE_OFFSET_G_M, dtype=np.float64)
    contact_count = DEFAULT_CONTACT_COUNT
    pose_count = contact_count

    failures: list[str] = []
    checks: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="tig_synthetic_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        positions, rotations = make_random_pose_set(
            count=pose_count,
            seed=42,
            position_center_m=np.array([-0.1, 0.2, 0.4]),
            position_spread_m=0.05,
        )
        fixed_point_world_m = np.array([0.15, -0.05, 0.02])

        # ---- 无噪声：验证 + 单轴 compute_rotation（沿 Z 对齐） + kabsch ----
        verify_session = synthesize_session_directory(
            root_dir=tmp_root,
            session_name="verify_noise_free",
            rotations=rotations,
            fixed_point_world_m=fixed_point_world_m,
            T_I_G_true=T_I_G_true,
            tip_G_m=probe_offset_G_m,
            seed=0,
            add_noise=False,
        )
        verify_result, _ = run_verify(
            demo_dir=verify_session.parent,
            T_I_G=T_I_G_true,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        expected_q_I = T_I_G_true[:3, :3] @ probe_offset_G_m + T_I_G_true[:3, 3]
        q_I_error = float(np.linalg.norm(verify_result.q_I_m - expected_q_I))
        checks.append(("verify noiseless q_I recovery", f"{q_I_error:.3e} m"))
        if q_I_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"verify noiseless q_I error {q_I_error:.3e} m")

        # ---- 无噪声：滑移直线拟合（沿 Z 对齐的 4 级滑移） ----
        slide_lengths_mm = [0.0, 8.0, 16.0, 25.0]
        slide_session_paths: list[Path] = []
        for slide_index, length_mm in enumerate(slide_lengths_mm):
            tip_G = probe_offset_G_m + np.array([0.0, 0.0, length_mm / 1000.0])
            slide_session_paths.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"slide_z_noise_free_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=tip_G,
                    seed=slide_index,
                    add_noise=False,
                ).parent
            )
        slide_fit = fit_slide_axis(
            session_dirs=slide_session_paths,
            lengths_mm=slide_lengths_mm,
            detection_T_I_G=T_I_G_true,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        direction_unit = slide_fit.direction_I / (np.linalg.norm(slide_fit.direction_I) + 1e-12)
        true_z_column = T_I_G_true[:3, :3] @ np.array([0.0, 0.0, 1.0])
        column_error = float(np.linalg.norm(direction_unit - true_z_column))
        checks.append(("compute noiseless z-column recovery", f"{column_error:.3e}"))
        if column_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"compute noiseless column error {column_error:.3e}")
        if abs(slide_fit.scale_error) > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"compute noiseless scale error {slide_fit.scale_error:.3e}")

        # ---- 无噪声：倾斜轴 u 的滑移（验证直线方向 = R @ u，不受倾斜影响） ----
        tilted_u = np.array([0.05, -0.03, 1.0])
        tilted_u = tilted_u / np.linalg.norm(tilted_u)
        tilted_sessions: list[Path] = []
        for slide_index, length_mm in enumerate(slide_lengths_mm):
            tip_G = probe_offset_G_m + tilted_u * (length_mm / 1000.0)
            tilted_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"slide_tilted_noise_free_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=tip_G,
                    seed=slide_index + 100,
                    add_noise=False,
                ).parent
            )
        tilted_fit = fit_slide_axis(
            session_dirs=tilted_sessions,
            lengths_mm=slide_lengths_mm,
            detection_T_I_G=T_I_G_true,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        tilted_direction = tilted_fit.direction_I / (np.linalg.norm(tilted_fit.direction_I) + 1e-12)
        true_R_u = T_I_G_true[:3, :3] @ tilted_u
        tilted_direction_error = float(np.linalg.norm(tilted_direction - true_R_u))
        checks.append(
            ("compute noiseless tilted direction (R@u)", f"{tilted_direction_error:.3e}")
        )
        if tilted_direction_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"tilted direction error {tilted_direction_error:.3e}")
        tilt_angle_deg = float(
            np.degrees(np.arccos(np.clip(tilted_direction @ true_z_column, -1.0, 1.0)))
        )
        checks.append(("tilt vs z-column angle (expected >0)", f"{tilt_angle_deg:.2f} deg"))

        # ---- 无噪声：双轴滑移组装完整旋转矩阵（z + x） ----
        z_axis_sessions: list[Path] = []
        x_axis_sessions: list[Path] = []
        for slide_index, length_mm in enumerate(slide_lengths_mm):
            z_tip = probe_offset_G_m + np.array([0.0, 0.0, length_mm / 1000.0])
            z_axis_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"two_axis_z_noise_free_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=z_tip,
                    seed=slide_index + 500,
                    add_noise=False,
                ).parent
            )
            x_tip = probe_offset_G_m + np.array([length_mm / 1000.0, 0.0, 0.0])
            x_axis_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"two_axis_x_noise_free_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=x_tip,
                    seed=slide_index + 600,
                    add_noise=False,
                ).parent
            )
        two_axis_candidate = run_compute_rotation_with_two_axes(
            axis_sessions={
                "z": [str(path) for path in z_axis_sessions],
                "x": [str(path) for path in x_axis_sessions],
            },
            axis_lengths_mm={"z": slide_lengths_mm, "x": slide_lengths_mm},
            axis_directions={"z": 1, "x": 1},  # 合成用例沿 +Z/+X 滑移
            current_T_I_G=T_I_G_true,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
            translation_tip_G_m=probe_offset_G_m,
        )
        if two_axis_candidate.T_I_G_candidate is None:
            failures.append("two-axis candidate matrix is None")
        else:
            rotation_recovered = two_axis_candidate.T_I_G_candidate[:3, :3]
            rotation_error_2 = float(np.linalg.norm(rotation_recovered - T_I_G_true[:3, :3]))
            checks.append(("compute noiseless two-axis R error", f"{rotation_error_2:.3e}"))
            if rotation_error_2 > SYNTHETIC_NOISELESS_TOL_M:
                failures.append(f"two-axis R error {rotation_error_2:.3e}")
            translation_recovered = two_axis_candidate.T_I_G_candidate[:3, 3]
            translation_error_2 = float(
                np.linalg.norm(translation_recovered - T_I_G_true[:3, 3])
            )
            checks.append(("compute noiseless two-axis t error", f"{translation_error_2:.3e} m"))
            if translation_error_2 > SYNTHETIC_NOISELESS_TOL_M:
                failures.append(f"two-axis t error {translation_error_2:.3e}")

        # ---- 无噪声：meta 驱动的 -Z/+X 方向恢复（模拟真实采集流程） ----
        # 采集程序写入 tig_meta.json（z 维 direction=-1，x 维 direction=+1），
        # 验证程序据此恢复旋转列。z 维牙签尖端 tip_G = (0,0,-L)，对应向下伸出。
        meta_z_sessions: list[Path] = []
        meta_x_sessions: list[Path] = []
        z_length_m = 31.93 / 1000.0
        x_length_m = 25.0 / 1000.0
        for slide_index, length_mm in enumerate(slide_lengths_mm):
            z_tip = np.array([0.0, 0.0, -(z_length_m + length_mm / 1000.0)])
            meta_z_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"meta_z_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=z_tip,
                    seed=slide_index + 700,
                    add_noise=False,
                ).parent
            )
            write_tig_meta(
                meta_z_sessions[-1],
                axis="z",
                slide_mm=length_mm,
                direction=-1,
                toothpick_length_mm=z_length_m * 1000.0,
            )
            x_tip = np.array([x_length_m + length_mm / 1000.0, 0.0, 0.0])
            meta_x_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"meta_x_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=x_tip,
                    seed=slide_index + 800,
                    add_noise=False,
                ).parent
            )
            write_tig_meta(
                meta_x_sessions[-1],
                axis="x",
                slide_mm=length_mm,
                direction=1,
                toothpick_length_mm=x_length_m * 1000.0,
            )

        meta_z_fit = fit_slide_axis(
            session_dirs=meta_z_sessions,
            lengths_mm=slide_lengths_mm,
            detection_T_I_G=T_I_G_true,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        meta_z_direction = meta_z_fit.direction_I / (np.linalg.norm(meta_z_fit.direction_I) + 1e-12)
        meta_z_direction = -meta_z_direction  # direction=-1 → 旋转列 = -d_unit
        true_z_column = T_I_G_true[:3, :3] @ np.array([0.0, 0.0, 1.0])
        meta_z_error = float(np.linalg.norm(meta_z_direction - true_z_column))
        checks.append(("compute noiseless meta -Z column recovery", f"{meta_z_error:.3e}"))
        if meta_z_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"meta -Z column error {meta_z_error:.3e}")

        meta_x_fit = fit_slide_axis(
            session_dirs=meta_x_sessions,
            lengths_mm=slide_lengths_mm,
            detection_T_I_G=T_I_G_true,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        meta_x_direction = meta_x_fit.direction_I / (np.linalg.norm(meta_x_fit.direction_I) + 1e-12)
        true_x_column = T_I_G_true[:3, :3] @ np.array([1.0, 0.0, 0.0])
        meta_x_error = float(np.linalg.norm(meta_x_direction - true_x_column))
        checks.append(("compute noiseless meta +X column recovery", f"{meta_x_error:.3e}"))
        if meta_x_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"meta +X column error {meta_x_error:.3e}")

        # meta 读取函数本身：分组与 tip_G 构造。
        meta_groups = group_meta_sessions(tmp_root, meta_z_sessions + meta_x_sessions)
        if len(meta_groups["z"]) != len(slide_lengths_mm) or len(meta_groups["x"]) != len(slide_lengths_mm):
            failures.append("meta grouping failed")
        else:
            tip_z = tip_G_from_meta(meta_groups["z"][0][1])
            tip_x = tip_G_from_meta(meta_groups["x"][0][1])
            if abs(tip_z[2] + z_length_m) > 1e-9 or abs(tip_x[0] - x_length_m) > 1e-9:
                failures.append("meta tip_G construction failed")
            else:
                checks.append(("meta tip_G construction", "ok"))

        # meta 驱动的完整矩阵组装：z(direction=-1) + x(direction=+1)。
        meta_two_axis = run_compute_rotation_with_two_axes(
            axis_sessions={
                "z": [str(path) for path in meta_z_sessions],
                "x": [str(path) for path in meta_x_sessions],
            },
            axis_lengths_mm={"z": slide_lengths_mm, "x": slide_lengths_mm},
            axis_directions={"z": -1, "x": 1},
            current_T_I_G=T_I_G_true,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
            translation_tip_G_m=np.array([0.0, 0.0, -z_length_m]),
        )
        if meta_two_axis.T_I_G_candidate is None:
            failures.append("meta two-axis candidate is None")
        else:
            meta_R_error = float(
                np.linalg.norm(meta_two_axis.T_I_G_candidate[:3, :3] - T_I_G_true[:3, :3])
            )
            meta_t_error = float(
                np.linalg.norm(meta_two_axis.T_I_G_candidate[:3, 3] - T_I_G_true[:3, 3])
            )
            checks.append(("meta two-axis R error", f"{meta_R_error:.3e}"))
            checks.append(("meta two-axis t error", f"{meta_t_error:.3e} m"))
            if meta_R_error > SYNTHETIC_NOISELESS_TOL_M:
                failures.append(f"meta two-axis R error {meta_R_error:.3e}")
            if meta_t_error > SYNTHETIC_NOISELESS_TOL_M:
                failures.append(f"meta two-axis t error {meta_t_error:.3e}")

        # ---- 无噪声：Kabsch 多尖端整体配准（4 个非共面点，比 3 点更稳） ----
        tip_g_three = np.array(
            [
                [0.0, 0.0, 0.03193],
                [0.02, 0.0, 0.03193],
                [0.0, 0.015, 0.045],
                [0.01, 0.005, 0.012],
            ],
            dtype=np.float64,
        )
        kabsch_sessions: list[Path] = []
        for tip_index, tip_G in enumerate(tip_g_three):
            kabsch_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"kabsch_noise_free_{tip_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=tip_G,
                    seed=tip_index + 200,
                    add_noise=False,
                ).parent
            )
        rotation_fit, translation_fit, q_I_points, _, rotation_diff, fit_rms = run_kabsch(
            kabsch_sessions=[str(path) for path in kabsch_sessions],
            tip_g_list=[tip.tolist() for tip in tip_g_three],
            current_T_I_G=T_I_G_true,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        rotation_error = float(np.linalg.norm(rotation_fit - T_I_G_true[:3, :3]))
        translation_error = float(np.linalg.norm(translation_fit - T_I_G_true[:3, 3]))
        checks.append(("kabsch noiseless R error", f"{rotation_error:.3e}"))
        checks.append(("kabsch noiseless t error", f"{translation_error:.3e} m"))
        if rotation_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"kabsch noiseless R error {rotation_error:.3e}")
        if translation_error > SYNTHETIC_NOISELESS_TOL_M:
            failures.append(f"kabsch noiseless t error {translation_error:.3e}")

        # ---- 加噪声：端到端鲁棒性 ----
        noisy_verify_session = synthesize_session_directory(
            root_dir=tmp_root,
            session_name="verify_noisy",
            rotations=rotations,
            fixed_point_world_m=fixed_point_world_m,
            T_I_G_true=T_I_G_true,
            tip_G_m=probe_offset_G_m,
            seed=1,
            add_noise=True,
        )
        noisy_verify, _ = run_verify(
            demo_dir=noisy_verify_session.parent,
            T_I_G=T_I_G_true,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        noisy_q_I_error = float(np.linalg.norm(noisy_verify.q_I_m - expected_q_I))
        checks.append(("verify noisy q_I recovery", f"{noisy_q_I_error:.3f} mm"))
        if noisy_q_I_error > SYNTHETIC_NOISY_TOL_M:
            failures.append(f"verify noisy q_I error {noisy_q_I_error:.3f} mm")

        noisy_slide_sessions: list[Path] = []
        for slide_index, length_mm in enumerate(slide_lengths_mm):
            tip_G = probe_offset_G_m + np.array([0.0, 0.0, length_mm / 1000.0])
            noisy_slide_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"slide_z_noisy_{slide_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=tip_G,
                    seed=slide_index + 300,
                    add_noise=True,
                ).parent
            )
        noisy_fit = fit_slide_axis(
            session_dirs=noisy_slide_sessions,
            lengths_mm=slide_lengths_mm,
            detection_T_I_G=T_I_G_true,
            detection_probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        noisy_direction = noisy_fit.direction_I / (np.linalg.norm(noisy_fit.direction_I) + 1e-12)
        noisy_column_error = float(np.linalg.norm(noisy_direction - true_z_column))
        checks.append(("compute noisy z-column error", f"{noisy_column_error:.3f}"))
        if noisy_column_error > SYNTHETIC_NOISY_ROT_TOL_DEG / 57.3:
            failures.append(f"compute noisy column error {noisy_column_error:.3f}")

        noisy_kabsch_sessions: list[Path] = []
        for tip_index, tip_G in enumerate(tip_g_three):
            noisy_kabsch_sessions.append(
                synthesize_session_directory(
                    root_dir=tmp_root,
                    session_name=f"kabsch_noisy_{tip_index}",
                    rotations=rotations,
                    fixed_point_world_m=fixed_point_world_m,
                    T_I_G_true=T_I_G_true,
                    tip_G_m=tip_G,
                    seed=tip_index + 400,
                    add_noise=True,
                ).parent
            )
        noisy_rotation_fit, _, _, _, _, _ = run_kabsch(
            kabsch_sessions=[str(path) for path in noisy_kabsch_sessions],
            tip_g_list=[tip.tolist() for tip in tip_g_three],
            current_T_I_G=T_I_G_true,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=contact_count,
        )
        noisy_rotation_diff_deg = rotation_angle_degrees(
            noisy_rotation_fit, T_I_G_true[:3, :3]
        )
        checks.append(("kabsch noisy R angle error", f"{noisy_rotation_diff_deg:.3f} deg"))
        if noisy_rotation_diff_deg > SYNTHETIC_NOISY_ROT_TOL_DEG:
            failures.append(
                f"kabsch noisy R angle error {noisy_rotation_diff_deg:.3f} deg"
            )

    print("\n合成自测明细：")
    for name, value in checks:
        print(f"  {name}: {value}")

    print("\n" + ("合成自测 FAIL" if failures else "合成自测 PASS"))
    for failure in failures:
        print(f"  FAIL: {failure}")
    return 1 if failures else 0


# ============================================================
# 9. 命令行入口
# ============================================================


def _parse_comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tip_g(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--tip-g 需要 x,y,z 三个值，当前是：{value}")
    numbers = [float(part) for part in parts]
    if not np.all(np.isfinite(numbers)):
        raise ValueError(f"--tip-g 必须是有限数值：{value}")
    return numbers


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="T_I_G 旋转矩阵验证与实验标定（详见脚本 docstring）。"
    )
    parser.add_argument(
        "--stray-root",
        type=Path,
        default=stray.DEFAULT_STRAY_ROOT,
        help=f"Stray 数据根目录，默认：{stray.DEFAULT_STRAY_ROOT}",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=None,
        help="验证/自检模式使用的 session 文件夹；不填时使用最新含 odometry 的文件夹。",
    )
    parser.add_argument(
        "--tig-json",
        type=Path,
        default=None,
        help="验证/标定使用的 T_I_G JSON（含 T_I_G_candidate 或 T_I_G 字段）；"
        "不填时使用当前正式外参。",
    )
    parser.add_argument(
        "--contact-count",
        type=int,
        default=DEFAULT_CONTACT_COUNT,
        help=f"每个 session 期望的接触次数，默认：{DEFAULT_CONTACT_COUNT}",
    )
    parser.add_argument(
        "--max-stable-speed",
        type=float,
        default=DEFAULT_MAX_STABLE_SPEED_M_PER_S,
        help=f"稳定接触速度阈值 m/s，默认：{DEFAULT_MAX_STABLE_SPEED_M_PER_S}",
    )
    parser.add_argument(
        "--min-contact-duration",
        type=float,
        default=DEFAULT_MIN_CONTACT_DURATION_S,
        help=f"每次接触最短持续时间 s，默认：{DEFAULT_MIN_CONTACT_DURATION_S}",
    )
    parser.add_argument(
        "--probe-offset-x",
        type=float,
        default=DEFAULT_PROBE_OFFSET_G_M[0],
        help="牙签尖端相对 TCP 在夹爪 X_G 轴上的偏移 m，默认 0。",
    )
    parser.add_argument(
        "--probe-offset-y",
        type=float,
        default=DEFAULT_PROBE_OFFSET_G_M[1],
        help="牙签尖端相对 TCP 在夹爪 Y_G 轴上的偏移 m，默认 0。",
    )
    parser.add_argument(
        "--probe-offset-z",
        type=float,
        default=DEFAULT_PROBE_OFFSET_G_M[2],
        help=(
            "牙签尖端相对 TCP 在夹爪 Z_G 轴上的偏移 m，"
            f"默认 {DEFAULT_PROBE_OFFSET_G_M[2]}。"
        ),
    )

    mode_group = parser.add_argument_group("模式（默认 --verify-only）")
    mode_group.add_argument(
        "--verify-only",
        action="store_true",
        help="验证当前/指定 T_I_G 的旋转（默认模式）。",
    )
    mode_group.add_argument(
        "--compute-rotation",
        action="store_true",
        help="用多滑移直线拟合从实验算旋转列/矩阵。",
    )
    mode_group.add_argument(
        "--kabsch",
        action="store_true",
        help="用已知绝对尖端偏移做 Umeyama 整体配准。",
    )
    mode_group.add_argument(
        "--line-check",
        action="store_true",
        help="画线自检：检验实验管线本身（不依赖 T_I_G）。",
    )
    mode_group.add_argument(
        "--synthetic",
        action="store_true",
        help="合成数据自测（无需硬件）。",
    )

    compute_group = parser.add_argument_group("--compute-rotation 参数")
    compute_group.add_argument(
        "--z-sessions",
        type=str,
        default=None,
        help="沿 +Z_G 滑移的 3~5 个 session 目录，逗号分隔，按滑移量递增。",
    )
    compute_group.add_argument(
        "--z-lengths-mm",
        type=str,
        default=None,
        help="与 --z-sessions 对应的滑移量 mm，逗号分隔（例如 0,8,16,25）。",
    )
    compute_group.add_argument(
        "--x-sessions",
        type=str,
        default=None,
        help="沿 +X_G 滑移的 3~5 个 session 目录（可选，用于组合完整矩阵）。",
    )
    compute_group.add_argument(
        "--x-lengths-mm",
        type=str,
        default=None,
        help="与 --x-sessions 对应的滑移量 mm。",
    )
    compute_group.add_argument(
        "--y-sessions",
        type=str,
        default=None,
        help="沿 +Y_G 滑移的 3~5 个 session 目录（可选）。",
    )
    compute_group.add_argument(
        "--y-lengths-mm",
        type=str,
        default=None,
        help="与 --y-sessions 对应的滑移量 mm。",
    )

    kabsch_group = parser.add_argument_group("--kabsch 参数")
    kabsch_group.add_argument(
        "--kabsch-sessions",
        type=str,
        default=None,
        help="3+ 个 session 目录，逗号分隔，每个 session 对应一个已知尖端位置。",
    )
    kabsch_group.add_argument(
        "--tip-g",
        action="append",
        type=_parse_tip_g,
        default=None,
        help="与 --kabsch-sessions 对应的已知尖端绝对偏移 x,y,z（m），可重复传。",
    )

    line_group = parser.add_argument_group("--line-check 参数")
    line_group.add_argument(
        "--line-sessions",
        type=str,
        default=None,
        help="沿直线触碰的 3+ 个点的 session 目录，逗号分隔。",
    )
    line_group.add_argument(
        "--line-spacings-mm",
        type=str,
        default=None,
        help="相邻点之间的尺子间距 mm，共 N-1 个，逗号分隔（例如 50,50）。",
    )

    return parser.parse_args()


def _resolve_tig_and_offset(args: argparse.Namespace) -> tuple[np.ndarray, str, np.ndarray]:
    """统一的 T_I_G 与探针偏移解析。"""

    T_I_G, description = tig.load_tig_matrix(args.tig_json)
    probe_offset_G_m = np.array(
        [args.probe_offset_x, args.probe_offset_y, args.probe_offset_z],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(probe_offset_G_m)):
        raise ValueError("探针偏移必须是有限数值。")
    return T_I_G, description, probe_offset_G_m


def _output_dir_for(demo_dir: Path) -> Path:
    return Path(demo_dir).resolve() / DEFAULT_OUTPUT_DIR_NAME


def _print_verify_result(verify_result: VerifyResult) -> None:
    print("\n验证结果：")
    print(f"  实验 q_I（mm）：{np.array2string(verify_result.q_I_m * 1000.0, precision=3, separator=', ')}")
    print(
        "  预测 q_I（mm）："
        f"{np.array2string(verify_result.predicted_q_I_m * 1000.0, precision=3, separator=', ')}"
    )
    print(f"  残差（mm）：{np.array2string(verify_result.residual_m * 1000.0, precision=3, separator=', ')}")
    print(f"  G 系分解：z_dir={verify_result.residual_z_dir_mm:.2f}, "
          f"x_dir={verify_result.residual_x_dir_mm:.2f}, "
          f"y_dir={verify_result.residual_y_dir_mm:.2f} mm")
    print(f"  判定：{verify_result.judgement}")


# ============================================================
# 9.5 交互式菜单（VSCode 终端输入数字选择模式）
# ============================================================


def _ask(prompt: str, default: str | None = None) -> str:
    """读取一行输入；空回车且提供默认值时返回默认值。"""

    try:
        raw = input(prompt)
    except EOFError:
        raise SystemExit(0)
    if raw.strip() == "" and default is not None:
        return str(default)
    return raw.strip()


def _ask_float(prompt: str, default: float) -> float:
    """读取浮点数；输入无效时回退到默认值。"""

    raw = _ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        print(f"  输入无效，使用默认 {default}。")
        return float(default)


def _ask_int(prompt: str, default: int) -> int:
    """读取整数；输入无效时回退到默认值。"""

    raw = _ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"  输入无效，使用默认 {default}。")
        return int(default)


def _ask_yes_no(prompt: str, default: bool) -> bool:
    """读取 y/n；空回车时使用默认值。"""

    hint = "y" if default else "n"
    raw = _ask(f"{prompt} [{'Y' if default else 'y'}/{'n' if not default else 'N'}]：", "").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def _ask_csv_floats(prompt: str) -> list[float]:
    """读取逗号分隔的浮点数列表。"""

    while True:
        raw = _ask(prompt)
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        try:
            values = [float(item) for item in parts]
        except ValueError:
            print("  格式错误：请用逗号分隔的数字，例如 0,8,16,25")
            continue
        if len(values) < 1:
            print("  请输入至少一个数字。")
            continue
        return values


def find_session_candidates(root_dir: Path) -> list[Path]:
    """
    找出数据根目录下的所有 session 文件夹（顶层子目录）。

    兼容两种结构：session/odometry.csv 与 session/<导出文件夹>/odometry.csv，
    统一收缩到 session 顶层目录展示，避免同一个 session 出现两次。
    """

    root_dir = Path(root_dir).resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    ignored_dir_names = {
        "processed_umi",
        "tig_rotation_calibration",
        "tig_fixed_point_validation",
        "test",
    }

    candidates: list[Path] = []
    for path in root_dir.rglob("*"):
        if not path.is_dir():
            continue
        if any(part.lower() in ignored_dir_names for part in path.parts):
            continue
        if (path / "odometry.csv").exists():
            relative_parts = path.relative_to(root_dir).parts
            top_level = root_dir / relative_parts[0]
            if top_level not in candidates:
                candidates.append(top_level)

    # 按修改时间降序排列：最新（最近录制/最近处理）的 session 排在第一位，
    # 保证"直接回车 = 用最新数据"。往 session 写输出文件会刷新 mtime，所以
    # 列表顺序可能随处理过程变化，编号以当前屏幕显示为准即可。
    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )


def _choose_sessions(
    prompt: str,
    root_dir: Path,
    min_count: int,
    max_count: int | None = None,
) -> list[Path] | None:
    """列出 session 供编号选择（支持逗号分隔多选）。"""

    candidates = find_session_candidates(root_dir)
    if not candidates:
        print(
            f"  {root_dir} 下没有找到任何包含 odometry.csv 的 session 文件夹。\n"
            "  请先把 Stray 导出的文件夹放进数据根目录。"
        )
        return None

    print("  可用 session（最新在前）：")
    for index, candidate in enumerate(candidates):
        print(f"    [{index + 1}] {candidate}")

    while True:
        if min_count == 1 and max_count == 1:
            action_hint = "直接回车 = 最新一条"
        else:
            action_hint = "如 1,2,3"
        raw = _ask(f"{prompt}（编号，多个用逗号分隔，{action_hint}）：")
        if raw == "" and min_count == 1 and max_count == 1:
            return [candidates[0]]
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        try:
            indices = [int(item) - 1 for item in parts]
        except ValueError:
            print("  请输入数字编号。")
            continue
        if len(indices) < min_count:
            print(f"  至少需要选择 {min_count} 个 session。")
            continue
        if max_count is not None and len(indices) > max_count:
            print(f"  最多选择 {max_count} 个 session。")
            continue
        if any(index < 0 or index >= len(candidates) for index in indices):
            print("  编号超出范围。")
            continue
        return [candidates[index] for index in indices]


def _choose_candidate_json(root_dir: Path) -> Path | None:
    """列出数据根下的候选 T_I_G JSON 供选择，或让用户输入路径。"""

    candidates = sorted(
        Path(root_dir).rglob("T_I_G_candidate.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print("  找到的候选 JSON（最近的在前）：")
        for index, candidate in enumerate(candidates[:10]):
            print(f"    [{index + 1}] {candidate}")
        while True:
            raw = _ask("  输入编号（直接回车取消）：")
            if raw == "":
                return None
            try:
                index = int(raw) - 1
            except ValueError:
                print("  请输入编号。")
                continue
            if 0 <= index < min(len(candidates), 10):
                return candidates[index]
            print("  编号超出范围。")
            continue

    print("  没有找到任何 T_I_G_candidate.json。")
    raw = _ask("  请输入候选 JSON 的完整路径（直接回车取消）：")
    if raw == "":
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        print(f"  路径不存在：{path}")
        return None
    return path


def _print_judgement_hint(verify_result: VerifyResult) -> None:
    """根据验证判定给出下一步操作建议。"""

    if verify_result.contact_quality_warning:
        print("  建议：接触质量差，重新录制并确保每次保持 1.5~2 秒、姿态差异足够大。")
    elif verify_result.residual_xy_mm > DEFAULT_VERIFY_XY_THRESHOLD_MM:
        print("  建议：旋转矩阵可疑，进入模式 2 用滑移实验重算旋转。")
    elif abs(verify_result.residual_z_dir_mm) > DEFAULT_VERIFY_Z_THRESHOLD_MM:
        print("  建议：Z 方向残差大而 X/Y 正常，是牙签伸出长度假设问题；")
        print("        量实际长度后重新运行本模式（顶部可改伸出长度）。")
    else:
        print("  建议：当前旋转矩阵通过验证，可直接使用。")


def _interactive_verify(root_dir: Path, probe_offset_G_m: np.ndarray) -> None:
    """菜单模式 1：验证旋转矩阵。"""

    use_candidate = _ask_yes_no("是否使用候选 T_I_G 矩阵验证（默认使用当前正式外参）：", False)
    if use_candidate:
        candidate = _choose_candidate_json(root_dir)
        if candidate is None:
            # 还没有候选矩阵（候选是模式 2/3 跑完后才生成的）。引导回正式外参，
            # 避免用户因为不知道这一点而卡在路径输入上。
            use_official = _ask_yes_no(
                "没有找到可用的候选矩阵（候选由模式 2/3 生成）。\n"
                "是否改用当前正式外参继续验证？",
                True,
            )
            if not use_official:
                return
            T_I_G_used, tig_description = tig.load_tig_matrix(None)
        else:
            T_I_G_used, tig_description = tig.load_tig_matrix(candidate)
    else:
        T_I_G_used, tig_description = tig.load_tig_matrix(None)

    sessions = _choose_sessions("选择验证用的 session", root_dir, 1, 1)
    if sessions is None:
        return
    demo_dir = sessions[0]
    contact_count = _ask_int("期望接触次数（回车默认 8，接触段不足时可调 6）：", DEFAULT_CONTACT_COUNT)

    print(f"\n正在验证 {demo_dir} ...")
    verify_result, solve_result = run_verify(
        demo_dir=demo_dir,
        T_I_G=T_I_G_used,
        probe_offset_G_m=probe_offset_G_m,
        contact_count=contact_count,
    )
    _print_verify_result(verify_result)
    _print_judgement_hint(verify_result)

    output_dir = _output_dir_for(demo_dir)
    summary_path = save_verify_outputs(
        output_dir=output_dir,
        verify_result=verify_result,
        T_I_G_used=T_I_G_used,
        tig_source_description=tig_description,
        probe_offset_G_m=probe_offset_G_m,
        solve_result=solve_result,
    )
    print(f"\n验证摘要已保存：{summary_path}")


# ============================================================
# 9.6 采集元信息（tig_meta.json）读取
# ============================================================


def write_tig_meta(
    session_dir: Path,
    axis: str,
    slide_mm: float,
    direction: int,
    toothpick_length_mm: float,
) -> Path:
    """向 session 文件夹写入 tig_meta.json（与采集程序 tig_verify_capture.py 格式一致）。"""

    meta = {
        "format": "tig_verify_capture_v1",
        "axis": axis,
        "slide_mm": float(slide_mm),
        "direction": int(direction),
        "toothpick_length_mm": float(toothpick_length_mm),
    }
    meta_path = Path(session_dir) / "tig_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta_path


def load_tig_session_meta(session_dir: Path) -> dict | None:
    """
    读取 session 文件夹内的 tig_meta.json（tig_verify_capture.py 写入）。

    字段：axis / slide_mm / direction（+1 或 -1）/ toothpick_length_mm。
    没有该文件时返回 None（老数据或手动整理的数据，走手动输入流程）。
    """

    meta_path = Path(session_dir) / "tig_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    axis = str(meta.get("axis", "")).lower()
    if axis not in ("z", "x", "y"):
        return None
    slide = float(meta.get("slide_mm", float("nan")))
    direction = int(meta.get("direction", 1))
    length = float(meta.get("toothpick_length_mm", float("nan")))
    if direction not in (-1, 1):
        return None
    return {
        "axis": axis,
        "slide_mm": slide,
        "direction": direction,
        "toothpick_length_mm": length,
    }


def group_meta_sessions(
    root_dir: Path,
    session_dirs: list[Path],
) -> dict[str, list[tuple[Path, dict]]]:
    """把带 tig_meta.json 的 session 按轴分组，返回 {axis: [(session, meta), ...]}。"""

    groups: dict[str, list[tuple[Path, dict]]] = {"z": [], "x": [], "y": []}
    for session_dir in session_dirs:
        meta = load_tig_session_meta(session_dir)
        if meta is not None:
            groups[meta["axis"]].append((Path(session_dir), meta))
    for axis_name in groups:
        groups[axis_name].sort(key=lambda item: item[1]["slide_mm"])
    return groups


def tip_G_from_meta(meta: dict) -> np.ndarray:
    """
    由 meta 构造该维牙签尖端在夹爪系中的偏移（米）。

    牙签沿 direction * axis 方向伸出 toothpick_length：
        z 维 direction=-1 → tip_G = (0, 0, -L)
        x 维 direction=+1 → tip_G = (L, 0, 0)
    """

    slot = {"z": 2, "x": 0, "y": 1}[meta["axis"]]
    length_m = meta["toothpick_length_mm"] / 1000.0
    tip_G = np.zeros(3, dtype=np.float64)
    tip_G[slot] = meta["direction"] * length_m
    return tip_G


def _print_slide_axis_quality(
    axis_name: str,
    direction: int,
    fit_result: SlideFitResult,
    tilt_degrees: float,
) -> None:
    """打印一根轴的滑移拟合质量指标。"""

    direction_unit = fit_result.direction_I / (np.linalg.norm(fit_result.direction_I) + 1e-12)
    column = direction * direction_unit  # 恢复为 R·e_axis（方向符号翻转）
    print(f"\n轴 {axis_name} 滑移直线拟合：")
    print(
        f"  拟合方向（I 系，单位向量）：{np.array2string(direction_unit, precision=5, separator=', ')}"
    )
    print(
        f"  旋转列（已按方向符号 {direction:+d}{axis_name.upper()} 翻转）："
        f"{np.array2string(column, precision=5, separator=', ')}"
    )
    print(f"  尺度误差 norm(direction)-1：{fit_result.scale_error:.5f}（≈0 = 尺子读数可靠）")
    print(f"  与标称轴夹角（牙签倾斜量）：{tilt_degrees:.2f} deg")
    print("  每级滑移残差（mm）：")
    for index in range(len(fit_result.lengths_m)):
        print(
            f"    L={fit_result.lengths_m[index] * 1000.0:.1f} mm: "
            f"perp={fit_result.per_point_perp_mm[index]:.3f}, "
            f"along={fit_result.per_point_along_mm[index]:.3f}"
        )
    perp_max = float(np.max(fit_result.per_point_perp_mm))
    if perp_max < 5.0:
        quality_text = "质量好（perp < 5 mm）"
    elif perp_max < 12.0:
        quality_text = "质量一般（perp 5~12 mm，建议检查插入一致性）"
    else:
        quality_text = "质量差（perp >= 12 mm，插入方向不一致或接触不稳，建议重录）"
    print(f"  判定：{quality_text}")


def _interactive_compute_rotation(root_dir: Path, probe_offset_G_m: np.ndarray) -> None:
    """
    菜单模式 2：按维度分步做多滑移直线拟合。

    流程：
        1. 先做 z 维度（至少 3 组滑移），自动读取 tig_meta.json 的滑移量与
           方向，拟合并展示质量。
        2. 确认 z 维度质量可以后，再收 x 维度（同样至少 3 组）。
        3. 两个维度都完成后组装完整旋转矩阵并保存候选 JSON。
    没有 tig_meta.json 的 session 退回手动输入滑移量与方向。
    """

    contact_count = _ask_int("期望接触次数（回车默认 8）：", DEFAULT_CONTACT_COUNT)
    current_T_I_G, tig_description = tig.load_tig_matrix(None)

    candidates = find_session_candidates(root_dir)
    if not candidates:
        print(f"  {root_dir} 下没有找到任何包含 odometry.csv 的 session 文件夹。")
        return
    meta_groups = group_meta_sessions(root_dir, candidates)

    print("\n带采集元信息的 session 分组（来自 tig_verify_capture.py）：")
    any_meta = False
    for axis_name in ("z", "x", "y"):
        entries = meta_groups[axis_name]
        if not entries:
            continue
        any_meta = True
        direction = entries[0][1]["direction"]
        length = entries[0][1]["toothpick_length_mm"]
        slides = ", ".join(f"{entry[1]['slide_mm']:.0f}mm" for entry in entries)
        print(
            f"  {axis_name} 维度：{len(entries)} 组（方向 "
            f"{'+' if direction > 0 else '-'}{axis_name.upper()}，牙签 {length:.1f}mm，"
            f"滑移量 {slides}）"
        )
    if not any_meta:
        print("  （没有找到 tig_meta.json，将用手动输入模式）")

    # ---- 维度 1：z ----
    print("\n=== 维度 1：z 轴滑移（牙签沿 -Z 向下伸） ===")
    z_data = _collect_axis_group(root_dir, candidates, "z", contact_count, meta_groups["z"])
    if z_data is None:
        return

    z_sessions, z_lengths, z_direction, z_tip_G, z_fit = z_data
    z_tilt = float(
        np.degrees(
            np.arccos(
                np.clip(
                    abs((z_fit.direction_I / (np.linalg.norm(z_fit.direction_I) + 1e-12))[2]),
                    0.0,
                    1.0,
                )
            )
        )
    )
    _print_slide_axis_quality("z", z_direction, z_fit, z_tilt)
    if not _ask_yes_no("z 维度质量可以吗？（不可以则重录 z 维度后重跑本模式）", True):
        print("  已取消，请重录 z 维度后再运行模式 2。")
        return

    # ---- 维度 2：x ----
    print("\n=== 维度 2：x 轴滑移（牙签沿 +X 向前伸） ===")
    x_data = _collect_axis_group(root_dir, candidates, "x", contact_count, meta_groups["x"])
    if x_data is None:
        print(
            "  没有 x 维度数据，无法组装完整矩阵。候选 JSON 只会记录 z 列方向。"
        )
        x_fit = None
    else:
        x_sessions, x_lengths, x_direction, x_tip_G, x_fit = x_data
        x_tilt = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        abs((x_fit.direction_I / (np.linalg.norm(x_fit.direction_I) + 1e-12))[0]),
                        0.0,
                        1.0,
                    )
                )
            )
        )
        _print_slide_axis_quality("x", x_direction, x_fit, x_tilt)
        if not _ask_yes_no("x 维度质量可以吗？（不可以则重录 x 维度后重跑本模式）", True):
            print("  已取消，请重录 x 维度后再运行模式 2。")
            return

    # ---- 组装 ----
    axis_sessions: dict[str, list[str]] = {}
    axis_lengths_mm: dict[str, list[float]] = {}
    axis_directions: dict[str, int] = {}
    for axis_name, fit_data in (("z", z_data), ("x", x_data)):
        if fit_data is None:
            continue
        sessions, lengths, direction, tip_G, _fit = fit_data
        axis_sessions[axis_name] = [str(path) for path in sessions]
        axis_lengths_mm[axis_name] = [float(value) for value in lengths]
        axis_directions[axis_name] = direction
    if not axis_sessions:
        return

    print("\n开始组装旋转矩阵 ...")
    candidate = run_compute_rotation_with_two_axes(
        axis_sessions=axis_sessions,
        axis_lengths_mm=axis_lengths_mm,
        axis_directions=axis_directions,
        current_T_I_G=current_T_I_G,
        probe_offset_G_m=probe_offset_G_m,
        contact_count=contact_count,
        translation_tip_G_m=z_tip_G,
    )
    first_session = axis_sessions[list(axis_sessions.keys())[0]][0]
    output_dir = _output_dir_for(first_session)
    candidate_path = save_rotation_candidate_json(
        output_dir=output_dir,
        candidate=candidate,
        source_T_I_G=current_T_I_G,
        source_description=tig_description,
        probe_offset_G_m=probe_offset_G_m,
    )
    print(f"\n候选矩阵已保存：{candidate_path}")
    if candidate.T_I_G_candidate is not None:
        print("下一步：用一条全新 session 运行模式 1，选择该候选 JSON 做独立验证。")
    else:
        print("提示：只有 z 轴数据，T_I_G_candidate 为 null；补 x 维度后重跑即可组合完整矩阵。")


def _collect_axis_group(
    root_dir: Path,
    candidates: list[Path],
    axis_name: str,
    contact_count: int,
    meta_entries: list[tuple[Path, dict]],
) -> tuple[list[Path], list[float], int, np.ndarray] | None:
    """
    收集一根轴的多级滑移数据：优先读 tig_meta.json，否则手动输入。

    返回 (session_dirs, lengths_mm, direction, tip_G, fit_result) 或 None（用户取消）。
    """

    meta_entries = list(meta_entries)
    if len(meta_entries) >= DEFAULT_MIN_SLIDE_SESSIONS:
        print(
            f"  自动读取 {axis_name} 维度 {len(meta_entries)} 组元信息"
            f"（滑移量 {[entry[1]['slide_mm'] for entry in meta_entries]} mm，"
            f"方向 {meta_entries[0][1]['direction']:+d}）。"
        )
        sessions = [entry[0] for entry in meta_entries]
        lengths = [entry[1]["slide_mm"] for entry in meta_entries]
        direction = meta_entries[0][1]["direction"]
        if any(entry[1]["direction"] != direction for entry in meta_entries):
            print(f"  警告：{axis_name} 维度内各 session 的滑移方向不一致，已用第一个。")
        tip_G = tip_G_from_meta(meta_entries[0][1])
        if not _ask_yes_no(f"  使用这 {len(meta_entries)} 组 {axis_name} 数据？", True):
            print("  跳过。")
            return None
        fit_result = fit_slide_axis(
            session_dirs=sessions,
            lengths_mm=lengths,
            detection_T_I_G=tig.load_tig_matrix(None)[0],
            detection_probe_offset_G_m=np.asarray(DEFAULT_PROBE_OFFSET_G_M),
            contact_count=contact_count,
        )
        return sessions, lengths, direction, tip_G, fit_result

    print(f"  {axis_name} 维度可用 session（手动输入滑移量）：")
    sessions = _choose_sessions(
        f"选择 {axis_name} 轴滑移的 session（按累计滑移量从小到大）",
        root_dir,
        DEFAULT_MIN_SLIDE_SESSIONS,
    )
    if sessions is None:
        return None
    lengths = _ask_csv_floats(
        f"依次输入 {len(sessions)} 个 session 的累计滑移量 mm（逗号分隔，如 0,10,20,30）："
    )
    if len(lengths) != len(sessions):
        print(f"  滑移量数量（{len(lengths)}）与 session 数量（{len(sessions)}）不一致，已取消。")
        return None
    direction = DEFAULT_DIRECTION[axis_name] if axis_name in DEFAULT_DIRECTION else 1
    direction = -direction if not _ask_yes_no(
        f"牙签是否沿 {'+' if direction > 0 else '-'}{axis_name.upper()} 方向延伸？", True
    ) else direction
    length_mm = _ask_float(
        f"{axis_name} 维牙签尖端伸出长度 mm（回车默认 "
        f"{DEFAULT_PROBE_OFFSET_G_M[2] * 1000.0 if axis_name == 'z' else 25.0}）：",
        DEFAULT_PROBE_OFFSET_G_M[2] * 1000.0 if axis_name == "z" else 25.0,
    )
    tip_G = np.zeros(3, dtype=np.float64)
    tip_G[{"z": 2, "x": 0, "y": 1}[axis_name]] = direction * length_mm / 1000.0
    fit_result = fit_slide_axis(
        session_dirs=sessions,
        lengths_mm=lengths,
        detection_T_I_G=tig.load_tig_matrix(None)[0],
        detection_probe_offset_G_m=np.asarray(DEFAULT_PROBE_OFFSET_G_M),
        contact_count=contact_count,
    )
    return sessions, lengths, direction, tip_G, fit_result


def _interactive_kabsch(root_dir: Path, probe_offset_G_m: np.ndarray) -> None:
    """菜单模式 3：Kabsch 整体配准。"""

    tip_count = _ask_int(
        "尖端点数（建议 4 个以上、明显非共面，回车默认 4）：",
        DEFAULT_MIN_KABSCH_TIPS + 1,
    )
    sessions = _choose_sessions(
        "选择每个尖端对应的 session",
        root_dir,
        tip_count,
        tip_count,
    )
    if sessions is None:
        return

    tip_g_list: list[list[float]] = []
    for index, session in enumerate(sessions):
        tip_mm = _ask_csv_floats(
            f"尖端 {index + 1} 在夹爪系中的绝对偏移 x,y,z（mm，逗号分隔，如 0,0,31.93）："
        )
        if len(tip_mm) != 3:
            print("  偏移需要 x,y,z 三个值，已取消。")
            return
        tip_g_list.append([value / 1000.0 for value in tip_mm])

    contact_count = _ask_int("期望接触次数（回车默认 8）：", DEFAULT_CONTACT_COUNT)
    T_I_G_used, tig_description = tig.load_tig_matrix(None)

    rotation_fit, translation_fit, _, residuals, rotation_diff, fit_rms = run_kabsch(
        kabsch_sessions=[str(path) for path in sessions],
        tip_g_list=tip_g_list,
        current_T_I_G=T_I_G_used,
        probe_offset_G_m=probe_offset_G_m,
        contact_count=contact_count,
    )
    print("\nKabsch 整体配准结果：")
    print(f"  R_I_G（与当前矩阵夹角 {rotation_diff:.2f} deg）：")
    for row in rotation_fit:
        print("   " + np.array2string(row, precision=6, separator=", "))
    print(
        "  t_I_G（mm）："
        f"{np.array2string(translation_fit * 1000.0, precision=3, separator=', ')}"
    )
    print(f"  拟合 RMS（mm）：{fit_rms:.3f}")
    print(
        "  逐点残差（mm）："
        f"{np.array2string(np.linalg.norm(residuals, axis=1) * 1000.0, precision=3, separator=', ')}"
    )

    candidate_matrix = np.eye(4, dtype=np.float64)
    candidate_matrix[:3, :3] = rotation_fit
    candidate_matrix[:3, 3] = translation_fit
    candidate = RotationCandidate(
        T_I_G_candidate=candidate_matrix,
        columns_I={},
        column_source_axes={},
        column_scale_errors={},
        column_tilt_degrees={},
        axis_pair_angle_degrees=None,
        used_slide_lengths_mm={},
        per_slide_perp_mm={},
        translation_source="Umeyama 整体配准得到的 t_I_G（包含牙签尖端绝对偏移约束）。",
    )
    output_dir = _output_dir_for(sessions[0])
    candidate_path = save_rotation_candidate_json(
        output_dir=output_dir,
        candidate=candidate,
        source_T_I_G=T_I_G_used,
        source_description=tig_description,
        probe_offset_G_m=probe_offset_G_m,
    )
    print(f"\n候选矩阵已保存：{candidate_path}")
    print("下一步：用一条全新 session 运行模式 1，选择该候选 JSON 做独立验证。")


def _interactive_line_check(root_dir: Path, probe_offset_G_m: np.ndarray) -> None:
    """菜单模式 4：画线自检。"""

    sessions = _choose_sessions(
        "选择沿直线触碰的 3 个以上点（每个点一个 session）",
        root_dir,
        3,
    )
    if sessions is None:
        return
    spacings = _ask_csv_floats(
        f"相邻点间距 mm（{len(sessions) - 1} 个，逗号分隔，如 50,50）："
    )
    if len(spacings) != len(sessions) - 1:
        print(f"  间距数量（{len(spacings)}）应为 {len(sessions) - 1}，已取消。")
        return

    contact_count = _ask_int("期望接触次数（回车默认 8）：", DEFAULT_CONTACT_COUNT)
    T_I_G_used, _ = tig.load_tig_matrix(None)
    result = run_line_check(
        line_sessions=[str(path) for path in sessions],
        line_spacings_mm=spacings,
        current_T_I_G=T_I_G_used,
        probe_offset_G_m=probe_offset_G_m,
        contact_count=contact_count,
    )
    output_dir = _output_dir_for(sessions[0])
    path = save_line_check_txt(output_dir, result)
    print(f"\n自检结果已保存：{path}")


def interactive_main() -> int:
    """交互式菜单主循环。"""

    print("=" * 60)
    print("T_I_G 旋转矩阵验证与实验标定（交互式）")
    print("=" * 60)
    root_raw = _ask(f"数据根目录（回车默认 {DEFAULT_INTERACTIVE_ROOT}）：", str(DEFAULT_INTERACTIVE_ROOT))
    stray_root = Path(root_raw).expanduser()
    probe_z_cm = _ask_float(
        f"牙签伸出长度 cm（回车默认 {DEFAULT_PROBE_OFFSET_G_M[2] * 100.0:.3f}，实测 31.93mm 直接回车）：",
        DEFAULT_PROBE_OFFSET_G_M[2] * 100.0,
    )
    probe_offset_G_m = np.array([0.0, 0.0, probe_z_cm / 100.0], dtype=np.float64)

    # 启动时提示当前数据根目录下有哪些 session，避免"数据放没放对位置"的困惑。
    session_candidates = find_session_candidates(stray_root)
    if session_candidates:
        print(f"\n数据根目录：{stray_root}（找到 {len(session_candidates)} 个 session）")
        print("  " + ", ".join(f"[{index + 1}]{candidate.name}" for index, candidate in enumerate(session_candidates)))
    else:
        print(f"\n注意：{stray_root} 下没有找到任何包含 odometry.csv 的 session 文件夹。")
        print("请先把录好的 Stray 数据放进数据根目录，否则模式 1/2/3/4 无法运行。")

    while True:
        print("\n" + "=" * 60)
        print("请选择模式：")
        print("  1. 验证旋转矩阵（固定点接触）")
        print("  2. 算旋转矩阵（多滑移直线拟合）")
        print("  3. Kabsch 整体配准（已知尖端绝对偏移）")
        print("  4. 画线自检（检验实验管线，无需知道旋转）")
        print("  5. 合成数据自测（无需硬件）")
        print("  0. 退出")
        choice = _ask("输入数字后回车：")
        try:
            if choice == "1":
                _interactive_verify(stray_root, probe_offset_G_m)
            elif choice == "2":
                _interactive_compute_rotation(stray_root, probe_offset_G_m)
            elif choice == "3":
                _interactive_kabsch(stray_root, probe_offset_G_m)
            elif choice == "4":
                _interactive_line_check(stray_root, probe_offset_G_m)
            elif choice == "5":
                run_synthetic(None)
            elif choice == "0":
                print("再见。")
                return 0
            else:
                print("无效选择，请输入 0-5。")
        except KeyboardInterrupt:
            print("\n已取消，返回菜单。")
        except Exception as error:
            print(f"\n执行失败：{type(error).__name__}: {error}")
            print("已返回菜单，可根据上面的提示调整后重试。")


def main() -> int:
    """完整流程入口。"""

    # 不带任何命令行参数时进入交互式菜单；带参数时走原命令行流程。
    if len(sys.argv) <= 1:
        return interactive_main()

    args = parse_args()
    T_I_G_used, tig_source_description, probe_offset_G_m = _resolve_tig_and_offset(args)

    if args.contact_count < 1:
        raise ValueError("--contact-count 必须大于 0。")

    mode_count = sum(
        [
            args.verify_only,
            args.compute_rotation,
            args.kabsch,
            args.line_check,
            args.synthetic,
        ]
    )
    if args.synthetic:
        return run_synthetic(args.tig_json)
    if mode_count > 1:
        raise ValueError("一次只能使用一个模式。")

    if args.compute_rotation:
        axis_configs: dict[str, list[str]] = {}
        axis_lengths: dict[str, list[float]] = {}
        for axis_name in ("z", "x", "y"):
            sessions_value = getattr(args, f"{axis_name}_sessions")
            lengths_value = getattr(args, f"{axis_name}_lengths_mm")
            if (sessions_value is None) != (lengths_value is None):
                raise ValueError(
                    f"--{axis_name}-sessions 与 --{axis_name}-lengths-mm 必须同时提供。"
                )
            if sessions_value is not None:
                axis_configs[axis_name] = _parse_comma_separated(sessions_value)
                axis_lengths[axis_name] = [
                    float(item) for item in _parse_comma_separated(lengths_value)
                ]
                if len(axis_configs[axis_name]) < DEFAULT_MIN_SLIDE_SESSIONS:
                    raise ValueError(
                        f"轴 {axis_name} 至少需要 {DEFAULT_MIN_SLIDE_SESSIONS} 级滑移。"
                    )
        if not axis_configs:
            raise ValueError("--compute-rotation 需要至少一个滑移轴（--z-sessions 等）。")

        # CLI 路径：方向符号用默认约定（z 沿 -Z、x/y 沿 +），
        # 与采集程序的默认一致；交互菜单会按 tig_meta.json 传实际方向。
        candidate = run_compute_rotation_with_two_axes(
            axis_sessions=axis_configs,
            axis_lengths_mm=axis_lengths,
            axis_directions={
                axis_name: DEFAULT_DIRECTION[axis_name] for axis_name in axis_configs
            },
            current_T_I_G=T_I_G_used,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=args.contact_count,
        )
        first_session = next(iter(axis_configs.values()))[0]
        output_dir = _output_dir_for(first_session)
        candidate_path = save_rotation_candidate_json(
            output_dir=output_dir,
            candidate=candidate,
            source_T_I_G=T_I_G_used,
            source_description=tig_source_description,
            probe_offset_G_m=probe_offset_G_m,
        )
        print(f"\n候选矩阵已保存：{candidate_path}")
        print("注意：候选矩阵未经独立验证，请用全新 session 跑 --verify-only --tig-json 验证。")
        return 0

    if args.kabsch:
        if args.kabsch_sessions is None or args.tip_g is None:
            raise ValueError("--kabsch 需要 --kabsch-sessions 与 --tip-g。")
        kabsch_session_list = _parse_comma_separated(args.kabsch_sessions)
        rotation_fit, translation_fit, _, residuals, rotation_diff, fit_rms = run_kabsch(
            kabsch_sessions=kabsch_session_list,
            tip_g_list=args.tip_g,
            current_T_I_G=T_I_G_used,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=args.contact_count,
        )
        print("\nKabsch 整体配准结果：")
        print(f"  R_I_G（与当前矩阵夹角 {rotation_diff:.2f} deg）：")
        for row in rotation_fit:
            print("   " + np.array2string(row, precision=6, separator=", "))
        print(f"  t_I_G（mm）：{np.array2string(translation_fit * 1000.0, precision=3, separator=', ')}")
        print(f"  拟合 RMS（mm）：{fit_rms:.3f}")
        print(f"  逐点残差（mm）：{np.array2string(np.linalg.norm(residuals, axis=1) * 1000.0, precision=3, separator=', ')}")

        candidate_matrix = np.eye(4, dtype=np.float64)
        candidate_matrix[:3, :3] = rotation_fit
        candidate_matrix[:3, 3] = translation_fit
        candidate = RotationCandidate(
            T_I_G_candidate=candidate_matrix,
            columns_I={},
            column_source_axes={},
            column_scale_errors={},
            column_tilt_degrees={},
            axis_pair_angle_degrees=None,
            used_slide_lengths_mm={},
            per_slide_perp_mm={},
            translation_source="Umeyama 整体配准得到的 t_I_G（包含牙签尖端绝对偏移约束）。",
        )
        output_dir = _output_dir_for(kabsch_session_list[0])
        candidate_path = save_rotation_candidate_json(
            output_dir=output_dir,
            candidate=candidate,
            source_T_I_G=T_I_G_used,
            source_description=tig_source_description,
            probe_offset_G_m=probe_offset_G_m,
        )
        print(f"\n候选矩阵已保存：{candidate_path}")
        print("注意：候选矩阵未经独立验证，请用全新 session 跑 --verify-only --tig-json 验证。")
        return 0

    if args.line_check:
        if args.line_sessions is None or args.line_spacings_mm is None:
            raise ValueError("--line-check 需要 --line-sessions 与 --line-spacings-mm。")
        line_session_list = _parse_comma_separated(args.line_sessions)
        line_spacings = [
            float(item) for item in _parse_comma_separated(args.line_spacings_mm)
        ]
        result = run_line_check(
            line_sessions=line_session_list,
            line_spacings_mm=line_spacings,
            current_T_I_G=T_I_G_used,
            probe_offset_G_m=probe_offset_G_m,
            contact_count=args.contact_count,
        )
        output_dir = _output_dir_for(line_session_list[0])
        path = save_line_check_txt(output_dir, result)
        print(f"\n自检结果已保存：{path}")
        return 0

    # 默认：--verify-only
    if args.demo_dir is None:
        demo_dir = stray.find_latest_demo_dir_containing_odometry(
            root_dir=args.stray_root,
            output_dir_name=stray.DEFAULT_OUTPUT_DIR_NAME,
        )
    else:
        demo_dir = Path(args.demo_dir)
    verify_result, solve_result = run_verify(
        demo_dir=demo_dir,
        T_I_G=T_I_G_used,
        probe_offset_G_m=probe_offset_G_m,
        contact_count=args.contact_count,
    )
    _print_verify_result(verify_result)

    output_dir = _output_dir_for(demo_dir)
    summary_path = save_verify_outputs(
        output_dir=output_dir,
        verify_result=verify_result,
        T_I_G_used=T_I_G_used,
        tig_source_description=tig_source_description,
        probe_offset_G_m=probe_offset_G_m,
        solve_result=solve_result,
    )
    print(f"\n验证摘要已保存：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
