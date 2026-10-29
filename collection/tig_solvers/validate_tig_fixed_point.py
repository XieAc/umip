"""
T_I_G 固定点接触验证与 TCP 平移微调脚本。

用途：
    1. 自动找到最新 session 里的 Stray Scanner odometry.csv。
    2. 使用 stray_shift_position.py 里当前的 T_I_G 定标矩阵。
    3. 根据牙签尖端相对 TCP 的已知偏移，计算每一帧牙签尖端世界坐标。
    4. 自动寻找 10 段“低速、稳定、持续约 1 秒”的固定点接触片段。
    5. 输出每次接触的牙签尖端平均位置、误差统计和可视化图片。
    6. 可选：保持 T_I_G 旋转不变，仅最小二乘微调 TCP 平移，并保存候选矩阵。

推荐采集方式：
    1. 打开 Stray Scanner 录制。
    2. 用固定在夹爪上的牙签尖端轻轻碰桌面固定点，保持约 1 秒。
    3. 移开，换姿态，再碰同一个点。
    4. 重复 10 次左右。
    5. 将 iPhone/Stray 导出的文件夹放入对应的 data_MMDD_HHMM session 文件夹。
    6. 运行本脚本。

运行方式：
    python validate_tig_fixed_point.py --fit-translation

可选参数示例：
    python validate_tig_fixed_point.py --demo-dir "E:\\机器人\\umi\\Stray数据\\data_0621_1922"
    python validate_tig_fixed_point.py --contact-count 10
    python validate_tig_fixed_point.py --max-stable-speed 0.008
    python validate_tig_fixed_point.py --min-contact-duration 0.7
    python validate_tig_fixed_point.py --probe-offset-z -0.03193
    python validate_tig_fixed_point.py --tig-json "...\\T_I_G_candidate.json"
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import stray_shift_position as stray


# ============================================================
# 1. 默认参数
# ============================================================

# 输出目录名称。它会创建在当前 session 文件夹下。
DEFAULT_OUTPUT_DIR_NAME = "tig_fixed_point_validation"

# 默认要求验证 10 次接触。
DEFAULT_CONTACT_COUNT = 10

# 牙签接触端相对“原夹爪 TCP”的默认偏移，单位 m，使用夹爪/TCP 坐标系 G 表达。
# 当前默认假设牙签从 TCP 沿 +Z_G 方向伸出 3.193 cm（实测 31.93 mm）。运行时会打印该向量；
# 如果实际安装方向相反，应传 --probe-offset-z -0.03193，而不是修改 T_I_G。
DEFAULT_PROBE_OFFSET_G_M = (0.0, 0.0, 0.03193)

# 接触段至少持续多久。你每次保持约 1 秒，所以默认用 0.6 秒过滤掉短暂停顿。
DEFAULT_MIN_CONTACT_DURATION_SECONDS = 0.6

# 自动检测稳定接触段的速度阈值，单位 m/s。
# 牙签确实压在固定点并保持静止时，尖端速度应非常低。
# 原来的 0.15 m/s 会把普通减速和短暂停顿误当成接触，因此默认收紧到 0.02 m/s。
# 如果检测不到足够接触，可先确认每次保持满 1 秒，再小幅调大到 0.03。
DEFAULT_MAX_STABLE_SPEED_M_PER_S = 0.02

# 说明：
# 这里检测的是牙签尖端世界坐标速度，不是 iPhone 本体平移速度。
# 手持验证时只要有轻微转腕，iPhone 到牙签尖端的长力臂就会产生位置速度，
# 因此阈值不能压得过低；0.02 m/s 是“减少误检”和“允许轻微手抖”的折中。

# 相邻接触段之间至少间隔多久，避免同一次接触被切成两段。
DEFAULT_MIN_GAP_BETWEEN_CONTACTS_SECONDS = 0.35

# 自动检测时，优先从每个稳定片段中间取这一段时间计算平均位置。
DEFAULT_CENTER_WINDOW_SECONDS = 0.5

# TCP 平移微调至少需要若干个方向明显不同的接触姿态；姿态相同只会得到秩亏方程。
DEFAULT_MIN_FIT_CONTACT_COUNT = 6

# 单个接触均值偏离拟合固定点超过该值时，视为可能没有点准纸面标记。
DEFAULT_OUTLIER_FLOOR_MM = 8.0

# 本工具定位为“微调”。单次建议修正超过 10 mm 通常意味着牙签偏移方向、接触段或数据有误。
DEFAULT_MAX_TRANSLATION_CORRECTION_MM = 10.0


@dataclass(frozen=True)
class ContactSegment:
    """保存一次固定点接触片段的帧范围和时间范围。"""

    contact_index: int
    start_index: int
    end_index_exclusive: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass(frozen=True)
class ContactPointResult:
    """保存一次接触的牙签尖端位置统计。"""

    contact_index: int
    start_index: int
    end_index_exclusive: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    frame_count: int
    mean_x_m: float
    mean_y_m: float
    mean_z_m: float
    std_x_mm: float
    std_y_mm: float
    std_z_mm: float
    within_segment_rms_mm: float
    error_to_global_mean_mm: float


@dataclass(frozen=True)
class TranslationFitResult:
    """保存固定点最小二乘得到的 TCP 平移候选值与质量统计。"""

    candidate_T_I_G: np.ndarray
    fixed_point_world_m: np.ndarray
    used_contact_indices: tuple[int, ...]
    rejected_contact_indices: tuple[int, ...]
    current_errors_mm: np.ndarray
    candidate_errors_mm: np.ndarray
    matrix_rank: int
    condition_number: float
    correction_mm: np.ndarray
    safe_correction_magnitude: bool


# ============================================================
# 2. TCP 轨迹与稳定片段检测
# ============================================================


def build_process_paths_without_output_guard(
    stray_root: Path,
    explicit_demo_dir: Path | None,
) -> stray.ProcessPaths:
    """
    构造输入/输出路径。

    这个验证脚本允许反复运行，因此不使用 stray_shift_position.py 的 processed_umi 防重复保护。
    但 odometry.csv 搜索逻辑仍然复用同一套 session 查找规则。
    """

    demo_dir = (
        explicit_demo_dir
        if explicit_demo_dir is not None
        else stray.find_latest_demo_dir_containing_odometry(
            root_dir=stray_root,
            output_dir_name=stray.DEFAULT_OUTPUT_DIR_NAME,
        )
    )
    demo_dir = demo_dir.resolve()

    if not demo_dir.exists():
        raise FileNotFoundError(f"指定的 session 文件夹不存在：{demo_dir}")

    if not demo_dir.is_dir():
        raise NotADirectoryError(f"指定的 session 路径不是文件夹：{demo_dir}")

    odometry_csv_path = stray.find_best_odometry_csv(
        demo_dir=demo_dir,
        output_dir_name=stray.DEFAULT_OUTPUT_DIR_NAME,
    )
    output_dir = demo_dir / DEFAULT_OUTPUT_DIR_NAME
    test_dir = output_dir / "test"

    return stray.ProcessPaths(
        demo_dir=demo_dir,
        odometry_csv_path=odometry_csv_path,
        output_dir=output_dir,
        test_dir=test_dir,
    )


def validate_tig_matrix(T_I_G: np.ndarray, source_description: str) -> np.ndarray:
    """严格检查外参矩阵，避免错误 JSON 或非刚体矩阵进入微调计算。"""

    transform = np.asarray(T_I_G, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            f"{source_description} 中的 T_I_G 形状必须是 (4, 4)，当前为 {transform.shape}。"
        )
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{source_description} 中的 T_I_G 包含 NaN 或无穷值。")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{source_description} 中的 T_I_G 最后一行不是 [0, 0, 0, 1]。")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{source_description} 中的 T_I_G 旋转矩阵不正交。")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{source_description} 中的 T_I_G 旋转矩阵行列式不为 +1。")

    return transform.copy()


def load_tig_matrix(tig_json_path: Path | None) -> tuple[np.ndarray, str]:
    """
    加载本次验证使用的 T_I_G。

    不传 JSON 时使用 stray_shift_position.py 中当前正式矩阵；传入候选 JSON 时，
    优先读取 T_I_G_candidate，便于用一条全新的 session 做独立验证。
    """

    if tig_json_path is None:
        return (
            validate_tig_matrix(stray.get_calibrated_T_I_G(), "stray_shift_position.py"),
            "stray_shift_position.py current calibration",
        )

    resolved_path = tig_json_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"指定的 T_I_G JSON 不存在：{resolved_path}")

    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        matrix_data = data.get("T_I_G_candidate", data.get("T_I_G"))
    else:
        matrix_data = data
    if matrix_data is None:
        raise KeyError(
            f"{resolved_path} 中没有 T_I_G_candidate 或 T_I_G 字段。"
        )

    return validate_tig_matrix(np.asarray(matrix_data), str(resolved_path)), str(resolved_path)


def calculate_probe_positions_from_transforms(
    T_W_I: np.ndarray,
    T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
) -> np.ndarray:
    """
    计算牙签接触端的世界坐标。

    probe_offset_G_m 是“原 TCP -> 牙签接触端”的向量，使用夹爪坐标系 G 表达。
    牙签只是临时探针，最终拟合仍更新 T_I_G 的 TCP 平移，不会把牙签端当成正式 TCP。
    """

    probe_offset = np.asarray(probe_offset_G_m, dtype=np.float64)
    if probe_offset.shape != (3,) or not np.all(np.isfinite(probe_offset)):
        raise ValueError("probe_offset_G_m 必须是包含 3 个有限数值的向量。")

    T_W_G = stray.compute_tcp_pose_sequence(T_W_I=T_W_I, T_I_G=T_I_G)
    return (
        T_W_G[:, :3, 3]
        + np.einsum("nij,j->ni", T_W_G[:, :3, :3], probe_offset)
    )


def load_probe_trajectory_from_odometry(
    odometry_csv_path: Path,
    T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 odometry.csv 加载 iPhone 位姿，并计算牙签接触端世界轨迹。

    返回：
        timestamps_s:       [N]
        T_W_I:              [N, 4, 4]
        probe_positions_w:  [N, 3]
    """

    odometry_data = stray.load_odometry_csv(odometry_csv_path)
    T_W_I, timestamps_s = stray.build_iphone_pose_sequence(odometry_data)
    probe_positions_w_m = calculate_probe_positions_from_transforms(
        T_W_I=T_W_I,
        T_I_G=T_I_G,
        probe_offset_G_m=probe_offset_G_m,
    )
    return timestamps_s, T_W_I, probe_positions_w_m


def calculate_tcp_positions_from_odometry(odometry_csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """保留旧接口：使用零探针偏移计算正式 TCP 轨迹。"""

    T_I_G = stray.get_calibrated_T_I_G()
    timestamps_s, _, tcp_positions_w_m = load_probe_trajectory_from_odometry(
        odometry_csv_path=odometry_csv_path,
        T_I_G=T_I_G,
        probe_offset_G_m=np.zeros(3, dtype=np.float64),
    )
    return timestamps_s, tcp_positions_w_m


def calculate_motion_speed_m_per_s(
    timestamps_s: np.ndarray,
    positions_m: np.ndarray,
) -> np.ndarray:
    """
    计算每一帧附近的牙签尖端平移速度，单位 m/s。

    输出长度和 positions_m 一致：
        speed[i] 表示第 i 帧附近的速度估计。
    """

    if len(timestamps_s) != len(positions_m):
        raise ValueError(
            "timestamps_s 与 positions_m 长度不一致："
            f"{len(timestamps_s)} vs {len(positions_m)}"
        )

    if len(timestamps_s) < 2:
        raise ValueError("帧数太少，无法计算速度。")

    dt = np.diff(timestamps_s)
    displacement = np.linalg.norm(np.diff(positions_m, axis=0), axis=1)

    safe_dt = np.where(dt > 1e-9, dt, np.nan)
    segment_speed = displacement / safe_dt

    # 把相邻帧速度扩展成逐帧速度，首尾使用最近的速度，中间取左右平均。
    speed = np.empty(len(timestamps_s), dtype=np.float64)
    speed[0] = segment_speed[0]
    speed[-1] = segment_speed[-1]
    if len(timestamps_s) > 2:
        speed[1:-1] = 0.5 * (segment_speed[:-1] + segment_speed[1:])

    return np.nan_to_num(speed, nan=np.inf, posinf=np.inf, neginf=np.inf)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    将布尔 mask 转成连续 True 区间。

    返回：
        [(start_index, end_index_exclusive), ...]
    """

    runs: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, value in enumerate(mask):
        if value and start_index is None:
            start_index = index
        elif not value and start_index is not None:
            runs.append((start_index, index))
            start_index = None

    if start_index is not None:
        runs.append((start_index, len(mask)))

    return runs


def trim_segment_to_center_window(
    start_index: int,
    end_index_exclusive: int,
    timestamps_s: np.ndarray,
    center_window_seconds: float,
) -> tuple[int, int]:
    """
    从稳定片段中间截取一个更干净的窗口。

    接触刚开始和刚结束时常有轻微滑动，中间部分通常更能代表真正接触点。
    如果片段本身比 center_window_seconds 短，就保留整个片段。
    """

    if center_window_seconds <= 0:
        return start_index, end_index_exclusive

    start_time = float(timestamps_s[start_index])
    end_time = float(timestamps_s[end_index_exclusive - 1])
    duration = end_time - start_time
    if duration <= center_window_seconds:
        return start_index, end_index_exclusive

    center_time = 0.5 * (start_time + end_time)
    window_start_time = center_time - 0.5 * center_window_seconds
    window_end_time = center_time + 0.5 * center_window_seconds

    trimmed_start = int(np.searchsorted(timestamps_s, window_start_time, side="left"))
    trimmed_end = int(np.searchsorted(timestamps_s, window_end_time, side="right"))
    trimmed_start = max(start_index, min(trimmed_start, end_index_exclusive - 1))
    trimmed_end = max(trimmed_start + 1, min(trimmed_end, end_index_exclusive))

    return trimmed_start, trimmed_end


def detect_contact_segments(
    timestamps_s: np.ndarray,
    positions_m: np.ndarray,
    contact_count: int,
    max_stable_speed_m_per_s: float,
    min_contact_duration_s: float,
    min_gap_between_contacts_s: float,
    center_window_seconds: float,
) -> tuple[list[ContactSegment], np.ndarray]:
    """
    自动检测固定点接触片段。

    检测逻辑：
        1. 根据 TCP 世界坐标计算速度。
        2. 速度低于阈值的帧视为“稳定”。
        3. 连续稳定时间超过 min_contact_duration_s 才作为接触候选。
        4. 候选按时间排序，取前 contact_count 段。
    """

    if contact_count <= 0:
        raise ValueError(f"contact_count 必须大于 0，当前是：{contact_count}")

    speed_m_per_s = calculate_motion_speed_m_per_s(timestamps_s, positions_m)
    stable_mask = speed_m_per_s <= max_stable_speed_m_per_s
    stable_runs = contiguous_true_runs(stable_mask)

    segments: list[ContactSegment] = []
    last_end_time = -np.inf

    for start_index, end_index_exclusive in stable_runs:
        if end_index_exclusive <= start_index:
            continue

        raw_start_time = float(timestamps_s[start_index])
        raw_end_time = float(timestamps_s[end_index_exclusive - 1])
        raw_duration = raw_end_time - raw_start_time
        if raw_duration < min_contact_duration_s:
            continue

        if raw_start_time - last_end_time < min_gap_between_contacts_s:
            continue

        trimmed_start, trimmed_end = trim_segment_to_center_window(
            start_index=start_index,
            end_index_exclusive=end_index_exclusive,
            timestamps_s=timestamps_s,
            center_window_seconds=center_window_seconds,
        )
        start_time = float(timestamps_s[trimmed_start])
        end_time = float(timestamps_s[trimmed_end - 1])

        segments.append(
            ContactSegment(
                contact_index=len(segments) + 1,
                start_index=trimmed_start,
                end_index_exclusive=trimmed_end,
                start_time_s=start_time,
                end_time_s=end_time,
                duration_s=end_time - start_time,
            )
        )
        last_end_time = raw_end_time

        if len(segments) >= contact_count:
            break

    if len(segments) < contact_count:
        raise RuntimeError(
            f"只检测到 {len(segments)} 段稳定接触，少于要求的 {contact_count} 段。\n"
            f"当前速度阈值：{max_stable_speed_m_per_s} m/s，"
            f"最短持续时间：{min_contact_duration_s} s。\n"
            "可以尝试：\n"
            "  1. 采集时每次接触保持更久、更稳。\n"
            "  2. 运行时小幅调大 --max-stable-speed，例如 0.03。\n"
            "  3. 运行时调小 --min-contact-duration，例如 0.4。"
        )

    return segments, speed_m_per_s


DROP_HIGHEST_VARIANCE_CONTACT = False


def discard_highest_variance_contact(segments, positions_m):
    """Drop one contact using within-window total positional variance (mm^2)."""
    if len(segments) < 4:
        raise ValueError("Need at least four contacts to discard one and retain three.")
    variances = []
    for segment in segments:
        points = np.asarray(positions_m[segment.start_index:segment.end_index_exclusive])
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Contact variance requires at least two finite position samples.")
        variances.append(float(np.var(points, axis=0, ddof=0).sum() * 1e6))
    worst = int(np.argmax(variances))
    return ([segment for i, segment in enumerate(segments) if i != worst],
            segments[worst], variances)


def calculate_contact_point_results(
    segments: list[ContactSegment],
    positions_m: np.ndarray,
) -> tuple[list[ContactPointResult], np.ndarray]:
    """根据检测到的接触段计算每次接触的平均 TCP 位置和误差。"""

    mean_positions = []
    segment_std_mm = []
    segment_rms_mm = []

    for segment in segments:
        segment_positions = positions_m[segment.start_index:segment.end_index_exclusive]
        mean_position = np.mean(segment_positions, axis=0)
        centered_positions = segment_positions - mean_position
        std_xyz_mm = np.std(segment_positions, axis=0) * 1000.0
        rms_mm = float(np.sqrt(np.mean(np.sum(centered_positions * centered_positions, axis=1))) * 1000.0)

        mean_positions.append(mean_position)
        segment_std_mm.append(std_xyz_mm)
        segment_rms_mm.append(rms_mm)

    mean_positions_array = np.asarray(mean_positions, dtype=np.float64)
    global_mean_position = np.mean(mean_positions_array, axis=0)
    errors_mm = np.linalg.norm(mean_positions_array - global_mean_position, axis=1) * 1000.0

    results: list[ContactPointResult] = []
    for index, segment in enumerate(segments):
        mean_position = mean_positions_array[index]
        std_xyz_mm = segment_std_mm[index]
        frame_count = segment.end_index_exclusive - segment.start_index

        results.append(
            ContactPointResult(
                contact_index=segment.contact_index,
                start_index=segment.start_index,
                end_index_exclusive=segment.end_index_exclusive,
                start_time_s=segment.start_time_s,
                end_time_s=segment.end_time_s,
                duration_s=segment.duration_s,
                frame_count=frame_count,
                mean_x_m=float(mean_position[0]),
                mean_y_m=float(mean_position[1]),
                mean_z_m=float(mean_position[2]),
                std_x_mm=float(std_xyz_mm[0]),
                std_y_mm=float(std_xyz_mm[1]),
                std_z_mm=float(std_xyz_mm[2]),
                within_segment_rms_mm=segment_rms_mm[index],
                error_to_global_mean_mm=float(errors_mm[index]),
            )
        )

    return results, global_mean_position


def average_rotation_matrices(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    对同一稳定接触窗口内的旋转矩阵求 SO(3) 投影均值。

    直接逐元素平均会破坏旋转矩阵正交性，因此先平均，再用 SVD 投影回最近的
    合法旋转矩阵。每个接触最终只贡献一个平均位姿，避免保持更久的接触权重更大。
    """

    mean_matrix = np.mean(np.asarray(rotation_matrices, dtype=np.float64), axis=0)
    left_vectors, _, right_vectors_transposed = np.linalg.svd(mean_matrix)
    mean_rotation = left_vectors @ right_vectors_transposed
    if np.linalg.det(mean_rotation) < 0.0:
        left_vectors[:, -1] *= -1.0
        mean_rotation = left_vectors @ right_vectors_transposed
    return mean_rotation


def calculate_contact_mean_iphone_poses(
    T_W_I: np.ndarray,
    segments: list[ContactSegment],
) -> tuple[np.ndarray, np.ndarray]:
    """为每个接触窗口计算一个平均 iPhone 旋转和平移。"""

    mean_rotations = []
    mean_positions = []
    for segment in segments:
        segment_transforms = T_W_I[segment.start_index:segment.end_index_exclusive]
        if len(segment_transforms) == 0:
            raise ValueError(f"接触 {segment.contact_index} 没有包含任何位姿帧。")
        mean_rotations.append(average_rotation_matrices(segment_transforms[:, :3, :3]))
        mean_positions.append(np.mean(segment_transforms[:, :3, 3], axis=0))

    return (
        np.asarray(mean_rotations, dtype=np.float64),
        np.asarray(mean_positions, dtype=np.float64),
    )


def solve_probe_point_and_fixed_point(
    rotations_W_I: np.ndarray,
    positions_W_I_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    线性求解牙签尖端在 iPhone 坐标系中的位置，以及纸面固定点的世界坐标。

    对每个接触姿态满足：
        p_W_I_i + R_W_I_i * q_I_probe = c_W_fixed

    整理为：
        [R_W_I_i, -I] [q_I_probe, c_W_fixed]^T = -p_W_I_i

    只要接触姿态有足够旋转变化，该 6 维线性系统即可唯一求解。
    """

    if len(rotations_W_I) != len(positions_W_I_m):
        raise ValueError("接触旋转数量与平移数量不一致。")

    identity = np.eye(3, dtype=np.float64)
    design_matrix = np.concatenate(
        [
            np.concatenate([rotation, -identity], axis=1)
            for rotation in rotations_W_I
        ],
        axis=0,
    )
    target_vector = -np.asarray(positions_W_I_m, dtype=np.float64).reshape(-1)
    solution, _, rank, singular_values = np.linalg.lstsq(
        design_matrix,
        target_vector,
        rcond=None,
    )

    if rank < 6:
        raise RuntimeError(
            "接触姿态变化不足，TCP 平移方程秩亏。请增加俯仰、滚转和偏航变化后重新采集。"
        )

    smallest_singular_value = float(singular_values[-1])
    condition_number = (
        float(singular_values[0] / smallest_singular_value)
        if smallest_singular_value > 1e-12
        else float("inf")
    )
    return solution[:3], solution[3:], int(rank), condition_number


def calculate_fixed_point_errors_mm(
    rotations_W_I: np.ndarray,
    positions_W_I_m: np.ndarray,
    point_in_iphone_m: np.ndarray,
    fixed_point_world_m: np.ndarray,
) -> np.ndarray:
    """计算每个接触均值到拟合固定点的三维欧氏误差，单位 mm。"""

    predicted_world_points = (
        positions_W_I_m
        + np.einsum("nij,j->ni", rotations_W_I, point_in_iphone_m)
    )
    return np.linalg.norm(
        predicted_world_points - fixed_point_world_m,
        axis=1,
    ) * 1000.0


def fit_tcp_translation_from_fixed_point_contacts(
    T_W_I: np.ndarray,
    segments: list[ContactSegment],
    current_T_I_G: np.ndarray,
    probe_offset_G_m: np.ndarray,
    max_translation_correction_mm: float,
) -> TranslationFitResult:
    """
    保持 T_I_G 旋转不变，仅根据固定点接触微调原 TCP 平移。

    先求牙签尖端在 iPhone 坐标系中的位置 q_I_probe，再减去已知的
    R_I_G @ probe_offset_G，得到原夹爪 TCP 的新平移 t_I_G。
    """

    if len(segments) < DEFAULT_MIN_FIT_CONTACT_COUNT:
        raise RuntimeError(
            f"至少需要 {DEFAULT_MIN_FIT_CONTACT_COUNT} 个接触才能微调，当前只有 {len(segments)} 个。"
        )

    mean_rotations, mean_positions = calculate_contact_mean_iphone_poses(
        T_W_I=T_W_I,
        segments=segments,
    )
    all_contact_mask = np.ones(len(segments), dtype=bool)

    # 第一遍使用全部接触拟合，再根据每段的三维残差做一次稳健异常点剔除。
    probe_point_I_m, fixed_point_W_m, rank, condition_number = (
        solve_probe_point_and_fixed_point(mean_rotations, mean_positions)
    )
    first_pass_errors_mm = calculate_fixed_point_errors_mm(
        rotations_W_I=mean_rotations,
        positions_W_I_m=mean_positions,
        point_in_iphone_m=probe_point_I_m,
        fixed_point_world_m=fixed_point_W_m,
    )
    median_error_mm = float(np.median(first_pass_errors_mm))
    mad_mm = float(np.median(np.abs(first_pass_errors_mm - median_error_mm)))
    robust_sigma_mm = 1.4826 * mad_mm
    outlier_threshold_mm = max(
        DEFAULT_OUTLIER_FLOOR_MM,
        median_error_mm + 3.0 * robust_sigma_mm,
    )
    candidate_mask = first_pass_errors_mm <= outlier_threshold_mm

    # 只有在保留点仍足够时才执行第二遍；否则保留全部点并让最终大残差显式暴露。
    if (
        np.count_nonzero(candidate_mask) >= DEFAULT_MIN_FIT_CONTACT_COUNT
        and not np.all(candidate_mask)
    ):
        all_contact_mask = candidate_mask
        probe_point_I_m, fixed_point_W_m, rank, condition_number = (
            solve_probe_point_and_fixed_point(
                mean_rotations[all_contact_mask],
                mean_positions[all_contact_mask],
            )
        )

    rotation_I_G = current_T_I_G[:3, :3]
    probe_offset = np.asarray(probe_offset_G_m, dtype=np.float64)
    candidate_translation_I_G_m = probe_point_I_m - rotation_I_G @ probe_offset

    candidate_T_I_G = current_T_I_G.copy()
    candidate_T_I_G[:3, 3] = candidate_translation_I_G_m
    correction_mm = (
        candidate_translation_I_G_m - current_T_I_G[:3, 3]
    ) * 1000.0

    # 当前矩阵与候选矩阵都用各自最合适的固定点均值计算误差，保证比较公平。
    current_probe_point_I_m = (
        current_T_I_G[:3, 3] + rotation_I_G @ probe_offset
    )
    current_predicted_points = (
        mean_positions
        + np.einsum("nij,j->ni", mean_rotations, current_probe_point_I_m)
    )
    current_fixed_point_W_m = np.mean(
        current_predicted_points[all_contact_mask],
        axis=0,
    )
    current_errors_mm = np.linalg.norm(
        current_predicted_points - current_fixed_point_W_m,
        axis=1,
    ) * 1000.0
    candidate_errors_mm = calculate_fixed_point_errors_mm(
        rotations_W_I=mean_rotations,
        positions_W_I_m=mean_positions,
        point_in_iphone_m=probe_point_I_m,
        fixed_point_world_m=fixed_point_W_m,
    )

    used_indices = tuple(
        segments[index].contact_index
        for index in np.flatnonzero(all_contact_mask)
    )
    rejected_indices = tuple(
        segments[index].contact_index
        for index in np.flatnonzero(~all_contact_mask)
    )
    correction_norm_mm = float(np.linalg.norm(correction_mm))

    return TranslationFitResult(
        candidate_T_I_G=candidate_T_I_G,
        fixed_point_world_m=fixed_point_W_m,
        used_contact_indices=used_indices,
        rejected_contact_indices=rejected_indices,
        current_errors_mm=current_errors_mm,
        candidate_errors_mm=candidate_errors_mm,
        matrix_rank=rank,
        condition_number=condition_number,
        correction_mm=correction_mm,
        safe_correction_magnitude=correction_norm_mm <= max_translation_correction_mm,
    )


# ============================================================
# 3. 保存数据与可视化
# ============================================================


def save_contact_points_csv(
    output_dir: Path,
    results: list[ContactPointResult],
    filename: str = "contact_points.csv",
) -> Path:
    """保存每次接触的统计结果；候选矩阵可使用单独文件名避免覆盖。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    fieldnames = [
        "contact_index",
        "start_index",
        "end_index_exclusive",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "frame_count",
        "mean_x_m",
        "mean_y_m",
        "mean_z_m",
        "std_x_mm",
        "std_y_mm",
        "std_z_mm",
        "within_segment_rms_mm",
        "error_to_global_mean_mm",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return csv_path


def save_contact_frames_csv(
    output_dir: Path,
    timestamps_s: np.ndarray,
    positions_m: np.ndarray,
    speed_m_per_s: np.ndarray,
    segments: list[ContactSegment],
) -> Path:
    """保存所有接触窗口内的逐帧 TCP 位置，便于后续手动复查。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "contact_frames.csv"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "contact_index",
                "frame_index",
                "timestamp_s",
                "probe_tip_x_m",
                "probe_tip_y_m",
                "probe_tip_z_m",
                "speed_m_per_s",
            ]
        )

        for segment in segments:
            for frame_index in range(segment.start_index, segment.end_index_exclusive):
                position = positions_m[frame_index]
                writer.writerow(
                    [
                        segment.contact_index,
                        frame_index,
                        f"{timestamps_s[frame_index]:.9f}",
                        f"{position[0]:.9f}",
                        f"{position[1]:.9f}",
                        f"{position[2]:.9f}",
                        f"{speed_m_per_s[frame_index]:.9f}",
                    ]
                )

    return csv_path


def summarize_errors(errors_mm: np.ndarray, used_indices_zero_based: np.ndarray) -> dict[str, float]:
    """把指定接触子集的误差整理成 JSON 可保存的统计值。"""

    selected = np.asarray(errors_mm, dtype=np.float64)[used_indices_zero_based]
    return {
        "mean_mm": float(np.mean(selected)),
        "median_mm": float(np.median(selected)),
        "p95_mm": float(np.percentile(selected, 95.0)),
        "max_mm": float(np.max(selected)),
    }


def save_translation_candidate_json(
    output_dir: Path,
    source_T_I_G: np.ndarray,
    source_description: str,
    probe_offset_G_m: np.ndarray,
    fit_result: TranslationFitResult,
    max_translation_correction_mm: float,
) -> Path:
    """
    保存可独立验证的 TCP 平移候选矩阵，但绝不自动覆盖正式外参。

    下一条全新 session 可通过 --tig-json 指向此文件，只做验证；验证通过后再人工决定
    是否把候选平移写入 stray_shift_position.py。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "T_I_G_candidate.json"
    used_zero_based = np.asarray(fit_result.used_contact_indices, dtype=np.int64) - 1
    correction_norm_mm = float(np.linalg.norm(fit_result.correction_mm))
    candidate_data = {
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "matrix_semantics": "T_W_G = T_W_I @ T_I_G; probe point = TCP + R_I_G @ probe_offset_G_m",
        "source_T_I_G": source_description,
        "probe_offset_G_m": np.asarray(probe_offset_G_m, dtype=np.float64).tolist(),
        "T_I_G_current": np.asarray(source_T_I_G, dtype=np.float64).tolist(),
        "T_I_G_candidate": np.asarray(fit_result.candidate_T_I_G, dtype=np.float64).tolist(),
        "translation_correction_mm": fit_result.correction_mm.tolist(),
        "translation_correction_norm_mm": correction_norm_mm,
        "max_allowed_translation_correction_mm": float(max_translation_correction_mm),
        "safe_correction_magnitude": bool(fit_result.safe_correction_magnitude),
        "requires_independent_validation": True,
        "used_contact_indices": list(fit_result.used_contact_indices),
        "rejected_contact_indices": list(fit_result.rejected_contact_indices),
        "linear_system_rank": int(fit_result.matrix_rank),
        "linear_system_condition_number": float(fit_result.condition_number),
        "current_error_on_used_contacts": summarize_errors(
            fit_result.current_errors_mm,
            used_zero_based,
        ),
        "candidate_error_on_used_contacts": summarize_errors(
            fit_result.candidate_errors_mm,
            used_zero_based,
        ),
        "candidate_errors_all_contacts_mm": fit_result.candidate_errors_mm.tolist(),
        "notes": [
            "旋转矩阵保持不变，只微调了原夹爪 TCP 的平移。",
            "本文件来自拟合数据，不能代替独立验证。",
            "用全新 session 运行 --tig-json 本文件路径；验证通过后才能采用。",
        ],
    }
    candidate_path.write_text(
        json.dumps(candidate_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidate_path


def classify_mean_error(mean_error_mm: float) -> str:
    """根据固定点接触平均误差给出经验判断。"""

    if mean_error_mm < 2.0:
        return "很好：平均误差 < 2 mm"
    if mean_error_mm < 5.0:
        return "可以用：平均误差 2~5 mm"
    if mean_error_mm < 10.0:
        return "第一版勉强能用，但要小心：平均误差 5~10 mm"
    return "T_I_G 很可能有问题：平均误差 > 10 mm"


def save_summary_txt(
    output_dir: Path,
    paths: stray.ProcessPaths,
    T_I_G_used: np.ndarray,
    tig_source_description: str,
    probe_offset_G_m: np.ndarray,
    results: list[ContactPointResult],
    global_mean_position_m: np.ndarray,
    speed_threshold_m_per_s: float,
    min_contact_duration_s: float,
    center_window_seconds: float,
) -> Path:
    """保存人类可读的验证摘要。"""

    errors_mm = np.array([result.error_to_global_mean_mm for result in results], dtype=np.float64)
    within_rms_mm = np.array([result.within_segment_rms_mm for result in results], dtype=np.float64)
    mean_error_mm = float(np.mean(errors_mm))
    max_error_mm = float(np.max(errors_mm))
    std_error_mm = float(np.std(errors_mm))

    lines = [
        "T_I_G toothpick fixed-point contact validation summary",
        "",
        f"demo_dir: {paths.demo_dir}",
        f"odometry_csv_path: {paths.odometry_csv_path}",
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
        "  offset_semantics: original_TCP_to_toothpick_contact_tip_in_G_frame",
        f"  offset_G_m: {np.asarray(probe_offset_G_m, dtype=np.float64).tolist()}",
        "",
        "auto_detection_parameters:",
        f"  speed_threshold_m_per_s: {speed_threshold_m_per_s}",
        f"  min_contact_duration_s: {min_contact_duration_s}",
        f"  center_window_seconds: {center_window_seconds}",
        "",
        "global_mean_toothpick_tip_position_m:",
        f"  x: {float(global_mean_position_m[0]):.9f}",
        f"  y: {float(global_mean_position_m[1]):.9f}",
        f"  z: {float(global_mean_position_m[2]):.9f}",
        "",
        "contact_error_statistics_mm:",
        f"  mean: {mean_error_mm:.6f}",
        f"  max : {max_error_mm:.6f}",
        f"  std : {std_error_mm:.6f}",
        f"  within_segment_rms_mean: {float(np.mean(within_rms_mm)):.6f}",
        "",
        f"judgement: {classify_mean_error(mean_error_mm)}",
        "",
        "per_contact_points:",
    ]

    for result in results:
        lines.append(
            "  "
            f"{result.contact_index:02d}: "
            f"p=[{result.mean_x_m:.9f}, {result.mean_y_m:.9f}, {result.mean_z_m:.9f}] m, "
            f"error={result.error_to_global_mean_mm:.3f} mm, "
            f"frames={result.frame_count}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def save_validation_plots(
    output_dir: Path,
    timestamps_s: np.ndarray,
    positions_m: np.ndarray,
    speed_m_per_s: np.ndarray,
    results: list[ContactPointResult],
    global_mean_position_m: np.ndarray,
) -> bool:
    """保存固定点接触验证图。"""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warning_path = output_dir / "plot_warning.txt"
        warning_path.write_text(
            "当前环境缺少 matplotlib，所以没有生成验证图片。\n"
            "如需图片，请安装：python -m pip install matplotlib\n",
            encoding="utf-8",
        )
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_positions = np.array(
        [[result.mean_x_m, result.mean_y_m, result.mean_z_m] for result in results],
        dtype=np.float64,
    )
    errors_mm = np.array([result.error_to_global_mean_mm for result in results], dtype=np.float64)
    contact_indices = np.array([result.contact_index for result in results], dtype=np.int32)

    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        contact_positions[:, 0],
        contact_positions[:, 1],
        contact_positions[:, 2],
        label="contact mean points",
    )
    axis.scatter(
        [global_mean_position_m[0]],
        [global_mean_position_m[1]],
        [global_mean_position_m[2]],
        marker="x",
        s=80,
        label="global mean",
    )
    for result in results:
        axis.text(result.mean_x_m, result.mean_y_m, result.mean_z_m, str(result.contact_index))
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_zlabel("z / m")
    axis.set_title("Fixed-point toothpick-tip contact positions")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "contact_points_3d.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 6))
    plt.scatter(contact_positions[:, 0], contact_positions[:, 1])
    plt.scatter(global_mean_position_m[0], global_mean_position_m[1], marker="x", s=80)
    for result in results:
        plt.text(result.mean_x_m, result.mean_y_m, str(result.contact_index))
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title("Contact points XY")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "contact_points_xy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(contact_indices, errors_mm)
    plt.axhline(2.0, color="green", linestyle="--", label="2 mm")
    plt.axhline(5.0, color="orange", linestyle="--", label="5 mm")
    plt.axhline(10.0, color="red", linestyle="--", label="10 mm")
    plt.xlabel("contact index")
    plt.ylabel("error to mean / mm")
    plt.title("Fixed-point contact error")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "contact_error_mm.png", dpi=200)
    plt.close()

    relative_time = timestamps_s - timestamps_s[0]
    plt.figure(figsize=(10, 5))
    plt.plot(relative_time, positions_m[:, 0], label="probe tip x")
    plt.plot(relative_time, positions_m[:, 1], label="probe tip y")
    plt.plot(relative_time, positions_m[:, 2], label="probe tip z")
    for result in results:
        plt.axvspan(
            result.start_time_s - timestamps_s[0],
            result.end_time_s - timestamps_s[0],
            alpha=0.18,
        )
    plt.xlabel("time / s")
    plt.ylabel("toothpick-tip position / m")
    plt.title("Toothpick-tip position over time; shaded windows are detected contacts")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tcp_position_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(relative_time, speed_m_per_s)
    plt.xlabel("time / s")
    plt.ylabel("toothpick-tip speed / m/s")
    plt.title("Toothpick-tip speed used for stable contact detection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "tcp_speed_time.png", dpi=200)
    plt.close()

    return True


# ============================================================
# 4. 命令行入口
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Validate T_I_G with fixed-point TCP contacts.")
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
        help="指定某个时间戳 session 文件夹；不填时自动处理最新且包含 odometry.csv 的文件夹。",
    )
    parser.add_argument(
        "--contact-count",
        type=int,
        default=DEFAULT_CONTACT_COUNT,
        help=f"期望检测的固定点接触次数，默认：{DEFAULT_CONTACT_COUNT}",
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
        default=DEFAULT_MIN_CONTACT_DURATION_SECONDS,
        help=f"每次接触最短持续时间 s，默认：{DEFAULT_MIN_CONTACT_DURATION_SECONDS}",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=DEFAULT_MIN_GAP_BETWEEN_CONTACTS_SECONDS,
        help=f"两次接触之间最短间隔 s，默认：{DEFAULT_MIN_GAP_BETWEEN_CONTACTS_SECONDS}",
    )
    parser.add_argument(
        "--center-window",
        type=float,
        default=DEFAULT_CENTER_WINDOW_SECONDS,
        help=f"每次稳定片段中间用于计算接触点的窗口长度 s，默认：{DEFAULT_CENTER_WINDOW_SECONDS}",
    )
    parser.add_argument(
        "--probe-offset-x",
        type=float,
        default=DEFAULT_PROBE_OFFSET_G_M[0],
        help="原 TCP 到牙签接触端在夹爪 X_G 轴上的偏移 m，默认 0。",
    )
    parser.add_argument(
        "--probe-offset-y",
        type=float,
        default=DEFAULT_PROBE_OFFSET_G_M[1],
        help="原 TCP 到牙签接触端在夹爪 Y_G 轴上的偏移 m，默认 0。",
    )
    parser.add_argument(
        "--probe-offset-z",
        type=float,
        default=DEFAULT_PROBE_OFFSET_G_M[2],
        help=(
            "原 TCP 到牙签接触端在夹爪 Z_G 轴上的偏移 m，"
            f"默认 {DEFAULT_PROBE_OFFSET_G_M[2]}（+Z_G 方向 3.193 cm）。"
        ),
    )
    parser.add_argument(
        "--tig-json",
        type=Path,
        default=None,
        help="独立验证时指定上一轮生成的 T_I_G_candidate.json；不填则使用当前正式外参。",
    )
    parser.add_argument(
        "--fit-translation",
        action="store_true",
        help="保持旋转不变，根据本 session 接触数据拟合 TCP 平移并保存候选 JSON。",
    )
    parser.add_argument(
        "--max-translation-correction-mm",
        type=float,
        default=DEFAULT_MAX_TRANSLATION_CORRECTION_MM,
        help=(
            "单次微调允许的最大平移修正模长 mm；超过时仍输出候选供排错，"
            f"但标记为不安全。默认 {DEFAULT_MAX_TRANSLATION_CORRECTION_MM}。"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """完整验证流程入口。"""

    args = parse_args()
    paths = build_process_paths_without_output_guard(
        stray_root=args.stray_root,
        explicit_demo_dir=args.demo_dir,
    )
    probe_offset_G_m = np.array(
        [args.probe_offset_x, args.probe_offset_y, args.probe_offset_z],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(probe_offset_G_m)):
        raise ValueError("牙签偏移必须是有限数值。")
    if args.max_translation_correction_mm <= 0.0:
        raise ValueError("--max-translation-correction-mm 必须大于 0。")

    T_I_G_used, tig_source_description = load_tig_matrix(args.tig_json)

    print("正在验证牙签尖端固定点接触：")
    print(f"  session 文件夹：{paths.demo_dir}")
    print(f"  odometry.csv：{paths.odometry_csv_path}")
    print(f"  输出目录：{paths.output_dir}")
    print(f"  期望接触次数：{args.contact_count}")
    print(f"  T_I_G 来源：{tig_source_description}")
    print(
        "  原 TCP -> 牙签接触端偏移 [x, y, z]："
        f"{(probe_offset_G_m * 100.0).tolist()} cm（夹爪坐标系 G）"
    )
    if np.allclose(probe_offset_G_m, DEFAULT_PROBE_OFFSET_G_M, atol=1e-12):
        print("  当前默认假设：牙签从原 TCP 沿 +Z_G 方向伸出 3.193 cm（实测 31.93 mm）。")
        print("  如果实际方向相反，请使用 --probe-offset-z -0.03193。")

    timestamps_s, T_W_I, probe_positions_w_m = load_probe_trajectory_from_odometry(
        odometry_csv_path=paths.odometry_csv_path,
        T_I_G=T_I_G_used,
        probe_offset_G_m=probe_offset_G_m,
    )
    segments, speed_m_per_s = detect_contact_segments(
        timestamps_s=timestamps_s,
        positions_m=probe_positions_w_m,
        contact_count=args.contact_count,
        max_stable_speed_m_per_s=args.max_stable_speed,
        min_contact_duration_s=args.min_contact_duration,
        min_gap_between_contacts_s=args.min_gap,
        center_window_seconds=args.center_window,
    )
    if DROP_HIGHEST_VARIANCE_CONTACT:
        original_segments = segments
        segments, discarded, variances = discard_highest_variance_contact(segments, probe_positions_w_m)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        (paths.output_dir / 'discarded_contact.json').write_text(json.dumps({
            'contact_index': discarded.contact_index,
            'start_index': discarded.start_index,
            'end_index_exclusive': discarded.end_index_exclusive,
            'metric': 'sum_xyz_population_variance_mm2',
            'variances': {str(s.contact_index): v for s, v in zip(original_segments, variances)},
        }, indent=2), encoding='utf-8')
        print(f"Discard contact #{discarded.contact_index}; retain {len(segments)} contacts")
    results, global_mean_position_m = calculate_contact_point_results(
        segments=segments,
        positions_m=probe_positions_w_m,
    )

    contact_points_csv = save_contact_points_csv(paths.output_dir, results)
    contact_frames_csv = save_contact_frames_csv(
        output_dir=paths.output_dir,
        timestamps_s=timestamps_s,
        positions_m=probe_positions_w_m,
        speed_m_per_s=speed_m_per_s,
        segments=segments,
    )
    summary_path = save_summary_txt(
        output_dir=paths.output_dir,
        paths=paths,
        T_I_G_used=T_I_G_used,
        tig_source_description=tig_source_description,
        probe_offset_G_m=probe_offset_G_m,
        results=results,
        global_mean_position_m=global_mean_position_m,
        speed_threshold_m_per_s=args.max_stable_speed,
        min_contact_duration_s=args.min_contact_duration,
        center_window_seconds=args.center_window,
    )
    plots_saved = save_validation_plots(
        output_dir=paths.test_dir,
        timestamps_s=timestamps_s,
        positions_m=probe_positions_w_m,
        speed_m_per_s=speed_m_per_s,
        results=results,
        global_mean_position_m=global_mean_position_m,
    )

    errors_mm = np.array([result.error_to_global_mean_mm for result in results], dtype=np.float64)
    mean_error_mm = float(np.mean(errors_mm))
    max_error_mm = float(np.max(errors_mm))

    candidate_path = None
    if args.fit_translation:
        fit_result = fit_tcp_translation_from_fixed_point_contacts(
            T_W_I=T_W_I,
            segments=segments,
            current_T_I_G=T_I_G_used,
            probe_offset_G_m=probe_offset_G_m,
            max_translation_correction_mm=args.max_translation_correction_mm,
        )
        candidate_path = save_translation_candidate_json(
            output_dir=paths.output_dir,
            source_T_I_G=T_I_G_used,
            source_description=tig_source_description,
            probe_offset_G_m=probe_offset_G_m,
            fit_result=fit_result,
            max_translation_correction_mm=args.max_translation_correction_mm,
        )

        candidate_probe_positions_w_m = calculate_probe_positions_from_transforms(
            T_W_I=T_W_I,
            T_I_G=fit_result.candidate_T_I_G,
            probe_offset_G_m=probe_offset_G_m,
        )
        candidate_results, _ = calculate_contact_point_results(
            segments=segments,
            positions_m=candidate_probe_positions_w_m,
        )
        save_contact_points_csv(
            output_dir=paths.output_dir,
            results=candidate_results,
            filename="contact_points_candidate.csv",
        )

        used_zero_based = np.asarray(fit_result.used_contact_indices, dtype=np.int64) - 1
        current_fit_stats = summarize_errors(
            fit_result.current_errors_mm,
            used_zero_based,
        )
        candidate_fit_stats = summarize_errors(
            fit_result.candidate_errors_mm,
            used_zero_based,
        )
        print("\nTCP 平移微调候选已计算（旋转矩阵未改变）：")
        print(
            "  当前平移 mm："
            f"{(T_I_G_used[:3, 3] * 1000.0).round(3).tolist()}"
        )
        print(
            "  候选平移 mm："
            f"{(fit_result.candidate_T_I_G[:3, 3] * 1000.0).round(3).tolist()}"
        )
        print(f"  建议修正 mm：{fit_result.correction_mm.round(3).tolist()}")
        print(
            f"  当前误差 mean/max：{current_fit_stats['mean_mm']:.3f} / "
            f"{current_fit_stats['max_mm']:.3f} mm"
        )
        print(
            f"  拟合误差 mean/max：{candidate_fit_stats['mean_mm']:.3f} / "
            f"{candidate_fit_stats['max_mm']:.3f} mm"
        )
        print(f"  使用接触：{list(fit_result.used_contact_indices)}")
        print(f"  剔除接触：{list(fit_result.rejected_contact_indices)}")
        if not fit_result.safe_correction_magnitude:
            print(
                "  警告：修正模长超过微调安全阈值，先检查牙签偏移轴/正负号和接触段，"
                "不要采用该候选。"
            )

    print("\n验证完成。")
    print(f"  平均误差：{mean_error_mm:.3f} mm")
    print(f"  最大误差：{max_error_mm:.3f} mm")
    print(f"  判断：{classify_mean_error(mean_error_mm)}")
    print(f"  接触点数据：{contact_points_csv}")
    print(f"  接触逐帧数据：{contact_frames_csv}")
    print(f"  summary：{summary_path}")
    if candidate_path is not None:
        print(f"  TCP 平移候选：{candidate_path}")
        print("  该候选尚未写入正式外参；必须用全新 session 配合 --tig-json 独立验证。")
    if plots_saved:
        print(f"  可视化图片：{paths.test_dir}")
    else:
        print(f"  未生成图片，原因见：{paths.test_dir / 'plot_warning.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
