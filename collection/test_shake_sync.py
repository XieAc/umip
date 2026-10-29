# -*- coding: utf-8 -*-
"""
用“两次摇晃”验证 iPhone / Stray Scanner 时间轴和 D435i IMU 时间轴是否对齐。

推荐采集流程：
    1. D435i 开始录制。
    2. gripper width log 开始录制。
    3. iPhone / Stray Scanner 开始录制。
    4. 静止 2 秒。
    5. 快速晃动一次。
    6. 静止 2 秒。
    7. 再快速晃动一次。
    8. 静止 2 秒。
    9. 停止录制。

核心验证逻辑：
    1. iPhone 侧用 imu.csv 的 gyro norm 作为 shake 信号。
       如果没有 gyro 字段，则退回 accel norm。
    2. D435i 侧用 d435i/imu.csv 的 gyro norm 作为 shake 信号。
       如果没有 gyro 数据，则退回 accel norm。
       D435i IMU 的时间轴使用 system_time_s，也就是电脑时间。
    3. 用第一次摇晃估计 offset：
           offset = t_I,shake1 - t_D,shake1
           t_iphone = t_d435i + offset
    4. 把第二次 iPhone shake 转回 D435i 时间轴：
           t_D_from_iPhone = t_I,shake2 - offset
       再和 D435i IMU 检测到的第二次 shake 比较。

输出：
    demo_dir/processed_umi/sync.json
    demo_dir/processed_umi/shake_sync_debug.csv
    demo_dir/processed_umi/test/iphone_shake_signal.png
    demo_dir/processed_umi/test/d435i_motion_signal.png
    demo_dir/processed_umi/test/aligned_shake_signals.png
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xac_config import DATA_ROOT


# ============================================================
# 1. 默认路径与参数
# ============================================================

# 你的采集 session 默认都放在这个根目录下。
DEFAULT_STRAY_ROOT = DATA_ROOT

# 和现有 stray_shift_position.py 保持一致，后续处理链路会优先看 processed_umi。
DEFAULT_OUTPUT_DIR_NAME = "processed_umi"
DEFAULT_TEST_DIR_NAME = "test"

# 两次 shake 都应该出现在录制开始后的前十几秒。实际采集中第二次 shake
# 可能稍晚于第 8 秒，因此默认搜索 12 秒，避免因搜索窗口过短而漏检。
DEFAULT_SHAKE_SEARCH_SECONDS = 12.0

# 平滑窗口太小会保留噪声，太大会把尖峰抹平。0.15 秒对手动快速 shake 比较稳。
DEFAULT_SMOOTH_WINDOW_SECONDS = 0.15

# 两次 shake event 之间至少间隔多少秒，用来避免把同一次晃动拆成多次同步事件。
DEFAULT_MIN_EVENT_SEPARATION_SECONDS = 1.0

# 连续超过这个相对阈值的区域会被视为一个 shake event。
# 阈值计算方式：
#     threshold = static_baseline + (global_peak - static_baseline) * ratio
# 0.25 的含义是：只要超过“静止基线到最大峰值高度”的 25%，就认为进入剧烈运动段。
DEFAULT_EVENT_THRESHOLD_RATIO = 0.25

# 有些 shake event 内部会短暂低于阈值，例如“晃动 -> 回弹 -> 停住”的中间谷值。
# 如果两个高于阈值的片段中间间隔很短，就把它们合并为同一个 shake event。
DEFAULT_MERGE_EVENT_GAP_SECONDS = 0.35

# 太短的阈值穿越通常是噪声或单个异常采样点，不作为一次 shake event。
DEFAULT_MIN_EVENT_DURATION_SECONDS = 0.05

# 递归找 iPhone imu.csv 时应跳过这些输出目录和缓存目录。
IGNORED_SEARCH_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    DEFAULT_OUTPUT_DIR_NAME,
    "d435i",
}


# ============================================================
# 2. 数据结构
# ============================================================


@dataclass(frozen=True)
class MotionSignal:
    """保存某一路 shake 信号，timestamps_s 和 signal 数组一一对应。"""

    name: str
    source_path: Path
    time_column: str
    signal_kind: str
    timestamps_s: np.ndarray
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray


@dataclass(frozen=True)
class ShakePeak:
    """保存一次 shake 峰值及其局部区域信息。"""

    peak_order: int
    sample_index: int
    shake_time_s: float
    sample_time_s: float
    peak_signal: float
    baseline_signal: float
    event_threshold_signal: float
    peak_to_static_ratio: float
    region_start_time_s: float
    region_end_time_s: float


# ============================================================
# 3. 通用 CSV 与列名处理
# ============================================================


def normalize_column_name(column_name: str) -> str:
    """
    把 CSV 列名归一化为只含小写字母和数字的形式。

    例如：
        "alpha_x" -> "alphax"
        "gyro-x"  -> "gyrox"

    这样可以兼容 Stray Scanner、RealSense 或不同导出工具的命名差异。
    """

    return "".join(char.lower() for char in column_name.strip() if char.isalnum())


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV，并把表头和每一行都做 strip，减少空格造成的解析错误。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头：{csv_path}")

        columns = [column.strip() for column in reader.fieldnames]
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            clean_row = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                clean_row[key.strip()] = value.strip() if isinstance(value, str) else value
            rows.append(clean_row)

    return columns, rows


def build_normalized_column_map(columns: list[str]) -> dict[str, str]:
    """建立 normalized_column_name -> original_column_name 的映射。"""

    return {normalize_column_name(column): column for column in columns}


def parse_float_or_nan(value) -> float:
    """把 CSV 单元格转成 float；失败时返回 NaN，后面统一过滤。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def choose_time_column(columns: list[str]) -> tuple[str, float]:
    """
    自动选择时间列。

    返回：
        (原始列名, scale)

    scale 用于把原始时间单位换算成秒。例如 timestamp_ms 需要乘 0.001。
    """

    normalized_map = build_normalized_column_map(columns)
    candidates = [
        ("timestamp", 1.0),
        ("timestamps", 1.0),
        ("time", 1.0),
        ("times", 1.0),
        ("systemtimes", 1.0),
        ("computertimes", 1.0),
        ("systemtimems", 0.001),
        ("timestampms", 0.001),
        ("sensortimestampms", 0.001),
    ]

    for normalized_name, scale in candidates:
        if normalized_name in normalized_map:
            return normalized_map[normalized_name], scale

    raise KeyError(f"没有找到可用时间列。当前列名：{columns}")


def choose_d435i_system_time_column(columns: list[str]) -> tuple[str, float]:
    """
    选择 D435i IMU 的电脑时间列。

    D435i 的 imu.csv 同时有：
        capture_system_time_s 真实拍摄时刻(墙钟)，由传感器时间戳换算，无队列延迟
        system_time_s         出队时刻的电脑时间，和 RGB-D timestamps.csv 同轴
        sensor_timestamp_ms   RealSense 设备内部时间

    本同步验证要输出 t_iphone = t_d435i + offset，其中 t_d435i 后续会用于
    D435i 图像和 width log 对齐，因此优先使用 capture_system_time_s；
    旧文件没有该列时再回退到 system_time_s。
    """

    normalized_map = build_normalized_column_map(columns)
    candidates = [
        ("capturesystemtimes", 1.0),
        ("systemtimes", 1.0),
        ("computertimes", 1.0),
        ("systemtimems", 0.001),
    ]

    for normalized_name, scale in candidates:
        if normalized_name in normalized_map:
            return normalized_map[normalized_name], scale

    # 如果旧文件没有 system_time_s，再退回通用时间列；这种情况会在 sync.json
    # 里留下 time_column，方便你发现它不是最理想的数据源。
    return choose_time_column(columns)


# ============================================================
# 4. iPhone IMU shake 信号
# ============================================================


def find_axis_columns(columns: list[str], signal_kind: str) -> tuple[str, str, str] | None:
    """
    查找 gyro 或 accel 三轴列。

    Stray Scanner 当前常见格式是：
        a_x, a_y, a_z
        alpha_x, alpha_y, alpha_z

    这里把 alpha_* 当作 gyro / angular velocity 使用；如果将来导出列名变成
    gyro_x 或 angular_velocity_x，也能继续兼容。
    """

    normalized_map = build_normalized_column_map(columns)

    if signal_kind == "gyro":
        candidate_triples = [
            ("gx", "gy", "gz"),
            ("gyrox", "gyroy", "gyroz"),
            ("gyrx", "gyry", "gyrz"),
            ("alphax", "alphay", "alphaz"),
            ("angularvelocityx", "angularvelocityy", "angularvelocityz"),
            ("rotationratex", "rotationratey", "rotationratez"),
        ]
    elif signal_kind == "accel":
        candidate_triples = [
            ("ax", "ay", "az"),
            ("accelx", "accely", "accelz"),
            ("accelerometerx", "accelerometery", "accelerometerz"),
            ("accelerationx", "accelerationy", "accelerationz"),
            ("useraccelerationx", "useraccelerationy", "useraccelerationz"),
        ]
    else:
        raise ValueError(f"未知 signal_kind：{signal_kind}")

    for x_name, y_name, z_name in candidate_triples:
        if x_name in normalized_map and y_name in normalized_map and z_name in normalized_map:
            return normalized_map[x_name], normalized_map[y_name], normalized_map[z_name]

    return None


def build_signal_from_stream_rows(
    rows: list[dict[str, str]],
    columns: list[str],
    time_column: str,
    time_scale: float,
    signal_kind: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    解析 stream,x,y,z 格式的 IMU CSV。

    D435i 的 imu.csv 是这种格式：
        stream = accel / gyro
        x, y, z = 三轴读数

    如果以后 iPhone 导出的 IMU 也变成这种长表格式，本函数也可以直接复用。
    """

    normalized_map = build_normalized_column_map(columns)
    if "stream" not in normalized_map:
        return None

    x_column = normalized_map.get("x")
    y_column = normalized_map.get("y")
    z_column = normalized_map.get("z")
    stream_column = normalized_map.get("stream")
    if x_column is None or y_column is None or z_column is None or stream_column is None:
        return None

    timestamps = []
    values = []
    for row in rows:
        stream_text = str(row.get(stream_column, "")).strip().lower()
        if signal_kind == "gyro" and "gyro" not in stream_text:
            continue
        if signal_kind == "accel" and not ("accel" in stream_text or "accelerometer" in stream_text):
            continue

        timestamps.append(parse_float_or_nan(row.get(time_column)) * time_scale)
        values.append(
            [
                parse_float_or_nan(row.get(x_column)),
                parse_float_or_nan(row.get(y_column)),
                parse_float_or_nan(row.get(z_column)),
            ]
        )

    if not timestamps:
        return None

    values_array = np.asarray(values, dtype=np.float64)
    signal = np.linalg.norm(values_array, axis=1)
    return np.asarray(timestamps, dtype=np.float64), signal


def build_signal_from_axis_columns(
    rows: list[dict[str, str]],
    columns: list[str],
    time_column: str,
    time_scale: float,
    signal_kind: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """解析独立三轴列格式的 IMU CSV，并返回三轴模长。"""

    axis_columns = find_axis_columns(columns, signal_kind)
    if axis_columns is None:
        return None

    x_column, y_column, z_column = axis_columns
    timestamps = []
    values = []
    for row in rows:
        timestamps.append(parse_float_or_nan(row.get(time_column)) * time_scale)
        values.append(
            [
                parse_float_or_nan(row.get(x_column)),
                parse_float_or_nan(row.get(y_column)),
                parse_float_or_nan(row.get(z_column)),
            ]
        )

    values_array = np.asarray(values, dtype=np.float64)
    signal = np.linalg.norm(values_array, axis=1)
    return np.asarray(timestamps, dtype=np.float64), signal


def smooth_signal(timestamps_s: np.ndarray, raw_signal: np.ndarray, smooth_window_seconds: float) -> np.ndarray:
    """
    对非严格等间隔采样的信号做简单 moving average 平滑。

    这里用时间戳的中位数间隔估计窗口样本数，避免偶尔一两个异常 dt
    影响平滑窗口大小。
    """

    if len(raw_signal) < 3 or smooth_window_seconds <= 0:
        return raw_signal.copy()

    dt = np.diff(timestamps_s)
    positive_dt = dt[dt > 1e-9]
    if len(positive_dt) == 0:
        return raw_signal.copy()

    median_dt = float(np.median(positive_dt))
    window_size = max(1, int(round(smooth_window_seconds / median_dt)))
    if window_size <= 1:
        return raw_signal.copy()

    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(raw_signal, kernel, mode="same")


def load_imu_signal(
    imu_csv_path: Path,
    signal_name: str,
    prefer_d435i_system_time: bool,
    smooth_window_seconds: float,
) -> MotionSignal:
    """
    从 IMU CSV 中读取 shake 信号。

    优先使用 gyro，因为快速 shake 在角速度上通常峰值最明显；
    如果没有 gyro / alpha 字段，再退回加速度模长。

    参数：
        prefer_d435i_system_time:
            True  时优先选择 system_time_s，适合 D435i IMU。
            False 时优先选择 timestamp，适合 iPhone / Stray Scanner。
    """

    columns, rows = read_csv_rows(imu_csv_path)
    if prefer_d435i_system_time:
        time_column, time_scale = choose_d435i_system_time_column(columns)
    else:
        time_column, time_scale = choose_time_column(columns)

    selected_kind = None
    selected_timestamps = None
    selected_signal = None

    for signal_kind in ("gyro", "accel"):
        stream_signal = build_signal_from_stream_rows(
            rows=rows,
            columns=columns,
            time_column=time_column,
            time_scale=time_scale,
            signal_kind=signal_kind,
        )
        axis_signal = build_signal_from_axis_columns(
            rows=rows,
            columns=columns,
            time_column=time_column,
            time_scale=time_scale,
            signal_kind=signal_kind,
        )

        signal_result = stream_signal if stream_signal is not None else axis_signal
        if signal_result is None:
            continue

        selected_kind = signal_kind
        selected_timestamps, selected_signal = signal_result
        break

    if selected_kind is None or selected_timestamps is None or selected_signal is None:
        raise KeyError(f"没有在 IMU CSV 中找到 gyro/accel 三轴数据：{imu_csv_path}")

    finite_mask = np.isfinite(selected_timestamps) & np.isfinite(selected_signal)
    timestamps_s = selected_timestamps[finite_mask]
    raw_signal = selected_signal[finite_mask]
    if len(timestamps_s) < 3:
        raise ValueError(f"IMU 有效数据太少，无法检测 shake：{imu_csv_path}")

    order = np.argsort(timestamps_s)
    timestamps_s = timestamps_s[order]
    raw_signal = raw_signal[order]
    smoothed_signal = smooth_signal(timestamps_s, raw_signal, smooth_window_seconds)

    return MotionSignal(
        name=signal_name,
        source_path=imu_csv_path,
        time_column=time_column,
        signal_kind=selected_kind,
        timestamps_s=timestamps_s,
        raw_signal=raw_signal,
        smoothed_signal=smoothed_signal,
    )


def load_iphone_imu_signal(imu_csv_path: Path, smooth_window_seconds: float) -> MotionSignal:
    """读取 iPhone / Stray Scanner 的 imu.csv。"""

    return load_imu_signal(
        imu_csv_path=imu_csv_path,
        signal_name="iphone",
        prefer_d435i_system_time=False,
        smooth_window_seconds=smooth_window_seconds,
    )


def load_d435i_imu_signal(imu_csv_path: Path, smooth_window_seconds: float) -> MotionSignal:
    """
    读取 D435i 的 imu.csv。

    D435i IMU 必须优先使用 system_time_s，因为后续 RGB-D 图像和 gripper width
    都是用电脑时间对齐的。这样估计出来的 offset 可以直接用于：
        t_iphone = t_d435i_system_time + offset
    """

    return load_imu_signal(
        imu_csv_path=imu_csv_path,
        signal_name="d435i_imu",
        prefer_d435i_system_time=True,
        smooth_window_seconds=smooth_window_seconds,
    )


# ============================================================
# 5. 峰值检测与质量判断
# ============================================================


# ============================================================
# 6. 峰值检测与质量判断
# ============================================================


def robust_static_baseline(values: np.ndarray) -> float:
    """
    估计静止段基线。

    直接取全局均值会被 shake 尖峰拉高，所以这里取较低 30% 样本的中位数，
    更接近“静止时的运动强度”。
    """

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return 0.0

    cutoff = np.percentile(finite_values, 30.0)
    low_values = finite_values[finite_values <= cutoff]
    if len(low_values) == 0:
        return float(np.median(finite_values))

    return float(np.median(low_values))


def find_shake_event_regions(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    search_indices: np.ndarray,
    baseline: float,
    event_threshold_ratio: float,
    merge_event_gap_seconds: float,
    min_event_duration_seconds: float,
) -> tuple[list[tuple[int, int]], float]:
    """
    从平滑后的 motion signal 中检测 shake event 区域。

    这里的核心思想是：
        1. 先计算一个全局阈值。
        2. 所有连续超过阈值的采样区间，先视为候选 event。
        3. 如果两个候选 event 之间只隔了很短时间，则合并为一个 event。
        4. 过滤掉持续时间太短的噪声 event。

    返回：
        (event_regions, threshold)

    event_regions 里的每个元素是闭区间索引：
        (start_index, end_index)
    """

    search_values = values[search_indices]
    search_peak = float(np.max(search_values))
    threshold = baseline + max(0.0, search_peak - baseline) * event_threshold_ratio
    above_threshold = search_values >= threshold

    raw_regions: list[tuple[int, int]] = []
    region_start_offset: int | None = None
    for offset, is_active in enumerate(above_threshold):
        if is_active and region_start_offset is None:
            region_start_offset = offset
        elif not is_active and region_start_offset is not None:
            raw_regions.append(
                (
                    int(search_indices[region_start_offset]),
                    int(search_indices[offset - 1]),
                )
            )
            region_start_offset = None

    if region_start_offset is not None:
        raw_regions.append(
            (
                int(search_indices[region_start_offset]),
                int(search_indices[-1]),
            )
        )

    if not raw_regions:
        return [], threshold

    merged_regions: list[tuple[int, int]] = []
    current_start, current_end = raw_regions[0]
    for next_start, next_end in raw_regions[1:]:
        gap_seconds = float(timestamps_s[next_start] - timestamps_s[current_end])
        if gap_seconds <= merge_event_gap_seconds:
            current_end = next_end
        else:
            merged_regions.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_regions.append((current_start, current_end))

    filtered_regions: list[tuple[int, int]] = []
    for start_index, end_index in merged_regions:
        duration_seconds = float(timestamps_s[end_index] - timestamps_s[start_index])
        if duration_seconds >= min_event_duration_seconds:
            filtered_regions.append((start_index, end_index))

    return filtered_regions, threshold


def find_two_shake_events(
    signal: MotionSignal,
    shake_search_seconds: float,
    min_event_separation_seconds: float,
    event_threshold_ratio: float,
    merge_event_gap_seconds: float,
    min_event_duration_seconds: float,
) -> list[ShakePeak]:
    """
    在录制开头搜索窗口中找两次 shake event。

    做法：
        1. 只看 timestamps[0] 到 timestamps[0] + shake_search_seconds。
        2. 根据阈值把“连续剧烈运动区域”合并成 shake event。
        3. 每个 event 内只取最大值，最大值时间就是 shake time。
        4. 最后按时间从早到晚排序，第 1 个用于估计 offset，第 2 个用于验证。
    """

    timestamps = signal.timestamps_s
    values = signal.smoothed_signal

    search_start_time = float(timestamps[0])
    search_end_time = search_start_time + shake_search_seconds if shake_search_seconds > 0 else float(timestamps[-1])
    search_mask = (timestamps >= search_start_time) & (timestamps <= search_end_time)
    search_indices = np.where(search_mask)[0]
    if len(search_indices) < 5:
        raise ValueError(
            f"{signal.name} 搜索窗口内样本太少，无法检测两次 shake："
            f"sample_count={len(search_indices)}, source={signal.source_path}"
        )

    baseline = robust_static_baseline(values[search_indices])

    event_regions, event_threshold = find_shake_event_regions(
        timestamps_s=timestamps,
        values=values,
        search_indices=search_indices,
        baseline=baseline,
        event_threshold_ratio=event_threshold_ratio,
        merge_event_gap_seconds=merge_event_gap_seconds,
        min_event_duration_seconds=min_event_duration_seconds,
    )

    # 如果两个 event 的间隔很短，保守地合并，避免一次长晃动被尾巴拆成两次。
    merged_by_separation: list[tuple[int, int]] = []
    for start_index, end_index in event_regions:
        if not merged_by_separation:
            merged_by_separation.append((start_index, end_index))
            continue

        previous_start, previous_end = merged_by_separation[-1]
        separation_seconds = float(timestamps[start_index] - timestamps[previous_end])
        if separation_seconds < min_event_separation_seconds:
            merged_by_separation[-1] = (previous_start, end_index)
        else:
            merged_by_separation.append((start_index, end_index))

    if len(merged_by_separation) < 2:
        raise RuntimeError(
            f"{signal.name} 只检测到 {len(merged_by_separation)} 个 shake event，少于需要的 2 个。\n"
            f"可以尝试：\n"
            f"  1. 录制时按 静止->摇晃->静止->摇晃->静止 的流程重新录。\n"
            f"  2. 增大 --shake-search-seconds，例如 10 或 12。\n"
            f"  3. 减小 --min-event-separation-seconds，例如 0.7。\n"
            f"  4. 减小 --event-threshold-ratio，例如 0.2。\n"
            f"  4. 晃动幅度更大，让峰值明显高于静止段。"
        )

    peaks: list[ShakePeak] = []
    for peak_order, (event_start_index, event_end_index) in enumerate(merged_by_separation[:2], start=1):
        event_slice = slice(event_start_index, event_end_index + 1)
        local_peak_offset = int(np.argmax(values[event_slice]))
        selected_index = int(event_start_index + local_peak_offset)

        peak_signal = float(values[selected_index])
        peak_to_static_ratio = peak_signal / max(abs(baseline), 1e-9)
        peaks.append(
            ShakePeak(
                peak_order=peak_order,
                sample_index=int(selected_index),
                shake_time_s=float(timestamps[selected_index]),
                sample_time_s=float(timestamps[selected_index]),
                peak_signal=peak_signal,
                baseline_signal=baseline,
                event_threshold_signal=event_threshold,
                peak_to_static_ratio=peak_to_static_ratio,
                region_start_time_s=float(timestamps[event_start_index]),
                region_end_time_s=float(timestamps[event_end_index]),
            )
        )

    return peaks


def classify_sync_quality(validation_error_s: float) -> tuple[str, str]:
    """按照 D435i 一帧时间附近的经验阈值给同步质量分级。"""

    if validation_error_s < 0.015:
        return "excellent", "很好，小于 15 ms"
    if validation_error_s < 0.033:
        return "good", "可以，小于约一帧 60/30 FPS 图像时间"
    if validation_error_s < 0.050:
        return "usable", "第一版勉强可用，但已经接近明显错位"
    if validation_error_s < 0.100:
        return "poor", "不太可靠，建议检查峰值或改用两点线性映射"
    return "bad", "基本不可用，动作标签可能明显错位"


def robust_normalize_for_plot(values: np.ndarray) -> np.ndarray:
    """把不同量纲的信号归一化到 0~1，便于画在同一张图上。"""

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return np.zeros_like(values)

    low = float(np.percentile(finite_values, 5.0))
    high = float(np.percentile(finite_values, 95.0))
    if high <= low + 1e-12:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))

    if high <= low + 1e-12:
        return np.zeros_like(values)

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.2)


# ============================================================
# 7. 路径查找
# ============================================================


def is_path_inside_ignored_directory(path: Path, search_root: Path, ignored_names: set[str]) -> bool:
    """判断某个路径是否位于需要忽略的目录下面。"""

    try:
        relative_parts = path.relative_to(search_root).parts
    except ValueError:
        return False

    ignored_lower = {name.lower() for name in ignored_names}
    return any(part.lower() in ignored_lower for part in relative_parts[:-1])


def find_iphone_imu_csv(demo_dir: Path, odometry_csv_path: Path | None = None) -> Path:
    """
    在 session 里寻找 iPhone / Stray Scanner 的 imu.csv。

    优先使用 odometry.csv 同目录下的 imu.csv，因为这通常就是 Stray 导出文件夹。
    如果没有指定 odometry.csv，则递归查找非 d435i、非 processed_umi 目录下的 imu.csv。
    """

    if odometry_csv_path is not None:
        sibling_imu = odometry_csv_path.parent / "imu.csv"
        if sibling_imu.exists():
            return sibling_imu

    imu_candidates = []
    for path in demo_dir.rglob("imu.csv"):
        if not path.is_file():
            continue
        if is_path_inside_ignored_directory(path, demo_dir, IGNORED_SEARCH_DIR_NAMES):
            continue
        imu_candidates.append(path)

    if not imu_candidates:
        raise FileNotFoundError(
            f"没有找到 iPhone / Stray Scanner 的 imu.csv。请确认 Stray 导出文件夹已放入：{demo_dir}"
        )

    return sorted(imu_candidates, key=lambda item: (item.stat().st_mtime, str(item)))[-1]


def count_csv_data_rows(csv_path: Path) -> int:
    """统计 CSV 数据行数，不包含表头；用于自动跳过空录制 session。"""

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        row_count = sum(1 for _ in csv_file)

    return max(0, row_count - 1)


def find_latest_demo_dir_with_required_files(stray_root: Path) -> Path:
    """
    自动寻找最新的可测试 session。

    必须同时满足：
        1. 存在 d435i/imu.csv。
        2. d435i/imu.csv 至少有 3 行数据，才能检测 shake。
        3. session 内能找到 iPhone imu.csv。
    """

    if not stray_root.exists():
        raise FileNotFoundError(f"Stray 数据根目录不存在：{stray_root}")
    if not stray_root.is_dir():
        raise NotADirectoryError(f"Stray 数据根路径不是文件夹：{stray_root}")

    candidate_dirs = sorted(
        [path for path in stray_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for candidate_dir in candidate_dirs:
        d435i_imu_csv = candidate_dir / "d435i" / "imu.csv"
        if count_csv_data_rows(d435i_imu_csv) < 3:
            continue
        try:
            find_iphone_imu_csv(candidate_dir)
        except FileNotFoundError:
            continue
        return candidate_dir

    raise FileNotFoundError(
        f"没有在 {stray_root} 下找到同时包含 D435i imu.csv 和 iPhone imu.csv 的 session。"
    )


# ============================================================
# 8. 输出 CSV、JSON 和检查图
# ============================================================


def peak_to_dict(peak: ShakePeak) -> dict:
    """把 ShakePeak 转成 JSON 可保存的普通字典。"""

    return {
        "peak_order": peak.peak_order,
        "sample_index": peak.sample_index,
        "shake_time_s": peak.shake_time_s,
        "sample_time_s": peak.sample_time_s,
        "peak_signal": peak.peak_signal,
        "baseline_signal": peak.baseline_signal,
        "event_threshold_signal": peak.event_threshold_signal,
        "peak_to_static_ratio": peak.peak_to_static_ratio,
        "region_start_time_s": peak.region_start_time_s,
        "region_end_time_s": peak.region_end_time_s,
    }


def save_debug_csv(
    debug_csv_path: Path,
    d435i_signal: MotionSignal,
    iphone_signal: MotionSignal,
    offset_iphone_minus_d435i: float,
) -> None:
    """
    保存两侧 shake 信号，方便你之后不用重新读图片也能复查峰值。

    对 iPhone 行额外保存 time_on_d435i_s：
        time_on_d435i_s = t_iphone - offset
    """

    debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "source",
                "time_s",
                "time_on_d435i_s",
                "relative_time_s",
                "raw_signal",
                "smoothed_signal",
                "normalized_signal",
                "signal_kind",
                "source_path",
            ]
        )

        for source_name, signal in (("d435i_imu", d435i_signal), ("iphone", iphone_signal)):
            normalized = robust_normalize_for_plot(signal.smoothed_signal)
            start_time = float(signal.timestamps_s[0])
            for time_s, raw_value, smooth_value, normalized_value in zip(
                signal.timestamps_s,
                signal.raw_signal,
                signal.smoothed_signal,
                normalized,
            ):
                if source_name == "iphone":
                    time_on_d435i_s = time_s - offset_iphone_minus_d435i
                else:
                    time_on_d435i_s = time_s

                writer.writerow(
                    [
                        source_name,
                        f"{time_s:.9f}",
                        f"{time_on_d435i_s:.9f}",
                        f"{time_s - start_time:.9f}",
                        f"{raw_value:.9f}",
                        f"{smooth_value:.9f}",
                        f"{normalized_value:.9f}",
                        signal.signal_kind,
                        str(signal.source_path),
                    ]
                )


def load_matplotlib():
    """延迟导入 matplotlib，并切换到无界面后端，适合终端批处理。"""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，请先安装：pip install matplotlib") from exc

    return plt


def save_single_signal_plot(
    output_path: Path,
    signal: MotionSignal,
    peaks: list[ShakePeak],
    title: str,
    ylabel: str,
) -> None:
    """保存单侧 shake 信号图，并标出检测到的两次峰值。"""

    plt = load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    relative_time_s = signal.timestamps_s - signal.timestamps_s[0]
    plt.figure(figsize=(11, 5))
    plt.plot(relative_time_s, signal.raw_signal, color="0.75", linewidth=1.0, label="raw")
    plt.plot(relative_time_s, signal.smoothed_signal, color="tab:blue", linewidth=1.8, label="smoothed")

    for peak in peaks:
        event_start_x = peak.region_start_time_s - signal.timestamps_s[0]
        event_end_x = peak.region_end_time_s - signal.timestamps_s[0]
        plt.axvspan(event_start_x, event_end_x, alpha=0.10)
        peak_x = peak.shake_time_s - signal.timestamps_s[0]
        plt.axvline(peak_x, linestyle="--", linewidth=1.5, label=f"shake {peak.peak_order}")
        plt.scatter([peak_x], [peak.peak_signal], s=45)

    plt.xlabel("time from this recording start / s")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_aligned_signal_plot(
    output_path: Path,
    d435i_signal: MotionSignal,
    iphone_signal: MotionSignal,
    d435i_peaks: list[ShakePeak],
    iphone_peaks: list[ShakePeak],
    offset_iphone_minus_d435i: float,
    validation_error_s: float,
    quality: str,
) -> None:
    """
    保存最重要的对齐检查图。

    iPhone 时间轴会先转换到 D435i 时间轴：
        t_D_from_iPhone = t_I - offset
    然后和 D435i IMU shake 信号画在同一张图上。
    """

    plt = load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    d435i_start_time = float(d435i_signal.timestamps_s[0])
    d435i_x = d435i_signal.timestamps_s - d435i_start_time
    iphone_time_on_d435i = iphone_signal.timestamps_s - offset_iphone_minus_d435i
    iphone_x = iphone_time_on_d435i - d435i_start_time

    d435i_y = robust_normalize_for_plot(d435i_signal.smoothed_signal)
    iphone_y = robust_normalize_for_plot(iphone_signal.smoothed_signal)

    plt.figure(figsize=(12, 6))
    plt.plot(d435i_x, d435i_y, linewidth=1.8, label="D435i IMU gyro/accel")
    plt.plot(iphone_x, iphone_y, linewidth=1.8, label="iPhone gyro/accel mapped to D435i time")

    for peak in d435i_peaks:
        event_start_x = peak.region_start_time_s - d435i_start_time
        event_end_x = peak.region_end_time_s - d435i_start_time
        plt.axvspan(event_start_x, event_end_x, color="tab:blue", alpha=0.08)
        peak_x = peak.shake_time_s - d435i_start_time
        plt.axvline(peak_x, color="tab:blue", linestyle="--", alpha=0.7)
        plt.text(peak_x, 1.05, f"D{peak.peak_order}", color="tab:blue", ha="center")

    for peak in iphone_peaks:
        event_start_x = (peak.region_start_time_s - offset_iphone_minus_d435i) - d435i_start_time
        event_end_x = (peak.region_end_time_s - offset_iphone_minus_d435i) - d435i_start_time
        plt.axvspan(event_start_x, event_end_x, color="tab:orange", alpha=0.08)
        peak_x = (peak.shake_time_s - offset_iphone_minus_d435i) - d435i_start_time
        plt.axvline(peak_x, color="tab:orange", linestyle=":", alpha=0.9)
        plt.text(peak_x, 0.92, f"I{peak.peak_order}", color="tab:orange", ha="center")

    plt.xlabel("D435i time axis, from first D435i motion sample / s")
    plt.ylabel("normalized signal")
    plt.title(
        "Aligned shake signals "
        f"(second-shake error = {validation_error_s * 1000.0:.1f} ms, quality = {quality})"
    )
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_sync_json(sync_json_path: Path, sync_info: dict) -> None:
    """保存 sync.json，供后续处理和人工复查。"""

    sync_json_path.parent.mkdir(parents=True, exist_ok=True)
    sync_json_path.write_text(
        json.dumps(sync_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 9. 主流程
# ============================================================


def run_shake_sync_test(
    demo_dir: Path,
    output_dir_name: str,
    test_dir_name: str,
    odometry_csv_path: Path | None,
    shake_search_seconds: float,
    smooth_window_seconds: float,
    min_event_separation_seconds: float,
    event_threshold_ratio: float,
    merge_event_gap_seconds: float,
    min_event_duration_seconds: float,
) -> dict:
    """执行完整的两次 shake 同步测试。"""

    demo_dir = demo_dir.resolve()
    d435i_imu_csv = demo_dir / "d435i" / "imu.csv"
    if not d435i_imu_csv.exists():
        raise FileNotFoundError(f"没有找到 D435i IMU 文件：{d435i_imu_csv}")

    iphone_imu_csv = find_iphone_imu_csv(demo_dir=demo_dir, odometry_csv_path=odometry_csv_path)

    output_dir = demo_dir / output_dir_name
    test_dir = output_dir / test_dir_name

    iphone_signal = load_iphone_imu_signal(
        imu_csv_path=iphone_imu_csv,
        smooth_window_seconds=smooth_window_seconds,
    )
    d435i_signal = load_d435i_imu_signal(
        imu_csv_path=d435i_imu_csv,
        smooth_window_seconds=smooth_window_seconds,
    )

    iphone_peaks = find_two_shake_events(
        signal=iphone_signal,
        shake_search_seconds=shake_search_seconds,
        min_event_separation_seconds=min_event_separation_seconds,
        event_threshold_ratio=event_threshold_ratio,
        merge_event_gap_seconds=merge_event_gap_seconds,
        min_event_duration_seconds=min_event_duration_seconds,
    )
    d435i_peaks = find_two_shake_events(
        signal=d435i_signal,
        shake_search_seconds=shake_search_seconds,
        min_event_separation_seconds=min_event_separation_seconds,
        event_threshold_ratio=event_threshold_ratio,
        merge_event_gap_seconds=merge_event_gap_seconds,
        min_event_duration_seconds=min_event_duration_seconds,
    )

    iphone_first_peak, iphone_second_peak = iphone_peaks[0], iphone_peaks[1]
    d435i_first_peak, d435i_second_peak = d435i_peaks[0], d435i_peaks[1]

    offset_iphone_minus_d435i = iphone_first_peak.shake_time_s - d435i_first_peak.shake_time_s
    iphone_second_time_on_d435i = iphone_second_peak.shake_time_s - offset_iphone_minus_d435i
    validation_error_s = abs(iphone_second_time_on_d435i - d435i_second_peak.shake_time_s)
    quality, quality_description = classify_sync_quality(validation_error_s)

    # 两次 shake 也可以刚好解出一条线性映射 t_iphone = a * t_d435i + b。
    # 这里仅作为诊断输出：如果 single offset 的第二次误差很大，而 affine 的 a 明显偏离 1，
    # 就说明可能存在时间轴漂移或两侧峰值定义不一致。
    d_delta = d435i_second_peak.shake_time_s - d435i_first_peak.shake_time_s
    i_delta = iphone_second_peak.shake_time_s - iphone_first_peak.shake_time_s
    if abs(d_delta) > 1e-9:
        affine_a = i_delta / d_delta
        affine_b = iphone_first_peak.shake_time_s - affine_a * d435i_first_peak.shake_time_s
    else:
        affine_a = float("nan")
        affine_b = float("nan")

    sync_info = {
        "method": "two_shakes_d435i_imu_validation",
        "mapping": "t_iphone = t_d435i + offset_iphone_minus_d435i",
        "aligned_plot_mapping": "t_d435i_from_iphone = t_iphone - offset_iphone_minus_d435i",
        "offset_iphone_minus_d435i": offset_iphone_minus_d435i,
        "validation_error_s": validation_error_s,
        "validation_error_ms": validation_error_s * 1000.0,
        "quality": quality,
        "quality_description": quality_description,
        "first_shake": {
            "d435i_shake_time_s": d435i_first_peak.shake_time_s,
            "iphone_shake_time_s": iphone_first_peak.shake_time_s,
            "iphone_time_on_d435i_s": iphone_first_peak.shake_time_s - offset_iphone_minus_d435i,
            "used_for": "estimate_offset",
        },
        "second_shake": {
            "d435i_shake_time_s": d435i_second_peak.shake_time_s,
            "iphone_shake_time_s": iphone_second_peak.shake_time_s,
            "iphone_time_on_d435i_s": iphone_second_time_on_d435i,
            "used_for": "validate_offset",
        },
        "affine_mapping_candidate": {
            "mapping": "t_iphone = a * t_d435i + b",
            "a": affine_a,
            "b": affine_b,
            "note": "这里只作为诊断参考；正式验证仍使用第一次 shake 得到的 single offset。",
        },
        "d435i_peaks": [peak_to_dict(peak) for peak in d435i_peaks],
        "iphone_peaks": [peak_to_dict(peak) for peak in iphone_peaks],
        "parameters": {
            "shake_search_seconds": shake_search_seconds,
            "smooth_window_seconds": smooth_window_seconds,
            "min_event_separation_seconds": min_event_separation_seconds,
            "event_threshold_ratio": event_threshold_ratio,
            "merge_event_gap_seconds": merge_event_gap_seconds,
            "min_event_duration_seconds": min_event_duration_seconds,
        },
        "input_files": {
            "demo_dir": str(demo_dir),
            "d435i_imu_csv": str(d435i_imu_csv),
            "iphone_imu_csv": str(iphone_imu_csv),
        },
        "output_files": {
            "sync_json": str(output_dir / "sync.json"),
            "debug_csv": str(output_dir / "shake_sync_debug.csv"),
            "iphone_shake_signal_png": str(test_dir / "iphone_shake_signal.png"),
            "d435i_motion_signal_png": str(test_dir / "d435i_motion_signal.png"),
            "aligned_shake_signals_png": str(test_dir / "aligned_shake_signals.png"),
        },
    }

    write_sync_json(output_dir / "sync.json", sync_info)
    save_debug_csv(
        debug_csv_path=output_dir / "shake_sync_debug.csv",
        d435i_signal=d435i_signal,
        iphone_signal=iphone_signal,
        offset_iphone_minus_d435i=offset_iphone_minus_d435i,
    )
    save_single_signal_plot(
        output_path=test_dir / "iphone_shake_signal.png",
        signal=iphone_signal,
        peaks=iphone_peaks,
        title="iPhone shake signal",
        ylabel=f"{iphone_signal.signal_kind} norm",
    )
    save_single_signal_plot(
        output_path=test_dir / "d435i_motion_signal.png",
        signal=d435i_signal,
        peaks=d435i_peaks,
        title="D435i IMU shake signal",
        ylabel=f"{d435i_signal.signal_kind} norm",
    )
    save_aligned_signal_plot(
        output_path=test_dir / "aligned_shake_signals.png",
        d435i_signal=d435i_signal,
        iphone_signal=iphone_signal,
        d435i_peaks=d435i_peaks,
        iphone_peaks=iphone_peaks,
        offset_iphone_minus_d435i=offset_iphone_minus_d435i,
        validation_error_s=validation_error_s,
        quality=quality,
    )

    return sync_info


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Validate iPhone-D435i time sync with two shakes and D435i IMU."
    )
    parser.add_argument(
        "--stray-root",
        type=Path,
        default=DEFAULT_STRAY_ROOT,
        help=f"Stray 数据根目录，默认：{DEFAULT_STRAY_ROOT}",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=None,
        help="指定某个 data_MMDD_HHMM session；不填时自动寻找最新有效 session。",
    )
    parser.add_argument(
        "--odometry-csv",
        type=Path,
        default=None,
        help="可选：指定 odometry.csv，用它同目录下的 imu.csv 作为 iPhone IMU。",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default=DEFAULT_OUTPUT_DIR_NAME,
        help=f"输出目录名，默认：{DEFAULT_OUTPUT_DIR_NAME}",
    )
    parser.add_argument(
        "--test-dir-name",
        type=str,
        default=DEFAULT_TEST_DIR_NAME,
        help=f"检查图目录名，默认：{DEFAULT_TEST_DIR_NAME}",
    )
    parser.add_argument(
        "--shake-search-seconds",
        type=float,
        default=DEFAULT_SHAKE_SEARCH_SECONDS,
        help=f"从每个信号开头搜索 shake 的秒数，默认：{DEFAULT_SHAKE_SEARCH_SECONDS}",
    )
    parser.add_argument(
        "--smooth-window-seconds",
        type=float,
        default=DEFAULT_SMOOTH_WINDOW_SECONDS,
        help=f"信号平滑窗口，默认：{DEFAULT_SMOOTH_WINDOW_SECONDS}",
    )
    parser.add_argument(
        "--min-event-separation-seconds",
        type=float,
        default=DEFAULT_MIN_EVENT_SEPARATION_SECONDS,
        help=f"两次 shake event 的最小间隔，默认：{DEFAULT_MIN_EVENT_SEPARATION_SECONDS}",
    )
    parser.add_argument(
        "--event-threshold-ratio",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD_RATIO,
        help=f"shake event 检测阈值相对高度，默认：{DEFAULT_EVENT_THRESHOLD_RATIO}",
    )
    parser.add_argument(
        "--merge-event-gap-seconds",
        type=float,
        default=DEFAULT_MERGE_EVENT_GAP_SECONDS,
        help=f"低于阈值的短间隔小于该值时合并为同一个 event，默认：{DEFAULT_MERGE_EVENT_GAP_SECONDS}",
    )
    parser.add_argument(
        "--min-event-duration-seconds",
        type=float,
        default=DEFAULT_MIN_EVENT_DURATION_SECONDS,
        help=f"shake event 最短持续时间，默认：{DEFAULT_MIN_EVENT_DURATION_SECONDS}",
    )
    parser.add_argument(
        "--min-peak-separation-seconds",
        dest="min_event_separation_seconds",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--peak-region-ratio",
        dest="event_threshold_ratio",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    demo_dir = args.demo_dir
    if demo_dir is None:
        demo_dir = find_latest_demo_dir_with_required_files(args.stray_root)

    sync_info = run_shake_sync_test(
        demo_dir=demo_dir,
        output_dir_name=args.output_dir_name,
        test_dir_name=args.test_dir_name,
        odometry_csv_path=args.odometry_csv,
        shake_search_seconds=args.shake_search_seconds,
        smooth_window_seconds=args.smooth_window_seconds,
        min_event_separation_seconds=args.min_event_separation_seconds,
        event_threshold_ratio=args.event_threshold_ratio,
        merge_event_gap_seconds=args.merge_event_gap_seconds,
        min_event_duration_seconds=args.min_event_duration_seconds,
    )

    output_files = sync_info["output_files"]
    print("shake 同步测试完成：")
    print(f"  demo_dir：{sync_info['input_files']['demo_dir']}")
    print(f"  offset_iphone_minus_d435i：{sync_info['offset_iphone_minus_d435i']:.9f} s")
    print(f"  第二次 shake 验证误差：{sync_info['validation_error_ms']:.3f} ms")
    print(f"  质量判断：{sync_info['quality']}，{sync_info['quality_description']}")
    print(f"  sync.json：{output_files['sync_json']}")
    print(f"  aligned_shake_signals.png：{output_files['aligned_shake_signals_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
