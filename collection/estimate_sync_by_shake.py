"""
用一次摇晃估计 iPhone 时间轴和 D435i/电脑时间轴的偏移量。

核心思路：
    1. 读取 D435i 的 imu.csv，使用 system_time_s 作为电脑时间轴。
    2. 读取 iPhone/Stray 导出的 imu.csv，使用 timestamp 作为 iPhone 时间轴。
    3. 优先用 gyro 三轴模长作为摇晃信号；没有 gyro 时再用 accel。
    4. 只在录制开头的 --shake-search-seconds 秒内找第一个 shake event。
       连续超过阈值的一整段剧烈运动只算一个 event，event 内取最大峰值时间。
    5. 得到：
           offset = t_iphone_shake - t_d435i_shake
       后续映射：
           t_iphone = t_d435i + offset

输出：
    processed_umi/sync.json
    processed_umi/shake_debug.csv
    processed_umi/test/shake_signal.png
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xac_config import DATA_ROOT


DEFAULT_STRAY_ROOT = DATA_ROOT
DEFAULT_OUTPUT_DIR_NAME = "processed_umi"
DEFAULT_TEST_DIR_NAME = "test"
DEFAULT_SHAKE_SEARCH_SECONDS = 8.0
DEFAULT_SMOOTH_WINDOW_SECONDS = 0.15

# 连续超过这个相对阈值的区域会被视为一个 shake event。
# 阈值计算方式：
#     threshold = static_baseline + (global_peak - static_baseline) * ratio
DEFAULT_EVENT_THRESHOLD_RATIO = 0.25

# 同一次摇晃内部可能短暂低于阈值；如果两个高阈值片段间隔很短，就合并为一个 event。
DEFAULT_MERGE_EVENT_GAP_SECONDS = 0.35

# 太短的阈值穿越通常是噪声或单个异常采样点，不作为同步 shake event。
DEFAULT_MIN_EVENT_DURATION_SECONDS = 0.05

# 保留旧常量名，兼容之前已经调用 estimate_sync_for_demo(..., peak_threshold_ratio=...) 的代码。
DEFAULT_PEAK_THRESHOLD_RATIO = DEFAULT_EVENT_THRESHOLD_RATIO

IGNORED_SEARCH_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    DEFAULT_OUTPUT_DIR_NAME,
}


@dataclass(frozen=True)
class MotionSignal:
    """保存某个 IMU 文件解析得到的摇晃信号。"""

    source_path: Path
    signal_kind: str
    time_column: str
    timestamps_s: np.ndarray
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray


@dataclass(frozen=True)
class ShakePeak:
    """保存单侧 IMU 检测到的 shake event 及其内部最大峰值。"""

    shake_time_s: float
    peak_time_s: float
    peak_signal: float
    baseline_signal: float
    event_threshold_signal: float
    region_start_time_s: float
    region_end_time_s: float
    search_start_time_s: float
    search_end_time_s: float
    sample_count_in_search: int


def normalize_column_name(name: str) -> str:
    """把 CSV 列名归一化，方便兼容不同 IMU 导出格式。"""

    return "".join(char.lower() for char in name.strip() if char.isalnum())


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV，并把每一行的 key/value 都 strip。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头：{csv_path}")

        columns = [column.strip() for column in reader.fieldnames]
        rows = []
        for raw_row in reader:
            rows.append(
                {
                    key.strip(): value.strip() if isinstance(value, str) else value
                    for key, value in raw_row.items()
                    if key is not None
                }
            )

    return columns, rows


def build_normalized_column_map(columns: list[str]) -> dict[str, str]:
    """建立 normalized_name -> original_name 的映射。"""

    return {normalize_column_name(column): column for column in columns}


def choose_time_column(columns: list[str], prefer_computer_time: bool) -> tuple[str, float]:
    """
    选择时间列。

    返回：
        (原始列名, scale)

    scale 用来把原始数值转换成秒；如果列名是 *_ms，则 scale=0.001。

    对电脑时间轴(D435i)优先选择 capture_system_time_s：它是 record_435i.py 用
    RealSense 传感器时间戳换算出的真实拍摄时刻(墙钟)，不含帧队列积压延迟；
    旧文件没有该列时依次回退 system_time_s / 其它时间列。
    """

    normalized_map = build_normalized_column_map(columns)

    if prefer_computer_time:
        candidates = [
            ("capturesystemtimes", 1.0),
            ("systemtimes", 1.0),
            ("computertimes", 1.0),
            ("times", 1.0),
            ("timestamp", 1.0),
            ("sensortimestampms", 0.001),
            ("timestampms", 0.001),
        ]
    else:
        candidates = [
            ("timestamp", 1.0),
            ("timestamps", 1.0),
            ("time", 1.0),
            ("times", 1.0),
            ("sensortimestampms", 0.001),
            ("timestampms", 0.001),
        ]

    for normalized_name, scale in candidates:
        if normalized_name in normalized_map:
            return normalized_map[normalized_name], scale

    raise KeyError(f"没有找到可用时间列。当前列名：{columns}")


def parse_float_or_nan(value) -> float:
    """把 CSV 字符串解析成 float，失败则返回 NaN。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def find_axis_columns(columns: list[str], signal_kind: str) -> tuple[str, str, str] | None:
    """
    查找 gyro/accel 三轴列。

    支持常见命名：
        gx, gy, gz
        gyro_x, gyro_y, gyro_z
        alpha_x, alpha_y, alpha_z (Stray Scanner rotation rate)
        angularVelocityX, angularVelocityY, angularVelocityZ
        ax, ay, az
        accel_x, accel_y, accel_z
    """

    normalized_map = build_normalized_column_map(columns)

    if signal_kind == "gyro":
        candidate_triples = [
            ("gx", "gy", "gz"),
            ("gyrox", "gyroy", "gyroz"),
            ("gyrx", "gyry", "gyrz"),
            ("angularvelocityx", "angularvelocityy", "angularvelocityz"),
            ("rotationratex", "rotationratey", "rotationratez"),
            ("alphax", "alphay", "alphaz"),
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

    D435i 的 imu.csv 就是这个格式：
        stream = accel / gyro
        x, y, z = 三轴读数
    """

    normalized_map = build_normalized_column_map(columns)
    if "stream" not in normalized_map:
        return None

    x_column = normalized_map.get("x")
    y_column = normalized_map.get("y")
    z_column = normalized_map.get("z")
    if x_column is None or y_column is None or z_column is None:
        return None

    stream_column = normalized_map["stream"]
    timestamps = []
    values = []

    for row in rows:
        stream_text = str(row.get(stream_column, "")).strip().lower()
        if signal_kind == "gyro" and "gyro" not in stream_text:
            continue
        if signal_kind == "accel" and not ("accel" in stream_text or "accelerometer" in stream_text):
            continue

        timestamp = parse_float_or_nan(row.get(time_column)) * time_scale
        x_value = parse_float_or_nan(row.get(x_column))
        y_value = parse_float_or_nan(row.get(y_column))
        z_value = parse_float_or_nan(row.get(z_column))
        timestamps.append(timestamp)
        values.append([x_value, y_value, z_value])

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
    """解析独立三轴列格式的 IMU CSV。"""

    axis_columns = find_axis_columns(columns, signal_kind)
    if axis_columns is None:
        return None

    x_column, y_column, z_column = axis_columns
    timestamps = []
    values = []

    for row in rows:
        timestamp = parse_float_or_nan(row.get(time_column)) * time_scale
        x_value = parse_float_or_nan(row.get(x_column))
        y_value = parse_float_or_nan(row.get(y_column))
        z_value = parse_float_or_nan(row.get(z_column))
        timestamps.append(timestamp)
        values.append([x_value, y_value, z_value])

    values_array = np.asarray(values, dtype=np.float64)
    signal = np.linalg.norm(values_array, axis=1)
    return np.asarray(timestamps, dtype=np.float64), signal


def smooth_signal(timestamps_s: np.ndarray, raw_signal: np.ndarray, smooth_window_seconds: float) -> np.ndarray:
    """对 IMU 信号做简单 moving average 平滑。"""

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


def load_motion_signal(
    imu_csv_path: Path,
    prefer_computer_time: bool,
    smooth_window_seconds: float = DEFAULT_SMOOTH_WINDOW_SECONDS,
) -> MotionSignal:
    """
    从 IMU CSV 中加载 shake 信号。

    优先选择 gyro；如果 CSV 没有 gyro 字段，再退回 accel。
    """

    columns, rows = read_csv_rows(imu_csv_path)
    time_column, time_scale = choose_time_column(columns, prefer_computer_time=prefer_computer_time)

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
        raise ValueError(f"IMU 有效数据太少，无法找 shake：{imu_csv_path}")

    order = np.argsort(timestamps_s)
    timestamps_s = timestamps_s[order]
    raw_signal = raw_signal[order]
    smoothed_signal = smooth_signal(timestamps_s, raw_signal, smooth_window_seconds)

    return MotionSignal(
        source_path=imu_csv_path,
        signal_kind=selected_kind,
        time_column=time_column,
        timestamps_s=timestamps_s,
        raw_signal=raw_signal,
        smoothed_signal=smoothed_signal,
    )


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

    一次摇晃里可能有多个局部峰，例如“晃过去、回弹、停住”。
    这里先把连续超过阈值的区域视为一个候选 event，再把短间隔候选区域合并，
    最后每个 event 内只取最大峰值时间作为 shake time。
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


def estimate_peak_time(
    signal: MotionSignal,
    shake_search_seconds: float,
    peak_threshold_ratio: float = DEFAULT_EVENT_THRESHOLD_RATIO,
    merge_event_gap_seconds: float = DEFAULT_MERGE_EVENT_GAP_SECONDS,
    min_event_duration_seconds: float = DEFAULT_MIN_EVENT_DURATION_SECONDS,
) -> ShakePeak:
    """
    在开头搜索窗口内估计 shake event 的时间。

    注意：
        这里不是把每个局部峰都当成一次 shake。
        程序会先检测连续剧烈运动区域，再在第一个 shake event 内取最大峰值时间。
    """

    timestamps = signal.timestamps_s
    smoothed = signal.smoothed_signal

    search_start_time = float(timestamps[0])
    if shake_search_seconds > 0:
        search_end_time = search_start_time + shake_search_seconds
        search_mask = timestamps <= search_end_time
    else:
        search_end_time = float(timestamps[-1])
        search_mask = np.ones_like(timestamps, dtype=bool)

    search_indices = np.where(search_mask)[0]
    if len(search_indices) < 3:
        raise ValueError(
            f"shake 搜索窗口内 IMU 样本太少：{signal.source_path}，"
            f"shake_search_seconds={shake_search_seconds}"
        )

    baseline = robust_static_baseline(smoothed[search_indices])
    event_regions, event_threshold = find_shake_event_regions(
        timestamps_s=timestamps,
        values=smoothed,
        search_indices=search_indices,
        baseline=baseline,
        event_threshold_ratio=peak_threshold_ratio,
        merge_event_gap_seconds=merge_event_gap_seconds,
        min_event_duration_seconds=min_event_duration_seconds,
    )
    if not event_regions:
        raise RuntimeError(
            f"没有在 shake 搜索窗口内检测到 shake event：{signal.source_path}\n"
            f"可以尝试：\n"
            f"  1. 采集开头做一次更明显、更短促的 shake。\n"
            f"  2. 增大 --shake-search-seconds，例如 10 或 12。\n"
            f"  3. 减小 event threshold ratio，例如 0.2。"
        )

    # 正式处理阶段只需要一次同步 shake；如果同一段连续运动内有多个峰，
    # 只在第一个 shake event 内取最大值，不把它们拆成多次同步。
    event_start_index, event_end_index = event_regions[0]
    event_slice = slice(event_start_index, event_end_index + 1)
    local_peak_offset = int(np.argmax(smoothed[event_slice]))
    peak_index = int(event_start_index + local_peak_offset)
    peak_signal = float(smoothed[peak_index])
    if not np.isfinite(peak_signal) or peak_signal <= 0:
        raise ValueError(f"shake 峰值无效：{signal.source_path}")

    last_search_index = int(search_indices[-1])

    return ShakePeak(
        shake_time_s=float(timestamps[peak_index]),
        peak_time_s=float(timestamps[peak_index]),
        peak_signal=peak_signal,
        baseline_signal=baseline,
        event_threshold_signal=event_threshold,
        region_start_time_s=float(timestamps[event_start_index]),
        region_end_time_s=float(timestamps[event_end_index]),
        search_start_time_s=search_start_time,
        search_end_time_s=float(timestamps[last_search_index]),
        sample_count_in_search=int(len(search_indices)),
    )


def is_path_inside_ignored_directory(path: Path, search_root: Path, ignored_names: set[str]) -> bool:
    """判断路径是否位于应忽略的目录内。"""

    try:
        relative_parts = path.relative_to(search_root).parts
    except ValueError:
        return False

    ignored_lower = {name.lower() for name in ignored_names}
    return any(part.lower() in ignored_lower for part in relative_parts[:-1])


def find_iphone_imu_csv(demo_dir: Path, odometry_csv_path: Path | None = None) -> Path:
    """
    在 session 中寻找 iPhone/Stray 的 imu.csv。

    优先使用 odometry.csv 同目录下的 imu.csv。
    其次递归查找非 d435i、非 processed_umi 目录下的 imu.csv。
    """

    if odometry_csv_path is not None:
        sibling_imu = odometry_csv_path.parent / "imu.csv"
        if sibling_imu.exists():
            return sibling_imu

    ignored_names = set(IGNORED_SEARCH_DIR_NAMES)
    ignored_names.add("d435i")

    imu_candidates = []
    for path in demo_dir.rglob("imu.csv"):
        if not path.is_file():
            continue
        if is_path_inside_ignored_directory(path, demo_dir, ignored_names):
            continue
        imu_candidates.append(path)

    if not imu_candidates:
        raise FileNotFoundError(
            f"没有找到 iPhone/Stray 的 imu.csv。请确认 iPhone 导出文件夹已放入：{demo_dir}"
        )

    return sorted(imu_candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1]


def find_d435i_imu_csv(demo_dir: Path) -> Path:
    """返回 d435i/imu.csv。"""

    imu_path = demo_dir / "d435i" / "imu.csv"
    if not imu_path.exists():
        raise FileNotFoundError(f"没有找到 D435i IMU 文件：{imu_path}")
    return imu_path


def find_latest_demo_dir_with_required_files(stray_root: Path) -> Path:
    """单独运行本脚本时，自动寻找最新且包含 d435i/imu.csv 的 session。"""

    if not stray_root.exists():
        raise FileNotFoundError(f"Stray 数据根目录不存在：{stray_root}")

    candidate_dirs = [path for path in stray_root.iterdir() if path.is_dir()]
    valid_dirs = []
    for candidate_dir in candidate_dirs:
        if (candidate_dir / "d435i" / "imu.csv").exists():
            valid_dirs.append(candidate_dir)

    if not valid_dirs:
        raise FileNotFoundError(f"没有在 {stray_root} 下找到包含 d435i/imu.csv 的 session。")

    return max(valid_dirs, key=lambda path: path.stat().st_mtime)


def save_shake_debug_csv(output_dir: Path, d435i_signal: MotionSignal, iphone_signal: MotionSignal) -> Path:
    """保存两侧 shake 信号，方便人工复查峰值。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "shake_debug.csv"

    with debug_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source", "time_s", "raw_signal", "smoothed_signal", "signal_kind", "source_path"])

        for signal_name, signal in (("d435i", d435i_signal), ("iphone", iphone_signal)):
            for time_s, raw_value, smooth_value in zip(
                signal.timestamps_s,
                signal.raw_signal,
                signal.smoothed_signal,
            ):
                writer.writerow(
                    [
                        signal_name,
                        f"{time_s:.9f}",
                        f"{raw_value:.9f}",
                        f"{smooth_value:.9f}",
                        signal.signal_kind,
                        str(signal.source_path),
                    ]
                )

    return debug_path


def save_shake_signal_plot(
    test_dir: Path,
    d435i_signal: MotionSignal,
    iphone_signal: MotionSignal,
    d435i_peak: ShakePeak,
    iphone_peak: ShakePeak,
) -> bool:
    """保存 shake 信号图。"""

    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (test_dir / "shake_plot_warning.txt").write_text(
            "当前环境缺少 matplotlib，所以没有生成 shake_signal.png。\n",
            encoding="utf-8",
        )
        return False

    plt.figure(figsize=(11, 6))

    d435i_relative_time = d435i_signal.timestamps_s - d435i_signal.timestamps_s[0]
    iphone_relative_time = iphone_signal.timestamps_s - iphone_signal.timestamps_s[0]
    plt.plot(
        d435i_relative_time,
        d435i_signal.smoothed_signal,
        label=f"D435i {d435i_signal.signal_kind}",
    )
    plt.plot(
        iphone_relative_time,
        iphone_signal.smoothed_signal,
        label=f"iPhone {iphone_signal.signal_kind}",
    )
    plt.axvspan(
        d435i_peak.region_start_time_s - d435i_signal.timestamps_s[0],
        d435i_peak.region_end_time_s - d435i_signal.timestamps_s[0],
        color="tab:blue",
        alpha=0.10,
    )
    plt.axvspan(
        iphone_peak.region_start_time_s - iphone_signal.timestamps_s[0],
        iphone_peak.region_end_time_s - iphone_signal.timestamps_s[0],
        color="tab:orange",
        alpha=0.10,
    )
    plt.axvline(
        d435i_peak.shake_time_s - d435i_signal.timestamps_s[0],
        color="tab:blue",
        linestyle="--",
        label="D435i shake",
    )
    plt.axvline(
        iphone_peak.shake_time_s - iphone_signal.timestamps_s[0],
        color="tab:orange",
        linestyle="--",
        label="iPhone shake",
    )
    plt.xlabel("time from each IMU start / s")
    plt.ylabel("smoothed IMU norm")
    plt.title("Single-shake sync signal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(test_dir / "shake_signal.png", dpi=200)
    plt.close()
    return True


def write_sync_json(output_dir: Path, sync_info: dict) -> Path:
    """保存 sync.json。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    sync_path = output_dir / "sync.json"
    sync_path.write_text(json.dumps(sync_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sync_path


def estimate_sync_for_demo(
    demo_dir: Path,
    odometry_csv_path: Path | None,
    output_dir: Path,
    test_dir: Path,
    shake_search_seconds: float = DEFAULT_SHAKE_SEARCH_SECONDS,
    smooth_window_seconds: float = DEFAULT_SMOOTH_WINDOW_SECONDS,
    peak_threshold_ratio: float = DEFAULT_PEAK_THRESHOLD_RATIO,
    merge_event_gap_seconds: float = DEFAULT_MERGE_EVENT_GAP_SECONDS,
    min_event_duration_seconds: float = DEFAULT_MIN_EVENT_DURATION_SECONDS,
) -> dict:
    """
    对单条 demo 估计 iPhone 时间轴到 D435i/电脑时间轴的 offset。

    返回的 sync_info 可直接用于：
        t_iphone = t_d435i + offset_iphone_minus_d435i
    """

    d435i_imu_path = find_d435i_imu_csv(demo_dir)
    iphone_imu_path = find_iphone_imu_csv(demo_dir, odometry_csv_path=odometry_csv_path)

    d435i_signal = load_motion_signal(
        d435i_imu_path,
        prefer_computer_time=True,
        smooth_window_seconds=smooth_window_seconds,
    )
    iphone_signal = load_motion_signal(
        iphone_imu_path,
        prefer_computer_time=False,
        smooth_window_seconds=smooth_window_seconds,
    )

    d435i_peak = estimate_peak_time(
        d435i_signal,
        shake_search_seconds=shake_search_seconds,
        peak_threshold_ratio=peak_threshold_ratio,
        merge_event_gap_seconds=merge_event_gap_seconds,
        min_event_duration_seconds=min_event_duration_seconds,
    )
    iphone_peak = estimate_peak_time(
        iphone_signal,
        shake_search_seconds=shake_search_seconds,
        peak_threshold_ratio=peak_threshold_ratio,
        merge_event_gap_seconds=merge_event_gap_seconds,
        min_event_duration_seconds=min_event_duration_seconds,
    )
    offset = iphone_peak.shake_time_s - d435i_peak.shake_time_s

    sync_info = {
        "method": "single_shake_event_imu",
        "mapping": "t_iphone = t_d435i + offset_iphone_minus_d435i",
        "offset_iphone_minus_d435i": offset,
        "d435i_shake_time_s": d435i_peak.shake_time_s,
        "iphone_shake_time_s": iphone_peak.shake_time_s,
        "shake_search_seconds": shake_search_seconds,
        "smooth_window_seconds": smooth_window_seconds,
        "event_threshold_ratio": peak_threshold_ratio,
        "merge_event_gap_seconds": merge_event_gap_seconds,
        "min_event_duration_seconds": min_event_duration_seconds,
        # 保留旧键名，方便已有脚本或旧 summary 读取。
        "peak_threshold_ratio": peak_threshold_ratio,
        "d435i_imu_csv": str(d435i_imu_path),
        "iphone_imu_csv": str(iphone_imu_path),
        "d435i_signal_kind": d435i_signal.signal_kind,
        "iphone_signal_kind": iphone_signal.signal_kind,
        "d435i_peak": d435i_peak.__dict__,
        "iphone_peak": iphone_peak.__dict__,
    }

    write_sync_json(output_dir, sync_info)
    save_shake_debug_csv(output_dir, d435i_signal=d435i_signal, iphone_signal=iphone_signal)
    save_shake_signal_plot(
        test_dir=test_dir,
        d435i_signal=d435i_signal,
        iphone_signal=iphone_signal,
        d435i_peak=d435i_peak,
        iphone_peak=iphone_peak,
    )

    return sync_info


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Estimate iPhone-D435i time offset from one shake.")
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
        help="指定某个时间戳 session 文件夹；不填时自动处理最新包含 d435i/imu.csv 的文件夹。",
    )
    parser.add_argument(
        "--odometry-csv",
        type=Path,
        default=None,
        help="指定 iPhone odometry.csv，用于优先定位同目录 imu.csv。",
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
        help=f"调试图目录名，默认：{DEFAULT_TEST_DIR_NAME}",
    )
    parser.add_argument(
        "--shake-search-seconds",
        type=float,
        default=DEFAULT_SHAKE_SEARCH_SECONDS,
        help=f"只在 IMU 开头多少秒内找 shake event；默认：{DEFAULT_SHAKE_SEARCH_SECONDS}",
    )
    parser.add_argument(
        "--smooth-window-seconds",
        type=float,
        default=DEFAULT_SMOOTH_WINDOW_SECONDS,
        help=f"IMU 信号平滑窗口；默认：{DEFAULT_SMOOTH_WINDOW_SECONDS}",
    )
    parser.add_argument(
        "--event-threshold-ratio",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD_RATIO,
        help=f"shake event 检测阈值相对高度；默认：{DEFAULT_EVENT_THRESHOLD_RATIO}",
    )
    parser.add_argument(
        "--merge-event-gap-seconds",
        type=float,
        default=DEFAULT_MERGE_EVENT_GAP_SECONDS,
        help=f"低于阈值的短间隔小于该值时合并为同一个 event；默认：{DEFAULT_MERGE_EVENT_GAP_SECONDS}",
    )
    parser.add_argument(
        "--min-event-duration-seconds",
        type=float,
        default=DEFAULT_MIN_EVENT_DURATION_SECONDS,
        help=f"shake event 最短持续时间；默认：{DEFAULT_MIN_EVENT_DURATION_SECONDS}",
    )
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    demo_dir = args.demo_dir if args.demo_dir is not None else find_latest_demo_dir_with_required_files(args.stray_root)
    demo_dir = demo_dir.resolve()
    output_dir = demo_dir / args.output_dir_name
    test_dir = output_dir / args.test_dir_name

    sync_info = estimate_sync_for_demo(
        demo_dir=demo_dir,
        odometry_csv_path=args.odometry_csv,
        output_dir=output_dir,
        test_dir=test_dir,
        shake_search_seconds=args.shake_search_seconds,
        smooth_window_seconds=args.smooth_window_seconds,
        peak_threshold_ratio=args.event_threshold_ratio,
        merge_event_gap_seconds=args.merge_event_gap_seconds,
        min_event_duration_seconds=args.min_event_duration_seconds,
    )

    print("shake 时间同步完成：")
    print(f"  demo_dir：{demo_dir}")
    print(f"  offset_iphone_minus_d435i：{sync_info['offset_iphone_minus_d435i']:.9f} s")
    print(f"  sync.json：{output_dir / 'sync.json'}")
    print(f"  shake_signal.png：{test_dir / 'shake_signal.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
