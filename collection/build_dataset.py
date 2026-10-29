"""
把一条 Stray + D435i + 夹爪采集记录整理成可训练的数据集 episode。

这个脚本是新的统一入口，正常使用时不需要再单独运行：
    - estimate_sync_by_shake.py
    - stray_shift_position.py

处理流程：
    1. 读取 E:\\机器人\\umi\\数据集 下已有的一级子文件夹，让用户选择数据集类别。
    2. 在选中的类别文件夹中自动生成下一个 episode 编号（001、002、003……）。
    3. 显示最新五个采集 session；可输入序号选择，也可手动输入其他时间戳。
    4. 使用一次 shake 估计 iPhone 与 D435i 的时间偏移。
    5. 从两端 IMU 确认整段 shake 完成后的短暂停顿开始，按最后公共有效时刻结束。
    6. 复用 stray_shift_position.py 的插值、外参、TCP 位姿和相对动作逻辑。
    7. 生成 224x224 RGB、机器人状态和 7 维未来相对动作。
    8. 先写入临时目录，全部成功后再原子重命名为正式 episode 编号。

7 维动作定义：
    [dx, dy, dz, rx, ry, rz, future_gripper_width_m]1

其中相对 TCP 位姿为：
    inverse(T_W_G[t]) @ T_W_G[t + k],  k = 1..H

直接运行：
    python build_dataset.py

也支持用命令行参数跳过交互，例如：
    python build_dataset.py --dataset-group "大方块001" --timestamp 0801_2038_001 --yes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from estimate_sync_by_shake import (
    DEFAULT_EVENT_THRESHOLD_RATIO,
    DEFAULT_MERGE_EVENT_GAP_SECONDS,
    DEFAULT_MIN_EVENT_DURATION_SECONDS,
    DEFAULT_SMOOTH_WINDOW_SECONDS,
    estimate_peak_time,
    find_d435i_imu_csv,
    find_iphone_imu_csv,
    load_motion_signal,
)
from stray_shift_position import (
    DEFAULT_ACTION_HORIZON,
    D435iFrameData,
    PoseAlignmentResult,
    RelativeActionResult,
    WidthAlignmentResult,
    align_gripper_width_to_d435i_frames,
    calculate_finite_transform_mask,
    calculate_timestamp_statistics,
    compute_tcp_pose_sequence,
    find_d435i_timestamps_csv,
    find_gripper_width_csv,
    find_odometry_csv_candidates,
    get_calibrated_T_I_G,
    interpolate_iphone_pose_to_d435i_frames,
    load_d435i_timestamps_csv,
    load_gripper_width_csv,
    load_odometry_csv,
    make_relative_actions,
    rotation_matrix_to_rotvec,
)
from xac_config import TIG_CALIBRATION_PATH, DATASET_ROOT, DATA_ROOT
from shake_timing import find_shake_pause, refine_burst_offset, refine_task_offset


# ============================================================
# 1. 默认路径与处理参数
# ============================================================

DEFAULT_STRAY_ROOT = DATA_ROOT
DEFAULT_DATASET_ROOT = DATASET_ROOT

# 缓冲包含静止确认，保留 shake 后的完整任务接近过程。
DEFAULT_START_DELAY_SECONDS = .4

# 只需要第一次 shake。默认在录制开头 12 秒内寻找，避免把后续正常操作误认为同步 shake。
DEFAULT_SHAKE_SEARCH_SECONDS = 12.0

# 与 stray_shift_position.py 当前默认值保持一致。
DEFAULT_MAX_IPHONE_GAP_SECONDS = 0.25

DEFAULT_IMAGE_SIZE = 224
LATEST_DEMO_SESSION_COUNT = 5
OLD_OUTPUT_DIR_NAME = "processed_umi"
EPISODE_DIR_PATTERN = re.compile(r"^\d{3,}$")


@dataclass(frozen=True)
class InputPaths:
    """集中保存本条 episode 的所有输入和输出路径。"""

    dataset_group_dir: Path
    episode_id: str
    final_episode_dir: Path
    temporary_episode_dir: Path
    demo_dir: Path
    odometry_csv_path: Path
    d435i_timestamps_csv_path: Path
    gripper_width_csv_path: Path


@dataclass(frozen=True)
class CroppedTimeline:
    """保存按 shake 起点和按 S 停止终点裁剪后的 D435i 主时间轴。"""

    frame_data: D435iFrameData
    rgb_paths: list[Path]
    rgb_file_valid_mask: np.ndarray
    start_time_s: float
    end_time_s: float
    original_frame_count: int


@dataclass(frozen=True)
class AlignedEpisode:
    """保存完成时间对齐、外参转换和状态转换后的 episode 数据。"""

    frame_data: D435iFrameData
    rgb_paths: list[Path]
    pose_alignment: PoseAlignmentResult
    width_alignment: WidthAlignmentResult
    T_I_G: np.ndarray
    T_W_G: np.ndarray
    robot0_eef_pos: np.ndarray
    robot0_eef_rot_axis_angle: np.ndarray
    base_valid_mask: np.ndarray


# ============================================================
# 2. 命令行与交互选择
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析可选命令行参数；不提供选择参数时按用户要求进行交互询问。"""

    parser = argparse.ArgumentParser(
        description="Build one UMI-style dataset episode from Stray, D435i and gripper logs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"数据集类别根目录，默认：{DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--stray-root",
        type=Path,
        default=DEFAULT_STRAY_ROOT,
        help=f"原始 Stray session 根目录，默认：{DEFAULT_STRAY_ROOT}",
    )
    parser.add_argument(
        "--dataset-group",
        type=str,
        default=None,
        help="数据集类别文件夹名称，或交互列表中的序号；不填时首先询问。",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="采集 session 编号，例如 001；也兼容旧时间戳或完整 session 路径。",
    )
    parser.add_argument(
        "--odometry-csv",
        type=Path,
        default=None,
        help="显式指定 odometry.csv；session 内只有一个候选时无需填写。",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help=f"未来相对动作长度 H，默认：{DEFAULT_ACTION_HORIZON}",
    )
    parser.add_argument(
        "--shake-search-seconds",
        type=float,
        default=DEFAULT_SHAKE_SEARCH_SECONDS,
        help=f"从 IMU 开头搜索第一次 shake 的时间，默认：{DEFAULT_SHAKE_SEARCH_SECONDS} 秒。",
    )
    parser.add_argument(
        "--start-source", choices=("shake-pause", "realtime-valid"), default="shake-pause",
        help="shake-pause 使用完整 shake 后的短暂停顿；realtime-valid 复现录制时提示的旧起点。",
    )
    parser.add_argument(
        "--sync-method", choices=("burst-correlation", "burst-task-correlation", "single-peak"),
        default="burst-correlation",
        help="默认用整段 gyro 信号细化时间同步；burst-task-correlation 再用 shake 后的任务段"
             "（±0.3 s）复核并在明显改善时修正；single-peak 复现旧单峰对齐。",
    )
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=DEFAULT_START_DELAY_SECONDS,
        help=f"shake 完全结束后的等待时间，默认：{DEFAULT_START_DELAY_SECONDS} 秒。",
    )
    parser.add_argument(
        "--max-iphone-gap-seconds",
        type=float,
        default=DEFAULT_MAX_IPHONE_GAP_SECONDS,
        help=f"允许插值的最大 iPhone odometry 间隔，默认：{DEFAULT_MAX_IPHONE_GAP_SECONDS} 秒。",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"RGB 输出边长，默认：{DEFAULT_IMAGE_SIZE}。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="显示路径后直接开始处理，不再询问最后一次确认。",
    )
    return parser.parse_args()


def natural_sort_key(text: str) -> list[Any]:
    """把字符串中的数字按数值排序，避免 group2 被排在 group10 后面。"""

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


def read_numbered_choice(prompt: str, candidates: list[Path]) -> Path:
    """显示带编号的路径列表，并持续询问直到用户输入有效序号。"""

    if not candidates:
        raise ValueError("候选列表不能为空。")

    print(prompt)
    for index, candidate in enumerate(candidates, start=1):
        print(f"  [{index}] {candidate.name}")

    while True:
        raw_value = input("请输入序号：").strip()
        try:
            selected_index = int(raw_value) - 1
        except ValueError:
            print("输入必须是列表中的整数序号，请重新输入。")
            continue

        if 0 <= selected_index < len(candidates):
            return candidates[selected_index]

        print(f"请输入 1 到 {len(candidates)} 之间的序号。")


def select_dataset_group(dataset_root: Path, requested_group: str | None) -> Path:
    """
    读取数据集根目录下已有的一级子文件夹。

    这些一级文件夹表示任务/物体类别；episode 的 001、002 等编号会创建在其中，
    因此本函数不会在根目录自动新建类别，避免把拼写错误变成新的数据集。
    """

    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集根目录不存在：{dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"数据集根路径不是文件夹：{dataset_root}")

    group_dirs = sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and not path.name.endswith(".tmp") and not path.name.startswith(".")
        ],
        key=lambda path: natural_sort_key(path.name),
    )
    if not group_dirs:
        raise FileNotFoundError(
            f"{dataset_root} 下没有数据集类别文件夹。\n"
            "请先手动创建类别文件夹，例如“大方块001”，然后重新运行。"
        )

    if requested_group is None:
        return read_numbered_choice("请选择数据集类别文件夹：", group_dirs)

    requested_group = requested_group.strip()
    if requested_group.isdigit():
        selected_index = int(requested_group) - 1
        if 0 <= selected_index < len(group_dirs):
            return group_dirs[selected_index]
        raise ValueError(f"--dataset-group 序号超出范围：{requested_group}")

    exact_matches = [path for path in group_dirs if path.name == requested_group]
    if len(exact_matches) == 1:
        return exact_matches[0]

    available_names = "、".join(path.name for path in group_dirs)
    raise FileNotFoundError(
        f"没有找到数据集类别文件夹：{requested_group}\n当前可选类别：{available_names}"
    )


def get_next_episode_id(dataset_group_dir: Path) -> str:
    """扫描类别文件夹中的纯数字 episode，返回当前最大编号加一并至少补足三位。"""

    existing_numbers = [
        int(path.name)
        for path in dataset_group_dir.iterdir()
        if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
    ]
    next_number = max(existing_numbers, default=0) + 1
    return f"{next_number:03d}"


def find_latest_demo_dirs(
    stray_root: Path,
    limit: int = LATEST_DEMO_SESSION_COUNT,
) -> list[Path]:
    """
    返回 Stray 根目录下最近更新的若干个采集 session。

    新格式是 ``001, 002, ...``；同时继续识别旧的 ``data_*`` 目录。

    录制结束后通常还会向 session 补充 iPhone 导出文件，因此以目录的最近修改时间
    从新到旧排序，能覆盖刚完成录制、刚完成导入这两种常见流程。名称用作修改时间
    相同时的稳定次级排序键，保证不同文件系统精度下的显示顺序一致。
    """

    if limit <= 0:
        raise ValueError("最新 session 的显示数量必须大于 0。")
    if not stray_root.is_dir():
        return []

    session_dirs = [
        path
        for path in stray_root.iterdir()
        if path.is_dir()
        and (
            EPISODE_DIR_PATTERN.fullmatch(path.name)
            or path.name.startswith("data_")
        )
    ]
    if not session_dirs:
        return []

    return sorted(
        session_dirs,
        key=lambda path: (path.stat().st_mtime_ns, natural_sort_key(path.name)),
        reverse=True,
    )[:limit]


def find_latest_demo_dir(stray_root: Path) -> Path | None:
    """返回最近更新的一个 session；没有候选时返回 ``None``。"""

    latest_demo_dirs = find_latest_demo_dirs(stray_root, limit=1)
    return latest_demo_dirs[0] if latest_demo_dirs else None


def resolve_demo_dir_once(stray_root: Path, timestamp_value: str) -> Path:
    """
    根据一个已经输入的时间戳或绝对路径解析 session，只尝试一次。

    把单次解析与交互重试分开，可以让命令行 ``--timestamp`` 在写错时立即报错，
    同时让人工输入在未找到 session 时回到输入提示，而不是退出整个程序。
    """

    timestamp_value = timestamp_value.strip()
    if not timestamp_value:
        raise ValueError("采集时间戳不能为空。")

    explicit_path = Path(timestamp_value)
    if explicit_path.is_absolute():
        if not explicit_path.is_dir():
            raise FileNotFoundError(f"指定的 session 文件夹不存在：{explicit_path}")
        return explicit_path

    if not stray_root.exists():
        raise FileNotFoundError(f"Stray 数据根目录不存在：{stray_root}")
    if not stray_root.is_dir():
        raise NotADirectoryError(f"Stray 数据根路径不是文件夹：{stray_root}")

    if timestamp_value.isdigit():
        session_names = {f"{int(timestamp_value):03d}"}
    else:
        legacy_name = timestamp_value if timestamp_value.startswith("data_") else f"data_{timestamp_value}"
        session_names = {legacy_name}

    candidates = sorted(
        [
            path
            for path in stray_root.iterdir()
            if path.is_dir()
            and any(
                path.name == session_name or path.name.startswith(f"{session_name}_")
                for session_name in session_names
            )
        ],
        key=lambda path: natural_sort_key(path.name),
    )
    if not candidates:
        raise FileNotFoundError(
            f"没有找到与 {timestamp_value} 对应的 session。\n"
            f"已检查：{', '.join(str(stray_root / name) for name in sorted(session_names))}。"
        )
    if len(candidates) == 1:
        return candidates[0]

    return read_numbered_choice("找到多个相同时间戳的 session，请选择：", candidates)


def resolve_demo_dir(stray_root: Path, timestamp_value: str | None) -> Path:
    """
    根据用户输入寻找 session。

    输入 001 会检查数字编号 session；输入 0801_2038 时仍兼容旧的
    data_0801_2038 和 data_0801_2038_*。
    如果存在多个匹配项，必须人工选择，不再根据修改时间自动猜测。
    """

    # 显式命令行参数属于非交互调用：只解析一次，失败后把错误交给主程序处理。
    if timestamp_value is not None:
        return resolve_demo_dir_once(stray_root, timestamp_value)

    if not stray_root.exists():
        raise FileNotFoundError(f"Stray 数据根目录不存在：{stray_root}")
    if not stray_root.is_dir():
        raise NotADirectoryError(f"Stray 数据根路径不是文件夹：{stray_root}")

    latest_demo_dirs = find_latest_demo_dirs(stray_root)
    if latest_demo_dirs:
        print(f"最新 {len(latest_demo_dirs)} 个采集 session（从新到旧）：")
        for index, demo_dir in enumerate(latest_demo_dirs, start=1):
            latest_marker = "（最新）" if index == 1 else ""
            print(f"  [{index}] {demo_dir.name}{latest_marker}")
    else:
        print(f"{stray_root} 下暂时没有采集 session，请手动输入编号或旧时间戳。")

    while True:
        if latest_demo_dirs:
            raw_value = input(
                f"请输入 1-{len(latest_demo_dirs)} 的序号或采集时间戳"
                "（直接回车使用 [1]）："
            ).strip()
        else:
            raw_value = input("请输入采集编号，例如 001：").strip()

        # 保留原来的快捷操作：直接回车仍然选择最新的一条记录。
        if not raw_value:
            if latest_demo_dirs:
                selected_demo_dir = latest_demo_dirs[0]
                print(f"已选择采集 session：{selected_demo_dir}")
                return selected_demo_dir
            print("采集时间戳不能为空，请重新输入。")
            continue

        # 三位以上纯数字优先解释为 session 编号；1、2 这样的短数字
        # 仍作为上方列表序号。
        exact_numbered_dir = stray_root / f"{int(raw_value):03d}" if raw_value.isdigit() else None
        if raw_value.isdigit() and len(raw_value) >= 3 and exact_numbered_dir.is_dir():
            print(f"已选择采集 session：{exact_numbered_dir}")
            return exact_numbered_dir

        if raw_value.isdigit() and latest_demo_dirs:
            selected_index = int(raw_value) - 1
            if 0 <= selected_index < len(latest_demo_dirs):
                selected_demo_dir = latest_demo_dirs[selected_index]
                print(f"已选择采集 session：{selected_demo_dir}")
                return selected_demo_dir
            print(f"序号超出范围，请输入 1 到 {len(latest_demo_dirs)}，或输入完整时间戳。")
            continue

        try:
            selected_demo_dir = resolve_demo_dir_once(stray_root, raw_value)
        except FileNotFoundError as exc:
            # 人工输入拼错时只提示本次错误，随后继续循环，不让整个构建程序退出。
            print(f"{exc}\n请重新输入采集时间戳或上方列表序号。")
            continue

        print(f"已选择采集 session：{selected_demo_dir}")
        return selected_demo_dir


def select_odometry_csv(demo_dir: Path, requested_path: Path | None) -> Path:
    """选择当前 session 的 odometry.csv；多个候选时必须由用户确认。"""

    if requested_path is not None:
        resolved_path = requested_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"指定的 odometry.csv 不存在：{resolved_path}")
        return resolved_path

    candidates = find_odometry_csv_candidates(
        search_dir=demo_dir,
        output_dir_name=OLD_OUTPUT_DIR_NAME,
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {demo_dir} 内没有找到 odometry.csv。\n"
            "请先把 Stray/iPhone 导出文件夹放到这个时间戳 session 中。"
        )
    if len(candidates) == 1:
        return candidates[0]

    print("找到多个 odometry.csv：")
    for index, path in enumerate(candidates, start=1):
        print(f"  [{index}] {path.relative_to(demo_dir)}")
    while True:
        raw_value = input("请选择本次要使用的 odometry.csv 序号：").strip()
        try:
            selected_index = int(raw_value) - 1
        except ValueError:
            print("请输入整数序号。")
            continue
        if 0 <= selected_index < len(candidates):
            return candidates[selected_index]
        print(f"请输入 1 到 {len(candidates)} 之间的序号。")


def build_input_paths(args: argparse.Namespace) -> InputPaths:
    """按照“先选数据集类别，再输入时间戳”的顺序构造本次处理路径。"""

    dataset_group_dir = select_dataset_group(args.dataset_root, args.dataset_group)
    episode_id = get_next_episode_id(dataset_group_dir)
    final_episode_dir = dataset_group_dir / episode_id
    temporary_episode_dir = dataset_group_dir / f"{episode_id}.tmp"

    if final_episode_dir.exists():
        raise FileExistsError(f"目标 episode 已存在，拒绝覆盖：{final_episode_dir}")
    if temporary_episode_dir.exists():
        raise FileExistsError(
            f"检测到上一次未完成的临时目录：{temporary_episode_dir}\n"
            "请先人工检查该目录，确认不需要后再处理。"
        )

    print(f"\n已选择数据集类别：{dataset_group_dir.name}")
    print(f"本条数据将自动编号为：{episode_id}")

    demo_dir = resolve_demo_dir(args.stray_root, args.timestamp)
    odometry_csv_path = select_odometry_csv(demo_dir, args.odometry_csv)
    d435i_timestamps_csv_path = find_d435i_timestamps_csv(demo_dir)
    gripper_width_csv_path = find_gripper_width_csv(
        demo_dir=demo_dir,
        output_dir_name=OLD_OUTPUT_DIR_NAME,
    )

    return InputPaths(
        dataset_group_dir=dataset_group_dir,
        episode_id=episode_id,
        final_episode_dir=final_episode_dir,
        temporary_episode_dir=temporary_episode_dir,
        demo_dir=demo_dir,
        odometry_csv_path=odometry_csv_path,
        d435i_timestamps_csv_path=d435i_timestamps_csv_path,
        gripper_width_csv_path=gripper_width_csv_path,
    )


# ============================================================
# 3. 单次 shake 同步与 D435i 时间范围
# ============================================================


def estimate_single_shake_sync(
    demo_dir: Path,
    odometry_csv_path: Path,
    shake_search_seconds: float,
) -> dict[str, Any]:
    """
    复用 estimate_sync_by_shake.py 的核心函数完成一次 shake 同步。

    这里故意不调用 estimate_sync_for_demo()，因为它会创建 shake_debug.csv 和 test 图片；
    当前数据集流程只需要同步结果，不需要 test 目录。
    """

    d435i_imu_path = find_d435i_imu_csv(demo_dir)
    iphone_imu_path = find_iphone_imu_csv(demo_dir, odometry_csv_path=odometry_csv_path)

    d435i_signal = load_motion_signal(
        d435i_imu_path,
        prefer_computer_time=True,
        smooth_window_seconds=DEFAULT_SMOOTH_WINDOW_SECONDS,
    )
    iphone_signal = load_motion_signal(
        iphone_imu_path,
        prefer_computer_time=False,
        smooth_window_seconds=DEFAULT_SMOOTH_WINDOW_SECONDS,
    )

    common_peak_arguments = {
        "shake_search_seconds": shake_search_seconds,
        "peak_threshold_ratio": DEFAULT_EVENT_THRESHOLD_RATIO,
        "merge_event_gap_seconds": DEFAULT_MERGE_EVENT_GAP_SECONDS,
        "min_event_duration_seconds": DEFAULT_MIN_EVENT_DURATION_SECONDS,
    }
    d435i_peak = estimate_peak_time(d435i_signal, **common_peak_arguments)
    iphone_peak = estimate_peak_time(iphone_signal, **common_peak_arguments)
    offset_seconds = float(iphone_peak.shake_time_s - d435i_peak.shake_time_s)

    return {
        "method": "single_shake_event_imu",
        "mapping": "t_iphone = t_d435i + offset_iphone_minus_d435i",
        "offset_iphone_minus_d435i": offset_seconds,
        "d435i_shake_time_s": float(d435i_peak.shake_time_s),
        "iphone_shake_time_s": float(iphone_peak.shake_time_s),
        "shake_search_seconds": float(shake_search_seconds),
        "smooth_window_seconds": float(DEFAULT_SMOOTH_WINDOW_SECONDS),
        "event_threshold_ratio": float(DEFAULT_EVENT_THRESHOLD_RATIO),
        "merge_event_gap_seconds": float(DEFAULT_MERGE_EVENT_GAP_SECONDS),
        "min_event_duration_seconds": float(DEFAULT_MIN_EVENT_DURATION_SECONDS),
        # 保留 estimate_sync_by_shake.py 已有的旧键名。
        "peak_threshold_ratio": float(DEFAULT_EVENT_THRESHOLD_RATIO),
        "d435i_imu_csv": str(d435i_imu_path),
        "iphone_imu_csv": str(iphone_imu_path),
        "d435i_signal_kind": d435i_signal.signal_kind,
        "iphone_signal_kind": iphone_signal.signal_kind,
        "d435i_peak": dict(d435i_peak.__dict__),
        "iphone_peak": dict(iphone_peak.__dict__),
        "second_shake_validation": None,
        "second_shake_note": "本数据集流程按要求只使用第一次 shake，不做第二次 shake 验证。",
    }


def prepare_shake_timing(demo_dir, sync_info, start_source, sync_method, delay_seconds):
    """Separate synchronization, shake-end detection and training-start selection."""
    events = load_realtime_recording_events(demo_dir)
    if events is not None and not events.get('events', {}).get('valid_data_started'):
        raise RuntimeError('录制未完成实时 shake 流程，保留待审核，不自动纳入训练。')
    if start_source == 'realtime-valid' and sync_method == 'single-peak':
        return
    d435i = load_motion_signal(Path(sync_info['d435i_imu_csv']), True, DEFAULT_SMOOTH_WINDOW_SECONDS)
    iphone = load_motion_signal(Path(sync_info['iphone_imu_csv']), False, DEFAULT_SMOOTH_WINDOW_SECONDS)
    window_start = sync_info['d435i_peak']['region_start_time_s'] - .5
    window_end = sync_info['d435i_peak']['search_end_time_s']
    if events is not None:
        window_end = min(window_end, events['events']['valid_data_started']['system_time_s'])
    if sync_method in ('burst-correlation', 'burst-task-correlation'):
        # Use the recorded end only as a search bound, not as the training start.
        correlation_end = window_end
        if events is not None and 'shake_ended' in events.get('events', {}):
            correlation_end = min(correlation_end, events['events']['shake_ended']['system_time_s'] + .2)
        offset, details = refine_burst_offset(d435i, iphone,
            sync_info['offset_iphone_minus_d435i'], window_start, correlation_end)
        sync_info['offset_iphone_minus_d435i'] = offset
        sync_info['burst_alignment'] = details
        sync_info['method'] = 'gyro_burst_correlation_refined_from_single_shake'
    def shake_pause():
        return find_shake_pause(d435i, iphone,
            sync_info['offset_iphone_minus_d435i'], sync_info['d435i_shake_time_s'],
            window_start, window_end, sync_info['d435i_peak']['peak_signal'],
            sync_info['iphone_peak']['peak_signal'], delay_seconds)
    if start_source == 'shake-pause':
        sync_info['shake_pause'] = shake_pause()
    if sync_method == 'burst-task-correlation':
        if start_source != 'shake-pause':
            raise ValueError('burst-task-correlation requires --start-source shake-pause')
        task_start = sync_info['shake_pause']['dataset_start_time_s']
        task_end = min(d435i.timestamps_s[-1],
                       iphone.timestamps_s[-1] - sync_info['offset_iphone_minus_d435i'])
        offset, details = refine_task_offset(d435i, iphone,
            sync_info['offset_iphone_minus_d435i'], task_start, task_end)
        sync_info['task_alignment'] = details
        if details['applied']:
            sync_info['offset_iphone_minus_d435i'] = offset
            sync_info['method'] = 'gyro_burst_then_task_correlation'
            sync_info['shake_pause'] = shake_pause()


def load_d435i_meta(demo_dir: Path) -> dict[str, Any]:
    """读取 record_435i.py 在按 S 停止时写入的 meta.json；没有该文件时返回空字典。"""

    meta_path = demo_dir / "d435i" / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 D435i meta.json：{meta_path}，原因：{exc}") from exc


def load_realtime_recording_events(demo_dir: Path) -> dict[str, Any] | None:
    """
    读取新版 record_435i.py 保存的实时事件。

    旧session没有 recording_events.json 时返回None，继续使用离线shake区域+1.5秒；
    新session如果存在事件文件，就应使用终端实际提示“开始收有效数据”的同一时间点。
    """

    events_path = demo_dir / "d435i" / "recording_events.json"
    if not events_path.exists():
        return None
    try:
        return json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"无法读取实时录制事件：{events_path}，原因：{exc}"
        ) from exc


def load_rgb_path_map(timestamps_csv_path: Path) -> dict[int, Path]:
    """
    从 D435i timestamps.csv 读取 frame_idx -> RGB 图片路径。

    优先使用 csv 的 rgb_path 列；旧记录没有该列时，兼容 rgb/000123.png 命名。
    """

    d435i_dir = timestamps_csv_path.parent
    rgb_path_by_frame: dict[int, Path] = {}
    with timestamps_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"D435i timestamps.csv 没有表头：{timestamps_csv_path}")

        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                key.strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
                if key is not None
            }
            try:
                frame_index = int(float(row["frame_idx"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"D435i timestamps.csv 第 {line_number} 行 frame_idx 无效：{row}"
                ) from exc

            relative_rgb_path = row.get("rgb_path") or f"rgb/{frame_index:06d}.png"
            normalized_relative_path = Path(relative_rgb_path.replace("/", "\\"))
            rgb_path_by_frame[frame_index] = d435i_dir / normalized_relative_path

    return rgb_path_by_frame


def crop_d435i_timeline(
    paths: InputPaths,
    sync_info: dict[str, Any],
    odometry_timestamps_s: np.ndarray,
    gripper_timestamps_s: np.ndarray,
    start_delay_seconds: float,
    action_horizon: int,
) -> CroppedTimeline:
    """
    按第一次 shake 结束时间和按 S 停止时间裁剪 D435i 主时间轴。

    结束点不是盲目使用某一路数据的最后值，而是取 D435i、iPhone odometry、
    夹爪日志和 meta.json 可共同覆盖的最晚时间，避免在某一路停止后进行外推。
    """

    full_frame_data = load_d435i_timestamps_csv(paths.d435i_timestamps_csv_path)
    if np.any(np.diff(full_frame_data.timestamps_s) <= 0.0):
        raise ValueError("D435i timestamps.csv 不是严格递增时间轴，不能安全生成动作。")

    d435i_peak = sync_info["d435i_peak"]
    realtime_events = load_realtime_recording_events(paths.demo_dir)
    if 'shake_pause' in sync_info:
        start_time_s = float(sync_info['shake_pause']['dataset_start_time_s'])
        start_time_source = 'joint_gyro_complete_burst_then_pause'
    elif realtime_events is None:
        start_time_s = (
            float(d435i_peak["region_end_time_s"]) + float(start_delay_seconds)
        )
        start_time_source = "offline_d435i_shake_region_end_plus_delay"
    else:
        valid_start_event = realtime_events.get("events", {}).get("valid_data_started")
        if valid_start_event is None:
            stopped_event = realtime_events.get("events", {}).get("recording_stopped", {})
            detector_state = stopped_event.get(
                "detector_state_before_stop",
                realtime_events.get("state", "unknown"),
            )
            raise RuntimeError(
                "本条新录制存在 recording_events.json，但录制时没有完成实时shake流程。\n"
                "终端没有进入“现在开始收有效任务数据”阶段，因此拒绝打包。\n"
                f"实时检测最终状态：{detector_state}"
            )
        start_time_s = float(valid_start_event["system_time_s"])
        start_time_source = "recording_events.json valid_data_started"
    offset_seconds = float(sync_info["offset_iphone_minus_d435i"])

    end_candidates = {
        "d435i_last_frame": float(full_frame_data.timestamps_s[-1]),
        "iphone_odometry_last_mapped_to_d435i": float(odometry_timestamps_s[-1] - offset_seconds),
        "gripper_width_last": float(gripper_timestamps_s[-1]),
    }

    d435i_meta = load_d435i_meta(paths.demo_dir)
    meta_last_time = d435i_meta.get("last_system_time_s")
    if meta_last_time is not None:
        end_candidates["d435i_meta_stop"] = float(meta_last_time)

    end_time_s = min(end_candidates.values())
    frame_mask = (
        (full_frame_data.timestamps_s >= start_time_s)
        & (full_frame_data.timestamps_s <= end_time_s)
    )

    # 如果同一个 d435i 文件夹被追加录制，meta.json 会记录最后一次按 S 录制的帧编号范围。
    # 在正常的一次-session结构中 start_frame_idx 通常为 0，此筛选不会改变结果。
    start_frame_index = d435i_meta.get("start_frame_idx")
    frame_count_this_run = d435i_meta.get("frame_count_this_run")
    if start_frame_index is not None and frame_count_this_run is not None:
        first_frame = int(start_frame_index)
        last_frame_exclusive = first_frame + int(frame_count_this_run)
        frame_mask &= (
            (full_frame_data.frame_indices >= first_frame)
            & (full_frame_data.frame_indices < last_frame_exclusive)
        )

    selected_indices = np.where(frame_mask)[0]
    if len(selected_indices) <= action_horizon:
        raise RuntimeError(
            "shake 后可用数据太短，无法生成未来动作：\n"
            f"  开始时间：{start_time_s:.6f}\n"
            f"  结束时间：{end_time_s:.6f}\n"
            f"  保留帧数：{len(selected_indices)}\n"
            f"  action horizon：{action_horizon}"
        )

    cropped_frame_data = D435iFrameData(
        frame_indices=full_frame_data.frame_indices[selected_indices].copy(),
        timestamps_s=full_frame_data.timestamps_s[selected_indices].copy(),
    )
    rgb_path_map = load_rgb_path_map(paths.d435i_timestamps_csv_path)
    rgb_paths = [
        rgb_path_map.get(
            int(frame_index),
            paths.d435i_timestamps_csv_path.parent / "rgb" / f"{int(frame_index):06d}.png",
        )
        for frame_index in cropped_frame_data.frame_indices
    ]
    rgb_file_valid_mask = np.asarray(
        [path.is_file() and path.stat().st_size > 0 for path in rgb_paths],
        dtype=bool,
    )

    sync_info["dataset_start_d435i_time_s"] = float(cropped_frame_data.timestamps_s[0])
    sync_info["dataset_requested_start_after_shake_s"] = float(start_time_s)
    sync_info["dataset_start_time_source"] = start_time_source
    sync_info["realtime_recording_events_used"] = realtime_events is not None
    sync_info["dataset_end_d435i_time_s"] = float(cropped_frame_data.timestamps_s[-1])
    sync_info["dataset_end_candidates_s"] = end_candidates
    sync_info['actual_pause_seconds'] = (
        sync_info['shake_pause']['pause_seconds'] if 'shake_pause' in sync_info else
        realtime_events.get('parameters', {}).get('valid_data_delay_after_shake_seconds')
        if realtime_events is not None else float(start_delay_seconds))
    sync_info["start_delay_after_shake_seconds"] = sync_info['actual_pause_seconds']
    sync_info["stop_rule"] = "last common timestamp after the S key stopped recording"

    return CroppedTimeline(
        frame_data=cropped_frame_data,
        rgb_paths=rgb_paths,
        rgb_file_valid_mask=rgb_file_valid_mask,
        start_time_s=float(cropped_frame_data.timestamps_s[0]),
        end_time_s=float(cropped_frame_data.timestamps_s[-1]),
        original_frame_count=len(full_frame_data.timestamps_s),
    )


# ============================================================
# 4. 位姿、夹爪、RGB和7维相对动作
# ============================================================


def extract_robot_state(T_W_G: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从每帧 T_W_G 提取 float32 位置和旋转轴角，并返回转换有效掩码。"""

    frame_count = len(T_W_G)
    positions = np.full((frame_count, 3), np.nan, dtype=np.float32)
    rotations = np.full((frame_count, 3), np.nan, dtype=np.float32)
    state_valid_mask = np.zeros(frame_count, dtype=bool)

    for frame_index, transform in enumerate(T_W_G):
        if not np.all(np.isfinite(transform)):
            continue
        try:
            positions[frame_index] = transform[:3, 3].astype(np.float32)
            rotations[frame_index] = rotation_matrix_to_rotvec(
                transform[:3, :3]
            ).astype(np.float32)
            state_valid_mask[frame_index] = True
        except (ValueError, FloatingPointError):
            # 单帧姿态损坏时保留 NaN 并在 valid_mask 中标记为无效，防止进入训练窗口。
            continue

    return positions, rotations, state_valid_mask


def align_episode_data(
    cropped_timeline: CroppedTimeline,
    odometry_data: Any,
    width_data: Any,
    sync_offset_seconds: float,
    max_iphone_gap_seconds: float,
) -> AlignedEpisode:
    """复用 stray_shift_position.py 完成D435i主时间轴上的位姿和夹爪宽度对齐。"""

    timestamps_s = cropped_timeline.frame_data.timestamps_s
    pose_alignment = interpolate_iphone_pose_to_d435i_frames(
        odometry_data=odometry_data,
        d435i_timestamps_s=timestamps_s,
        sync_offset_iphone_minus_d435i=sync_offset_seconds,
        max_iphone_gap_seconds=max_iphone_gap_seconds,
    )
    width_alignment = align_gripper_width_to_d435i_frames(
        width_data=width_data,
        d435i_timestamps_s=timestamps_s,
    )

    T_I_G = get_calibrated_T_I_G()
    T_W_G = compute_tcp_pose_sequence(pose_alignment.T_W_I, T_I_G)
    finite_transform_mask = calculate_finite_transform_mask(T_W_G)
    robot0_eef_pos, robot0_eef_rot_axis_angle, state_valid_mask = extract_robot_state(T_W_G)

    # width_alignment.valid_mask 还要求 d_left/d_right/sync_error 三个诊断列有效；
    # 但旧版夹爪日志可能只有必须的 width_m。训练标签只依赖 width_m，
    # 因此这里以实际对齐后的宽度是否有限为准，附加诊断列仍原样保存在 auxiliary。
    gripper_width_valid_mask = np.isfinite(width_alignment.width_m)
    base_valid_mask = (
        pose_alignment.valid_mask
        & gripper_width_valid_mask
        & finite_transform_mask
        & state_valid_mask
        & cropped_timeline.rgb_file_valid_mask
    )

    return AlignedEpisode(
        frame_data=cropped_timeline.frame_data,
        rgb_paths=cropped_timeline.rgb_paths,
        pose_alignment=pose_alignment,
        width_alignment=width_alignment,
        T_I_G=T_I_G,
        T_W_G=T_W_G,
        robot0_eef_pos=robot0_eef_pos,
        robot0_eef_rot_axis_angle=robot0_eef_rot_axis_angle,
        base_valid_mask=base_valid_mask,
    )


def import_cv2():
    """延迟导入 OpenCV，使缺少依赖时能够给出明确的 umi_py39 安装提示。"""

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境没有 OpenCV，无法整理 camera0_rgb。\n"
            "请确认使用 umi_py39 环境，并安装 opencv-python。"
        ) from exc
    return cv2


def read_image_unicode_safe(cv2: Any, image_path: Path) -> np.ndarray | None:
    """使用 bytes + imdecode 读取图片，兼容 Windows 中文路径。"""

    try:
        encoded_data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if encoded_data.size == 0:
        return None
    return cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)


def resize_rgb_with_letterbox(cv2: Any, bgr_image: np.ndarray, image_size: int) -> np.ndarray:
    """
    保持完整视野等比例缩放，再居中补黑边到正方形，最后从 BGR 转成 RGB。

    D435i 原图通常为 640x480；直接拉伸为 224x224 会改变物体形状，中心裁剪又会丢失视野。
    letterbox 能保留完整画面，但训练和机器人部署时必须使用完全相同的预处理。
    """

    source_height, source_width = bgr_image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("RGB 图片宽高无效。")

    scale = min(image_size / source_width, image_size / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_bgr = cv2.resize(
        bgr_image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas_bgr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    x_offset = (image_size - resized_width) // 2
    y_offset = (image_size - resized_height) // 2
    canvas_bgr[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized_bgr
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def save_camera0_rgb(
    output_path: Path,
    rgb_paths: list[Path],
    image_size: int,
) -> np.ndarray:
    """
    逐帧读取并写入 NPY memmap，避免同时把全部原图和全部224图像放进内存。

    返回实际成功解码的图片掩码。缺失或损坏图片写入全黑占位，但对应帧会被标为无效。
    """

    cv2 = import_cv2()
    frame_count = len(rgb_paths)
    camera_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, image_size, image_size, 3),
    )
    decoded_valid_mask = np.zeros(frame_count, dtype=bool)
    progress_interval = max(1, frame_count // 10)

    for frame_index, image_path in enumerate(rgb_paths):
        bgr_image = read_image_unicode_safe(cv2, image_path)
        if bgr_image is None:
            camera_array[frame_index] = 0
        else:
            try:
                camera_array[frame_index] = resize_rgb_with_letterbox(
                    cv2=cv2,
                    bgr_image=bgr_image,
                    image_size=image_size,
                )
                decoded_valid_mask[frame_index] = True
            except (ValueError, cv2.error):
                camera_array[frame_index] = 0

        completed_count = frame_index + 1
        if completed_count % progress_interval == 0 or completed_count == frame_count:
            print(f"  RGB处理进度：{completed_count}/{frame_count}")

    camera_array.flush()
    del camera_array
    return decoded_valid_mask


def make_actions_relative_7d(
    T_W_G: np.ndarray,
    timestamps_s: np.ndarray,
    gripper_width_m: np.ndarray,
    valid_mask: np.ndarray,
    action_horizon: int,
) -> tuple[RelativeActionResult, np.ndarray]:
    """
    在已有 6 维相对 TCP 动作后拼接未来绝对夹爪宽度，得到 [N-H,H,7]。

    夹爪维度使用 future_index=t+k 的 width，而不是当前 width，也不是 width delta；
    这与 UMI/DP 常用的“末端相对运动 + 未来夹爪命令”表达一致。
    """

    relative_action_result = make_relative_actions(
        T_W_G=T_W_G,
        action_start_timestamps_source=timestamps_s,
        action_horizon=action_horizon,
        valid_mask=valid_mask,
    )
    actions_relative_6d = relative_action_result.actions_relative_6d
    actions_relative_7d = np.full(
        actions_relative_6d.shape[:-1] + (7,),
        np.nan,
        dtype=np.float32,
    )

    for start_index, is_valid in enumerate(relative_action_result.actions_valid_mask):
        if not is_valid:
            continue
        future_width = gripper_width_m[
            start_index + 1:start_index + action_horizon + 1
        ]
        if len(future_width) != action_horizon or not np.all(np.isfinite(future_width)):
            continue
        actions_relative_7d[start_index, :, :6] = actions_relative_6d[
            start_index
        ].astype(np.float32)
        actions_relative_7d[start_index, :, 6] = future_width.astype(np.float32)

    return relative_action_result, actions_relative_7d


# ============================================================
# 5. 输出、元数据与最终校验
# ============================================================


def write_json(json_path: Path, data: dict[str, Any]) -> None:
    """以UTF-8和可读缩进保存JSON。"""

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_final_arrays(
    aligned_episode: AlignedEpisode,
    valid_mask: np.ndarray,
    relative_action_result: RelativeActionResult,
    actions_relative_7d: np.ndarray,
    image_size: int,
) -> None:
    """在正式命名 episode 前检查所有关键长度、形状和有效动作数量。"""

    frame_count = len(aligned_episode.frame_data.timestamps_s)
    expected_frame_shapes = {
        "robot0_eef_pos": (frame_count, 3),
        "robot0_eef_rot_axis_angle": (frame_count, 3),
        "robot0_gripper_width": (frame_count, 1),
        "valid_mask": (frame_count,),
    }
    actual_frame_shapes = {
        "robot0_eef_pos": aligned_episode.robot0_eef_pos.shape,
        "robot0_eef_rot_axis_angle": aligned_episode.robot0_eef_rot_axis_angle.shape,
        "robot0_gripper_width": aligned_episode.width_alignment.width_m[:, None].shape,
        "valid_mask": valid_mask.shape,
    }
    for name, expected_shape in expected_frame_shapes.items():
        if actual_frame_shapes[name] != expected_shape:
            raise ValueError(
                f"{name} 形状错误：{actual_frame_shapes[name]}，期望：{expected_shape}"
            )

    expected_action_shape = (
        frame_count - actions_relative_7d.shape[1],
        actions_relative_7d.shape[1],
        7,
    )
    if actions_relative_7d.shape != expected_action_shape:
        raise ValueError(
            f"actions_relative_7d 形状错误：{actions_relative_7d.shape}，"
            f"期望：{expected_action_shape}"
        )
    if relative_action_result.actions_valid_mask.shape != (actions_relative_7d.shape[0],):
        raise ValueError("actions_valid_mask 与 actions_relative_7d 第一维不一致。")
    if not np.any(relative_action_result.actions_valid_mask):
        raise RuntimeError("没有任何完整有效的未来动作窗口，本条数据不保存。")
    if image_size <= 0:
        raise ValueError(f"image_size 必须大于0，当前为：{image_size}")


def save_episode_outputs(
    paths: InputPaths,
    args: argparse.Namespace,
    sync_info: dict[str, Any],
    cropped_timeline: CroppedTimeline,
    aligned_episode: AlignedEpisode,
) -> dict[str, Any]:
    """
    写入临时 episode 目录，重新结合实际RGB解码结果生成mask和动作，再保存元数据。

    只有这个函数成功返回后，调用方才会把 001.tmp 重命名为 001。
    """

    temporary_dir = paths.temporary_episode_dir
    data_dir = temporary_dir / "data"
    auxiliary_dir = temporary_dir / "auxiliary"
    metadata_dir = temporary_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=False)
    auxiliary_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir.mkdir(parents=True, exist_ok=False)

    print("\n正在整理 D435i RGB：")
    decoded_rgb_valid_mask = save_camera0_rgb(
        output_path=data_dir / "camera0_rgb.npy",
        rgb_paths=aligned_episode.rgb_paths,
        image_size=args.image_size,
    )
    valid_mask = aligned_episode.base_valid_mask & decoded_rgb_valid_mask

    relative_action_result, actions_relative_7d = make_actions_relative_7d(
        T_W_G=aligned_episode.T_W_G,
        timestamps_s=aligned_episode.frame_data.timestamps_s,
        gripper_width_m=aligned_episode.width_alignment.width_m,
        valid_mask=valid_mask,
        action_horizon=args.horizon,
    )
    validate_final_arrays(
        aligned_episode=aligned_episode,
        valid_mask=valid_mask,
        relative_action_result=relative_action_result,
        actions_relative_7d=actions_relative_7d,
        image_size=args.image_size,
    )

    # ------------------------
    # 训练直接使用的数据
    # ------------------------
    np.save(data_dir / "robot0_eef_pos.npy", aligned_episode.robot0_eef_pos.astype(np.float32))
    np.save(
        data_dir / "robot0_eef_rot_axis_angle.npy",
        aligned_episode.robot0_eef_rot_axis_angle.astype(np.float32),
    )
    np.save(
        data_dir / "robot0_gripper_width.npy",
        aligned_episode.width_alignment.width_m[:, None].astype(np.float32),
    )
    # 沿用 stray_shift_position.py 已有命名，而不是另存为含义不明确的 timestamps.npy。
    np.save(data_dir / "d435i_timestamps.npy", aligned_episode.frame_data.timestamps_s)
    np.save(data_dir / "valid_mask.npy", valid_mask.astype(bool))
    np.save(data_dir / "actions_relative_7d.npy", actions_relative_7d)
    np.save(
        data_dir / "actions_valid_mask.npy",
        relative_action_result.actions_valid_mask.astype(bool),
    )
    np.save(
        data_dir / "action_start_timestamps.npy",
        relative_action_result.action_start_timestamps,
    )

    # ------------------------
    # 沿用旧脚本名称的可追溯中间结果
    # ------------------------
    np.save(auxiliary_dir / "d435i_frame_indices.npy", aligned_episode.frame_data.frame_indices)
    np.save(
        auxiliary_dir / "iphone_timestamps.npy",
        aligned_episode.pose_alignment.iphone_query_timestamps,
    )
    np.save(auxiliary_dir / "T_W_I.npy", aligned_episode.pose_alignment.T_W_I)
    np.save(auxiliary_dir / "T_I_G.npy", aligned_episode.T_I_G)
    np.save(auxiliary_dir / "T_W_G.npy", aligned_episode.T_W_G)
    np.save(
        auxiliary_dir / "gripper_width_aligned.npy",
        aligned_episode.width_alignment.width_m.astype(np.float32),
    )
    np.save(
        auxiliary_dir / "gripper_d_left_aligned.npy",
        aligned_episode.width_alignment.d_left_m.astype(np.float32),
    )
    np.save(
        auxiliary_dir / "gripper_d_right_aligned.npy",
        aligned_episode.width_alignment.d_right_m.astype(np.float32),
    )
    np.save(
        auxiliary_dir / "gripper_sync_error_aligned.npy",
        aligned_episode.width_alignment.sync_error_m.astype(np.float32),
    )
    np.save(
        auxiliary_dir / "actions_relative_4x4.npy",
        relative_action_result.actions_relative_4x4,
    )
    np.save(
        auxiliary_dir / "actions_relative_6d.npy",
        relative_action_result.actions_relative_6d,
    )

    frame_count = len(aligned_episode.frame_data.timestamps_s)
    valid_frame_count = int(np.count_nonzero(valid_mask))
    valid_action_count = int(np.count_nonzero(relative_action_result.actions_valid_mask))
    timestamp_statistics = calculate_timestamp_statistics(
        aligned_episode.frame_data.timestamps_s
    )
    episode_ends = np.asarray([frame_count], dtype=np.int64)
    np.save(metadata_dir / "episode_ends.npy", episode_ends)

    source_info = {
        "dataset_group": paths.dataset_group_dir.name,
        "episode_id": paths.episode_id,
        "source_session_name": paths.demo_dir.name,
        "source_session_dir": str(paths.demo_dir),
        "odometry_csv": str(paths.odometry_csv_path),
        "d435i_timestamps_csv": str(paths.d435i_timestamps_csv_path),
        "gripper_width_csv": str(paths.gripper_width_csv_path),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(metadata_dir / "source.json", source_info)
    write_json(metadata_dir / "sync.json", sync_info)

    processing_info = {
        "builder": "build_dataset.py",
        "integrated_functions": [
            "estimate_sync_by_shake.py single-shake synchronization",
            "stray_shift_position.py pose/width alignment, T_I_G, T_W_G and relative actions",
        ],
        "image_preprocess": {
            "color_order": "RGB",
            "dtype": "uint8",
            "output_shape": [frame_count, args.image_size, args.image_size, 3],
            "resize": "keep aspect ratio and center letterbox with black pixels",
        },
        "action": {
            "file": "data/actions_relative_7d.npy",
            "shape": list(actions_relative_7d.shape),
            "horizon": int(args.horizon),
            "definition": [
                "relative_position_x_m",
                "relative_position_y_m",
                "relative_position_z_m",
                "relative_rotvec_x_rad",
                "relative_rotvec_y_rad",
                "relative_rotvec_z_rad",
                "future_gripper_width_m",
            ],
            "future_steps": "t+1 ... t+H",
        },
        "state_shapes": {
            "robot0_eef_pos": list(aligned_episode.robot0_eef_pos.shape),
            "robot0_eef_rot_axis_angle": list(
                aligned_episode.robot0_eef_rot_axis_angle.shape
            ),
            "robot0_gripper_width": [frame_count, 1],
            "d435i_timestamps": [frame_count],
            "valid_mask": [frame_count],
        },
        "start_delay_after_shake_seconds": sync_info['actual_pause_seconds'],
        "start_time_source": sync_info['dataset_start_time_source'],
        "synchronization_method": sync_info['method'],
        "max_iphone_gap_seconds": float(args.max_iphone_gap_seconds),
        "position_unit": "meter",
        "rotation_format": "axis_angle_rotvec",
        "rotation_unit": "radian",
        "gripper_width_unit": "meter",
        "calibration_source": str(TIG_CALIBRATION_PATH),
        "success_failure_label": "not_included",
        "test_directory": "not_generated",
    }
    write_json(metadata_dir / "processing.json", processing_info)

    summary_lines = [
        "UMI dataset episode processing summary",
        "",
        f"dataset_group: {paths.dataset_group_dir.name}",
        f"episode_id: {paths.episode_id}",
        f"source_session: {paths.demo_dir}",
        f"odometry_csv: {paths.odometry_csv_path}",
        "",
        "single_shake_sync:",
        f"  offset_iphone_minus_d435i_s: {float(sync_info['offset_iphone_minus_d435i']):.9f}",
        f"  d435i_shake_time_s: {float(sync_info['d435i_shake_time_s']):.9f}",
        f"  iphone_shake_time_s: {float(sync_info['iphone_shake_time_s']):.9f}",
        f"  start_delay_after_shake_s: {float(sync_info['actual_pause_seconds']):.3f}",
        "  second_shake_validation: not used",
        "",
        "cropped_timeline:",
        f"  original_d435i_frame_count: {cropped_timeline.original_frame_count}",
        f"  dataset_frame_count: {frame_count}",
        f"  dataset_start_time_s: {cropped_timeline.start_time_s:.9f}",
        f"  dataset_end_time_s: {cropped_timeline.end_time_s:.9f}",
        f"  duration_s: {cropped_timeline.end_time_s - cropped_timeline.start_time_s:.3f}",
        f"  estimated_fps: {timestamp_statistics['estimated_fps']:.3f}",
        f"  max_frame_interval_s: {timestamp_statistics['max_dt_seconds']:.6f}",
        "",
        "validity:",
        f"  valid_frames: {valid_frame_count}/{frame_count}",
        f"  valid_action_chunks: {valid_action_count}/{len(relative_action_result.actions_valid_mask)}",
        "",
        "output_shapes:",
        f"  camera0_rgb: ({frame_count}, {args.image_size}, {args.image_size}, 3) uint8",
        f"  robot0_eef_pos: {aligned_episode.robot0_eef_pos.shape} float32 meter",
        f"  robot0_eef_rot_axis_angle: {aligned_episode.robot0_eef_rot_axis_angle.shape} float32 radian",
        f"  robot0_gripper_width: ({frame_count}, 1) float32 meter",
        f"  actions_relative_7d: {actions_relative_7d.shape} float32",
        f"  episode_ends: {episode_ends.tolist()}",
        "",
        "notes:",
        "  This episode was built directly from raw data; standalone sync/shift scripts are not required.",
        "  No test directory and no success/failure label were generated.",
    ]
    (metadata_dir / "process_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    return {
        "frame_count": frame_count,
        "valid_frame_count": valid_frame_count,
        "action_count": len(relative_action_result.actions_valid_mask),
        "valid_action_count": valid_action_count,
        "action_shape": tuple(actions_relative_7d.shape),
        "estimated_fps": float(timestamp_statistics["estimated_fps"]),
    }


def print_processing_inputs(paths: InputPaths, args: argparse.Namespace) -> None:
    """在真正开始处理前显示所有关键路径和固定参数，便于人工发现选错数据。"""

    print("\n准备构建数据集 episode：")
    print(f"  数据集类别：{paths.dataset_group_dir}")
    print(f"  episode编号：{paths.episode_id}")
    print(f"  原始session：{paths.demo_dir}")
    print(f"  odometry.csv：{paths.odometry_csv_path}")
    print(f"  D435i时间戳：{paths.d435i_timestamps_csv_path}")
    print(f"  夹爪宽度：{paths.gripper_width_csv_path}")
    print(f"  最终输出：{paths.final_episode_dir}")
    print(f"  shake搜索：前 {args.shake_search_seconds:.1f} 秒")
    print(f"  shake后延迟：{args.start_delay_seconds:.1f} 秒")
    print(f"  起点策略：{args.start_source}；同步策略：{args.sync_method}")
    print(f"  action horizon：{args.horizon}")
    print(f"  RGB输出：{args.image_size}x{args.image_size}")
    print("  第二次shake：不使用")
    print("  test目录：不生成")


def run_dataset_build(args: argparse.Namespace) -> Path:
    """执行一次完整数据集构建，并返回最终 episode 路径。"""

    if args.horizon <= 0:
        raise ValueError(f"--horizon 必须大于0，当前为：{args.horizon}")
    if args.shake_search_seconds <= 0:
        raise ValueError("--shake-search-seconds 必须大于0。")
    if args.start_delay_seconds < 0:
        raise ValueError("--start-delay-seconds 不能小于0。")
    if args.max_iphone_gap_seconds <= 0:
        raise ValueError("--max-iphone-gap-seconds 必须大于0。")
    if args.image_size <= 0:
        raise ValueError("--image-size 必须大于0。")

    paths = build_input_paths(args)
    print_processing_inputs(paths, args)
    if not args.yes:
        confirmation = input("\n确认使用以上数据开始处理吗？[Y/n]：").strip().lower()
        if confirmation in {"n", "no", "否"}:
            print("已取消，本次没有写入数据。")
            return paths.final_episode_dir

    print("\n1/5 估计第一次 shake 的时间偏移……")
    sync_info = estimate_single_shake_sync(
        demo_dir=paths.demo_dir,
        odometry_csv_path=paths.odometry_csv_path,
        shake_search_seconds=args.shake_search_seconds,
    )
    prepare_shake_timing(paths.demo_dir, sync_info, args.start_source,
                         args.sync_method, args.start_delay_seconds)
    print(
        "  offset_iphone_minus_d435i："
        f"{float(sync_info['offset_iphone_minus_d435i']):.9f} s"
    )

    print("2/5 读取 odometry 和夹爪宽度……")
    odometry_data = load_odometry_csv(paths.odometry_csv_path)
    width_data = load_gripper_width_csv(paths.gripper_width_csv_path)

    print(
        "3/5 按 shake+"
        f"{args.start_delay_seconds:g}秒 到按S停止位置裁剪D435i时间轴……"
    )
    cropped_timeline = crop_d435i_timeline(
        paths=paths,
        sync_info=sync_info,
        odometry_timestamps_s=odometry_data.timestamps,
        gripper_timestamps_s=width_data.timestamps_s,
        start_delay_seconds=args.start_delay_seconds,
        action_horizon=args.horizon,
    )
    print(
        f"  保留 {len(cropped_timeline.frame_data.timestamps_s)} 帧，"
        f"时间 {cropped_timeline.start_time_s:.6f} -> {cropped_timeline.end_time_s:.6f}"
    )

    print(f"4/5 对齐位姿和夹爪宽度，并应用本设备 {TIG_CALIBRATION_PATH}……")
    aligned_episode = align_episode_data(
        cropped_timeline=cropped_timeline,
        odometry_data=odometry_data,
        width_data=width_data,
        sync_offset_seconds=float(sync_info["offset_iphone_minus_d435i"]),
        max_iphone_gap_seconds=args.max_iphone_gap_seconds,
    )

    print("5/5 写入RGB、状态、7维动作和元数据……")
    temporary_dir_created = False
    try:
        # save_episode_outputs 内部首先创建 001.tmp；只有本次调用创建成功后才允许清理它。
        build_statistics = save_episode_outputs(
            paths=paths,
            args=args,
            sync_info=sync_info,
            cropped_timeline=cropped_timeline,
            aligned_episode=aligned_episode,
        )
        temporary_dir_created = True

        if paths.final_episode_dir.exists():
            raise FileExistsError(f"正式输出目录在处理期间被创建：{paths.final_episode_dir}")
        paths.temporary_episode_dir.rename(paths.final_episode_dir)
        temporary_dir_created = False
    except Exception:
        # 临时目录在本次运行前已经被拒绝，因此这里删除的只可能是本次新生成的不完整输出。
        if paths.temporary_episode_dir.exists():
            temporary_dir_created = True
        if temporary_dir_created:
            shutil.rmtree(paths.temporary_episode_dir)
        raise

    print("\n数据集构建完成：")
    print(f"  输出目录：{paths.final_episode_dir}")
    print(
        f"  有效帧：{build_statistics['valid_frame_count']}/"
        f"{build_statistics['frame_count']}"
    )
    print(
        f"  有效动作窗口：{build_statistics['valid_action_count']}/"
        f"{build_statistics['action_count']}"
    )
    print(f"  actions_relative_7d：{build_statistics['action_shape']}")
    print(f"  估计帧率：{build_statistics['estimated_fps']:.3f} FPS")
    print("  已集成单次shake同步和stray_shift_position处理，无需再运行旧脚本。")
    print("  未生成test目录，也未写入成功/失败标签。")
    return paths.final_episode_dir


def main() -> int:
    """命令行入口；错误时输出简洁原因并返回非零状态码。"""

    args = parse_args()
    try:
        run_dataset_build(args)
    except KeyboardInterrupt:
        print("\n用户中断，本次数据集构建未完成。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n数据集构建失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
