"""
Stray Scanner odometry.csv -> UMI-style relative action labels.

默认用途：
    1. 自动找到 D:\\Users\\dell\\Desktop\\umi\\0811data 里面最新且包含 odometry.csv 的时间戳 session 文件夹。
    2. 自动递归寻找 session 文件夹内部的 odometry.csv。
    3. 将 iPhone 位姿转换成 4x4 齐次变换矩阵 T_W_I。
    4. 使用定标得到的 T_I_G，表示 iPhone 坐标系到夹爪 TCP 坐标系的外参。
    5. 计算 TCP 位姿 T_W_G。
    6. 生成 UMI 风格的相对轨迹动作标签 actions_relative_6d。

输出目录结构：
    最新时间戳 session 文件夹/
    ├── d435i/
    ├── gripper_width_log.csv
    ├── 某个 iPhone Stray 导出文件夹/
    │   └── odometry.csv
    └── processed_umi/
        ├── T_W_I.npy
        ├── iphone_timestamps.npy
        ├── T_I_G.npy
        ├── T_W_G.npy
        ├── actions_relative_4x4.npy
        ├── actions_relative_6d.npy
        ├── action_start_timestamps.npy
        ├── process_summary.txt
        └── test/
            ├── iphone_position_xyz.png
            ├── iphone_traj_xz.png
            ├── relative_translation_norm.png
            └── relative_rotation_norm.png

运行方式：
    python stray_shift_position.py

可选参数示例：
    python stray_shift_position.py --horizon 16
    python stray_shift_position.py --stray-root "D:\\Users\\dell\\Desktop\\umi\\0811data"
    python stray_shift_position.py --demo-dir "D:\\Users\\dell\\Desktop\\umi\\0811data\\data_0811_2100"
    python stray_shift_position.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from estimate_sync_by_shake import estimate_sync_for_demo
from xac_config import DATA_ROOT, TIG_CALIBRATION_PATH


# ============================================================
# 1. 默认路径与参数
# ============================================================

# Stray Scanner 导出的数据根目录。脚本默认会从这里挑选“最新的子文件夹”。
DEFAULT_STRAY_ROOT = DATA_ROOT

# 输出目录名称。它会创建在最新 Stray 文件夹下面。
DEFAULT_OUTPUT_DIR_NAME = "processed_umi"

# 调试图目录名称。调试图不直接放在 processed_umi 根目录，避免和后续训练需要的 npy 文件混在一起。
DEFAULT_TEST_DIR_NAME = "test"

# UMI 相对轨迹动作的 horizon。H=16 表示每个时刻生成未来 16 帧的相对 TCP 运动。
DEFAULT_ACTION_HORIZON = 16

# odometry.csv 必须包含这些基础列。
REQUIRED_ODOMETRY_COLUMNS = ("timestamp", "x", "y", "z", "qx", "qy", "qz", "qw")

# 四元数范数太小时说明数据损坏，继续转换会得到不可信的旋转矩阵。
MIN_QUATERNION_NORM = 1e-8

# 递归搜索 odometry.csv 时跳过这些目录，避免误把输出目录或缓存目录里的文件当成原始输入。
IGNORED_ODOMETRY_SEARCH_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
}


@dataclass(frozen=True)
class ProcessPaths:
    """集中保存本次处理会用到的关键路径，减少函数之间传参混乱。"""

    demo_dir: Path
    odometry_csv_path: Path
    output_dir: Path
    test_dir: Path


@dataclass(frozen=True)
class OdometryData:
    """保存 odometry.csv 中生成动作标签真正需要的数据。"""

    columns: list[str]
    timestamps: np.ndarray
    positions_xyz: np.ndarray
    quaternions_xyzw: np.ndarray


@dataclass(frozen=True)
class D435iFrameData:
    """保存 D435i 每一帧的编号和电脑时间。"""

    frame_indices: np.ndarray
    timestamps_s: np.ndarray


@dataclass(frozen=True)
class GripperWidthData:
    """保存夹爪宽度日志。"""

    timestamps_s: np.ndarray
    width_m: np.ndarray
    d_left_m: np.ndarray
    d_right_m: np.ndarray
    sync_error_m: np.ndarray


@dataclass(frozen=True)
class PoseAlignmentResult:
    """保存 iPhone pose 对齐到 D435i 时间轴后的结果。"""

    T_W_I: np.ndarray
    iphone_query_timestamps: np.ndarray
    iphone_idx_before: np.ndarray
    iphone_idx_after: np.ndarray
    alpha: np.ndarray
    valid_mask: np.ndarray
    reasons: list[str]


@dataclass(frozen=True)
class WidthAlignmentResult:
    """保存 gripper width 对齐到 D435i 时间轴后的结果。"""

    width_m: np.ndarray
    d_left_m: np.ndarray
    d_right_m: np.ndarray
    sync_error_m: np.ndarray
    valid_mask: np.ndarray
    reasons: list[str]


@dataclass(frozen=True)
class RelativeActionResult:
    """保存相对动作生成结果，便于后续统一保存与统计。"""

    actions_relative_4x4: np.ndarray
    actions_relative_6d: np.ndarray
    action_start_timestamps: np.ndarray
    actions_valid_mask: np.ndarray


# ============================================================
# 2. 文件夹与 CSV 读取
# ============================================================


def find_latest_subdirectory(root_dir: Path) -> Path:
    """
    在 Stray 数据根目录下找修改时间最新的子文件夹。

    这样你每次采集完新的 Stray 数据后，不需要手动改脚本路径；
    只要新数据文件夹在 D:\\Users\\dell\\Desktop\\umi\\0811data 里面，脚本就会自动处理最新的那一个。
    """

    if not root_dir.exists():
        raise FileNotFoundError(f"Stray 数据根目录不存在：{root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Stray 数据根路径不是文件夹：{root_dir}")

    candidate_dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"Stray 数据根目录下没有任何子文件夹：{root_dir}")

    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)


def is_path_inside_ignored_directory(path: Path, search_root: Path, output_dir_name: str) -> bool:
    """
    判断某个路径是否位于不应该参与 odometry 搜索的目录下。

    当前采集结构里，processed_umi 是本脚本的输出目录；
    如果以后重新处理数据，不应该从旧输出目录里再找输入。
    """

    ignored_dir_names = {
        name.lower() for name in IGNORED_ODOMETRY_SEARCH_DIR_NAMES
    }
    ignored_dir_names.add(output_dir_name.lower())

    try:
        relative_parts = path.relative_to(search_root).parts
    except ValueError:
        return False

    return any(part.lower() in ignored_dir_names for part in relative_parts[:-1])


def find_odometry_csv_candidates(search_dir: Path, output_dir_name: str) -> list[Path]:
    """
    在某个文件夹内递归查找所有 odometry.csv 候选。

    兼容两种结构：
        001/odometry.csv

    以及现在的新结构：
        001/某个 iPhone Stray 导出文件夹/odometry.csv

    注意：
        这里只找输入数据，不找 processed_umi 里的旧输出。
    """

    if not search_dir.exists():
        return []

    odometry_paths: list[Path] = []
    for path in search_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower() != "odometry.csv":
            continue
        if is_path_inside_ignored_directory(path, search_dir, output_dir_name):
            continue
        odometry_paths.append(path)

    return sorted(
        odometry_paths,
        key=lambda path: (path.stat().st_mtime, str(path)),
    )


def find_latest_demo_dir_containing_odometry(root_dir: Path, output_dir_name: str) -> Path:
    """
    在 Stray 数据根目录下，找到“最新且内部包含 odometry.csv”的 session 文件夹。

    现在你的流程是先由相机程序创建：
        D:\\Users\\dell\\Desktop\\umi\\0811data\\001

    然后再把 iPhone/Stray 导出的文件夹放进这个时间戳文件夹里。
    因此最新的有效 demo 不一定是 odometry.csv 自己所在的文件夹，
    而是包含它的外层数字编号 session 文件夹。
    """

    if not root_dir.exists():
        raise FileNotFoundError(f"Stray 数据根目录不存在：{root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Stray 数据根路径不是文件夹：{root_dir}")

    candidate_dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"Stray 数据根目录下没有任何子文件夹：{root_dir}")

    dirs_with_odometry: list[tuple[float, Path]] = []
    for candidate_dir in candidate_dirs:
        odometry_candidates = find_odometry_csv_candidates(
            search_dir=candidate_dir,
            output_dir_name=output_dir_name,
        )
        if not odometry_candidates:
            continue

        latest_odometry_mtime = max(path.stat().st_mtime for path in odometry_candidates)
        dirs_with_odometry.append((latest_odometry_mtime, candidate_dir))

    if not dirs_with_odometry:
        latest_dir = find_latest_subdirectory(root_dir)
        raise FileNotFoundError(
            "没有在任何 Stray 数据子文件夹内找到 odometry.csv。\n"
            f"当前最新子文件夹是：{latest_dir}\n"
            "请确认已经把 iPhone/Stray Scanner 导出的文件夹放进对应的数字编号 session 目录。"
        )

    return max(dirs_with_odometry, key=lambda item: item[0])[1]


def find_best_odometry_csv(demo_dir: Path, output_dir_name: str) -> Path:
    """
    在 demo 文件夹内递归查找 odometry.csv，并选择最合适的一个。

    如果只找到一个，直接使用。
    如果找到多个，选择修改时间最新的 odometry.csv，并打印候选列表方便人工检查。
    """

    odometry_paths = find_odometry_csv_candidates(
        search_dir=demo_dir,
        output_dir_name=output_dir_name,
    )

    if not odometry_paths:
        raise FileNotFoundError(f"在该 Stray 文件夹内没有找到 odometry.csv：{demo_dir}")

    if len(odometry_paths) > 1:
        joined_paths = "\n".join(str(path) for path in odometry_paths)
        chosen_path = odometry_paths[-1]
        print(
            "找到多个 odometry.csv，已自动使用修改时间最新的一个：\n"
            f"  {chosen_path}\n"
            f"候选列表：\n{joined_paths}"
        )
        return chosen_path

    return odometry_paths[0]


def build_process_paths(
    stray_root: Path,
    explicit_demo_dir: Path | None,
    output_dir_name: str,
    test_dir_name: str,
    overwrite_existing_output: bool,
) -> ProcessPaths:
    """
    根据用户参数构造完整输入/输出路径。

    如果 explicit_demo_dir 不为空，就处理用户指定的数据文件夹；
    否则自动处理 stray_root 下面最新且内部包含 odometry.csv 的 session 文件夹。
    """

    demo_dir = (
        explicit_demo_dir
        if explicit_demo_dir is not None
        else find_latest_demo_dir_containing_odometry(
            root_dir=stray_root,
            output_dir_name=output_dir_name,
        )
    )
    demo_dir = demo_dir.resolve()

    if not demo_dir.exists():
        raise FileNotFoundError(f"指定的 Stray 数据文件夹不存在：{demo_dir}")

    if not demo_dir.is_dir():
        raise NotADirectoryError(f"指定的 Stray 数据路径不是文件夹：{demo_dir}")

    odometry_csv_path = find_best_odometry_csv(
        demo_dir=demo_dir,
        output_dir_name=output_dir_name,
    )
    output_dir = demo_dir / output_dir_name
    test_dir = output_dir / test_dir_name

    # 防止重复处理同一个 Stray 文件夹，把之前已经生成好的结果覆盖掉。
    # 如果确实需要重新生成，运行时显式加 --overwrite。
    if output_dir.exists() and not overwrite_existing_output:
        raise FileExistsError(
            f"检测到该 Stray 文件夹已经存在输出目录：{output_dir}\n"
            "为防止重复生成或覆盖旧结果，本次已停止。\n"
            "如果你确认要重新生成，请运行：python stray_to_umi_actions.py --overwrite"
        )

    return ProcessPaths(
        demo_dir=demo_dir,
        odometry_csv_path=odometry_csv_path,
        output_dir=output_dir,
        test_dir=test_dir,
    )


def parse_required_float(row: dict[str, str], column: str, line_number: int) -> float:
    """
    从 CSV 行里读取一个必要的浮点数。

    单独封装这个函数，是为了在数据异常时给出明确错误：
    哪一行、哪一列、原始内容是什么。
    """

    raw_value = row.get(column)
    if raw_value is None or raw_value.strip() == "":
        raise ValueError(f"odometry.csv 第 {line_number} 行缺少必要列 {column} 的数值。")

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"odometry.csv 第 {line_number} 行的 {column} 不是有效数字：{raw_value!r}"
        ) from exc


def load_odometry_csv(odometry_csv_path: Path) -> OdometryData:
    """
    读取并校验 Stray Scanner 的 odometry.csv。

    目前只依赖 timestamp、平移 x/y/z、四元数 qx/qy/qz/qw；
    fx/fy/cx/cy 等相机内参列暂时不参与动作标签生成。
    """

    timestamps: list[float] = []
    positions_xyz: list[list[float]] = []
    quaternions_xyzw: list[list[float]] = []

    with odometry_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        # skipinitialspace=True 可以处理 "timestamp, frame, x" 这种逗号后带空格的表头。
        csv_reader = csv.DictReader(csv_file, skipinitialspace=True)

        if csv_reader.fieldnames is None:
            raise ValueError(f"odometry.csv 没有表头：{odometry_csv_path}")

        columns = [column.strip() for column in csv_reader.fieldnames]
        missing_columns = [
            column for column in REQUIRED_ODOMETRY_COLUMNS if column not in columns
        ]
        if missing_columns:
            raise KeyError(
                f"odometry.csv 缺少必要列：{missing_columns}\n"
                f"当前实际列名：{columns}"
            )

        for line_number, raw_row in enumerate(csv_reader, start=2):
            # DictReader 使用原始表头做 key。这里统一 strip，避免 " x" 这种列名影响读取。
            row = {
                key.strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
                if key is not None
            }

            timestamps.append(parse_required_float(row, "timestamp", line_number))
            positions_xyz.append(
                [
                    parse_required_float(row, "x", line_number),
                    parse_required_float(row, "y", line_number),
                    parse_required_float(row, "z", line_number),
                ]
            )
            quaternions_xyzw.append(
                [
                    parse_required_float(row, "qx", line_number),
                    parse_required_float(row, "qy", line_number),
                    parse_required_float(row, "qz", line_number),
                    parse_required_float(row, "qw", line_number),
                ]
            )

    if len(timestamps) < 2:
        raise ValueError(f"odometry.csv 帧数太少，无法生成相对动作：{len(timestamps)}")

    return OdometryData(
        columns=columns,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        positions_xyz=np.asarray(positions_xyz, dtype=np.float64),
        quaternions_xyzw=np.asarray(quaternions_xyzw, dtype=np.float64),
    )


def find_d435i_timestamps_csv(demo_dir: Path) -> Path:
    """找到 D435i 的 timestamps.csv。"""

    timestamps_path = demo_dir / "d435i" / "timestamps.csv"
    if not timestamps_path.exists():
        raise FileNotFoundError(f"没有找到 D435i timestamps.csv：{timestamps_path}")
    return timestamps_path


def load_d435i_timestamps_csv(timestamps_csv_path: Path) -> D435iFrameData:
    """
    读取 D435i 每帧时间。

    优先使用 capture_system_time_s：它是 record_435i.py 用 color 流传感器时间戳
    换算出的真实拍摄时刻(墙钟)，和 gripper_width_log.csv 的 computer_time_s 同轴，
    且不含帧队列积压延迟。旧文件没有该列时回退到 system_time_s。
    """

    frame_indices: list[int] = []
    timestamps_s: list[float] = []

    with timestamps_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"D435i timestamps.csv 没有表头：{timestamps_csv_path}")

        columns = [column.strip() for column in reader.fieldnames]
        required_columns = {"frame_idx", "system_time_s"}
        missing_columns = sorted(required_columns - set(columns))
        if missing_columns:
            raise KeyError(
                f"D435i timestamps.csv 缺少必要列：{missing_columns}\n"
                f"当前实际列名：{columns}"
            )
        # 新采集文件带真实拍摄时刻列; 旧文件没有则回退到出队时刻 system_time_s。
        capture_column = (
            "capture_system_time_s"
            if "capture_system_time_s" in columns
            else "system_time_s"
        )

        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                key.strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
                if key is not None
            }
            try:
                frame_indices.append(int(float(row["frame_idx"])))
                timestamps_s.append(float(row[capture_column]))
            except ValueError as exc:
                raise ValueError(f"D435i timestamps.csv 第 {line_number} 行格式错误：{row}") from exc

    if len(frame_indices) < 2:
        raise ValueError(f"D435i 帧数太少，无法对齐：{len(frame_indices)}")

    frame_indices_array = np.asarray(frame_indices, dtype=np.int64)
    timestamps_array = np.asarray(timestamps_s, dtype=np.float64)
    timestamp_diffs = np.diff(timestamps_array)

    # RealSense 的 global timestamp 刚初始化时，偶尔会让最前面的两个 RGB-D
    # 帧得到完全相同的 capture_system_time_s。它们位于 shake/有效任务之前，
    # 丢弃后一个重复启动帧比让整条有效 demo 失败更合理。真正的时间倒退仍然
    # 拒绝处理，避免把乱序任务帧用于动作标签。
    if np.any(timestamp_diffs < 0.0):
        first_bad_index = int(np.flatnonzero(timestamp_diffs < 0.0)[0])
        raise ValueError(
            "D435i timestamps.csv 存在时间倒退，不能安全生成动作："
            f"frame {frame_indices_array[first_bad_index]} -> "
            f"{frame_indices_array[first_bad_index + 1]}"
        )

    keep_mask = np.ones(len(timestamps_array), dtype=bool)
    keep_mask[1:] = timestamp_diffs > 0.0
    duplicate_count = int(np.count_nonzero(~keep_mask))
    if duplicate_count:
        print(f"  D435i启动时间戳去重：丢弃 {duplicate_count} 个重复帧。")
        frame_indices_array = frame_indices_array[keep_mask]
        timestamps_array = timestamps_array[keep_mask]

    if len(frame_indices_array) < 2:
        raise ValueError("D435i 时间戳去重后帧数不足，无法对齐。")

    return D435iFrameData(
        frame_indices=frame_indices_array,
        timestamps_s=timestamps_array,
    )


def find_gripper_width_csv(demo_dir: Path, output_dir_name: str) -> Path:
    """
    找到夹爪宽度日志。

    兼容：
        gripper_width_log.csv
        gripper_width.csv
    """

    direct_candidates = [
        demo_dir / "gripper_width_log.csv",
        demo_dir / "gripper_width.csv",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    candidates = []
    for path in demo_dir.rglob("*.csv"):
        if path.name.lower() not in {"gripper_width_log.csv", "gripper_width.csv"}:
            continue
        if is_path_inside_ignored_directory(path, demo_dir, output_dir_name):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"没有找到夹爪宽度日志 gripper_width_log.csv / gripper_width.csv：{demo_dir}"
        )

    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1]


def load_gripper_width_csv(width_csv_path: Path) -> GripperWidthData:
    """读取夹爪宽度日志。"""

    timestamps_s: list[float] = []
    width_m: list[float] = []
    d_left_m: list[float] = []
    d_right_m: list[float] = []
    sync_error_m: list[float] = []

    with width_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"夹爪宽度 CSV 没有表头：{width_csv_path}")

        columns = [column.strip() for column in reader.fieldnames]
        required_columns = {"computer_time_s", "width_m"}
        missing_columns = sorted(required_columns - set(columns))
        if missing_columns:
            raise KeyError(
                f"夹爪宽度 CSV 缺少必要列：{missing_columns}\n"
                f"当前实际列名：{columns}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                key.strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
                if key is not None
            }
            try:
                timestamps_s.append(float(row["computer_time_s"]))
                width_m.append(float(row["width_m"]))
                d_left_m.append(float(row.get("d_left", "nan")))
                d_right_m.append(float(row.get("d_right", "nan")))
                sync_error_m.append(float(row.get("sync_error", "nan")))
            except ValueError as exc:
                raise ValueError(f"夹爪宽度 CSV 第 {line_number} 行格式错误：{row}") from exc

    if len(timestamps_s) < 2:
        raise ValueError(f"夹爪宽度日志太短，无法对齐：{len(timestamps_s)}")

    timestamps_array = np.asarray(timestamps_s, dtype=np.float64)
    order = np.argsort(timestamps_array)

    return GripperWidthData(
        timestamps_s=timestamps_array[order],
        width_m=np.asarray(width_m, dtype=np.float64)[order],
        d_left_m=np.asarray(d_left_m, dtype=np.float64)[order],
        d_right_m=np.asarray(d_right_m, dtype=np.float64)[order],
        sync_error_m=np.asarray(sync_error_m, dtype=np.float64)[order],
    )


# ============================================================
# 3. 位姿矩阵转换
# ============================================================


def quaternions_xyzw_to_rotation_matrices(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """
    将 [qx, qy, qz, qw] 四元数批量转换成旋转矩阵。

    这里不用 scipy，而是直接写公式，减少环境依赖。
    输入形状：
        [N, 4]
    输出形状：
        [N, 3, 3]
    """

    if quaternions_xyzw.ndim != 2 or quaternions_xyzw.shape[1] != 4:
        raise ValueError(
            "quaternions_xyzw 形状必须是 [N, 4]，当前是："
            f"{quaternions_xyzw.shape}"
        )

    quaternion_norms = np.linalg.norm(quaternions_xyzw, axis=1)
    invalid_mask = quaternion_norms < MIN_QUATERNION_NORM
    if invalid_mask.any():
        first_bad_index = int(np.where(invalid_mask)[0][0])
        raise ValueError(
            "发现四元数范数过小，说明 odometry.csv 里可能有坏数据："
            f"index={first_bad_index}, quat={quaternions_xyzw[first_bad_index]}"
        )

    normalized_quaternions = quaternions_xyzw / quaternion_norms[:, None]
    qx = normalized_quaternions[:, 0]
    qy = normalized_quaternions[:, 1]
    qz = normalized_quaternions[:, 2]
    qw = normalized_quaternions[:, 3]

    rotation_matrices = np.empty((len(normalized_quaternions), 3, 3), dtype=np.float64)

    # 标准四元数转旋转矩阵公式，四元数顺序为 x, y, z, w。
    rotation_matrices[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    rotation_matrices[:, 0, 1] = 2.0 * (qx * qy - qz * qw)
    rotation_matrices[:, 0, 2] = 2.0 * (qx * qz + qy * qw)

    rotation_matrices[:, 1, 0] = 2.0 * (qx * qy + qz * qw)
    rotation_matrices[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    rotation_matrices[:, 1, 2] = 2.0 * (qy * qz - qx * qw)

    rotation_matrices[:, 2, 0] = 2.0 * (qx * qz - qy * qw)
    rotation_matrices[:, 2, 1] = 2.0 * (qy * qz + qx * qw)
    rotation_matrices[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)

    return rotation_matrices


def rotation_matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    将单个 3x3 旋转矩阵转换成 [qx, qy, qz, qw] 四元数。

    这个函数主要服务于 rotation_matrix_to_rotvec。
    使用矩阵迹 trace 的经典分支写法，比直接 arccos 更能处理接近 180 度的情况。
    """

    matrix = rotation_matrix
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion_norm = np.linalg.norm(quaternion_xyzw)
    if quaternion_norm < MIN_QUATERNION_NORM:
        raise ValueError(f"旋转矩阵无法转换成有效四元数：\n{rotation_matrix}")

    quaternion_xyzw = quaternion_xyzw / quaternion_norm

    # q 和 -q 表示同一个旋转。这里强制 qw >= 0，让 rotvec 使用较短旋转方向。
    if quaternion_xyzw[3] < 0.0:
        quaternion_xyzw = -quaternion_xyzw

    return quaternion_xyzw


def rotation_matrix_to_rotvec(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    将单个 3x3 旋转矩阵转换成旋转向量。

    输出 rotvec 的方向是旋转轴，长度是旋转角，单位为弧度。
    """

    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation_matrix)
    vector_part = quaternion_xyzw[:3]
    scalar_part = float(np.clip(quaternion_xyzw[3], -1.0, 1.0))
    vector_norm = float(np.linalg.norm(vector_part))

    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(vector_norm, scalar_part)
    return vector_part / vector_norm * angle


def build_iphone_pose_sequence(odometry_data: OdometryData) -> tuple[np.ndarray, np.ndarray]:
    """
    将全部 odometry 数据转换成 iPhone 位姿序列。

    坐标含义：
        W = Stray / ARKit 世界坐标系
        I = iPhone 相机坐标系

    T_W_I 表示：
        把一个点从 iPhone 坐标系 I 变换到世界坐标系 W。

    返回：
        T_W_I: [N, 4, 4]
        iphone_timestamps: [N]
    """

    frame_count = len(odometry_data.timestamps)
    rotation_matrices = quaternions_xyzw_to_rotation_matrices(
        odometry_data.quaternions_xyzw
    )

    T_W_I = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], frame_count, axis=0)
    T_W_I[:, :3, :3] = rotation_matrices
    T_W_I[:, :3, 3] = odometry_data.positions_xyz

    return T_W_I, odometry_data.timestamps.copy()


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """归一化单个 [qx, qy, qz, qw] 四元数。"""

    norm = float(np.linalg.norm(quaternion))
    if norm < MIN_QUATERNION_NORM:
        raise ValueError(f"四元数范数过小，无法归一化：{quaternion}")
    return quaternion / norm


def slerp_quaternion_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    对 [qx, qy, qz, qw] 四元数做球面线性插值。

    q 和 -q 表示同一个旋转；如果点积为负，先翻转 q1，保证走较短旋转路径。
    """

    q0 = normalize_quaternion_xyzw(q0)
    q1 = normalize_quaternion_xyzw(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        interpolated = q0 + alpha * (q1 - q0)
        return normalize_quaternion_xyzw(interpolated)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    scale_0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale_1 = sin_theta / sin_theta_0
    return normalize_quaternion_xyzw(scale_0 * q0 + scale_1 * q1)


def transform_from_position_quaternion(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """根据位置和四元数构造单个 T_W_I。"""

    rotation_matrix = quaternions_xyzw_to_rotation_matrices(
        quaternion_xyzw.reshape(1, 4)
    )[0]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position_xyz
    return transform


def interpolate_iphone_pose_to_d435i_frames(
    odometry_data: OdometryData,
    d435i_timestamps_s: np.ndarray,
    sync_offset_iphone_minus_d435i: float,
    max_iphone_gap_seconds: float,
) -> PoseAlignmentResult:
    """
    以 D435i 每一帧为基准，把 iPhone pose 插值到对应时刻。

    映射关系：
        t_iphone = t_d435i + sync_offset_iphone_minus_d435i
    """

    iphone_times = odometry_data.timestamps
    positions = odometry_data.positions_xyz
    quaternions = odometry_data.quaternions_xyzw

    aligned_count = len(d435i_timestamps_s)
    T_W_I = np.full((aligned_count, 4, 4), np.nan, dtype=np.float64)
    iphone_query_times = d435i_timestamps_s + sync_offset_iphone_minus_d435i
    idx_before = np.full(aligned_count, -1, dtype=np.int64)
    idx_after = np.full(aligned_count, -1, dtype=np.int64)
    alphas = np.full(aligned_count, np.nan, dtype=np.float64)
    valid_mask = np.zeros(aligned_count, dtype=bool)
    reasons = ["not_processed"] * aligned_count

    for frame_index, query_time in enumerate(iphone_query_times):
        insert_index = int(np.searchsorted(iphone_times, query_time, side="left"))
        if insert_index <= 0:
            reasons[frame_index] = "iphone_time_before_odometry"
            continue
        if insert_index >= len(iphone_times):
            reasons[frame_index] = "iphone_time_after_odometry"
            continue

        before_index = insert_index - 1
        after_index = insert_index
        t0 = float(iphone_times[before_index])
        t1 = float(iphone_times[after_index])
        dt = t1 - t0
        if dt <= 1e-9:
            reasons[frame_index] = "iphone_duplicate_or_invalid_time"
            continue
        if dt > max_iphone_gap_seconds:
            reasons[frame_index] = "iphone_interpolation_gap_too_large"
            continue

        alpha = float((query_time - t0) / dt)
        position = (1.0 - alpha) * positions[before_index] + alpha * positions[after_index]
        quaternion = slerp_quaternion_xyzw(
            quaternions[before_index],
            quaternions[after_index],
            alpha,
        )

        T_W_I[frame_index] = transform_from_position_quaternion(position, quaternion)
        idx_before[frame_index] = before_index
        idx_after[frame_index] = after_index
        alphas[frame_index] = alpha
        valid_mask[frame_index] = True
        reasons[frame_index] = "ok"

    return PoseAlignmentResult(
        T_W_I=T_W_I,
        iphone_query_timestamps=iphone_query_times,
        iphone_idx_before=idx_before,
        iphone_idx_after=idx_after,
        alpha=alphas,
        valid_mask=valid_mask,
        reasons=reasons,
    )


def get_calibrated_T_I_G() -> np.ndarray:
    """
    返回定标后的外参 T_I_G。

    坐标含义：
        I = iPhone 坐标系
        G = gripper / TCP 坐标系

    矩阵语义：
        T_W_G = T_W_I @ T_I_G
        因此 T_I_G 表示夹爪/TCP 坐标系 G 在 iPhone 坐标系 I 中的位姿，
        可把使用 G 坐标表达的点转换为使用 I 坐标表达的点。

    本设备矩阵从 calibration/T_I_G.npy 加载。矩阵必须通过实机外参标定获得，
    不允许复用 collection_umip 中另一套夹爪的矩阵。
    """

    calibration_path = TIG_CALIBRATION_PATH
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"缺少本设备外参：{calibration_path}。"
            "请完成 iPhone 到夹爪 TCP 的外参标定；禁止沿用参考设备矩阵。"
        )
    matrix = np.asarray(np.load(calibration_path), dtype=np.float64)
    validate_single_transform("T_I_G", matrix)
    return matrix


def compute_tcp_pose_sequence(T_W_I: np.ndarray, T_I_G: np.ndarray) -> np.ndarray:
    """
    根据 iPhone 位姿和外参计算 TCP 位姿。

    公式：
        T_W_G(t) = T_W_I(t) @ T_I_G

    T_I_G 由 TIG 定标给出，因此 T_W_G 是经过外参转换后的夹爪 TCP 位姿。
    """

    validate_transform_sequence("T_W_I", T_W_I)
    validate_single_transform("T_I_G", T_I_G)

    return T_W_I @ T_I_G


MAX_GRIPPER_NEAREST_GAP_SECONDS = 0.1


def select_nearest_one_series_to_d435i(
    query_times_s: np.ndarray,
    source_times_s: np.ndarray,
    source_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为每个 D435i 时刻选择时间最近的真实夹爪样本，不做插值。

    返回：
        nearest_values
        valid_mask
    """
    nearest_values = np.full(len(query_times_s), np.nan, dtype=np.float64)
    valid_mask = np.zeros(len(query_times_s), dtype=bool)

    finite_time_mask = np.isfinite(source_times_s)
    if np.count_nonzero(finite_time_mask) < 1:
        return nearest_values, valid_mask

    times = source_times_s[finite_time_mask]
    values = source_values[finite_time_mask]
    order = np.argsort(times)
    times = times[order]
    values = values[order]

    finite_query_indices = np.flatnonzero(np.isfinite(query_times_s))
    if len(finite_query_indices) == 0:
        return nearest_values, valid_mask

    queries = query_times_s[finite_query_indices]
    insert_indices = np.searchsorted(times, queries, side="left")
    right_indices = np.clip(insert_indices, 0, len(times) - 1)
    left_indices = np.clip(insert_indices - 1, 0, len(times) - 1)
    left_deltas = np.abs(queries - times[left_indices])
    right_deltas = np.abs(times[right_indices] - queries)
    choose_right = right_deltas < left_deltas
    nearest_indices = np.where(choose_right, right_indices, left_indices)
    nearest_deltas = np.abs(queries - times[nearest_indices])
    selected_values = values[nearest_indices]
    selected_valid = (
        np.isfinite(selected_values)
        & (nearest_deltas <= MAX_GRIPPER_NEAREST_GAP_SECONDS)
    )

    valid_query_indices = finite_query_indices[selected_valid]
    nearest_values[valid_query_indices] = selected_values[selected_valid]
    valid_mask[valid_query_indices] = True
    return nearest_values, valid_mask


def align_gripper_width_to_d435i_frames(
    width_data: GripperWidthData,
    d435i_timestamps_s: np.ndarray,
) -> WidthAlignmentResult:
    """把每帧 D435i 映射到最近的15 Hz夹爪实测样本，不做插值。"""

    width_m, width_valid = select_nearest_one_series_to_d435i(
        query_times_s=d435i_timestamps_s,
        source_times_s=width_data.timestamps_s,
        source_values=width_data.width_m,
    )
    d_left_m, left_valid = select_nearest_one_series_to_d435i(
        query_times_s=d435i_timestamps_s,
        source_times_s=width_data.timestamps_s,
        source_values=width_data.d_left_m,
    )
    d_right_m, right_valid = select_nearest_one_series_to_d435i(
        query_times_s=d435i_timestamps_s,
        source_times_s=width_data.timestamps_s,
        source_values=width_data.d_right_m,
    )
    sync_error_m, sync_valid = select_nearest_one_series_to_d435i(
        query_times_s=d435i_timestamps_s,
        source_times_s=width_data.timestamps_s,
        source_values=width_data.sync_error_m,
    )

    valid_mask = width_valid & left_valid & right_valid & sync_valid
    reasons = ["ok" if is_valid else "width_time_out_of_range" for is_valid in valid_mask]

    return WidthAlignmentResult(
        width_m=width_m,
        d_left_m=d_left_m,
        d_right_m=d_right_m,
        sync_error_m=sync_error_m,
        valid_mask=valid_mask,
        reasons=reasons,
    )


# ============================================================
# 4. UMI 风格相对动作生成
# ============================================================


def transform_to_action6d(relative_T: np.ndarray) -> np.ndarray:
    """
    将相对 SE(3) 变换转换成 6D 动作向量。

    输出格式：
        [dx, dy, dz, rx, ry, rz]

    其中：
        dx/dy/dz 是相对平移，单位是米；
        rx/ry/rz 是旋转向量，单位是弧度。
    """

    translation = relative_T[:3, 3]
    rotation_vector = rotation_matrix_to_rotvec(relative_T[:3, :3])
    return np.concatenate([translation, rotation_vector], axis=0)


def make_relative_actions(
    T_W_G: np.ndarray,
    action_start_timestamps_source: np.ndarray,
    action_horizon: int,
    valid_mask: np.ndarray | None = None,
) -> RelativeActionResult:
    """
    生成 UMI 风格的相对轨迹动作。

    对每个起点 t，生成未来 k=1..H 的相对 TCP 运动：
        a_{t,k} = inverse(T_W_G(t)) @ T_W_G(t+k)

    输出形状：
        actions_relative_4x4: [N-H, H, 4, 4]
        actions_relative_6d:  [N-H, H, 6]
        action_start_timestamps: [N-H]
        actions_valid_mask: [N-H]

    action_start_timestamps 用于记录每个动作 chunk 对应的起点时间；
    后续和图像、夹爪宽度做时间同步时会更方便。
    """

    validate_transform_sequence("T_W_G", T_W_G)

    if action_horizon <= 0:
        raise ValueError(f"action_horizon 必须大于 0，当前是：{action_horizon}")

    if len(T_W_G) <= action_horizon:
        raise ValueError(
            f"帧数 {len(T_W_G)} 必须大于 horizon {action_horizon}，否则无法生成相对动作。"
        )

    if len(action_start_timestamps_source) != len(T_W_G):
        raise ValueError(
            "action_start_timestamps_source 与 T_W_G 长度不一致："
            f"{len(action_start_timestamps_source)} vs {len(T_W_G)}"
        )

    if valid_mask is None:
        valid_mask = np.ones(len(T_W_G), dtype=bool)
    if len(valid_mask) != len(T_W_G):
        raise ValueError(
            "valid_mask 与 T_W_G 长度不一致："
            f"{len(valid_mask)} vs {len(T_W_G)}"
        )

    sample_count = len(T_W_G) - action_horizon
    actions_relative_4x4 = np.full(
        (sample_count, action_horizon, 4, 4),
        np.nan,
        dtype=np.float64,
    )
    actions_relative_6d = np.full(
        (sample_count, action_horizon, 6),
        np.nan,
        dtype=np.float64,
    )
    actions_valid_mask = np.zeros(sample_count, dtype=bool)

    for start_index in range(sample_count):
        chunk_valid = bool(np.all(valid_mask[start_index:start_index + action_horizon + 1]))
        if not chunk_valid:
            continue

        T_now_inv = np.linalg.inv(T_W_G[start_index])

        for horizon_index in range(action_horizon):
            future_index = start_index + horizon_index + 1
            relative_T = T_now_inv @ T_W_G[future_index]

            actions_relative_4x4[start_index, horizon_index] = relative_T
            actions_relative_6d[start_index, horizon_index] = transform_to_action6d(relative_T)

        actions_valid_mask[start_index] = True

    action_start_timestamps = action_start_timestamps_source[:sample_count].copy()

    return RelativeActionResult(
        actions_relative_4x4=actions_relative_4x4,
        actions_relative_6d=actions_relative_6d,
        action_start_timestamps=action_start_timestamps,
        actions_valid_mask=actions_valid_mask,
    )


# ============================================================
# 5. 数据校验与调试图
# ============================================================


def validate_single_transform(name: str, transform: np.ndarray) -> None:
    """检查单个 4x4 齐次变换矩阵的形状是否正确。"""

    if transform.shape != (4, 4):
        raise ValueError(f"{name} 形状必须是 (4, 4)，当前是：{transform.shape}")


def validate_transform_sequence(name: str, transforms: np.ndarray) -> None:
    """检查一组 4x4 齐次变换矩阵的形状是否正确。"""

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"{name} 形状必须是 [N, 4, 4]，当前是：{transforms.shape}")


def calculate_timestamp_statistics(timestamps: np.ndarray) -> dict[str, float]:
    """
    计算时间戳统计信息，便于判断采样频率是否正常。

    Stray Scanner 的 timestamp 通常是秒级时间；
    若相邻帧间隔平均约 0.0167 秒，说明大约是 60 FPS。
    """

    if len(timestamps) < 2:
        return {
            "mean_dt_seconds": float("nan"),
            "median_dt_seconds": float("nan"),
            "min_dt_seconds": float("nan"),
            "max_dt_seconds": float("nan"),
            "estimated_fps": float("nan"),
        }

    dt = np.diff(timestamps)
    mean_dt = float(np.mean(dt))

    if mean_dt > 0:
        estimated_fps = 1.0 / mean_dt
    else:
        estimated_fps = float("nan")

    return {
        "mean_dt_seconds": mean_dt,
        "median_dt_seconds": float(np.median(dt)),
        "min_dt_seconds": float(np.min(dt)),
        "max_dt_seconds": float(np.max(dt)),
        "estimated_fps": float(estimated_fps),
    }


def calculate_first_step_action_statistics(actions_relative_6d: np.ndarray) -> dict[str, float]:
    """
    计算 horizon 第一步动作的统计量。

    这些数值主要用于快速发现明显异常：
        - 单步平移突然达到几十厘米，通常说明 pose 跳变或解析错误。
        - 单步旋转突然很大，可能说明四元数顺序或坐标系理解有问题。
    """

    first_step_actions = actions_relative_6d[:, 0, :]
    finite_mask = np.all(np.isfinite(first_step_actions), axis=1)
    if not finite_mask.any():
        return {
            "first_step_translation_mean_m": float("nan"),
            "first_step_translation_max_m": float("nan"),
            "first_step_rotation_mean_rad": float("nan"),
            "first_step_rotation_max_rad": float("nan"),
        }

    first_step_actions = first_step_actions[finite_mask]
    translation_norm = np.linalg.norm(first_step_actions[:, :3], axis=1)
    rotation_norm = np.linalg.norm(first_step_actions[:, 3:6], axis=1)

    return {
        "first_step_translation_mean_m": float(np.mean(translation_norm)),
        "first_step_translation_max_m": float(np.max(translation_norm)),
        "first_step_rotation_mean_rad": float(np.mean(rotation_norm)),
        "first_step_rotation_max_rad": float(np.max(rotation_norm)),
    }


def save_debug_plots(
    T_W_I: np.ndarray,
    actions_relative_6d: np.ndarray,
    d435i_timestamps_s: np.ndarray,
    gripper_width_m: np.ndarray,
    valid_mask: np.ndarray,
    test_dir: Path,
) -> bool:
    """
    保存调试检查图到 processed_umi/test。

    这些图片不进入训练数据，只用于人工确认：
        1. iPhone 轨迹是否平滑。
        2. 相对动作是否存在离谱跳变。

    返回：
        True  = 调试图保存成功。
        False = 当前 Python 环境缺少 matplotlib，已跳过画图。
    """

    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warning_text = (
            "当前 Python 环境缺少 matplotlib，所以本次没有生成调试图片。\n"
            "核心 npy 文件已经可以正常生成。\n"
            "如需调试图片，请安装：python -m pip install matplotlib\n"
        )
        (test_dir / "plot_warning.txt").write_text(warning_text, encoding="utf-8")
        print("\n警告：缺少 matplotlib，已跳过调试图生成。")
        print(f"提示文件：{test_dir / 'plot_warning.txt'}")
        return False

    valid_indices = np.where(valid_mask)[0]
    iphone_positions = T_W_I[:, :3, 3]
    first_step_actions = actions_relative_6d[:, 0, :]
    action_finite_mask = np.all(np.isfinite(first_step_actions), axis=1)
    translation_norm = np.full(len(first_step_actions), np.nan, dtype=np.float64)
    rotation_norm = np.full(len(first_step_actions), np.nan, dtype=np.float64)
    translation_norm[action_finite_mask] = np.linalg.norm(first_step_actions[action_finite_mask, :3], axis=1)
    rotation_norm[action_finite_mask] = np.linalg.norm(first_step_actions[action_finite_mask, 3:6], axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(iphone_positions[:, 0], label="x")
    plt.plot(iphone_positions[:, 1], label="y")
    plt.plot(iphone_positions[:, 2], label="z")
    plt.xlabel("D435i frame")
    plt.ylabel("position / m")
    plt.title("Aligned iPhone position on D435i timeline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(test_dir / "aligned_iphone_position_xyz.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    if len(valid_indices) > 0:
        plt.plot(iphone_positions[valid_indices, 0], iphone_positions[valid_indices, 2])
    else:
        plt.plot(iphone_positions[:, 0], iphone_positions[:, 2])
    plt.xlabel("x / m")
    plt.ylabel("z / m")
    plt.title("Aligned iPhone trajectory: x-z")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(test_dir / "aligned_iphone_traj_xz.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    relative_time = d435i_timestamps_s - d435i_timestamps_s[0]
    plt.plot(relative_time, gripper_width_m)
    plt.xlabel("D435i time from start / s")
    plt.ylabel("width / m")
    plt.title("Aligned gripper width on D435i timeline")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(test_dir / "aligned_width.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(valid_mask.astype(np.int32))
    plt.xlabel("D435i frame")
    plt.ylabel("valid")
    plt.title("Aligned frame valid mask")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(test_dir / "valid_mask.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(translation_norm)
    plt.xlabel("sample")
    plt.ylabel("translation norm / m")
    plt.title("Relative translation norm, first horizon step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(test_dir / "relative_translation_norm.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rotation_norm)
    plt.xlabel("sample")
    plt.ylabel("rotation norm / rad")
    plt.title("Relative rotation norm, first horizon step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(test_dir / "relative_rotation_norm.png", dpi=200)
    plt.close()

    return True


# ============================================================
# 6. 保存输出与 summary
# ============================================================


def save_numpy_outputs(
    output_dir: Path,
    d435i_frame_data: D435iFrameData,
    pose_alignment: PoseAlignmentResult,
    T_I_G: np.ndarray,
    T_W_G: np.ndarray,
    width_alignment: WidthAlignmentResult,
    valid_mask: np.ndarray,
    relative_action_result: RelativeActionResult,
) -> None:
    """保存后续 UMI 数据集构建会用到的核心 npy 文件。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # D435i is the master timeline for training observations.
    np.save(output_dir / "d435i_frame_indices.npy", d435i_frame_data.frame_indices)
    np.save(output_dir / "d435i_timestamps.npy", d435i_frame_data.timestamps_s)

    # iphone_timestamps.npy is now the aligned iPhone query time for each D435i frame.
    np.save(output_dir / "iphone_timestamps.npy", pose_alignment.iphone_query_timestamps)
    np.save(output_dir / "T_W_I.npy", pose_alignment.T_W_I)
    np.save(output_dir / "T_I_G.npy", T_I_G)
    np.save(output_dir / "T_W_G.npy", T_W_G)
    np.save(output_dir / "gripper_width_aligned.npy", width_alignment.width_m)
    np.save(output_dir / "gripper_d_left_aligned.npy", width_alignment.d_left_m)
    np.save(output_dir / "gripper_d_right_aligned.npy", width_alignment.d_right_m)
    np.save(output_dir / "gripper_sync_error_aligned.npy", width_alignment.sync_error_m)
    np.save(output_dir / "valid_mask.npy", valid_mask)
    np.save(output_dir / "actions_relative_4x4.npy", relative_action_result.actions_relative_4x4)
    np.save(output_dir / "actions_relative_6d.npy", relative_action_result.actions_relative_6d)
    np.save(output_dir / "action_start_timestamps.npy", relative_action_result.action_start_timestamps)
    np.save(output_dir / "actions_valid_mask.npy", relative_action_result.actions_valid_mask)


def calculate_finite_transform_mask(transforms: np.ndarray) -> np.ndarray:
    """Return True for frames whose 4x4 transform contains only finite values."""

    validate_transform_sequence("transforms", transforms)
    return np.all(np.isfinite(transforms), axis=(1, 2))


def combine_alignment_reasons(
    pose_alignment: PoseAlignmentResult,
    width_alignment: WidthAlignmentResult,
    finite_transform_mask: np.ndarray,
) -> list[str]:
    """
    Merge pose, width, and transform validity into one readable reason per frame.
    The resulting strings are written to frame_map.csv for quick manual debugging.
    """

    frame_reasons: list[str] = []
    for frame_index in range(len(finite_transform_mask)):
        reasons = []

        if not pose_alignment.valid_mask[frame_index]:
            reasons.append(pose_alignment.reasons[frame_index])
        if not width_alignment.valid_mask[frame_index]:
            reasons.append(width_alignment.reasons[frame_index])
        if not finite_transform_mask[frame_index]:
            reasons.append("non_finite_transform")

        frame_reasons.append("ok" if not reasons else ";".join(reasons))

    return frame_reasons


def save_frame_map_csv(
    output_dir: Path,
    d435i_frame_data: D435iFrameData,
    pose_alignment: PoseAlignmentResult,
    width_alignment: WidthAlignmentResult,
    valid_mask: np.ndarray,
    frame_reasons: list[str],
) -> Path:
    """
    Save the per-frame mapping between D435i time, aligned iPhone pose, and width.
    This CSV is mainly for inspection; training code should prefer the npy files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_map_path = output_dir / "frame_map.csv"

    with frame_map_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "d435i_frame_idx",
                "d435i_time_s",
                "iphone_query_time_s",
                "iphone_idx_before",
                "iphone_idx_after",
                "iphone_alpha",
                "width_m",
                "d_left_m",
                "d_right_m",
                "sync_error_m",
                "valid",
                "reason",
            ]
        )

        for frame_index in range(len(d435i_frame_data.timestamps_s)):
            writer.writerow(
                [
                    int(d435i_frame_data.frame_indices[frame_index]),
                    f"{d435i_frame_data.timestamps_s[frame_index]:.9f}",
                    f"{pose_alignment.iphone_query_timestamps[frame_index]:.9f}",
                    int(pose_alignment.iphone_idx_before[frame_index]),
                    int(pose_alignment.iphone_idx_after[frame_index]),
                    f"{pose_alignment.alpha[frame_index]:.9f}",
                    f"{width_alignment.width_m[frame_index]:.9f}",
                    f"{width_alignment.d_left_m[frame_index]:.9f}",
                    f"{width_alignment.d_right_m[frame_index]:.9f}",
                    f"{width_alignment.sync_error_m[frame_index]:.9f}",
                    int(bool(valid_mask[frame_index])),
                    frame_reasons[frame_index],
                ]
            )

    return frame_map_path


def save_gripper_width_aligned_csv(
    output_dir: Path,
    d435i_frame_data: D435iFrameData,
    width_alignment: WidthAlignmentResult,
    valid_mask: np.ndarray,
) -> Path:
    """
    Save gripper width selected from the nearest real 15 Hz sample for each D435i frame.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    width_path = output_dir / "gripper_width_aligned.csv"

    with width_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "d435i_frame_idx",
                "d435i_time_s",
                "width_m",
                "d_left_m",
                "d_right_m",
                "sync_error_m",
                "valid",
            ]
        )

        for frame_index in range(len(d435i_frame_data.timestamps_s)):
            writer.writerow(
                [
                    int(d435i_frame_data.frame_indices[frame_index]),
                    f"{d435i_frame_data.timestamps_s[frame_index]:.9f}",
                    f"{width_alignment.width_m[frame_index]:.9f}",
                    f"{width_alignment.d_left_m[frame_index]:.9f}",
                    f"{width_alignment.d_right_m[frame_index]:.9f}",
                    f"{width_alignment.sync_error_m[frame_index]:.9f}",
                    int(bool(valid_mask[frame_index])),
                ]
            )

    return width_path


def format_dictionary_lines(items: dict[str, float], indent: str = "  ") -> list[str]:
    """将统计字典格式化成 summary 里的多行文本。"""

    return [f"{indent}{key}: {value}" for key, value in items.items()]


def legacy_write_process_summary_raw_iphone_timeline(
    paths: ProcessPaths,
    action_horizon: int,
    odometry_columns: Iterable[str],
    T_W_I: np.ndarray,
    iphone_timestamps: np.ndarray,
    T_I_G: np.ndarray,
    T_W_G: np.ndarray,
    relative_action_result: RelativeActionResult,
    debug_plots_saved: bool,
) -> None:
    """
    写入处理摘要，方便以后回看这次数据是怎么生成的。

    summary 不参与训练，但它对排查路径、帧数、horizon、动作统计非常有用。
    """

    timestamp_statistics = calculate_timestamp_statistics(iphone_timestamps)
    action_statistics = calculate_first_step_action_statistics(
        relative_action_result.actions_relative_6d
    )

    summary_lines = [
        "Stray Scanner -> UMI action processing summary",
        "",
        f"demo_dir: {paths.demo_dir}",
        f"odometry_csv_path: {paths.odometry_csv_path}",
        f"output_dir: {paths.output_dir}",
        f"test_dir: {paths.test_dir}",
        "",
        f"odometry_columns: {list(odometry_columns)}",
        f"action_horizon: {action_horizon}",
        "",
        f"T_W_I shape: {T_W_I.shape}",
        f"iphone_timestamps shape: {iphone_timestamps.shape}",
        f"T_I_G shape: {T_I_G.shape}",
        f"T_W_G shape: {T_W_G.shape}",
        f"actions_relative_4x4 shape: {relative_action_result.actions_relative_4x4.shape}",
        f"actions_relative_6d shape: {relative_action_result.actions_relative_6d.shape}",
        f"action_start_timestamps shape: {relative_action_result.action_start_timestamps.shape}",
        f"debug_plots_saved: {debug_plots_saved}",
        "",
        "timestamp_statistics:",
        *format_dictionary_lines(timestamp_statistics),
        "",
        "first_step_action_statistics:",
        *format_dictionary_lines(action_statistics),
        "",
        "notes:",
        "  T_I_G uses the calibrated iPhone-to-gripper/TCP transform.",
        "  Debug plots are saved in processed_umi/test and are not required for training.",
    ]

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "process_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 7. 命令行参数与主流程
# ============================================================


def write_process_summary(
    paths: ProcessPaths,
    d435i_timestamps_csv_path: Path,
    width_csv_path: Path,
    sync_info: dict,
    action_horizon: int,
    max_iphone_gap_seconds: float,
    odometry_columns: Iterable[str],
    d435i_frame_data: D435iFrameData,
    pose_alignment: PoseAlignmentResult,
    T_I_G: np.ndarray,
    T_W_G: np.ndarray,
    width_alignment: WidthAlignmentResult,
    valid_mask: np.ndarray,
    frame_reasons: list[str],
    relative_action_result: RelativeActionResult,
    debug_plots_saved: bool,
) -> None:
    """
    Write a readable summary for the aligned outputs.
    This definition intentionally replaces the older raw-iPhone-timeline summary above.
    """

    timestamp_statistics = calculate_timestamp_statistics(d435i_frame_data.timestamps_s)
    action_statistics = calculate_first_step_action_statistics(
        relative_action_result.actions_relative_6d
    )
    frame_reason_counts: dict[str, int] = {}
    for reason in frame_reasons:
        frame_reason_counts[reason] = frame_reason_counts.get(reason, 0) + 1

    sync_offset = float(sync_info["offset_iphone_minus_d435i"])
    valid_frame_count = int(np.count_nonzero(valid_mask))
    valid_action_count = int(np.count_nonzero(relative_action_result.actions_valid_mask))

    summary_lines = [
        "Stray Scanner + D435i + gripper width -> aligned UMI processing summary",
        "",
        f"demo_dir: {paths.demo_dir}",
        f"odometry_csv_path: {paths.odometry_csv_path}",
        f"d435i_timestamps_csv_path: {d435i_timestamps_csv_path}",
        f"width_csv_path: {width_csv_path}",
        f"output_dir: {paths.output_dir}",
        f"test_dir: {paths.test_dir}",
        "",
        "time_sync:",
        f"  method: {sync_info.get('method')}",
        "  mapping: t_iphone = t_d435i + offset_iphone_minus_d435i",
        f"  offset_iphone_minus_d435i: {sync_offset:.9f}",
        f"  shake_search_seconds: {sync_info.get('shake_search_seconds')}",
        f"  event_threshold_ratio: {sync_info.get('event_threshold_ratio')}",
        f"  merge_event_gap_seconds: {sync_info.get('merge_event_gap_seconds')}",
        f"  min_event_duration_seconds: {sync_info.get('min_event_duration_seconds')}",
        f"  d435i_imu_csv: {sync_info.get('d435i_imu_csv')}",
        f"  iphone_imu_csv: {sync_info.get('iphone_imu_csv')}",
        "",
        f"odometry_columns: {list(odometry_columns)}",
        f"action_horizon: {action_horizon}",
        f"max_iphone_gap_seconds: {max_iphone_gap_seconds}",
        "",
        f"d435i_frame_count: {len(d435i_frame_data.timestamps_s)}",
        f"valid_frame_count: {valid_frame_count}",
        f"invalid_frame_count: {len(valid_mask) - valid_frame_count}",
        f"valid_action_count: {valid_action_count}",
        f"invalid_action_count: {len(relative_action_result.actions_valid_mask) - valid_action_count}",
        "",
        f"T_W_I shape: {pose_alignment.T_W_I.shape}",
        f"iphone_timestamps shape: {pose_alignment.iphone_query_timestamps.shape}",
        f"T_I_G shape: {T_I_G.shape}",
        f"T_W_G shape: {T_W_G.shape}",
        f"gripper_width_aligned shape: {width_alignment.width_m.shape}",
        f"actions_relative_4x4 shape: {relative_action_result.actions_relative_4x4.shape}",
        f"actions_relative_6d shape: {relative_action_result.actions_relative_6d.shape}",
        f"action_start_timestamps shape: {relative_action_result.action_start_timestamps.shape}",
        f"actions_valid_mask shape: {relative_action_result.actions_valid_mask.shape}",
        f"debug_plots_saved: {debug_plots_saved}",
        "",
        "d435i_timestamp_statistics:",
        *format_dictionary_lines(timestamp_statistics),
        "",
        "first_step_action_statistics:",
        *format_dictionary_lines(action_statistics),
        "",
        "frame_reason_counts:",
        *format_dictionary_lines({key: float(value) for key, value in frame_reason_counts.items()}),
        "",
        "notes:",
        "  D435i frame time is the master timeline.",
        "  T_W_I and T_W_G are already aligned to D435i frames.",
        "  gripper_width_aligned.npy/csv uses the nearest real 15 Hz sample for each D435i frame.",
        "  valid_mask.npy marks frames where pose, width, and transform are all usable.",
        "  actions_valid_mask.npy marks action chunks whose whole future horizon is usable.",
        "  frame_map.csv is the easiest place to inspect per-frame alignment decisions.",
    ]

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "process_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。默认参数已经匹配当前项目路径。"""

    parser = argparse.ArgumentParser(
        description="Convert latest Stray Scanner odometry.csv to UMI-style relative actions."
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
        help="指定某个时间戳 session 文件夹；不填时自动处理 stray-root 里最新且包含 odometry.csv 的文件夹。",
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
        "--horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help=f"相对动作 horizon，默认：{DEFAULT_ACTION_HORIZON}",
    )
    parser.add_argument(
        "--shake-search-seconds",
        type=float,
        default=8.0,
        help="Search only this many seconds from the beginning of both IMU logs for the single shake peak.",
    )
    parser.add_argument(
        "--max-iphone-gap-seconds",
        type=float,
        default=0.25,
        help="Maximum allowed odometry gap when interpolating iPhone pose to a D435i frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标文件夹里已经有 processed_umi，允许覆盖重新生成；默认会停止。",
    )

    return parser.parse_args()


def legacy_main_raw_iphone_timeline() -> int:
    """完整处理流程入口。"""

    args = parse_args()

    try:
        paths = build_process_paths(
            stray_root=args.stray_root,
            explicit_demo_dir=args.demo_dir,
            output_dir_name=args.output_dir_name,
            test_dir_name=args.test_dir_name,
            overwrite_existing_output=args.overwrite,
        )
    except FileExistsError as exc:
        print(str(exc))
        print("本次没有写入任何新文件。")
        return 0

    print("正在处理 Stray Scanner 数据：")
    print(f"  数据文件夹：{paths.demo_dir}")
    print(f"  odometry.csv：{paths.odometry_csv_path}")
    print(f"  输出目录：{paths.output_dir}")
    print(f"  调试目录：{paths.test_dir}")
    print(f"  action horizon：{args.horizon}")

    odometry_data = load_odometry_csv(paths.odometry_csv_path)
    T_W_I, iphone_timestamps = build_iphone_pose_sequence(odometry_data)
    T_I_G = get_calibrated_T_I_G()
    T_W_G = compute_tcp_pose_sequence(T_W_I, T_I_G)

    relative_action_result = make_relative_actions(
        T_W_G=T_W_G,
        iphone_timestamps=iphone_timestamps,
        action_horizon=args.horizon,
    )

    save_numpy_outputs(
        output_dir=paths.output_dir,
        T_W_I=T_W_I,
        iphone_timestamps=iphone_timestamps,
        T_I_G=T_I_G,
        T_W_G=T_W_G,
        relative_action_result=relative_action_result,
    )

    debug_plots_saved = save_debug_plots(
        T_W_I=T_W_I,
        actions_relative_6d=relative_action_result.actions_relative_6d,
        test_dir=paths.test_dir,
    )

    write_process_summary(
        paths=paths,
        action_horizon=args.horizon,
        odometry_columns=odometry_data.columns,
        T_W_I=T_W_I,
        iphone_timestamps=iphone_timestamps,
        T_I_G=T_I_G,
        T_W_G=T_W_G,
        relative_action_result=relative_action_result,
        debug_plots_saved=debug_plots_saved,
    )

    action_statistics = calculate_first_step_action_statistics(
        relative_action_result.actions_relative_6d
    )

    print("\n处理完成。核心输出：")
    print(f"  {paths.output_dir / 'T_W_I.npy'}                 {T_W_I.shape}")
    print(f"  {paths.output_dir / 'iphone_timestamps.npy'}     {iphone_timestamps.shape}")
    print(f"  {paths.output_dir / 'T_I_G.npy'}                 {T_I_G.shape}")
    print(f"  {paths.output_dir / 'T_W_G.npy'}                 {T_W_G.shape}")
    print(
        "  "
        f"{paths.output_dir / 'actions_relative_4x4.npy'} "
        f"{relative_action_result.actions_relative_4x4.shape}"
    )
    print(
        "  "
        f"{paths.output_dir / 'actions_relative_6d.npy'}  "
        f"{relative_action_result.actions_relative_6d.shape}"
    )
    print(
        "  "
        f"{paths.output_dir / 'action_start_timestamps.npy'} "
        f"{relative_action_result.action_start_timestamps.shape}"
    )
    print(f"  {paths.output_dir / 'process_summary.txt'}")

    if debug_plots_saved:
        print("\n调试图已保存到：")
        print(f"  {paths.test_dir}")
    else:
        print("\n调试图未生成，但核心 npy 文件已保存。")
        print(f"  原因见：{paths.test_dir / 'plot_warning.txt'}")

    print("\n第一步相对动作快速检查：")
    print(
        "  平移 mean/max = "
        f"{action_statistics['first_step_translation_mean_m']:.6f} / "
        f"{action_statistics['first_step_translation_max_m']:.6f} m"
    )
    print(
        "  旋转 mean/max = "
        f"{action_statistics['first_step_rotation_mean_rad']:.6f} / "
        f"{action_statistics['first_step_rotation_max_rad']:.6f} rad"
    )

    return 0


def main() -> int:
    """
    New aligned processing entry point.

    Pipeline:
        1. Find the session and input CSV files.
        2. Estimate iPhone-D435i clock offset from the single shake.
        3. Use D435i frames as the master timeline.
        4. Interpolate iPhone pose and gripper width to that timeline.
        5. Save only aligned outputs for downstream UMI processing.
    """

    args = parse_args()

    try:
        paths = build_process_paths(
            stray_root=args.stray_root,
            explicit_demo_dir=args.demo_dir,
            output_dir_name=args.output_dir_name,
            test_dir_name=args.test_dir_name,
            overwrite_existing_output=args.overwrite,
        )
    except FileExistsError as exc:
        print(str(exc))
        print("This run did not write new files.")
        return 0

    if args.overwrite and paths.output_dir.exists():
        # --overwrite is explicit, but still verify that only the session output folder is deleted.
        resolved_demo_dir = paths.demo_dir.resolve()
        resolved_output_dir = paths.output_dir.resolve()
        if resolved_demo_dir not in resolved_output_dir.parents:
            raise RuntimeError(f"Refusing to remove output outside demo dir: {resolved_output_dir}")
        shutil.rmtree(paths.output_dir)

    d435i_timestamps_csv_path = find_d435i_timestamps_csv(paths.demo_dir)
    width_csv_path = find_gripper_width_csv(
        demo_dir=paths.demo_dir,
        output_dir_name=args.output_dir_name,
    )

    print("Processing aligned UMI data on the D435i timeline:")
    print(f"  demo_dir: {paths.demo_dir}")
    print(f"  odometry.csv: {paths.odometry_csv_path}")
    print(f"  d435i timestamps: {d435i_timestamps_csv_path}")
    print(f"  gripper width csv: {width_csv_path}")
    print(f"  output_dir: {paths.output_dir}")
    print(f"  shake_search_seconds: {args.shake_search_seconds}")
    print(f"  action_horizon: {args.horizon}")

    sync_info = estimate_sync_for_demo(
        demo_dir=paths.demo_dir,
        odometry_csv_path=paths.odometry_csv_path,
        output_dir=paths.output_dir,
        test_dir=paths.test_dir,
        shake_search_seconds=args.shake_search_seconds,
    )
    sync_offset = float(sync_info["offset_iphone_minus_d435i"])

    d435i_frame_data = load_d435i_timestamps_csv(d435i_timestamps_csv_path)
    odometry_data = load_odometry_csv(paths.odometry_csv_path)
    width_data = load_gripper_width_csv(width_csv_path)

    pose_alignment = interpolate_iphone_pose_to_d435i_frames(
        odometry_data=odometry_data,
        d435i_timestamps_s=d435i_frame_data.timestamps_s,
        sync_offset_iphone_minus_d435i=sync_offset,
        max_iphone_gap_seconds=args.max_iphone_gap_seconds,
    )
    width_alignment = align_gripper_width_to_d435i_frames(
        width_data=width_data,
        d435i_timestamps_s=d435i_frame_data.timestamps_s,
    )

    T_I_G = get_calibrated_T_I_G()
    T_W_G = compute_tcp_pose_sequence(pose_alignment.T_W_I, T_I_G)

    finite_transform_mask = calculate_finite_transform_mask(T_W_G)
    valid_mask = pose_alignment.valid_mask & width_alignment.valid_mask & finite_transform_mask
    frame_reasons = combine_alignment_reasons(
        pose_alignment=pose_alignment,
        width_alignment=width_alignment,
        finite_transform_mask=finite_transform_mask,
    )

    relative_action_result = make_relative_actions(
        T_W_G=T_W_G,
        action_start_timestamps_source=d435i_frame_data.timestamps_s,
        action_horizon=args.horizon,
        valid_mask=valid_mask,
    )

    save_numpy_outputs(
        output_dir=paths.output_dir,
        d435i_frame_data=d435i_frame_data,
        pose_alignment=pose_alignment,
        T_I_G=T_I_G,
        T_W_G=T_W_G,
        width_alignment=width_alignment,
        valid_mask=valid_mask,
        relative_action_result=relative_action_result,
    )
    frame_map_path = save_frame_map_csv(
        output_dir=paths.output_dir,
        d435i_frame_data=d435i_frame_data,
        pose_alignment=pose_alignment,
        width_alignment=width_alignment,
        valid_mask=valid_mask,
        frame_reasons=frame_reasons,
    )
    width_aligned_csv_path = save_gripper_width_aligned_csv(
        output_dir=paths.output_dir,
        d435i_frame_data=d435i_frame_data,
        width_alignment=width_alignment,
        valid_mask=valid_mask,
    )

    debug_plots_saved = save_debug_plots(
        T_W_I=pose_alignment.T_W_I,
        actions_relative_6d=relative_action_result.actions_relative_6d,
        d435i_timestamps_s=d435i_frame_data.timestamps_s,
        gripper_width_m=width_alignment.width_m,
        valid_mask=valid_mask,
        test_dir=paths.test_dir,
    )

    write_process_summary(
        paths=paths,
        d435i_timestamps_csv_path=d435i_timestamps_csv_path,
        width_csv_path=width_csv_path,
        sync_info=sync_info,
        action_horizon=args.horizon,
        max_iphone_gap_seconds=args.max_iphone_gap_seconds,
        odometry_columns=odometry_data.columns,
        d435i_frame_data=d435i_frame_data,
        pose_alignment=pose_alignment,
        T_I_G=T_I_G,
        T_W_G=T_W_G,
        width_alignment=width_alignment,
        valid_mask=valid_mask,
        frame_reasons=frame_reasons,
        relative_action_result=relative_action_result,
        debug_plots_saved=debug_plots_saved,
    )

    action_statistics = calculate_first_step_action_statistics(
        relative_action_result.actions_relative_6d
    )
    valid_frame_count = int(np.count_nonzero(valid_mask))
    valid_action_count = int(np.count_nonzero(relative_action_result.actions_valid_mask))

    print("\nAligned processing complete.")
    print(f"  sync offset iphone-minus-d435i: {sync_offset:.9f} s")
    print(f"  valid frames: {valid_frame_count} / {len(valid_mask)}")
    print(f"  valid action chunks: {valid_action_count} / {len(relative_action_result.actions_valid_mask)}")
    print(f"  T_W_I.npy: {pose_alignment.T_W_I.shape}")
    print(f"  T_W_G.npy: {T_W_G.shape}")
    print(f"  gripper_width_aligned.npy: {width_alignment.width_m.shape}")
    print(f"  actions_relative_6d.npy: {relative_action_result.actions_relative_6d.shape}")
    print(f"  frame_map.csv: {frame_map_path}")
    print(f"  gripper_width_aligned.csv: {width_aligned_csv_path}")
    print(f"  process_summary.txt: {paths.output_dir / 'process_summary.txt'}")

    if debug_plots_saved:
        print(f"  debug plots: {paths.test_dir}")
    else:
        print(f"  debug plots skipped: {paths.test_dir / 'plot_warning.txt'}")

    print(
        "  first-step translation mean/max: "
        f"{action_statistics['first_step_translation_mean_m']:.6f} / "
        f"{action_statistics['first_step_translation_max_m']:.6f} m"
    )
    print(
        "  first-step rotation mean/max: "
        f"{action_statistics['first_step_rotation_mean_rad']:.6f} / "
        f"{action_statistics['first_step_rotation_max_rad']:.6f} rad"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
