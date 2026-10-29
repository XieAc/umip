"""Read-only readiness and artifact checks for the XAC data pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibrate_tig import validate_transform
from xac_config import (
    CALIBRATION_DIR, WIDTH_CALIBRATION_PATH, TIG_CALIBRATION_PATH,
    DATA_ROOT, DATASET_ROOT, TIG_ROOT, load_width_calibration,
)


@dataclass
class Report:
    errors: list[str]
    warnings: list[str]
    passed: list[str]

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)


def csv_info(path: Path, required: set[str]) -> tuple[int, set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {field.strip() for field in (reader.fieldnames or []) if field}
        missing = required - fields
        if missing:
            raise ValueError(f"缺少列 {sorted(missing)}")
        rows = sum(1 for _ in reader)
    if rows == 0:
        raise ValueError("没有数据行")
    return rows, fields


def check_environment(report: Report) -> None:
    required = ("numpy", "scipy", "cv2", "zarr", "numcodecs")
    optional_hardware = ("pyrealsense2", "dynamixel_sdk")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        report.fail(f"缺少处理依赖：{', '.join(missing)}")
    else:
        report.ok("数据处理依赖完整")
    missing_hardware = [name for name in optional_hardware if importlib.util.find_spec(name) is None]
    if missing_hardware:
        report.warn(f"缺少硬件依赖（仅离线处理可用）：{', '.join(missing_hardware)}")
    else:
        report.ok("D435i 和 Dynamixel Python 依赖可导入")


def check_calibrations(report: Report) -> None:
    width_path = WIDTH_CALIBRATION_PATH
    try:
        calibration = load_width_calibration()
        report.ok(f"夹爪宽度标定有效：{width_path}（{len(calibration['samples'])} 点）")
        if len(calibration["samples"]) < 5:
            report.warn("夹爪宽度标定少于 5 点，只满足程序最低要求，不建议用于正式数据")
    except (OSError, ValueError, KeyError) as exc:
        report.fail(f"夹爪宽度标定无效：{exc}")

    tig_path = TIG_CALIBRATION_PATH
    try:
        matrix = np.asarray(np.load(tig_path, allow_pickle=False), dtype=np.float64)
        validate_transform(matrix)
        report.ok(f"T_I_G 几何检查通过：{tig_path}")
        metadata_path = tig_path.with_suffix(".json")
        if not metadata_path.exists():
            report.warn("T_I_G 缺少同名 JSON 来源记录")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            method = str(metadata.get("method", "unknown"))
            if method == "measured_mounting_pose":
                report.warn("当前 T_I_G 仅来自手工位置/欧拉角，尚无实验候选来源记录")
            else:
                report.ok(f"T_I_G 来源：{method}")
    except (OSError, ValueError) as exc:
        report.fail(f"T_I_G 无效：{exc}")


def check_session(path: Path, report: Report) -> None:
    session = path.expanduser().resolve()
    if not session.is_dir():
        report.fail(f"session 不存在：{session}")
        return
    checks = (
        (session / "d435i" / "timestamps.csv", {"frame_idx"}),
        (session / "d435i" / "imu.csv", set()),
        (session / "gripper_width_log.csv", {"computer_time_s", "width_m"}),
        (session / "odometry.csv", {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}),
        (session / "imu.csv", set()),
    )
    for file_path, columns in checks:
        try:
            rows, _ = csv_info(file_path, columns)
            report.ok(f"{file_path.relative_to(session)}：{rows} 行")
        except (OSError, ValueError) as exc:
            report.fail(f"{file_path.relative_to(session)} 无效：{exc}")

    for required in ("session.json", "d435i/meta.json", "d435i/intrinsics.json"):
        file_path = session / required
        try:
            json.loads(file_path.read_text(encoding="utf-8"))
            report.ok(f"{required} 可读取")
        except (OSError, ValueError) as exc:
            report.fail(f"{required} 无效：{exc}")

    events_path = session / "d435i" / "recording_events.json"
    if events_path.exists():
        try:
            events = json.loads(events_path.read_text(encoding="utf-8"))
            state = events.get("state")
            if state == "collecting_valid_data":
                report.ok("实时 shake 检测已进入有效数据阶段")
            else:
                report.warn(f"实时 shake 检测最终状态为 {state!r}")
        except (OSError, ValueError) as exc:
            report.warn(f"recording_events.json 无法读取：{exc}")
    else:
        report.warn("缺少 recording_events.json；旧 session 将使用离线 shake 检测")

    rgb_count = len(list((session / "d435i" / "rgb").glob("*.png")))
    depth_count = len(list((session / "d435i" / "depth").glob("*.png")))
    if rgb_count and rgb_count == depth_count:
        report.ok(f"RGB/Depth 文件数量一致：{rgb_count}")
    else:
        report.fail(f"RGB/Depth 文件数量异常：RGB={rgb_count}, Depth={depth_count}")


def check_episode(path: Path, report: Report) -> None:
    episode = path.expanduser().resolve()
    data = episode / "data"
    expected = {
        "camera0_rgb.npy": (4, (224, 224, 3)),
        "robot0_eef_pos.npy": (2, (3,)),
        "robot0_eef_rot_axis_angle.npy": (2, (3,)),
        "robot0_gripper_width.npy": (2, (1,)),
        "d435i_timestamps.npy": (1, ()),
        "valid_mask.npy": (1, ()),
    }
    lengths: dict[str, int] = {}
    for name, (ndim, tail) in expected.items():
        try:
            array = np.load(data / name, mmap_mode="r", allow_pickle=False)
            if array.ndim != ndim or tuple(array.shape[1:]) != tail:
                raise ValueError(f"shape={array.shape}, 期望 (*,{tail})")
            lengths[name] = len(array)
            report.ok(f"data/{name}：{array.shape}")
        except (OSError, ValueError) as exc:
            report.fail(f"data/{name} 无效：{exc}")
    if lengths and len(set(lengths.values())) != 1:
        report.fail(f"episode 主数组长度不一致：{lengths}")
    try:
        valid = np.load(data / "valid_mask.npy", allow_pickle=False).astype(bool)
        ratio = float(np.mean(valid)) if len(valid) else 0.0
        if ratio <= 0:
            report.fail("valid_mask 没有有效帧")
        elif ratio < 0.8:
            report.warn(f"有效帧比例偏低：{ratio:.1%}")
        else:
            report.ok(f"有效帧比例：{ratio:.1%}")
    except (OSError, ValueError):
        pass
    for required in ("metadata/source.json", "metadata/processing.json", "metadata/sync.json"):
        try:
            json.loads((episode / required).read_text(encoding="utf-8"))
            report.ok(f"{required} 可读取")
        except (OSError, ValueError) as exc:
            report.fail(f"{required} 无效：{exc}")


def print_report(report: Report) -> int:
    for message in report.passed:
        print(f"[PASS] {message}")
    for message in report.warnings:
        print(f"[WARN] {message}")
    for message in report.errors:
        print(f"[FAIL] {message}")
    print(
        f"\n汇总：PASS={len(report.passed)} WARN={len(report.warnings)} FAIL={len(report.errors)}"
    )
    return 1 if report.errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="检查一条已合并 iPhone 数据的原始 session")
    parser.add_argument("--episode", type=Path, help="检查一条 build_dataset.py 输出的 episode")
    parser.add_argument("--project", action="store_true", help="检查环境和两项正式标定")
    args = parser.parse_args()
    report = Report([], [], [])
    if args.project or (args.session is None and args.episode is None):
        print(f"CALIBRATION_DIR={CALIBRATION_DIR}\nDATA_ROOT={DATA_ROOT}\nDATASET_ROOT={DATASET_ROOT}\nTIG_ROOT={TIG_ROOT}\n")
        check_environment(report)
        check_calibrations(report)
    if args.session is not None:
        check_session(args.session, report)
    if args.episode is not None:
        check_episode(args.episode, report)
    return print_report(report)


if __name__ == "__main__":
    sys.exit(main())
