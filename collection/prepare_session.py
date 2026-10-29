"""Copy and validate the two iPhone CSV exports into an XAC session."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def validate_csv(path: Path, required_columns: set[str]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {column.strip() for column in (reader.fieldnames or []) if column}
        missing = required_columns - columns
        if missing:
            raise ValueError(f"CSV 缺少列 {sorted(missing)}：{path}")
        if next(reader, None) is None:
            raise ValueError(f"CSV 没有数据行：{path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("odometry", type=Path)
    parser.add_argument("imu", type=Path)
    args = parser.parse_args()
    if not args.session.is_dir():
        raise FileNotFoundError(f"session 不存在：{args.session}")
    if not (args.session / "d435i" / "timestamps.csv").exists():
        raise ValueError(f"不是完整的 XAC session（缺少 d435i/timestamps.csv）：{args.session}")
    copies = (
        (args.odometry, "odometry.csv", {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}),
        (args.imu, "imu.csv", set()),
    )
    # Validate every source and target before copying either file. This avoids
    # leaving a half-prepared session when the second input is invalid.
    for source, name, required_columns in copies:
        validate_csv(source, required_columns)
        target = args.session / name
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有文件：{target}")
    for source, name, _ in copies:
        target = args.session / name
        shutil.copy2(source, target)
        print(f"已复制：{target}")
    print(f"session 已就绪。下一步：python check_pipeline.py --session \"{args.session}\"")


if __name__ == "__main__":
    main()
