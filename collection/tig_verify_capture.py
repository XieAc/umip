"""Create metadata-backed XAC sessions for TIG probe calibration.

XAC is a hand-operated collector whose motor torque stays disabled.  This
tool therefore never opens the serial port or commands the gripper; it only
creates a session and records the probe geometry used for that Stray take.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from _tig_solver_bridge import tig_root

DEFAULT_DIRECTION = {"z": -1, "x": 1, "y": 1}


def ask(prompt: str, default: str = "") -> str:
    value = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return value or default


def ask_float(prompt: str, default: float) -> float:
    value = float(ask(prompt, str(default)))
    if not (-1e6 < value < 1e6):
        raise ValueError("数值超出合理范围")
    return value


def new_session(axis: str) -> Path:
    root = tig_root()
    direction = int(ask("延伸方向符号（+1/-1）", str(DEFAULT_DIRECTION[axis])))
    if direction not in (-1, 1):
        raise ValueError("方向只能是 +1 或 -1")
    slide_mm = ask_float("相对第一组的累计滑移量 mm", 0.0)
    length_mm = ask_float("本组 TCP 到探针尖端的实测长度 mm", 30.0)
    meta = {
        "format": "tig_verify_capture_v1",
        "device": "collect_xac",
        "created_at": datetime.now().astimezone().isoformat(),
        "axis": axis,
        "slide_mm": slide_mm,
        "direction": direction,
        "toothpick_length_mm": length_mm,
        "note": ask("备注（可留空）"),
    }

    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("data_%m%d_%H%M%S")
    session = root / stem
    suffix = 1
    while session.exists():
        session = root / f"{stem}_{suffix:02d}"
        suffix += 1
    session.mkdir()
    (session / "tig_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n已创建：{session}")
    print("请用 Stray 录制同一固定点的 8 次多姿态接触，再把 odometry.csv 放入此目录。")
    return session


def list_sessions() -> None:
    root = tig_root()
    print(f"\n标定数据目录：{root}")
    for path in sorted(root.glob("data_*")) if root.exists() else []:
        meta_path = path / "tig_meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            odometry = any(p.name.lower() == "odometry.csv" for p in path.rglob("*"))
            print(
                f"  {path.name}: {meta['axis']}轴, {meta['slide_mm']} mm, "
                f"方向 {meta['direction']:+d}, odometry={'有' if odometry else '无'}"
            )
        except (OSError, ValueError, KeyError):
            print(f"  {path.name}: 元数据无效")


def main() -> int:
    print("XAC T_I_G 探针标定采集助手（不控制电机）")
    print(f"数据目录：{tig_root()}")
    while True:
        print("\n1 新建 Z 轴组  2 新建 X 轴组  3 新建 Y 轴组  4 列出组  0 退出")
        choice = ask("选择")
        if choice == "0":
            return 0
        if choice in {"1", "2", "3"}:
            new_session({"1": "z", "2": "x", "3": "y"}[choice])
        elif choice == "4":
            list_sessions()
        else:
            print("无效选择。")


if __name__ == "__main__":
    raise SystemExit(main())
