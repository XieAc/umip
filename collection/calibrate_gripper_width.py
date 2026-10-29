"""Create a measured tick-to-opening calibration for the XAC gripper."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from xac_config import WIDTH_CALIBRATION_PATH, DXL_BAUDRATE, DXL_IDS, DXL_PORT

ADDR_PRESENT_POSITION = 132
ADDR_TORQUE_ENABLE = 64


def validate_samples(samples: list[dict]) -> None:
    if len(samples) < 2:
        raise ValueError("至少需要两个不同宽度的标定点")
    # The two hinged motors move in opposite directions.  Their mean can
    # therefore stay nearly constant (and is not a valid opening coordinate).
    try:
        paired_ticks = [float(row["m1"]) - float(row["m2"]) for row in samples]
        widths = [float(row["width_m"]) for row in samples]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("每个标定点必须包含 m1、m2、width_m 数值") from exc
    if not all(math.isfinite(value) for value in paired_ticks):
        raise ValueError("电机 tick 必须是有限数值")
    if not all(math.isfinite(value) and value >= 0 for value in widths):
        raise ValueError("开口宽度必须是有限的非负数")
    order = sorted(range(len(samples)), key=paired_ticks.__getitem__)
    sorted_ticks = [paired_ticks[index] for index in order]
    sorted_widths = [widths[index] for index in order]
    if any(right <= left for left, right in zip(sorted_ticks, sorted_ticks[1:])):
        raise ValueError("存在重复的平均电机 tick，请重新采集该点")
    differences = [right - left for left, right in zip(sorted_widths, sorted_widths[1:])]
    if differences and not (all(value > 0 for value in differences) or all(value < 0 for value in differences)):
        raise ValueError("双电机差值 tick 与实测开口不是严格单调关系，请检查输入或机械回差")


def save_calibration(path: Path, samples: list[dict]) -> None:
    validate_samples(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_{stamp}.bak.json")
        shutil.copy2(path, backup)
        print(f"旧标定已备份：{backup}")
    payload = {
        "schema_version": 1,
        "model": "paired_difference_tick",
        "width_definition": "fingertip_distance_m",
        "created_at": datetime.now().astimezone().isoformat(),
        "samples": samples,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

    port, packet = PortHandler(DXL_PORT), PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(DXL_BAUDRATE):
        raise RuntimeError(f"无法打开 {DXL_PORT} @ {DXL_BAUDRATE}")
    samples = []
    interrupted = False
    try:
        for motor in DXL_IDS:
            result, error = packet.write1ByteTxRx(port, motor, ADDR_TORQUE_ENABLE, 0)
            if result != COMM_SUCCESS or error:
                raise RuntimeError(f"关闭 ID={motor} 扭矩失败")
        print("电机扭矩已关闭。通过绑带手动调整夹爪到多个宽度。")
        print("输入卡尺测得的总开口(mm)，直接回车结束；至少需要两点。")
        while True:
            raw = input("开口宽度 mm: ").strip()
            if not raw:
                break
            try:
                width_mm = float(raw)
            except ValueError:
                print("请输入有限的数字（单位 mm），或直接回车结束。")
                continue
            if not math.isfinite(width_mm) or width_mm < 0:
                print("开口宽度必须是非负有限数值。")
                continue
            if width_mm > 300:
                confirmation = input(
                    f"警告：{width_mm:g} mm 对当前夹爪可能过大，确认单位确实是 mm？[y/N]: "
                ).strip().lower()
                if confirmation not in {"y", "yes"}:
                    print("已跳过该点，请重新测量并输入。")
                    continue
            positions = []
            for motor in DXL_IDS:
                value, result, error = packet.read4ByteTxRx(port, motor, ADDR_PRESENT_POSITION)
                if result != COMM_SUCCESS or error:
                    raise RuntimeError(f"读取 ID={motor} 失败")
                positions.append(value - 0x100000000 if value >= 0x80000000 else value)
            samples.append({"width_m": width_mm / 1000.0, "m1": positions[0], "m2": positions[1]})
            print(f"记录：m1={positions[0]}, m2={positions[1]}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n已中断输入，不会覆盖正式标定文件。")
    finally:
        for motor in DXL_IDS:
            packet.write1ByteTxRx(port, motor, ADDR_TORQUE_ENABLE, 0)
        port.closePort()
    path = WIDTH_CALIBRATION_PATH
    if interrupted:
        if samples:
            partial_path = path.with_name("gripper_width.partial.json")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps(
                    {"schema_version": 1, "model": "paired_difference_tick", "samples": samples},
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print(f"已保存未完成采样：{partial_path}；补采后再运行本程序写入正式文件。")
        return
    save_calibration(path, samples)
    print(f"标定已保存：{path}")


if __name__ == "__main__":
    main()
