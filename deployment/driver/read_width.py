"""Display encoder angles and calibrated fingertip width (mm)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xac_config import (
    COUNTS_PER_REV,
    DXL_BAUDRATE,
    DXL_IDS,
    DXL_PORT,
    load_width_calibration,
    positions_to_width,
)

ADDR_PRESENT_POSITION = 132


def format_reading(position_1: float, position_2: float, calibration: dict) -> str:
    angle_1 = (position_1 % COUNTS_PER_REV) * 360.0 / COUNTS_PER_REV
    angle_2 = (position_2 % COUNTS_PER_REV) * 360.0 / COUNTS_PER_REV
    width_m, _, _ = positions_to_width(position_1, position_2, calibration)
    width_text = f"{width_m * 1000.0:.2f} mm" if math.isfinite(width_m) else "超出标定范围"
    return f"ID={DXL_IDS[0]}: {angle_1:.2f}°  ID={DXL_IDS[1]}: {angle_2:.2f}°  width={width_text}"


def monitor(port_name: str, baudrate: int, interval: float, once: bool, calibration: dict) -> None:
    try:
        from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
    except ImportError as exc:
        raise RuntimeError("请先安装 dynamixel-sdk：pip install dynamixel-sdk") from exc

    port = PortHandler(port_name)
    packet = PacketHandler(2.0)
    if not port.openPort():
        raise RuntimeError(f"无法打开串口：{port_name}")
    try:
        if not port.setBaudRate(baudrate):
            raise RuntimeError(f"无法设置波特率：{baudrate}")
        if not once:
            print(f"串口 {port_name} @ {baudrate}，实时显示角度及宽度；Ctrl+C 退出。")
        while True:
            positions = []
            for motor_id in DXL_IDS:
                value, result, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
                if result != COMM_SUCCESS:
                    raise RuntimeError(f"读取 ID={motor_id} 失败：{packet.getTxRxResult(result)}")
                if error:
                    raise RuntimeError(f"读取 ID={motor_id} 失败：{packet.getRxPacketError(error)}")
                positions.append(value - 0x100000000 if value >= 0x80000000 else value)
            # Use raw signed positions for conversion, wrapped angles only for display.
            print(format_reading(*positions, calibration), flush=True)
            if once:
                return
            time.sleep(interval)
    finally:
        port.closePort()


def main() -> int:
    parser = argparse.ArgumentParser(description="读取两台电机角度，根据 calibration/gripper_width.json 显示夹爪宽度。")
    parser.add_argument("--port", default=DXL_PORT, help="串口，如 /dev/ttyUSB0 或 COM6")
    parser.add_argument("--baudrate", type=int, default=DXL_BAUDRATE)
    parser.add_argument("--interval", type=float, default=0.1, help="刷新间隔（秒），默认 0.1")
    parser.add_argument("--once", action="store_true", help="读取一次后退出")
    parser.add_argument("--angles", type=float, nargs=2, metavar=("ANGLE1", "ANGLE2"), help="输入 ID 1、2 的角度（度），离线计算宽度")
    args = parser.parse_args()
    if args.baudrate <= 0:
        parser.error("波特率必须大于 0")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("刷新间隔必须是大于 0 的有限数值")
    if args.angles and any(not math.isfinite(angle) or not 0 <= angle <= 360 for angle in args.angles):
        parser.error("角度必须在 0 到 360 度之间")
    try:
        calibration = load_width_calibration()
        if args.angles:
            positions = [angle * COUNTS_PER_REV / 360.0 for angle in args.angles]
            print(format_reading(*positions, calibration))
        else:
            monitor(args.port, args.baudrate, args.interval, args.once, calibration)
        return 0
    except KeyboardInterrupt:
        print("\n已停止读取。")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"读取宽度失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
