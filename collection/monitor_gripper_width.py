"""Read encoder angles and print calibrated fingertip width at 1 Hz."""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np

from xac_config import COUNTS_PER_REV, DXL_BAUDRATE, DXL_IDS, DXL_PORT, load_width_calibration

ADDR_PRESENT_POSITION = 132
PERIOD_S = 1.0


def angle_width_table(calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Express the existing measured tick calibration in degrees and mm."""
    if calibration.get("model") != "paired_difference_tick":
        raise ValueError("本脚本需要 paired_difference_tick 角度差标定，请重新运行 calibrate_gripper_width.py")
    samples = calibration["samples"]
    angle_diffs = np.asarray([
        (float(row["m1"]) - float(row["m2"])) * 360.0 / COUNTS_PER_REV
        for row in samples
    ])
    widths_mm = np.asarray([float(row["width_m"]) * 1000.0 for row in samples])
    order = np.argsort(angle_diffs)
    return angle_diffs[order], widths_mm[order]


def read_angle(packet, port, motor_id: int, comm_success: int) -> float:
    ticks, result, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
    if result != comm_success:
        raise RuntimeError(f"读取 ID={motor_id} 失败：{packet.getTxRxResult(result)}")
    if error:
        raise RuntimeError(f"读取 ID={motor_id} 失败：{packet.getRxPacketError(error)}")
    # Preserve signed, continuous position; do not wrap each angle at 360°.
    if ticks >= 0x80000000:
        ticks -= 0x100000000
    return ticks * 360.0 / COUNTS_PER_REV


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DXL_PORT, help="电机串口，默认使用 XAC_DXL_PORT / COM6")
    parser.add_argument("--baudrate", type=int, default=DXL_BAUDRATE)
    args = parser.parse_args()
    if args.baudrate <= 0:
        parser.error("--baudrate 必须为正整数")

    angle_diffs, widths_mm = angle_width_table(load_width_calibration())
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

    port = PortHandler(args.port)
    packet = PacketHandler(2.0)
    if not port.openPort():
        raise RuntimeError(f"无法打开串口 {args.port}")
    try:
        if not port.setBaudRate(args.baudrate):
            raise RuntimeError(f"无法设置波特率 {args.baudrate}")
        print(f"串口 {args.port} @ {args.baudrate}；每秒输出一次，Ctrl+C 退出。", flush=True)
        print("width 按 θ1−θ2 插值，单位 mm；仅读取编码器，不修改扭矩或工作模式。", flush=True)
        deadline = time.monotonic()
        while True:
            angle_1, angle_2 = (read_angle(packet, port, motor, COMM_SUCCESS) for motor in DXL_IDS)
            angle_diff = angle_1 - angle_2
            width_mm = float(np.interp(angle_diff, angle_diffs, widths_mm))
            note = " [超出标定范围，width 已限幅]" if not angle_diffs[0] <= angle_diff <= angle_diffs[-1] else ""
            print(
                f"{datetime.now():%H:%M:%S}  "
                f"ID={DXL_IDS[0]} angle={angle_1:.3f}deg  "
                f"ID={DXL_IDS[1]} angle={angle_2:.3f}deg  "
                f"diff={angle_diff:.3f}deg  width={width_mm:.2f}mm{note}",
                flush=True,
            )
            deadline += PERIOD_S
            now = time.monotonic()
            if deadline <= now:
                deadline = now + PERIOD_S
            time.sleep(deadline - now)
    finally:
        port.closePort()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止读取。")
    except Exception as exc:
        raise SystemExit(f"退出：{exc}") from exc
