"""Read-only XC330 USB/servo diagnostics for the local Linux setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from xac_config import DXL_BAUDRATE, DXL_IDS, DXL_PORT
except ModuleNotFoundError:  # direct ``python driver/check_xc330.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from xac_config import DXL_BAUDRATE, DXL_IDS, DXL_PORT


def read_diagnostics(port_name: str, baudrate: int, motor_ids: tuple[int, int]) -> list[dict]:
    try:
        from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
    except ImportError as exc:
        raise RuntimeError("Install dynamixel-sdk before checking XC330") from exc
    port = PortHandler(port_name)
    packet = PacketHandler(2.0)
    if not port.openPort():
        raise RuntimeError(f"Cannot open {port_name}")
    if not port.setBaudRate(baudrate):
        port.closePort()
        raise RuntimeError(f"Cannot set baudrate {baudrate} on {port_name}")

    def read(method, motor_id: int, address: int) -> int | None:
        value, result, error = method(port, motor_id, address)
        if result != COMM_SUCCESS or (error & 0x7F):
            return None
        return int(value)

    try:
        rows = []
        for motor_id in motor_ids:
            model, result, error = packet.ping(port, motor_id)
            rows.append(
                {
                    "id": motor_id,
                    "ping": bool(result == COMM_SUCCESS and (error & 0x7F) == 0),
                    "model_number": int(model) if result == COMM_SUCCESS else None,
                    "hardware_error": read(packet.read1ByteTxRx, motor_id, 70),
                    "torque_enabled": read(packet.read1ByteTxRx, motor_id, 64),
                    "present_position": read(packet.read4ByteTxRx, motor_id, 132),
                    "present_current_raw": read(packet.read2ByteTxRx, motor_id, 126),
                    "voltage_raw": read(packet.read2ByteTxRx, motor_id, 144),
                    "temperature_c": read(packet.read1ByteTxRx, motor_id, 146),
                }
            )
        return rows
    finally:
        port.closePort()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DXL_PORT)
    parser.add_argument("--baudrate", type=int, default=DXL_BAUDRATE)
    parser.add_argument("--ids", type=int, nargs=2, default=DXL_IDS)
    args = parser.parse_args()
    try:
        print(json.dumps({"port": args.port, "baudrate": args.baudrate,
                          "motors": read_diagnostics(args.port, args.baudrate, tuple(args.ids))},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"XC330 check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
