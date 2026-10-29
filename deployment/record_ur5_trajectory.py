"""Record a manually demonstrated UR5 TCP trajectory without commanding motion.

Use the pendant to demonstrate the motion. This script only reads RTDE state and
writes a CSV plus JSON metadata. The recorded pose is the controller's active
TCP H in UR5 Base coordinates. If the pendant has the flange TCP active, H=F;
if a calibrated gripper TCP is active, H=G.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rtde_receive import RTDEReceiveInterface

from replay.verify_tcp_transform import read_controller


DEFAULT_ROBOT_IP = "192.168.1.101"
DEFAULT_RATE_HZ = 125.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV output path; JSON metadata uses the same stem")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--duration-s", type=float,
                        help="Optional automatic stop duration; otherwise Ctrl+C stops")
    parser.add_argument("--label", default="pendant_demo",
                        help="Human-readable demonstration label")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not np.isfinite(args.rate_hz) or args.rate_hz <= 0 or args.rate_hz > 500:
        raise ValueError("--rate-hz must be in (0, 500]")
    if args.duration_s is not None and (not np.isfinite(args.duration_s) or args.duration_s <= 0):
        raise ValueError("--duration-s must be positive")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"output already exists: {output} or {metadata_path}")

    state = read_controller(args.robot_ip)
    tcp_offset = np.asarray(state["tcp_offset_m_rad"], dtype=float)
    initial_pose = np.asarray(state["actual_tcp_pose_m_rad"], dtype=float)
    if tcp_offset.shape != (6,) or initial_pose.shape != (6,):
        raise RuntimeError("controller returned an invalid TCP pose or offset")

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_ip": args.robot_ip,
        "label": args.label,
        "pose_frame": "H_in_UR5_Base",
        "active_tcp_definition": "controller active TCP H; H=F for flange-zero TCP, H=G for active gripper TCP",
        "tcp_offset_m_rad_at_start": tcp_offset.tolist(),
        "initial_active_tcp_pose_m_rad": initial_pose.tolist(),
        "sample_rate_hz_requested": float(args.rate_hz),
        "duration_s_requested": None if args.duration_s is None else float(args.duration_s),
        "motion_commands_sent": False,
        "csv": str(output),
    }

    stop_requested = False
    stop_reason = "operator_ctrl_c"

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = "operator_ctrl_c"

    old_sigint = signal.signal(signal.SIGINT, stop)
    receiver = None
    rows = 0
    start_wall = time.monotonic()
    try:
        receiver = RTDEReceiveInterface(args.robot_ip, args.rate_hz)
        fieldnames = [
            "sample_index", "wall_time_s", "controller_timestamp_s",
            "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad",
            "tcp_offset_tx_m", "tcp_offset_ty_m", "tcp_offset_tz_m",
            "tcp_offset_rx_rad", "tcp_offset_ry_rad", "tcp_offset_rz_rad",
            "speed_scaling",
        ]
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            print(f"Connected to UR5 {args.robot_ip}; recording to {output}")
            print("Use the pendant to demonstrate the motion. Press Ctrl+C to stop.", flush=True)
            period = 1.0 / args.rate_hz
            next_sample = time.monotonic()
            while not stop_requested:
                now = time.monotonic()
                if args.duration_s is not None and now - start_wall >= args.duration_s:
                    stop_reason = "duration"
                    break
                if now < next_sample:
                    time.sleep(min(next_sample - now, 0.002))
                    continue
                pose = np.asarray(receiver.getActualTCPPose(), dtype=float)
                controller_timestamp = float(receiver.getTimestamp())
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise RuntimeError(f"invalid TCP pose: {pose}")
                writer.writerow({
                    "sample_index": rows,
                    "wall_time_s": now - start_wall,
                    "controller_timestamp_s": controller_timestamp,
                    "x_m": pose[0], "y_m": pose[1], "z_m": pose[2],
                    "rx_rad": pose[3], "ry_rad": pose[4], "rz_rad": pose[5],
                    "tcp_offset_tx_m": tcp_offset[0], "tcp_offset_ty_m": tcp_offset[1],
                    "tcp_offset_tz_m": tcp_offset[2], "tcp_offset_rx_rad": tcp_offset[3],
                    "tcp_offset_ry_rad": tcp_offset[4], "tcp_offset_rz_rad": tcp_offset[5],
                    "speed_scaling": float(receiver.getSpeedScaling()),
                })
                rows += 1
                if rows % max(1, round(args.rate_hz)) == 0:
                    print(f"{now - start_wall:7.2f}s  H xyz(m)="
                          f"[{pose[0]: .4f}, {pose[1]: .4f}, {pose[2]: .4f}]", flush=True)
                next_sample += period
                if next_sample < now - period:
                    next_sample = now + period
        metadata["sample_count"] = rows
        metadata["actual_duration_s"] = time.monotonic() - start_wall
        metadata["stopped_by"] = stop_reason
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {rows} samples: {output}")
        print(f"Saved metadata: {metadata_path}")
        return 0
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        if receiver is not None:
            receiver.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
