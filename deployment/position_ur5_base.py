"""实时读取 UR5 TCP 在机器人 Base 坐标系中的位姿。

本脚本只使用 RTDEReceiveInterface 读取状态，不上传控制脚本，也不发送运动指令。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation

from replay.verify_tcp_transform import gripper_from_active, read_controller


DEFAULT_ROBOT_IP = "192.168.1.101"
DEFAULT_FREQUENCY_HZ = 1
DEFAULT_TCP_CALIBRATION = Path(__file__).resolve().parent / "calibration/tcp_gripper_01.json"


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def load_tcp_calibration(path: Path) -> np.ndarray:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    pose = np.asarray(data["tcp_offset_m_rad"], dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"无效的 TCP 标定：{path}")
    return pose_to_transform(pose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                        help=f"UR5 IP，默认 {DEFAULT_ROBOT_IP}")
    parser.add_argument("--frequency-hz", type=float, default=DEFAULT_FREQUENCY_HZ,
                        help=f"打印频率，默认 {DEFAULT_FREQUENCY_HZ:g} Hz")
    parser.add_argument("--tcp-calibration", type=Path, default=DEFAULT_TCP_CALIBRATION,
                        help="法兰到夹爪 G 的已有标定 JSON")
    parser.add_argument("--raw-active-tcp", action="store_true",
                        help="只输出控制器当前激活 TCP，不换算夹爪 G")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not np.isfinite(args.frequency_hz) or args.frequency_hz <= 0:
        raise ValueError("--frequency-hz must be a positive finite number")

    flange_to_gripper = None
    flange_to_active = None
    if not args.raw_active_tcp:
        flange_to_gripper = load_tcp_calibration(args.tcp_calibration)
        state = read_controller(args.robot_ip)
        flange_to_active = pose_to_transform(np.asarray(state["tcp_offset_m_rad"], dtype=np.float64))

    receiver = RTDEReceiveInterface(args.robot_ip, args.frequency_hz)
    period = 1.0 / args.frequency_hz
    print(f"已连接 UR5 {args.robot_ip}，按 Ctrl+C 退出。", flush=True)
    if args.raw_active_tcp:
        print("输出：控制器当前激活 TCP H（Base 坐标系）", flush=True)
    else:
        print(f"输出：夹爪 G（Base 坐标系），标定：{args.tcp_calibration}", flush=True)
    print("time_s, x_m, y_m, z_m, x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad", flush=True)
    try:
        while True:
            pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
            if pose.shape != (6,) or not np.isfinite(pose).all():
                raise RuntimeError(f"UR5 返回了无效 TCP 位姿：{pose}")
            if not args.raw_active_tcp:
                gripper = gripper_from_active(
                    pose_to_transform(pose), flange_to_gripper, flange_to_active
                )
                pose = np.r_[
                    gripper[:3, 3],
                    Rotation.from_matrix(gripper[:3, :3]).as_rotvec(),
                ]
            print(
                f"{time.time():.3f}, "
                f"{pose[0]: .6f}, {pose[1]: .6f}, {pose[2]: .6f}, "
                f"{pose[0] * 1000: .2f}, {pose[1] * 1000: .2f}, {pose[2] * 1000: .2f}, "
                f"{pose[3]: .6f}, {pose[4]: .6f}, {pose[5]: .6f}",
                flush=True,
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n已停止读取。", flush=True)
    finally:
        receiver.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
