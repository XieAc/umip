"""Read-only: print the G pose and its world-vertical depth for the soft floor.

Move the gripper (pendant / freedrive) to the lowest position it may reach,
for example fingertips just above the table, then run:

    python test/read_g_depth.py

It reads the active TCP on the passive state port and the pose over RTDE
receive; it uploads no script and sends no motion.  Depth is the G origin
projected on --world-down-in-base (default: the finger direction at go_home),
the same quantity --floor-depth-m / --max-depth-m limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay.replay_xac_ur5 import pose_to_transform, transform_to_pose  # noqa: E402
from replay.verify_tcp_transform import gripper_from_active, load_calibration, read_controller  # noqa: E402

GO_HOME_DOWN_IN_BASE = (-0.7071, 0.0, -0.7071)   # keep equal to test/test_model_inference.py


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", default="192.168.1.101")
    parser.add_argument("--tcp-calibration", type=Path,
                        default=ROOT / "calibration/tcp_gripper_01_gx_down_candidate.json")
    parser.add_argument("--world-down-in-base", type=float, nargs=3, default=GO_HOME_DOWN_IN_BASE)
    parser.add_argument("--fingertip-half-span-m", type=float, default=0.05)
    parser.add_argument("--margin-mm", type=float, default=0.5,
                        help="soft floor this far above the measured lowest position")
    args = parser.parse_args()
    from rtde_receive import RTDEReceiveInterface

    down = np.asarray(args.world_down_in_base, dtype=np.float64)
    down /= np.linalg.norm(down)
    state = read_controller(args.robot_ip)
    flange_to_active = pose_to_transform(np.asarray(state["tcp_offset_m_rad"], dtype=np.float64))
    flange_to_gripper, _ = load_calibration(args.tcp_calibration.expanduser().resolve())
    receiver = RTDEReceiveInterface(args.robot_ip)
    try:
        active = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
    finally:
        receiver.disconnect()
    g = transform_to_pose(gripper_from_active(pose_to_transform(active), flange_to_gripper, flange_to_active))
    from scipy.spatial.transform import Rotation
    side = Rotation.from_rotvec(g[3:]).apply([0.0, 1.0, 0.0])
    origin_depth = float(np.dot(g[:3], down))
    depth = origin_depth + args.fingertip_half_span_m * abs(float(np.dot(side, down)))   # lower fingertip
    floor = depth - args.margin_mm / 1000.0
    print(f"G pose (m/rad): {np.round(g, 5).tolist()}")
    print(f"base Z: {g[2]:.4f} m   G origin depth: {origin_depth:.4f} m   lower fingertip depth: {depth:.4f} m")
    print("If this is the lowest allowed position, use:")
    print(f"  SAFE_MAX_DEPTH={depth:.4f}   # hard stop")
    print(f"  SAFE_FLOOR_DEPTH={floor:.4f}   # soft floor, {args.margin_mm:g} mm higher")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
