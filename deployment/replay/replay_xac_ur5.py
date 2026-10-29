"""Safely replay one XAC episode on a UR5 using relative pose anchoring.

The script is dry-run only unless --enable-motion is supplied. Hardware
dependencies are imported lazily so trajectory checks work on the collection
computer without ur_rtde or Dynamixel SDK installed.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Keep direct execution (``python replay/replay_xac_ur5.py``) working after
# the replay tools are grouped into this package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xac_config import CALIBRATION_DIR, DXL_BAUDRATE, DXL_PORT
from calibrate_gripper_width import validate_samples
from replay.verify_tcp_transform import (
    active_from_gripper, gripper_from_active, load_calibration as load_tcp_calibration,
    read_controller as read_robot_state, ready as require_idle_robot,
)
from driver.xc330 import XC330Controller as _DriverXC330Controller
from driver.d435i import D435iConfig, D435iStream


@dataclass
class Trajectory:
    timestamps_s: np.ndarray
    positions_m: np.ndarray
    rotations_rotvec: np.ndarray
    gripper_width_m: np.ndarray
    source_indices: np.ndarray

    @property
    def length(self) -> int:
        return len(self.timestamps_s)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])


def _load_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode array: {path}")
    return np.asarray(np.load(path, allow_pickle=False))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_end = run_start = 0
    in_run = False
    for index, value in enumerate(np.append(mask, False)):
        if value and not in_run:
            run_start = index
            in_run = True
        elif not value and in_run:
            if index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            in_run = False
    return best_start, best_end


def load_trajectory(episode: Path) -> Trajectory:
    data = episode.expanduser().resolve() / "data"
    timestamps = _load_array(data, "d435i_timestamps.npy").astype(np.float64)
    positions = _load_array(data, "robot0_eef_pos.npy").astype(np.float64)
    rotations = _load_array(data, "robot0_eef_rot_axis_angle.npy").astype(np.float64)
    widths = _load_array(data, "robot0_gripper_width.npy").astype(np.float64).reshape(-1)
    valid = _load_array(data, "valid_mask.npy").astype(bool).reshape(-1)

    n = len(timestamps)
    expected = {
        "positions": positions.shape,
        "rotations": rotations.shape,
        "widths": widths.shape,
        "valid": valid.shape,
    }
    if timestamps.shape != (n,) or positions.shape != (n, 3) or rotations.shape != (n, 3):
        raise ValueError(f"Episode array shapes do not agree: timestamps={timestamps.shape}, {expected}")
    if widths.shape != (n,) or valid.shape != (n,):
        raise ValueError(f"Episode array lengths do not agree: timestamps={timestamps.shape}, {expected}")
    finite = (
        np.isfinite(timestamps)
        & np.all(np.isfinite(positions), axis=1)
        & np.all(np.isfinite(rotations), axis=1)
        & np.isfinite(widths)
    )
    valid &= finite
    start, end = _longest_true_run(valid)
    if end - start < 2:
        raise ValueError("Episode has fewer than two contiguous valid frames")
    selected_timestamps = timestamps[start:end]
    if np.any(np.diff(selected_timestamps) <= 0):
        raise ValueError("Timestamps in the selected valid run are not strictly increasing")
    auxiliary_transform_path = episode.expanduser().resolve() / "auxiliary" / "T_W_G.npy"
    if auxiliary_transform_path.is_file():
        transforms = np.asarray(np.load(auxiliary_transform_path, allow_pickle=False), dtype=np.float64)
        if transforms.shape != (n, 4, 4) or not np.isfinite(transforms).all():
            raise ValueError(f"Invalid auxiliary T_W_G shape/content: {auxiliary_transform_path}")
        reconstructed = poses_to_transforms(positions, rotations)
        max_error = float(np.max(np.abs(transforms - reconstructed)))
        if max_error > 2e-5:
            raise ValueError(
                f"Episode pose arrays disagree with auxiliary/T_W_G.npy (max matrix error {max_error:.3g})"
            )
    return Trajectory(
        timestamps_s=selected_timestamps,
        positions_m=positions[start:end],
        rotations_rotvec=rotations[start:end],
        gripper_width_m=widths[start:end],
        source_indices=np.arange(start, end, dtype=np.int64),
    )


def poses_to_transforms(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(positions), axis=0)
    transforms[:, :3, :3] = Rotation.from_rotvec(rotations).as_matrix()
    transforms[:, :3, 3] = positions
    return transforms


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]
    )


def anchor_trajectory(
    trajectory: Trajectory,
    base_start_pose: np.ndarray,
    translation_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Anchor the relative trajectory, optionally rotating translations in B.

    ``translation_rotation`` is an empirical alignment for position only. It
    is deliberately applied to the relative translation after extracting the
    source motion; recorded orientation replay remains unchanged.
    """

    base_start_pose = np.asarray(base_start_pose, dtype=np.float64)
    if base_start_pose.shape != (6,) or not np.all(np.isfinite(base_start_pose)):
        raise ValueError("Anchor must contain six finite values in metres/radians")
    world = poses_to_transforms(trajectory.positions_m, trajectory.rotations_rotvec)
    base_start = pose_to_transform(np.asarray(base_start_pose, dtype=np.float64))
    world_start_inverse = np.linalg.inv(world[0])
    relative = world_start_inverse[None, :, :] @ world
    if translation_rotation is not None:
        alignment = np.asarray(translation_rotation, dtype=np.float64)
        if alignment.shape != (3, 3) or not np.allclose(alignment.T @ alignment, np.eye(3), atol=1e-6):
            raise ValueError("translation_rotation must be a 3x3 rotation matrix")
        if np.linalg.det(alignment) < 0.0:
            raise ValueError("translation_rotation must preserve handedness")
        relative[:, :3, 3] = relative[:, :3, 3] @ alignment.T
    targets = base_start[None, :, :] @ relative
    return np.stack([transform_to_pose(item) for item in targets])


def trajectory_metrics(trajectory: Trajectory, target_poses: np.ndarray, speed_scale: float) -> dict[str, Any]:
    dt = np.diff(trajectory.timestamps_s)
    displacement = np.linalg.norm(np.diff(target_poses[:, :3], axis=0), axis=1)
    rotations = Rotation.from_rotvec(target_poses[:, 3:])
    angular_step = (rotations[1:] * rotations[:-1].inv()).magnitude()
    distance_from_start = np.linalg.norm(target_poses[:, :3] - target_poses[0, :3], axis=1)
    source_linear_speed = displacement / dt
    source_angular_speed = angular_step / dt
    return {
        "frames": trajectory.length,
        "source_index_start": int(trajectory.source_indices[0]),
        "source_index_end_inclusive": int(trajectory.source_indices[-1]),
        "source_duration_s": trajectory.duration_s,
        "wall_duration_at_speed_scale_s": trajectory.duration_s / speed_scale,
        "median_source_dt_s": float(np.median(dt)),
        "max_source_dt_s": float(np.max(dt)),
        "max_distance_from_start_m": float(np.max(distance_from_start)),
        "max_linear_speed_source_m_s": float(np.max(source_linear_speed)),
        "max_angular_speed_source_rad_s": float(np.max(source_angular_speed)),
        "max_linear_speed_replay_m_s": float(np.max(source_linear_speed) * speed_scale),
        "max_angular_speed_replay_rad_s": float(np.max(source_angular_speed) * speed_scale),
        "gripper_width_min_m": float(np.min(trajectory.gripper_width_m)),
        "gripper_width_max_m": float(np.max(trajectory.gripper_width_m)),
        "gripper_width_start_m": float(trajectory.gripper_width_m[0]),
        "target_xyz_min_m": np.min(target_poses[:, :3], axis=0).tolist(),
        "target_xyz_max_m": np.max(target_poses[:, :3], axis=0).tolist(),
        "start_target_pose": target_poses[0].tolist(),
        "end_target_pose": target_poses[-1].tolist(),
        "source_delta_xyz_m": (trajectory.positions_m[-1] - trajectory.positions_m[0]).tolist(),
        "target_delta_xyz_m": (target_poses[-1, :3] - target_poses[0, :3]).tolist(),
        "source_xyz_min_m": np.min(trajectory.positions_m, axis=0).tolist(),
        "source_xyz_max_m": np.max(trajectory.positions_m, axis=0).tolist(),
    }


def trajectory_diagnostic(
    trajectory: Trajectory,
    target_poses: np.ndarray,
    speed_scale: float,
) -> dict[str, Any]:
    """Return a compact axis-by-axis report for offline replay debugging.

    The source arrays are expressed in the recorded G/world frame.  Target
    arrays are expressed in UR Base after relative anchoring.  Keeping both in
    the report makes a wrong dataset axis distinguishable from a wrong TCP or
    anchor transform.
    """
    source = trajectory.positions_m
    target = target_poses[:, :3]
    source_steps = np.diff(source, axis=0)
    target_steps = np.diff(target, axis=0)
    dt = np.diff(trajectory.timestamps_s)
    source_start_rotation = Rotation.from_rotvec(trajectory.rotations_rotvec[0]).as_matrix()
    target_start_rotation = Rotation.from_rotvec(target_poses[0, 3:]).as_matrix()
    source_delta_world = source[-1] - source[0]
    target_delta_base = target[-1] - target[0]
    source_delta_start_g = source_start_rotation.T @ source_delta_world
    target_delta_start_g = target_start_rotation.T @ target_delta_base
    return {
        "frames": trajectory.length,
        "source_frame": "recorded_G_in_W",
        "target_frame": "anchored_G_in_UR_base",
        "source_delta_xyz_m": source_delta_world.tolist(),
        "target_delta_xyz_m": target_delta_base.tolist(),
        "source_delta_in_start_g_m": source_delta_start_g.tolist(),
        "target_delta_in_start_g_m": target_delta_start_g.tolist(),
        "source_range_xyz_m": np.ptp(source, axis=0).tolist(),
        "target_range_xyz_m": np.ptp(target, axis=0).tolist(),
        "source_mean_abs_step_xyz_m": np.mean(np.abs(source_steps), axis=0).tolist(),
        "target_mean_abs_step_xyz_m": np.mean(np.abs(target_steps), axis=0).tolist(),
        "source_max_abs_speed_xyz_m_s": np.max(np.abs(source_steps / dt[:, None]), axis=0).tolist(),
        "target_max_abs_speed_xyz_m_s": np.max(np.abs(target_steps / dt[:, None]) * speed_scale, axis=0).tolist(),
        "source_start_pose": np.concatenate([trajectory.positions_m[0], trajectory.rotations_rotvec[0]]).tolist(),
        "source_end_pose": np.concatenate([trajectory.positions_m[-1], trajectory.rotations_rotvec[-1]]).tolist(),
        "target_start_pose": target_poses[0].tolist(),
        "target_end_pose": target_poses[-1].tolist(),
    }


def print_trajectory_diagnostic(report: dict[str, Any]) -> None:
    """Print the high-signal fields operators need before any motion."""
    print("Replay trajectory diagnostic (no robot command):")
    print(f"  frames={report['frames']} source={report['source_frame']} target={report['target_frame']}")
    print(f"  source delta xyz (m): {np.asarray(report['source_delta_xyz_m']).round(6).tolist()}")
    print(f"  target delta xyz (m): {np.asarray(report['target_delta_xyz_m']).round(6).tolist()}")
    print(f"  source range xyz (m): {np.asarray(report['source_range_xyz_m']).round(6).tolist()}")
    print(f"  target range xyz (m): {np.asarray(report['target_range_xyz_m']).round(6).tolist()}")
    print(f"  target max abs speed xyz (m/s): {np.asarray(report['target_max_abs_speed_xyz_m_s']).round(6).tolist()}")


def enforce_limits(
    metrics: dict[str, Any],
    max_distance_m: float,
    min_z_m: float,
    max_z_m: float,
    max_linear_speed_m_s: float,
    max_angular_speed_rad_s: float,
    max_source_gap_s: float,
) -> None:
    violations: list[str] = []
    if metrics["max_distance_from_start_m"] > max_distance_m:
        violations.append(
            f"distance from start {metrics['max_distance_from_start_m']:.4f} m > {max_distance_m:.4f} m"
        )
    if metrics["target_xyz_min_m"][2] < min_z_m:
        violations.append(f"minimum G-origin Base Z {metrics['target_xyz_min_m'][2]:.4f} m < {min_z_m:.4f} m")
    if metrics["target_xyz_max_m"][2] > max_z_m:
        violations.append(f"maximum G-origin Base Z {metrics['target_xyz_max_m'][2]:.4f} m > {max_z_m:.4f} m")
    if metrics["max_linear_speed_replay_m_s"] > max_linear_speed_m_s:
        violations.append(
            f"linear speed {metrics['max_linear_speed_replay_m_s']:.4f} m/s > {max_linear_speed_m_s:.4f} m/s"
        )
    if metrics["max_angular_speed_replay_rad_s"] > max_angular_speed_rad_s:
        violations.append(
            f"angular speed {metrics['max_angular_speed_replay_rad_s']:.4f} rad/s > {max_angular_speed_rad_s:.4f} rad/s"
        )
    if metrics["max_source_dt_s"] > max_source_gap_s:
        violations.append(
            f"source timestamp gap {metrics['max_source_dt_s']:.4f} s > {max_source_gap_s:.4f} s"
        )
    if violations:
        raise ValueError("Safety limit violation:\n  " + "\n  ".join(violations))


def load_replay_calibration(path: Path) -> list[dict]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if data.get("model") != "paired_difference_tick":
        raise ValueError("Gripper calibration must use the paired_difference_tick model")
    samples = data.get("samples", [])
    validate_samples(samples)
    return sorted(samples, key=lambda row: float(row["width_m"]))


def validate_gripper_trajectory(trajectory: Trajectory, path: Path) -> None:
    samples = load_replay_calibration(path)
    low, high = float(samples[0]["width_m"]), float(samples[-1]["width_m"])
    if np.any(trajectory.gripper_width_m < low) or np.any(trajectory.gripper_width_m > high):
        raise ValueError(f"Episode gripper widths exceed calibration range [{low}, {high}] m")



# Keep the historical import name used by deployment scripts and tests while
# putting the actual hardware implementation under driver/.
XC330Controller = _DriverXC330Controller


def read_robot_pose(robot_ip: str) -> np.ndarray:
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("Install the UR dependency: pip install ur-rtde") from exc
    receiver = RTDEReceiveInterface(robot_ip)
    try:
        pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
    finally:
        receiver.disconnect()
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"UR5 returned an invalid TCP pose: {pose}")
    return pose


def execute_replay(
    args: argparse.Namespace,
    trajectory: Trajectory,
    target_poses: np.ndarray,
    trace_out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("Install the UR dependency: pip install ur-rtde") from exc

    control = None
    receiver = None
    gripper = None
    camera = None
    trace: list[dict[str, Any]] = trace_out if trace_out is not None else []
    software_tcp = args.tcp_calibration is not None
    flange_to_gripper = (getattr(args, "software_tcp_transform", None)
                         if software_tcp else None)
    if software_tcp and flange_to_gripper is None:
        flange_to_gripper = load_tcp_calibration(args.tcp_calibration)[0]
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        if args.enable_d435i:
            camera = D435iStream(
                D435iConfig(
                    width=args.d435i_width,
                    height=args.d435i_height,
                    fps=args.d435i_fps,
                    serial=args.d435i_serial,
                    enable_imu=not args.d435i_no_imu,
                )
            )
            camera.start(drain=True)
        if software_tcp:
            state = read_robot_state(args.robot_ip)
            require_idle_robot(state)
            expected_offset = getattr(args, "expected_active_tcp_offset", state["tcp_offset_m_rad"])
            if not np.allclose(pose_to_transform(np.asarray(state["tcp_offset_m_rad"])),
                               pose_to_transform(np.asarray(expected_offset)), atol=1e-8, rtol=0):
                raise RuntimeError("Active TCP offset changed after dry-run; rerun the preview")
        control = RTDEControlInterface(args.robot_ip, args.control_frequency_hz)
        receiver = RTDEReceiveInterface(args.robot_ip, args.control_frequency_hz)
        actual_tcp = np.asarray(control.getTCPOffset(), dtype=np.float64)
        if actual_tcp.shape != (6,) or not np.isfinite(actual_tcp).all():
            raise RuntimeError("Invalid active TCP offset")
        if software_tcp:
            if not np.allclose(pose_to_transform(actual_tcp), pose_to_transform(np.asarray(expected_offset)), atol=1e-8, rtol=0):
                raise RuntimeError("Active TCP offset changed when opening RTDEControl")
        elif not np.allclose(actual_tcp, args.tcp_offset, atol=1e-5, rtol=0):
            raise RuntimeError("Active UR TCP offset differs from --tcp-offset; configure the measured flange-to-G TCP on the pendant")
        flange_to_active = pose_to_transform(actual_tcp)

        def actual_gripper(raw):
            return (transform_to_pose(gripper_from_active(pose_to_transform(raw), flange_to_gripper, flange_to_active))
                    if software_tcp else raw)

        def controller_target(target):
            return (transform_to_pose(active_from_gripper(pose_to_transform(target), flange_to_gripper, flange_to_active))
                    if software_tcp else target)

        actual_start = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
        if actual_start.shape != (6,) or not np.all(np.isfinite(actual_start)):
            raise RuntimeError("UR5 returned an invalid TCP pose")
        actual_start = actual_gripper(actual_start)
        start_position_error = float(np.linalg.norm(actual_start[:3] - target_poses[0, :3]))
        start_rotation_error = float(
            (Rotation.from_rotvec(actual_start[3:]).inv() * Rotation.from_rotvec(target_poses[0, 3:])).magnitude()
        )
        if (
            start_position_error > args.start_pose_tolerance_m
            or start_rotation_error > args.start_rotation_tolerance_rad
        ):
            raise RuntimeError(
                "UR5 moved after the anchor was captured: "
                f"position error={start_position_error:.4f} m, "
                f"rotation error={start_rotation_error:.4f} rad"
            )
        if args.enable_gripper:
            gripper = XC330Controller(
                port_name=args.gripper_port,
                baudrate=args.gripper_baudrate,
                calibration_path=args.gripper_calibration,
                motor_ids=tuple(args.gripper_ids),
                current_limit_raw=args.gripper_current_limit_raw,
                start_tolerance_ticks=args.gripper_start_tolerance_ticks,
                goal_current_raw=args.gripper_goal_current_raw,
                hold_current_raw=args.gripper_hold_current_raw,
                profile_velocity=args.gripper_profile_velocity,
                profile_acceleration=args.gripper_profile_acceleration,
            )
            gripper.initialize(float(trajectory.gripper_width_m[0]))

        source_time = trajectory.timestamps_s - trajectory.timestamps_s[0]
        target_rotations = Rotation.from_rotvec(target_poses[:, 3:])
        rotation_interpolator = Slerp(source_time, target_rotations)
        source_duration = float(source_time[-1])
        replay_duration = source_duration / args.speed_scale
        control_period = 1.0 / args.control_frequency_hz
        replay_start = time.monotonic()
        iteration = 0
        next_gripper_command = 0.0
        next_current_check = 0.0
        next_progress = 0.0
        while True:
            if stop_requested:
                raise KeyboardInterrupt
            if camera is not None and not camera.healthy:
                raise RuntimeError(str(camera.last_error))
            elapsed = iteration * control_period
            wall_elapsed = time.monotonic() - replay_start
            if args.max_duration_s is not None and wall_elapsed >= args.max_duration_s:
                print(f"Stopped at wall-time limit {args.max_duration_s:.3f} s")
                break
            if elapsed > replay_duration:
                break
            sample_source_time = min(elapsed * args.speed_scale, source_duration)
            target = np.empty(6, dtype=np.float64)
            target[:3] = [
                np.interp(sample_source_time, source_time, target_poses[:, axis])
                for axis in range(3)
            ]
            target[3:] = rotation_interpolator([sample_source_time]).as_rotvec()[0]
            width = float(np.interp(sample_source_time, source_time, trajectory.gripper_width_m))
            source_index = int(np.searchsorted(source_time, sample_source_time, side="right") - 1)
            source_index = min(max(source_index, 0), trajectory.length - 1)
            cycle_start = control.initPeriod()
            command = controller_target(target)
            ok = control.servoL(
                command.tolist(),
                0.0,
                0.0,
                control_period,
                args.lookahead_time_s,
                args.servo_gain,
            )
            if not ok:
                raise RuntimeError(f"UR5 rejected servoL near source frame {trajectory.source_indices[source_index]}")
            if gripper is not None and elapsed >= next_gripper_command:
                check_current = elapsed >= next_current_check
                gripper.command_width(width, check_current=check_current)
                next_gripper_command = elapsed + 0.1
                if check_current:
                    next_current_check = elapsed + 0.2
            control.waitPeriod(cycle_start)
            actual = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
            if actual.shape != (6,) or not np.all(np.isfinite(actual)):
                raise RuntimeError("UR5 returned an invalid TCP pose during replay")
            actual_active = actual.copy()
            actual = actual_gripper(actual)
            if receiver.isProtectiveStopped() or receiver.isEmergencyStopped():
                raise RuntimeError("UR5 safety stop during replay")
            actual_distance = float(np.linalg.norm(actual[:3] - target_poses[0, :3]))
            following_position_error = float(np.linalg.norm(actual[:3] - target[:3]))
            following_rotation_error = float(
                (Rotation.from_rotvec(actual[3:]).inv() * Rotation.from_rotvec(target[3:])).magnitude()
            )
            if actual_distance > args.max_distance_from_start_m + args.runtime_position_margin_m:
                raise RuntimeError(
                    f"Runtime distance protection triggered: {actual_distance:.4f} m"
                )
            if actual[2] < args.minimum_base_z_m or actual[2] > args.maximum_base_z_m:
                raise RuntimeError(f"Runtime base-Z protection triggered: {actual[2]:.4f} m")
            if following_position_error > args.max_following_error_m:
                raise RuntimeError(
                    f"UR5 position following error is too large: {following_position_error:.4f} m"
                )
            if following_rotation_error > args.max_rotation_following_error_rad:
                raise RuntimeError(
                    f"UR5 rotation following error is too large: {following_rotation_error:.4f} rad"
                )
            if iteration % max(1, round(args.control_frequency_hz / 10.0)) == 0:
                trace.append(
                    {
                        "source_index": int(trajectory.source_indices[source_index]),
                        "wall_time_s": float(time.monotonic() - replay_start),
                        "target_pose": target.tolist(),
                        "actual_pose": actual.tolist(),
                        "pose_frame": "G_in_UR_base",
                        "controller_target_pose": command.tolist(),
                        "controller_actual_pose": actual_active.tolist(),
                        "active_tcp_offset_m_rad": actual_tcp.tolist(),
                        "gripper_width_target_m": width,
                    }
                )
            if wall_elapsed >= next_progress:
                print(f"Replay {wall_elapsed:.1f}s; source {sample_source_time:.2f}s; "
                      f"G movement {actual_distance*1000:.2f}mm; following error {following_position_error*1000:.2f}mm", flush=True)
                next_progress = wall_elapsed + 1
            iteration += 1
        return trace
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        try:
            if control is not None:
                control.servoStop(2.0)
        except Exception:
            pass
        try:
            if control is not None:
                control.stopScript()
        except Exception:
            pass
        try:
            if control is not None:
                control.disconnect()
        except Exception:
            pass
        try:
            if receiver is not None:
                receiver.disconnect()
        except Exception:
            pass
        if gripper is not None:
            gripper.close()
        if camera is not None:
            camera.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--robot-ip", help="UR5 IP; required for live motion or reading a live dry-run anchor")
    parser.add_argument(
        "--anchor-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Gripper G anchor pose in UR base coordinates; uses live gripper pose when omitted with --robot-ip",
    )
    parser.add_argument("--enable-motion", action="store_true")
    tcp_options = parser.add_mutually_exclusive_group()
    tcp_options.add_argument("--tcp-offset", type=float, nargs=6,
                        help="Legacy mode: measured flange-to-G offset already active on controller, m/rad")
    tcp_options.add_argument("--tcp-calibration", type=Path,
                        help="Computer-side T_F_G JSON; supports any actual active TCP without setting it")
    parser.add_argument("--speed-scale", type=float, default=0.1)
    parser.add_argument("--max-duration-s", type=float)
    parser.add_argument("--max-distance-from-start-m", type=float, default=0.12)
    parser.add_argument("--minimum-base-z-m", type=float)
    parser.add_argument("--maximum-base-z-m", type=float)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=0.5)
    parser.add_argument("--max-source-gap-s", type=float, default=0.25)
    parser.add_argument(
        "--translation-rotation-rpy-deg", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
        help="Optional position-only alignment rotation (degrees, XYZ order); orientation replay is unchanged",
    )
    parser.add_argument("--control-frequency-hz", type=float, default=125.0)
    parser.add_argument("--lookahead-time-s", type=float, default=0.1)
    parser.add_argument("--servo-gain", type=float, default=300.0)
    parser.add_argument("--start-pose-tolerance-m", type=float, default=0.002)
    parser.add_argument("--start-rotation-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--runtime-position-margin-m", type=float, default=0.005)
    parser.add_argument("--max-following-error-m", type=float, default=0.03)
    parser.add_argument("--max-rotation-following-error-rad", type=float, default=0.3)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--diagnose", action="store_true",
        help="print source/anchored XYZ diagnostics and exit without hardware commands",
    )
    parser.add_argument(
        "--dump-targets", type=Path,
        help="save the anchored target poses and source indices as a JSON diagnostic file",
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--gripper-port", default=DXL_PORT)
    parser.add_argument("--gripper-baudrate", type=int, default=DXL_BAUDRATE)
    parser.add_argument("--gripper-ids", type=int, nargs=2, default=(1, 2))
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=CALIBRATION_DIR / "gripper_width.json",
    )
    parser.add_argument(
        "--gripper-current-limit-raw", type=int, default=600,
        help="software trip threshold for present current (local XC330 default: 600 raw)",
    )
    parser.add_argument("--gripper-start-tolerance-ticks", type=int, default=80)
    parser.add_argument("--gripper-goal-current-raw", type=int, default=600)
    parser.add_argument("--gripper-hold-current-raw", type=int, default=100)
    parser.add_argument("--gripper-profile-velocity", type=int, default=30)
    parser.add_argument("--gripper-profile-acceleration", type=int, default=20)
    parser.add_argument(
        "--enable-d435i", action="store_true",
        help="start the local D435i and drain frames during live replay; frames are not used for control",
    )
    parser.add_argument("--d435i-serial", help="optional D435i serial number")
    parser.add_argument("--d435i-width", type=int, default=640)
    parser.add_argument("--d435i-height", type=int, default=480)
    parser.add_argument("--d435i-fps", type=int, default=30)
    parser.add_argument("--d435i-no-imu", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    for name in ("anchor_pose", "tcp_offset"):
        value = getattr(args, name)
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    for name in ("max_linear_speed_m_s", "max_angular_speed_rad_s", "start_pose_tolerance_m",
                 "start_rotation_tolerance_rad", "max_following_error_m",
                 "max_rotation_following_error_rad", "gripper_current_limit_raw", "gripper_baudrate"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.runtime_position_margin_m < 0 or args.gripper_start_tolerance_ticks < 0:
        raise ValueError("Runtime margin and gripper start tolerance must be nonnegative")
    if not 0.03 <= args.lookahead_time_s <= 0.2 or not 100 <= args.servo_gain <= 2000:
        raise ValueError("UR servo lookahead must be 0.03..0.2 s and gain 100..2000")
    if len(set(args.gripper_ids)) != 2 or any(i < 0 or i > 252 for i in args.gripper_ids):
        raise ValueError("Gripper IDs must be distinct and between 0 and 252")
    if not 0 < args.speed_scale <= 1:
        raise ValueError("--speed-scale must be in (0, 1]")
    if args.max_duration_s is not None and args.max_duration_s <= 0:
        raise ValueError("--max-duration-s must be positive")
    if args.control_frequency_hz < 10 or args.control_frequency_hz > 500:
        raise ValueError("--control-frequency-hz must be between 10 and 500")
    if args.max_distance_from_start_m <= 0:
        raise ValueError("--max-distance-from-start-m must be positive")
    if args.max_source_gap_s <= 0:
        raise ValueError("--max-source-gap-s must be positive")
    if args.translation_rotation_rpy_deg is not None and not np.isfinite(args.translation_rotation_rpy_deg).all():
        raise ValueError("--translation-rotation-rpy-deg values must be finite")
    if args.minimum_base_z_m is not None and args.maximum_base_z_m is not None:
        if args.minimum_base_z_m >= args.maximum_base_z_m:
            raise ValueError("--minimum-base-z-m must be smaller than --maximum-base-z-m")
    if args.enable_gripper and not args.enable_motion:
        raise ValueError("--enable-gripper requires --enable-motion")
    if args.enable_d435i and not args.enable_motion:
        raise ValueError("--enable-d435i requires --enable-motion")
    if args.diagnose and args.enable_motion:
        raise ValueError("--diagnose cannot be combined with --enable-motion")
    for name in ("gripper_goal_current_raw", "gripper_hold_current_raw",
                 "gripper_profile_velocity", "gripper_profile_acceleration",
                 "d435i_width", "d435i_height", "d435i_fps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.enable_motion:
        missing = []
        if not args.robot_ip:
            missing.append("--robot-ip")
        if args.tcp_offset is None and args.tcp_calibration is None:
            missing.append("--tcp-offset or --tcp-calibration")
        if args.minimum_base_z_m is None:
            missing.append("--minimum-base-z-m")
        if args.maximum_base_z_m is None:
            missing.append("--maximum-base-z-m")
        if missing:
            raise ValueError("Live replay requires " + ", ".join(missing))


def main() -> int:
    args = build_parser().parse_args()
    validate_arguments(args)
    trajectory = load_trajectory(args.episode)
    translation_rotation = None
    if args.translation_rotation_rpy_deg is not None:
        translation_rotation = Rotation.from_euler(
            "xyz", args.translation_rotation_rpy_deg, degrees=True
        ).as_matrix()
    software_tcp = args.tcp_calibration is not None
    if software_tcp:
        flange_to_gripper, _ = load_tcp_calibration(args.tcp_calibration)
        args.software_tcp_transform = flange_to_gripper
        if args.robot_ip:
            state = read_robot_state(args.robot_ip)
            args.expected_active_tcp_offset = state["tcp_offset_m_rad"]
            flange_to_active = pose_to_transform(np.asarray(state["tcp_offset_m_rad"]))
            live_gripper = transform_to_pose(gripper_from_active(
                pose_to_transform(np.asarray(state["actual_tcp_pose_m_rad"])), flange_to_gripper, flange_to_active))
        else:
            flange_to_active = np.eye(4)
    if args.enable_gripper:
        validate_gripper_trajectory(trajectory, args.gripper_calibration)
    if args.anchor_pose is not None:
        anchor = np.asarray(args.anchor_pose, dtype=np.float64)
        anchor_source = "command_line"
    elif args.robot_ip:
        anchor = live_gripper if software_tcp else read_robot_pose(args.robot_ip)
        anchor_source = "live_gripper_from_computer_calibration" if software_tcp else "live_ur5_tcp"
    else:
        anchor = np.concatenate([trajectory.positions_m[0], trajectory.rotations_rotvec[0]])
        anchor_source = "recorded_start_for_relative_only_dry_run"

    targets = anchor_trajectory(trajectory, anchor, translation_rotation)
    diagnostic = trajectory_diagnostic(trajectory, targets, args.speed_scale)
    if args.diagnose:
        print_trajectory_diagnostic(diagnostic)
    if args.dump_targets:
        dump_path = args.dump_targets.expanduser().resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps({
            "episode": str(args.episode.expanduser().resolve()),
            "anchor_source": anchor_source,
            "anchor_pose_g_m_rad": anchor.tolist(),
            "diagnostic": diagnostic,
            "samples": [
                {
                    "source_index": int(source_index),
                    "source_pose": np.concatenate([
                        trajectory.positions_m[index], trajectory.rotations_rotvec[index]
                    ]).tolist(),
                    "target_pose_g_in_ur_base": targets[index].tolist(),
                    "gripper_width_m": float(trajectory.gripper_width_m[index]),
                }
                for index, source_index in enumerate(trajectory.source_indices)
            ],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Target diagnostic saved: {dump_path}")
    metrics = trajectory_metrics(trajectory, targets, args.speed_scale)
    metrics["trajectory_diagnostic"] = diagnostic
    if software_tcp:
        command_targets = np.stack([transform_to_pose(active_from_gripper(
            pose_to_transform(target), flange_to_gripper, flange_to_active)) for target in targets])
        command_metrics = trajectory_metrics(trajectory, command_targets, args.speed_scale)
        metrics.update({"tcp_mode": "computer_side", "tcp_calibration": str(args.tcp_calibration.expanduser().resolve()),
                        "T_F_G": flange_to_gripper.tolist(), "active_tcp_offset_m_rad": transform_to_pose(flange_to_active).tolist(),
                        "live_active_tcp_pose_m_rad": state["actual_tcp_pose_m_rad"] if args.robot_ip else None,
                        "live_gripper_anchor_pose_m_rad": live_gripper.tolist() if args.robot_ip else None,
                        "controller_command_max_linear_speed_m_s": command_metrics["max_linear_speed_replay_m_s"],
                        "controller_command_start_pose_m_rad": command_targets[0].tolist(),
                        "controller_command_xyz_min_m": command_metrics["target_xyz_min_m"],
                        "controller_command_xyz_max_m": command_metrics["target_xyz_max_m"],
                        "offline_controller_offset_assumed_zero": not bool(args.robot_ip)})
        if command_metrics["max_linear_speed_replay_m_s"] > args.max_linear_speed_m_s:
            raise ValueError(f"Controller TCP speed after software offset compensation "
                             f"{command_metrics['max_linear_speed_replay_m_s']:.4f} m/s exceeds "
                             f"{args.max_linear_speed_m_s:.4f} m/s; lower --speed-scale")
    else:
        metrics["tcp_mode"] = "controller_side_legacy"
    metrics.update(
        {
            "episode": str(args.episode.expanduser().resolve()),
            "anchor_source": anchor_source,
            "motion_enabled": bool(args.enable_motion),
            "gripper_enabled": bool(args.enable_gripper),
            "gripper_port": args.gripper_port if args.enable_gripper else None,
            "gripper_ids": list(args.gripper_ids) if args.enable_gripper else None,
            "d435i_enabled": bool(args.enable_d435i),
            "speed_scale": float(args.speed_scale),
            "translation_rotation_rpy_deg": args.translation_rotation_rpy_deg,
        }
    )
    min_z = -math.inf if args.minimum_base_z_m is None else args.minimum_base_z_m
    max_z = math.inf if args.maximum_base_z_m is None else args.maximum_base_z_m
    enforce_limits(
        metrics,
        max_distance_m=args.max_distance_from_start_m,
        min_z_m=min_z,
        max_z_m=max_z,
        max_linear_speed_m_s=args.max_linear_speed_m_s,
        max_angular_speed_rad_s=args.max_angular_speed_rad_s,
        max_source_gap_s=args.max_source_gap_s,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Safety checks: PASS")
    if args.report:
        args.report.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report.expanduser().resolve().write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if not args.enable_motion:
        print("DRY RUN ONLY: no UR5 or gripper command was sent.")
        return 0

    print("\nLIVE REPLAY ARMED")
    print(f"UR5: {args.robot_ip}; anchor TCP: {anchor.tolist()}")
    print("Clear the workspace, set the pendant speed slider low, and keep the emergency stop in reach.")
    if input('Type exactly "RUN" to start motion: ').strip() != "RUN":
        print("Cancelled; no motion command was sent.")
        return 2

    trace_path = args.trace or (
        PROJECT_ROOT / "logs" / "replay" /
        f"replay_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    trace = []
    trace_payload = {"created_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
                     "completed": False, "samples": trace}
    try:
        execute_replay(args, trajectory, targets, trace_out=trace)
        trace_payload["completed"] = True
    except BaseException as exc:
        trace_payload["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        trace_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        trace_path.expanduser().resolve().write_text(
            json.dumps(trace_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Trace saved: {trace_path.expanduser().resolve()}")
    print(f"Replay stopped cleanly. Trace: {trace_path.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by operator.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
