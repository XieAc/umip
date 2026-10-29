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
from xac_config import WIDTH_CALIBRATION_PATH, DXL_BAUDRATE, DXL_PORT
from calibrate_gripper_width import validate_samples


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


def anchor_trajectory(trajectory: Trajectory, base_start_pose: np.ndarray) -> np.ndarray:
    """Apply T_B_G(t) = T_B_G_start @ inv(T_W_G_start) @ T_W_G(t)."""

    base_start_pose = np.asarray(base_start_pose, dtype=np.float64)
    if base_start_pose.shape != (6,) or not np.all(np.isfinite(base_start_pose)):
        raise ValueError("Anchor must contain six finite values in metres/radians")
    world = poses_to_transforms(trajectory.positions_m, trajectory.rotations_rotvec)
    base_start = pose_to_transform(np.asarray(base_start_pose, dtype=np.float64))
    world_start_inverse = np.linalg.inv(world[0])
    targets = base_start[None, :, :] @ world_start_inverse[None, :, :] @ world
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
        "target_xyz_min_m": np.min(target_poses[:, :3], axis=0).tolist(),
        "target_xyz_max_m": np.max(target_poses[:, :3], axis=0).tolist(),
        "start_target_pose": target_poses[0].tolist(),
        "end_target_pose": target_poses[-1].tolist(),
    }


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
        violations.append(f"minimum base Z {metrics['target_xyz_min_m'][2]:.4f} m < {min_z_m:.4f} m")
    if metrics["target_xyz_max_m"][2] > max_z_m:
        violations.append(f"maximum base Z {metrics['target_xyz_max_m'][2]:.4f} m > {max_z_m:.4f} m")
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


class XC330Controller:
    ADDR_OPERATING_MODE = 11
    ADDR_TORQUE = 64
    ADDR_PROFILE_ACCELERATION = 108
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_CURRENT = 126
    ADDR_PRESENT_POSITION = 132
    EXTENDED_POSITION_MODE = 4

    def __init__(
        self,
        port_name: str,
        baudrate: int,
        calibration_path: Path,
        motor_ids: tuple[int, int],
        current_limit_raw: int,
        start_tolerance_ticks: int,
    ) -> None:
        try:
            from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler
        except ImportError as exc:
            raise RuntimeError("Install the gripper dependency: pip install dynamixel-sdk") from exc
        self._comm_success = COMM_SUCCESS
        self._group_sync_write_cls = GroupSyncWrite
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(2.0)
        self.ids = motor_ids
        self.current_limit_raw = current_limit_raw
        self.start_tolerance_ticks = start_tolerance_ticks
        self.torque_enabled = False
        ordered = load_replay_calibration(calibration_path)
        self.widths = np.asarray([float(row["width_m"]) for row in ordered])
        self.m1 = np.asarray([float(row["m1"]) for row in ordered])
        self.m2 = np.asarray([float(row["m2"]) for row in ordered])
        if np.any(np.diff(self.widths) <= 0):
            raise ValueError("Gripper calibration widths must be strictly increasing")
        if not self.port.openPort() or not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise RuntimeError(f"Cannot open Dynamixel port {port_name} at {baudrate} baud")

    def _check(self, result: int, error: int, action: str) -> None:
        if result != self._comm_success:
            raise RuntimeError(f"{action}: {self.packet.getTxRxResult(result)}")
        if error:
            raise RuntimeError(f"{action}: {self.packet.getRxPacketError(error)}")

    def _write1(self, motor_id: int, address: int, value: int, action: str) -> None:
        result, error = self.packet.write1ByteTxRx(self.port, motor_id, address, value)
        self._check(result, error, action)

    def _write4(self, motor_id: int, address: int, value: int, action: str) -> None:
        result, error = self.packet.write4ByteTxRx(self.port, motor_id, address, value)
        self._check(result, error, action)

    def _read4_signed(self, motor_id: int, address: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(self.port, motor_id, address)
        self._check(result, error, f"read motor {motor_id}")
        return value - (1 << 32) if value >= (1 << 31) else value

    def _read2_signed(self, motor_id: int, address: int) -> int:
        value, result, error = self.packet.read2ByteTxRx(self.port, motor_id, address)
        self._check(result, error, f"read current motor {motor_id}")
        return value - (1 << 16) if value >= (1 << 15) else value

    def width_to_targets(
        self, width_m: float, reference: dict[int, int] | None = None
    ) -> dict[int, int]:
        if not self.widths[0] <= width_m <= self.widths[-1]:
            raise ValueError(
                f"Gripper width {width_m:.5f} m is outside calibration range "
                f"[{self.widths[0]:.5f}, {self.widths[-1]:.5f}] m"
            )
        targets = {
            self.ids[0]: int(round(np.interp(width_m, self.widths, self.m1))),
            self.ids[1]: int(round(np.interp(width_m, self.widths, self.m2))),
        }
        if reference is not None:
            for motor_id in self.ids:
                targets[motor_id] += 4096 * round((reference[motor_id] - targets[motor_id]) / 4096)
        return targets

    def initialize(self, initial_width_m: float) -> None:
        current = {motor_id: self._read4_signed(motor_id, self.ADDR_PRESENT_POSITION) for motor_id in self.ids}
        targets = self.width_to_targets(initial_width_m, reference=current)
        errors = {motor_id: abs(current[motor_id] - targets[motor_id]) for motor_id in self.ids}
        if max(errors.values()) > self.start_tolerance_ticks:
            raise RuntimeError(
                f"Gripper is not near the recorded start width: current={current}, "
                f"target={targets}, errors={errors}, limit={self.start_tolerance_ticks} ticks"
            )
        # Cleanup must disable both motors even if initialization fails halfway.
        self.torque_enabled = True
        for motor_id in self.ids:
            self._write1(motor_id, self.ADDR_TORQUE, 0, f"disable torque motor {motor_id}")
            self._write1(
                motor_id,
                self.ADDR_OPERATING_MODE,
                self.EXTENDED_POSITION_MODE,
                f"set mode motor {motor_id}",
            )
            self._write4(motor_id, self.ADDR_PROFILE_ACCELERATION, 40, f"set acceleration motor {motor_id}")
            self._write4(motor_id, self.ADDR_PROFILE_ACCELERATION + 4, 80, f"set velocity motor {motor_id}")
            self._write4(motor_id, self.ADDR_GOAL_POSITION, current[motor_id], f"hold motor {motor_id}")
            self._write1(motor_id, self.ADDR_TORQUE, 1, f"enable torque motor {motor_id}")
        self.torque_enabled = True
        self.last_targets = dict(targets)

    def command_width(self, width_m: float, check_current: bool) -> None:
        targets = self.width_to_targets(width_m, reference=self.last_targets)
        writer = self._group_sync_write_cls(self.port, self.packet, self.ADDR_GOAL_POSITION, 4)
        for motor_id, target in targets.items():
            encoded = int(target & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
            if not writer.addParam(motor_id, list(encoded)):
                writer.clearParam()
                raise RuntimeError(f"Cannot add Dynamixel sync-write parameter for motor {motor_id}")
        result = writer.txPacket()
        writer.clearParam()
        if result != self._comm_success:
            raise RuntimeError(f"Dynamixel sync write failed: {self.packet.getTxRxResult(result)}")
        self.last_targets = dict(targets)
        if check_current:
            currents = {motor_id: self._read2_signed(motor_id, self.ADDR_PRESENT_CURRENT) for motor_id in self.ids}
            if max(abs(value) for value in currents.values()) >= self.current_limit_raw:
                raise RuntimeError(f"Gripper current protection triggered: {currents}")

    def close(self) -> None:
        if self.torque_enabled:
            for motor_id in self.ids:
                try:
                    self._write1(motor_id, self.ADDR_TORQUE, 0, f"disable torque motor {motor_id}")
                except Exception:
                    pass
        try:
            self.port.closePort()
        except Exception:
            pass


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
) -> list[dict[str, Any]]:
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("Install the UR dependency: pip install ur-rtde") from exc

    control = None
    receiver = None
    gripper = None
    trace: list[dict[str, Any]] = []
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        control = RTDEControlInterface(args.robot_ip, args.control_frequency_hz)
        receiver = RTDEReceiveInterface(args.robot_ip, args.control_frequency_hz)
        actual_tcp = np.asarray(control.getTCPOffset(), dtype=np.float64)
        if not np.allclose(actual_tcp, args.tcp_offset, atol=1e-5, rtol=0):
            raise RuntimeError("Active UR TCP offset differs from --tcp-offset; configure the measured flange-to-G TCP on the pendant")
        actual_start = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
        if actual_start.shape != (6,) or not np.all(np.isfinite(actual_start)):
            raise RuntimeError("UR5 returned an invalid TCP pose")
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
        while True:
            if stop_requested:
                raise KeyboardInterrupt
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
            ok = control.servoL(
                target.tolist(),
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
                        "gripper_width_target_m": width,
                    }
                )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--robot-ip", help="UR5 IP; required for live motion or reading a live dry-run anchor")
    parser.add_argument(
        "--anchor-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Dry-run anchor pose in UR base coordinates; uses live UR TCP when omitted with --robot-ip",
    )
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--tcp-offset", type=float, nargs=6,
                        help="Measured flange-to-G TCP [x y z rx ry rz], m/rad; required for live replay")
    parser.add_argument("--speed-scale", type=float, default=0.1)
    parser.add_argument("--max-duration-s", type=float)
    parser.add_argument("--max-distance-from-start-m", type=float, default=0.12)
    parser.add_argument("--minimum-base-z-m", type=float)
    parser.add_argument("--maximum-base-z-m", type=float)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=0.5)
    parser.add_argument("--max-source-gap-s", type=float, default=0.25)
    parser.add_argument("--control-frequency-hz", type=float, default=125.0)
    parser.add_argument("--lookahead-time-s", type=float, default=0.1)
    parser.add_argument("--servo-gain", type=float, default=300.0)
    parser.add_argument("--start-pose-tolerance-m", type=float, default=0.002)
    parser.add_argument("--start-rotation-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--runtime-position-margin-m", type=float, default=0.005)
    parser.add_argument("--max-following-error-m", type=float, default=0.03)
    parser.add_argument("--max-rotation-following-error-rad", type=float, default=0.3)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--gripper-port", default=DXL_PORT)
    parser.add_argument("--gripper-baudrate", type=int, default=DXL_BAUDRATE)
    parser.add_argument("--gripper-ids", type=int, nargs=2, default=(1, 2))
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=WIDTH_CALIBRATION_PATH,
    )
    parser.add_argument("--gripper-current-limit-raw", type=int, default=350)
    parser.add_argument("--gripper-start-tolerance-ticks", type=int, default=80)
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
    if args.minimum_base_z_m is not None and args.maximum_base_z_m is not None:
        if args.minimum_base_z_m >= args.maximum_base_z_m:
            raise ValueError("--minimum-base-z-m must be smaller than --maximum-base-z-m")
    if args.enable_gripper and not args.enable_motion:
        raise ValueError("--enable-gripper requires --enable-motion")
    if args.enable_motion:
        missing = []
        if not args.robot_ip:
            missing.append("--robot-ip")
        if args.tcp_offset is None:
            missing.append("--tcp-offset")
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
    if args.enable_gripper:
        validate_gripper_trajectory(trajectory, args.gripper_calibration)
    if args.anchor_pose is not None:
        anchor = np.asarray(args.anchor_pose, dtype=np.float64)
        anchor_source = "command_line"
    elif args.robot_ip:
        anchor = read_robot_pose(args.robot_ip)
        anchor_source = "live_ur5_tcp"
    else:
        anchor = np.concatenate([trajectory.positions_m[0], trajectory.rotations_rotvec[0]])
        anchor_source = "recorded_start_for_relative_only_dry_run"

    targets = anchor_trajectory(trajectory, anchor)
    metrics = trajectory_metrics(trajectory, targets, args.speed_scale)
    metrics.update(
        {
            "episode": str(args.episode.expanduser().resolve()),
            "anchor_source": anchor_source,
            "motion_enabled": bool(args.enable_motion),
            "gripper_enabled": bool(args.enable_gripper),
            "speed_scale": float(args.speed_scale),
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

    trace = execute_replay(args, trajectory, targets)
    trace_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "samples": trace,
    }
    trace_path = args.trace or Path(f"replay_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    trace_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    trace_path.expanduser().resolve().write_text(
        json.dumps(trace_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
