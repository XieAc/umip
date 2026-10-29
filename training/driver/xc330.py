"""Small, replay-safe XC330-M288-T driver.

The collection program keeps the motors unpowered so the operator can move the
gripper by hand.  Replay is different: it uses current-based position mode,
maps the recorded opening in metres to the paired encoder targets, and disables
both motors on every exit path.

The Dynamixel SDK is imported lazily.  This keeps offline episode checks and
unit tests usable on machines without the USB adapter attached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class XC330Error(RuntimeError):
    """Raised for a communication, configuration, or safety failure."""


def load_width_calibration(path: Path) -> list[dict[str, Any]]:
    """Load and validate the paired motor-to-opening calibration."""

    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise XC330Error(f"Cannot read gripper calibration: {path}") from exc
    except json.JSONDecodeError as exc:
        raise XC330Error(f"Invalid gripper calibration JSON: {path}") from exc

    if data.get("model") != "paired_difference_tick":
        raise XC330Error("Gripper calibration must use the paired_difference_tick model")
    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise XC330Error("Gripper calibration needs at least two samples")

    required = ("width_m", "m1", "m2")
    checked: list[dict[str, Any]] = []
    for index, row in enumerate(samples, 1):
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise XC330Error(f"Calibration sample {index} is missing width_m/m1/m2")
        try:
            values = {key: float(row[key]) for key in required}
        except (TypeError, ValueError) as exc:
            raise XC330Error(f"Calibration sample {index} is not numeric") from exc
        if not all(np.isfinite(value) for value in values.values()):
            raise XC330Error(f"Calibration sample {index} is not finite")
        if values["width_m"] < 0:
            raise XC330Error(f"Calibration sample {index} has negative width")
        checked.append({key: values[key] for key in required})

    checked.sort(key=lambda row: row["width_m"])
    widths = np.asarray([row["width_m"] for row in checked], dtype=np.float64)
    if np.any(np.diff(widths) <= 0):
        raise XC330Error("Gripper calibration widths must be strictly increasing")
    differences = np.asarray([row["m1"] - row["m2"] for row in checked], dtype=np.float64)
    if np.any(np.diff(differences) == 0):
        raise XC330Error("Gripper calibration paired encoder differences must be unique")
    delta = np.diff(differences)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise XC330Error("Gripper calibration paired encoder differences must be monotonic")
    return checked


class XC330Controller:
    """Control the two local XC330 motors using the paired width calibration.

    ``current_limit_raw`` is the measured present-current trip threshold.  The
    goal current used by current-based position mode defaults to the same value,
    while ``hold_current_raw`` is used only when the motor is holding the
    initial position.  All values are Dynamixel raw current units (approximately
    mA for the installed XC330 firmware).
    """

    ADDR_DRIVE_MODE = 10
    ADDR_OPERATING_MODE = 11
    ADDR_PWM_LIMIT = 36
    ADDR_CURRENT_LIMIT = 38
    ADDR_TORQUE = 64
    ADDR_GOAL_CURRENT = 102
    ADDR_PROFILE_ACCELERATION = 108
    ADDR_PROFILE_VELOCITY = 112
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_CURRENT = 126
    ADDR_PRESENT_POSITION = 132

    CURRENT_BASED_POSITION_MODE = 5
    EXTENDED_POSITION_MODE = 4

    def __init__(
        self,
        port_name: str,
        baudrate: int,
        calibration_path: Path,
        motor_ids: tuple[int, int],
        current_limit_raw: int,
        start_tolerance_ticks: int,
        *,
        goal_current_raw: int | None = None,
        hold_current_raw: int = 100,
        profile_velocity: int = 80,
        profile_acceleration: int = 40,
        current_based_position: bool = True,
    ) -> None:
        if len(motor_ids) != 2 or motor_ids[0] == motor_ids[1]:
            raise ValueError("XC330 replay requires two distinct motor IDs")
        if baudrate <= 0:
            raise ValueError("XC330 baudrate must be positive")
        if current_limit_raw <= 0 or hold_current_raw <= 0:
            raise ValueError("XC330 current limits must be positive")
        if start_tolerance_ticks < 0:
            raise ValueError("XC330 start tolerance must be nonnegative")

        try:
            from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler
        except ImportError as exc:  # pragma: no cover - exercised on deployment host
            raise XC330Error("Install dynamixel-sdk before enabling the gripper") from exc

        self._comm_success = COMM_SUCCESS
        self._group_sync_write_cls = GroupSyncWrite
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(2.0)
        self.ids = tuple(int(value) for value in motor_ids)
        self.current_limit_raw = int(current_limit_raw)
        self.goal_current_raw = int(goal_current_raw if goal_current_raw is not None else current_limit_raw)
        self.hold_current_raw = int(hold_current_raw)
        self.profile_velocity = int(profile_velocity)
        self.profile_acceleration = int(profile_acceleration)
        self.current_based_position = bool(current_based_position)
        self.start_tolerance_ticks = int(start_tolerance_ticks)
        self.torque_enabled = False
        self.last_targets: dict[int, int] = {}
        self.hardware_current_limits: dict[int, int] = {}
        self._goal_current_active = False
        ordered = load_width_calibration(calibration_path)
        self.widths = np.asarray([row["width_m"] for row in ordered], dtype=np.float64)
        self.m1 = np.asarray([row["m1"] for row in ordered], dtype=np.float64)
        self.m2 = np.asarray([row["m2"] for row in ordered], dtype=np.float64)
        if not self.port.openPort() or not self.port.setBaudRate(baudrate):
            try:
                self.port.closePort()
            except Exception:
                pass
            raise XC330Error(f"Cannot open Dynamixel port {port_name} at {baudrate} baud")

    def _check(self, result: int, error: int, action: str) -> None:
        if result != self._comm_success:
            raise XC330Error(f"{action}: {self.packet.getTxRxResult(result)}")
        if error:
            raise XC330Error(f"{action}: {self.packet.getRxPacketError(error)}")

    def _write1(self, motor_id: int, address: int, value: int, action: str) -> None:
        result, error = self.packet.write1ByteTxRx(self.port, motor_id, address, int(value))
        self._check(result, error, action)

    def _write2(self, motor_id: int, address: int, value: int, action: str) -> None:
        result, error = self.packet.write2ByteTxRx(self.port, motor_id, address, int(value))
        self._check(result, error, action)

    def _write4(self, motor_id: int, address: int, value: int, action: str) -> None:
        result, error = self.packet.write4ByteTxRx(self.port, motor_id, address, int(value) & 0xFFFFFFFF)
        self._check(result, error, action)

    def _read1(self, motor_id: int, address: int, action: str) -> int:
        value, result, error = self.packet.read1ByteTxRx(self.port, motor_id, address)
        self._check(result, error, action)
        return int(value)

    def _read2(self, motor_id: int, address: int, action: str) -> int:
        value, result, error = self.packet.read2ByteTxRx(self.port, motor_id, address)
        self._check(result, error, action)
        return int(value)

    def _read4_signed(self, motor_id: int, address: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(self.port, motor_id, address)
        self._check(result, error, f"read motor {motor_id}")
        return int(value) - (1 << 32) if value >= (1 << 31) else int(value)

    def _read2_signed(self, motor_id: int, address: int) -> int:
        value = self._read2(motor_id, address, f"read motor {motor_id}")
        return value - (1 << 16) if value >= (1 << 15) else value

    def width_to_targets(
        self, width_m: float, reference: dict[int, int] | None = None
    ) -> dict[int, int]:
        width_m = float(width_m)
        if not np.isfinite(width_m) or not self.widths[0] <= width_m <= self.widths[-1]:
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
                targets[motor_id] += 4096 * round((int(reference[motor_id]) - targets[motor_id]) / 4096)
        return targets

    def _disable_all(self) -> None:
        errors: list[str] = []
        for motor_id in self.ids:
            try:
                self._write1(motor_id, self.ADDR_TORQUE, 0, f"disable torque motor {motor_id}")
            except Exception as exc:
                errors.append(str(exc))
        self.torque_enabled = False
        if errors:
            raise XC330Error("; ".join(errors))

    def initialize(self, initial_width_m: float) -> None:
        current = {motor_id: self._read4_signed(motor_id, self.ADDR_PRESENT_POSITION) for motor_id in self.ids}
        targets = self.width_to_targets(initial_width_m, reference=current)
        errors = {motor_id: abs(current[motor_id] - targets[motor_id]) for motor_id in self.ids}
        if max(errors.values()) > self.start_tolerance_ticks:
            raise XC330Error(
                f"Gripper is not near the recorded start width: current={current}, "
                f"target={targets}, errors={errors}, limit={self.start_tolerance_ticks} ticks"
            )

        # Always disable both motors before changing operating mode.  If any
        # write fails, close() will attempt to disable both again.
        self.torque_enabled = True
        current_based_position = getattr(self, "current_based_position", False)
        profile_acceleration = getattr(self, "profile_acceleration", 40)
        profile_velocity = getattr(self, "profile_velocity", 80)
        goal_current_raw = getattr(self, "goal_current_raw", 600)
        try:
            for motor_id in self.ids:
                self._write1(motor_id, self.ADDR_TORQUE, 0, f"disable torque motor {motor_id}")
            for motor_id in self.ids:
                if current_based_position:
                    drive_mode = self._read1(motor_id, self.ADDR_DRIVE_MODE, f"read drive mode motor {motor_id}")
                    pwm_limit = self._read2(motor_id, self.ADDR_PWM_LIMIT, f"read PWM limit motor {motor_id}")
                    hardware_current_limit = self._read2(
                        motor_id, self.ADDR_CURRENT_LIMIT, f"read current limit motor {motor_id}"
                    )
                    if not 0 <= pwm_limit <= 885:
                        raise XC330Error(f"Invalid PWM limit for motor {motor_id}: {pwm_limit}")
                    if not 0 < hardware_current_limit <= 2352:
                        raise XC330Error(
                            f"Invalid current limit for motor {motor_id}: {hardware_current_limit}"
                        )
                    if drive_mode & 0x0C:
                        self._write1(
                            motor_id,
                            self.ADDR_DRIVE_MODE,
                            drive_mode & ~0x0C,
                            f"set velocity profile mode motor {motor_id}",
                        )
                    self._write1(
                        motor_id,
                        self.ADDR_OPERATING_MODE,
                        self.CURRENT_BASED_POSITION_MODE,
                        f"set current-based position mode motor {motor_id}",
                    )
                    self.hardware_current_limits[motor_id] = hardware_current_limit
                    self._write2(
                        motor_id,
                        self.ADDR_PWM_LIMIT,
                        pwm_limit,
                        f"restore PWM limit motor {motor_id}",
                    )
                    self._write2(
                        motor_id,
                        self.ADDR_GOAL_CURRENT,
                        min(getattr(self, "hold_current_raw", 100), hardware_current_limit),
                        f"set hold current motor {motor_id}",
                    )
                else:
                    self._write1(
                        motor_id,
                        self.ADDR_OPERATING_MODE,
                        self.EXTENDED_POSITION_MODE,
                        f"set extended position mode motor {motor_id}",
                    )
                self._write4(motor_id, self.ADDR_PROFILE_ACCELERATION, profile_acceleration,
                             f"set acceleration motor {motor_id}")
                self._write4(motor_id, self.ADDR_PROFILE_VELOCITY, profile_velocity,
                             f"set velocity motor {motor_id}")
                self._write4(motor_id, self.ADDR_GOAL_POSITION, current[motor_id],
                             f"hold motor {motor_id}")
                self._write1(motor_id, self.ADDR_TORQUE, 1, f"enable torque motor {motor_id}")
            self.last_targets = dict(targets)
            self._goal_current_active = False
            self.torque_enabled = True
        except Exception:
            # Keep the flag set so the caller's finally/close path disables
            # every motor even if only the first one was configured.
            self.torque_enabled = True
            raise

    def command_width(self, width_m: float, check_current: bool) -> None:
        if not self.torque_enabled:
            raise XC330Error("Cannot command XC330 before initialize()")
        if getattr(self, "current_based_position", False) and not getattr(self, "_goal_current_active", False):
            for motor_id in self.ids:
                hardware_limit = self.hardware_current_limits.get(motor_id, 2352)
                self._write2(
                    motor_id,
                    self.ADDR_GOAL_CURRENT,
                    min(getattr(self, "goal_current_raw", 600), hardware_limit),
                    f"set motion current motor {motor_id}",
                )
            self._goal_current_active = True
        targets = self.width_to_targets(width_m, reference=self.last_targets)
        writer = self._group_sync_write_cls(self.port, self.packet, self.ADDR_GOAL_POSITION, 4)
        try:
            for motor_id, target in targets.items():
                encoded = int(target & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
                if not writer.addParam(motor_id, list(encoded)):
                    raise XC330Error(f"Cannot add sync-write parameter for motor {motor_id}")
            result = writer.txPacket()
            self._check(result, 0, "Dynamixel sync write")
        finally:
            writer.clearParam()
        self.last_targets = dict(targets)
        if check_current:
            currents = {motor_id: self._read2_signed(motor_id, self.ADDR_PRESENT_CURRENT) for motor_id in self.ids}
            if max(abs(value) for value in currents.values()) >= self.current_limit_raw:
                raise XC330Error(f"Gripper current protection triggered: {currents}")

    def close(self) -> None:
        if self.torque_enabled:
            for motor_id in self.ids:
                try:
                    self._write1(motor_id, self.ADDR_TORQUE, 0, f"disable torque motor {motor_id}")
                except Exception:
                    pass
        self.torque_enabled = False
        try:
            self.port.closePort()
        except Exception:
            pass
