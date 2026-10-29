"""Open/close (switch) gripper for policy deployment.

Grip and release reuse ``driver/gripper_torque_control.py`` unchanged: velocity
mode, both motors driven closed until they stall for ``STALL_TIME``, then held
with torque on; release reverses for 1 s and turns torque off.  The policy's
predicted width only decides when to switch, with hysteresis:

    open   -> grip    when the predicted width < close_below_m
    closed -> release when the predicted width > open_above_m

Widths are still read from the encoders with the paired calibration, so the
policy keeps seeing the measured opening.  ``command_width`` is non-blocking
and is called at the observation rate (about 10 Hz).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from driver.xc330 import XC330Controller, XC330Error

RELEASE_DURATION_S = 1.0     # same as gripper_torque_control.release()
RELEASE_TORQUE_OFF_S = 0.1   # pause before torque off, as in release()


class XC330SwitchGripper(XC330Controller):
    def __init__(self, port_name: str, baudrate: int, calibration_path: Path,
                 motor_ids: tuple[int, int], current_limit_raw: int, start_tolerance_ticks: int,
                 *, close_below_m: float, open_above_m: float, close_below_at_floor_m: float | None = None,
                 clock=time.monotonic) -> None:
        if not 0.0 < open_above_m < close_below_m <= 0.1:
            raise ValueError("Switch gripper needs 0 < open_above_m < close_below_m <= 0.1 m")
        if close_below_at_floor_m is not None and not close_below_m <= close_below_at_floor_m <= 0.1:
            raise ValueError("close_below_at_floor_m must lie between close_below_m and 0.1 m")
        from driver import gripper_torque_control as gtc
        super().__init__(port_name, baudrate, calibration_path, motor_ids,
                         current_limit_raw, start_tolerance_ticks, current_based_position=False)
        if set(self.ids) != set(gtc.MOTOR_DIRECTION):
            raise XC330Error(f"gripper_torque_control has directions for {sorted(gtc.MOTOR_DIRECTION)}, "
                             f"not motor IDs {self.ids}")
        self.gtc = gtc
        self.close_below_m = float(close_below_m)
        self.open_above_m = float(open_above_m)
        # Looser grip threshold while the arm rests on the soft floor: there the
        # policy predicts 80-93 mm but cannot descend further (2026-10-29).
        self.close_below_at_floor_m = None if close_below_at_floor_m is None else float(close_below_at_floor_m)
        self.at_floor_until = float("-inf")
        self.clock = clock
        self.state = "uninitialized"
        self.state_since = 0.0
        self.stall_since: float | None = None
        self.events: list[dict[str, Any]] = []

    # -- helpers wrapping gripper_torque_control (it prints and returns False on errors)
    def _ok(self, ok: bool, action: str) -> None:
        if not ok:
            raise XC330Error(f"XC330 {action} failed")

    def _set_velocities(self, sign: int, speed: int, action: str) -> None:
        for motor_id in self.ids:
            self._ok(self.gtc.set_velocity(self.packet, self.port, motor_id,
                                           sign * self.gtc.MOTOR_DIRECTION[motor_id] * speed), action)

    def _torque(self, enable: bool) -> None:
        value = self.gtc.TORQUE_ENABLE if enable else self.gtc.TORQUE_DISABLE
        for motor_id in self.ids:
            self._ok(self.gtc.write_torque(self.packet, self.port, motor_id, value),
                     f"torque {'on' if enable else 'off'} motor {motor_id}")
        self.torque_enabled = bool(enable) or self.torque_enabled

    def _enter(self, state: str, now: float, width_m: float, note: str = "") -> None:
        self.state, self.state_since, self.stall_since = state, now, None
        self.events.append({"monotonic_s": now, "state": state, "predicted_width_m": float(width_m), "note": note})
        print(f"[gripper] {state} (predicted width {width_m*1000:.1f} mm{', ' + note if note else ''})", flush=True)

    ADDR_PRESENT_VELOCITY = 128

    def _both_stalled(self) -> bool:
        # Same test as gripper_torque_control.grip() (|velocity| <= VELOCITY_THRESHOLD
        # on both motors), but read only the velocity: the SDK busy-waits on the
        # serial port holding the GIL, and extra reads slowed inference to a stop.
        for motor_id in self.ids:
            try:
                velocity = self._read4_signed(motor_id, self.ADDR_PRESENT_VELOCITY)
            except XC330Error:
                return False
            if abs(velocity) > self.gtc.VELOCITY_THRESHOLD:
                return False
        return True

    AT_FLOOR_HOLD_S = 0.5   # floor contact counts this long (the floor flag flickers step to step)

    def mark_at_floor(self) -> None:
        """Called by the servo loop whenever the soft floor clamps the arm."""
        self.at_floor_until = self.clock() + self.AT_FLOOR_HOLD_S

    def grip_threshold(self) -> tuple[float, bool]:
        at_floor = self.close_below_at_floor_m is not None and self.clock() < self.at_floor_until
        return (self.close_below_at_floor_m if at_floor else self.close_below_m), at_floor

    BUSY_STATES = ("closing", "releasing", "release_settling")

    def is_busy(self) -> bool:
        """True while gripping or releasing (the arm should hold still)."""
        return self.state in self.BUSY_STATES

    # -- interface used by test_model_inference.py
    def initialize(self, initial_width_m: float) -> None:
        current = {motor_id: self._read4_signed(motor_id, self.ADDR_PRESENT_POSITION) for motor_id in self.ids}
        targets = self.width_to_targets(initial_width_m, reference=current)
        errors = {motor_id: abs(current[motor_id] - targets[motor_id]) for motor_id in self.ids}
        if max(errors.values()) > self.start_tolerance_ticks:
            raise XC330Error(f"Gripper is not near the recorded start width: current={current}, "
                             f"target={targets}, errors={errors}, limit={self.start_tolerance_ticks} ticks")
        self.torque_enabled = True   # lets close() disable torque if setup fails part-way
        for motor_id in self.ids:
            self._ok(self.gtc.set_velocity_mode(self.packet, self.port, motor_id),
                     f"set velocity mode motor {motor_id}")
        self._torque(True)
        self._set_velocities(1, 0, "hold open")   # like gripper_torque_control.main(): torque on, still
        self._enter("open", self.clock(), initial_width_m, "velocity mode")

    def command_width(self, width_m: float | None, check_current: bool = True) -> None:
        """``width_m=None`` advances a grip/release already in progress but makes
        no open/close decision (the arm is holding and has no fresh prediction)."""
        if not self.torque_enabled and self.state != "open":
            raise XC330Error("Cannot command XC330 before initialize()")
        now = self.clock()
        decide = width_m is not None
        width_m = float("nan") if width_m is None else float(width_m)
        if self.state in ("open", "closed") and not decide:
            return
        if self.state == "open":
            threshold, at_floor = self.grip_threshold()
            if width_m < threshold:
                self._torque(True)                      # release() leaves torque off
                self._set_velocities(1, self.gtc.GRIP_SPEED, "grip")
                self._enter("closing", now, width_m,
                            f"at floor, threshold {threshold*1000:.0f} mm" if at_floor else "")
        elif self.state == "closing":
            if now - self.state_since > self.gtc.MAX_GRIP_TIME:
                self._set_velocities(1, 0, "hold")
                self._enter("closed", now, width_m, "max grip time")
            elif self._both_stalled():
                if self.stall_since is None:
                    self.stall_since = now
                elif now - self.stall_since >= self.gtc.STALL_TIME:
                    self._set_velocities(1, 0, "hold")
                    self._enter("closed", now, width_m, "both motors stalled")
            else:
                self.stall_since = None
        elif self.state == "closed":
            if width_m > self.open_above_m:
                self._set_velocities(-1, self.gtc.RELEASE_SPEED, "release")
                self._enter("releasing", now, width_m)
        elif self.state == "releasing":
            if now - self.state_since >= RELEASE_DURATION_S:
                self._set_velocities(1, 0, "stop release")
                self._enter("release_settling", now, width_m)
        elif self.state == "release_settling":
            if now - self.state_since >= RELEASE_TORQUE_OFF_S:
                self._torque(False)
                self.torque_enabled = False
                self._enter("open", now, width_m, "torque off")
        else:
            raise XC330Error("Cannot command XC330 before initialize()")

    def close(self) -> None:
        try:
            if self.torque_enabled:
                for motor_id in self.ids:
                    try:
                        self.gtc.set_velocity(self.packet, self.port, motor_id, 0)
                    except Exception:
                        pass
        finally:
            super().close()
