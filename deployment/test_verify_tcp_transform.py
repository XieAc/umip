import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_robot_tcp import transform
from replay.verify_tcp_transform import (
    active_from_gripper, gripper_from_active, load_calibration, main, path_segment,
    pose, run_translation_leg,
)


class TransformVerificationTest(unittest.TestCase):
    def setUp(self):
        self.fg, self.gp = load_calibration(Path(__file__).parent / "calibration/tcp_gripper_01.json")
        self.bg = transform([0.3, -0.5, 0.4, 0.6, -0.9, 0.2])

    def test_active_tcp_does_not_double_apply_offset(self):
        for fh in (np.eye(4), self.fg, transform([0.02, -0.01, 0.07, 0.3, -0.4, 0.2])):
            bh = active_from_gripper(self.bg, self.fg, fh)
            np.testing.assert_allclose(gripper_from_active(bh, self.fg, fh), self.bg, atol=1e-12)
        np.testing.assert_allclose(active_from_gripper(self.bg, self.fg, self.fg), self.bg, atol=1e-12)

    def test_translation_follows_gripper_axis_in_base_coordinates(self):
        for axis, index in zip("xyz", range(3)):
            at, _, _ = path_segment(self.bg, self.fg, np.eye(4), self.gp, "translation", axis, .003, .02, "gripper", .001, .01)
            target_g, target_f = at(1)
            np.testing.assert_allclose(target_g[:3, 3]-self.bg[:3, 3], .003*self.bg[:3, index], atol=1e-12)
            np.testing.assert_allclose(target_f @ self.fg, target_g, atol=1e-12)

    def test_rotation_preserves_pivot_and_compensates_flange_translation(self):
        for pivot, gc in (("gripper", np.eye(4)), ("probe", self.gp)):
            start_point = (self.bg @ gc)[:3, 3]
            for fh in (np.eye(4), self.fg):
                at, _, _ = path_segment(self.bg, self.fg, fh, self.gp, "rotation", "y", .003,
                    np.deg2rad(2), pivot, .001, np.deg2rad(.5))
                for s in np.linspace(0, 1, 11):
                    g, h = at(s)
                    np.testing.assert_allclose((g @ gc)[:3, 3], start_point, atol=1e-12)
                    np.testing.assert_allclose(h @ np.linalg.inv(fh) @ self.fg, g, atol=1e-12)
            if pivot == "gripper":
                tip_change = np.linalg.norm((at(1)[0] @ self.gp)[:3, 3]-(self.bg @ self.gp)[:3, 3])
                self.assertAlmostEqual(tip_change, 2*.08*np.sin(np.deg2rad(1)), places=12)

    def test_smooth_trajectory_respects_target_speed_limits(self):
        # Arbitrary active TCP and longer probe catch an underestimated rotation lever arm.
        fh = transform([-.3, .2, .1, .4, .2, -.5])
        gp = transform([0.08, 0, .4, 0, 0, 0])
        speed, omega = .001, np.deg2rad(.5)
        for pivot in ("gripper", "probe"):
            at, duration, _ = path_segment(self.bg, self.fg, fh, gp, "rotation", "y", .003,
                np.deg2rad(2), pivot, speed, omega)
            previous = None
            for t in np.linspace(0, 1, 2001):
                g, h = at(t*t*(3-2*t))
                f = g @ np.linalg.inv(self.fg)
                matrices = (g, h, f, g @ gp)
                if previous:
                    for current, old in zip(matrices, previous):
                        self.assertLessEqual(np.linalg.norm(current[:3, 3]-old[:3, 3])/(duration/2000), speed*1.00001)
                    angle = Rotation.from_matrix(previous[0][:3, :3].T @ g[:3, :3]).magnitude()
                    self.assertLessEqual(angle/(duration/2000), omega*1.00001)
                previous = matrices

    def test_default_preview_never_opens_control_interface(self):
        state = {"tcp_offset_m_rad": [0.]*6, "actual_tcp_pose_m_rad": [.3, -.5, .4, 0, 0, 0]}
        with patch("sys.argv", ["verify_tcp_transform.py"]), \
             patch("replay.verify_tcp_transform.read_controller", return_value=state), \
             patch("rtde_control.RTDEControlInterface", side_effect=AssertionError("preview uploaded script")), \
             patch("builtins.print"):
            self.assertEqual(main(), 0)

    def translation_fixture(self, follow=True):
        fh = np.eye(4)
        at, _, _ = path_segment(self.bg, self.fg, fh, self.gp, "translation", "x", .003, .02, "gripper", .001, .01)
        clock = {"now": 0., "actual": active_from_gripper(self.bg, self.fg, fh), "target": None, "speed": .001}
        receiver = MagicMock()
        receiver.isConnected.return_value = True
        receiver.isProtectiveStopped.return_value = False
        receiver.isEmergencyStopped.return_value = False
        receiver.getTimestamp.side_effect = lambda: clock["now"]
        receiver.getActualTCPPose.side_effect = lambda: pose(clock["actual"]).tolist()
        receiver.getActualQd.return_value = [0.] * 6
        receiver.getSpeedScaling.return_value = 0.

        def velocity():
            if clock["target"] is None or not follow:
                return [0.] * 6
            vector = clock["target"][:3, 3]-clock["actual"][:3, 3]
            distance = np.linalg.norm(vector)
            return np.r_[vector/distance*clock["speed"], [0., 0., 0.]].tolist() if distance > 1e-9 else [0.] * 6

        def advance(dt):
            clock["now"] += dt
            if clock["target"] is not None and follow:
                vector = clock["target"][:3, 3]-clock["actual"][:3, 3]
                distance = np.linalg.norm(vector)
                if distance > 1e-9:
                    clock["actual"][:3, 3] += vector/distance*min(distance, clock["speed"]*dt)

        def move(target, speed, acceleration, asynchronous):
            clock["target"] = transform(target)
            clock["speed"] = speed
            return True

        receiver.getActualTCPSpeed.side_effect = velocity
        control = MagicMock()
        control.isConnected.return_value = True
        control.isProgramRunning.return_value = True
        control.moveL.side_effect = move
        return receiver, control, at, clock, advance

    def test_moveL_allows_zero_startup_scaling_and_requires_actual_arrival(self):
        receiver, control, at, clock, advance = self.translation_fixture()
        with patch("replay.verify_tcp_transform.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("replay.verify_tcp_transform.time.sleep", side_effect=advance), patch("builtins.print"):
            actual = run_translation_leg(receiver, control, np.eye(4), self.fg, at, .001)
        np.testing.assert_allclose(actual, at(1)[0], atol=1e-10)
        args, kwargs = control.moveL.call_args
        np.testing.assert_allclose(transform(args[0]), at(1)[1], atol=1e-10)
        self.assertEqual(args[1:3], (.001, .01))
        self.assertEqual(kwargs, {"asynchronous": True})
        control.servoL.assert_not_called()

    def test_moveL_command_accepted_without_motion_times_out_and_stops(self):
        receiver, control, at, clock, advance = self.translation_fixture(follow=False)
        with patch("replay.verify_tcp_transform.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("replay.verify_tcp_transform.time.sleep", side_effect=advance), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "moveL 超时"):
                run_translation_leg(receiver, control, np.eye(4), self.fg, at, .001)
        control.stopL.assert_called_once_with(.1)


if __name__ == "__main__":
    unittest.main()
