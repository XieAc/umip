"""Offline deployment regression tests; never connect to robot hardware."""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from replay_xac_ur5 import (
    Trajectory, XC330Controller, anchor_trajectory, build_parser,
    execute_replay, validate_arguments, validate_gripper_trajectory,
)


class DeploymentTest(unittest.TestCase):
    def trajectory(self):
        return Trajectory(np.array([0., .1, .2]),
                          np.array([[0., 0., .5], [.001, 0., .5], [.002, 0., .5]]),
                          np.zeros((3, 3)), np.array([.02, .025, .03]), np.arange(3))

    def args(self, *extra):
        return build_parser().parse_args(["--episode", "unused", *extra])

    def test_nonfinite_and_invalid_limits_fail(self):
        for option, value in (("--minimum-base-z-m", "nan"),
                              ("--max-linear-speed-m-s", "inf"),
                              ("--max-following-error-m", "0"),
                              ("--lookahead-time-s", ".01")):
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_arguments(self.args(option, value))
        with self.assertRaises(ValueError):
            anchor_trajectory(self.trajectory(), np.array([0, 0, 0, 0, 0, np.nan]))

    def test_motion_requires_measured_tcp(self):
        args = self.args("--enable-motion", "--robot-ip", "unused",
                         "--minimum-base-z-m", ".1", "--maximum-base-z-m", "1")
        with self.assertRaisesRegex(ValueError, "tcp-offset"):
            validate_arguments(args)

    def test_entire_gripper_trajectory_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "width.json"
            samples = [{"m1": 100, "m2": 200, "width_m": .02},
                       {"m1": 200, "m2": 100, "width_m": .03}]
            path.write_text(json.dumps({"model": "paired_difference_tick", "samples": samples}))
            trajectory = self.trajectory()
            validate_gripper_trajectory(trajectory, path)
            trajectory.gripper_width_m[-1] = .04
            with self.assertRaisesRegex(ValueError, "range"):
                validate_gripper_trajectory(trajectory, path)
            samples[1]["m1"] = float("nan")
            path.write_text(json.dumps({"model": "paired_difference_tick", "samples": samples}))
            with self.assertRaises(ValueError):
                validate_gripper_trajectory(self.trajectory(), path)

    def test_tcp_mismatch_stops_before_servo(self):
        control, receiver = Mock(), Mock()
        control.getTCPOffset.return_value = [.1, 0, 0, 0, 0, 0]
        modules = {
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=Mock(return_value=control)),
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receiver)),
        }
        args = self.args("--robot-ip", "unused", "--tcp-offset", "0", "0", ".2", "0", "0", "0")
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "TCP offset"):
            execute_replay(args, self.trajectory(), np.zeros((3, 6)))
        control.servoL.assert_not_called()
        control.servoStop.assert_called_once()
        control.disconnect.assert_called_once()
        receiver.disconnect.assert_called_once()

    def test_partial_gripper_initialization_disables_both_motors(self):
        gripper = XC330Controller.__new__(XC330Controller)
        gripper.ids = (1, 2)
        gripper.torque_enabled = False
        gripper.start_tolerance_ticks = 80
        gripper.port = Mock()
        gripper._read4_signed = Mock(return_value=100)
        gripper.width_to_targets = Mock(return_value={1: 100, 2: 100})
        gripper._write4 = Mock()
        gripper._write1 = Mock(side_effect=[None, None, None, RuntimeError("motor 2 failed")])
        with self.assertRaises(RuntimeError):
            gripper.initialize(.02)
        gripper._write1 = Mock()
        gripper.close()
        self.assertEqual([call.args[:3] for call in gripper._write1.call_args_list],
                         [(1, 64, 0), (2, 64, 0)])
        gripper.port.closePort.assert_called_once()

    def test_portable_dry_run_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "episode" / "data"
            data.mkdir(parents=True)
            trajectory = self.trajectory()
            arrays = {"d435i_timestamps": trajectory.timestamps_s,
                      "robot0_eef_pos": trajectory.positions_m,
                      "robot0_eef_rot_axis_angle": trajectory.rotations_rotvec,
                      "robot0_gripper_width": trajectory.gripper_width_m,
                      "valid_mask": np.ones(3, dtype=bool)}
            for name, array in arrays.items():
                np.save(data / (name + ".npy"), array)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("deploy.py")),
                                     "replay", "--episode", str(data.parent),
                                     "--report", str(root / "report.json")],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertFalse(report["motion_enabled"])
            self.assertEqual(report["frames"], 3)
            self.assertIn("DRY RUN ONLY", result.stdout)


if __name__ == "__main__":
    unittest.main()
