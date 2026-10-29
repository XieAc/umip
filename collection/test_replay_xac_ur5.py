from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from replay_xac_ur5 import (
    anchor_trajectory,
    enforce_limits,
    load_trajectory,
    trajectory_metrics,
)


class ReplayTrajectoryTest(unittest.TestCase):
    def make_episode(self, root: Path) -> Path:
        data = root / "data"
        data.mkdir()
        np.save(data / "d435i_timestamps.npy", np.asarray([0.0, 0.1, 0.2, 0.3]))
        np.save(
            data / "robot0_eef_pos.npy",
            np.asarray([[9, 9, 9], [1, 2, 3], [1.01, 2, 3], [1.02, 2, 3]], dtype=np.float32),
        )
        np.save(
            data / "robot0_eef_rot_axis_angle.npy",
            np.asarray([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]], dtype=np.float32),
        )
        np.save(data / "robot0_gripper_width.npy", np.asarray([[0.09], [0.08], [0.07], [0.06]]))
        np.save(data / "valid_mask.npy", np.asarray([False, True, True, True]))
        return root

    def test_loads_longest_valid_run_and_anchors_relative_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = load_trajectory(self.make_episode(Path(temporary)))
        anchor = np.asarray([0.4, -0.5, 0.6, 0.2, -0.1, 0.3])
        targets = anchor_trajectory(trajectory, anchor)
        np.testing.assert_allclose(targets[0], anchor, atol=1e-8)
        recorded_relative_rotation = (
            Rotation.from_rotvec(trajectory.rotations_rotvec[0]).inv()
            * Rotation.from_rotvec(trajectory.rotations_rotvec[-1])
        ).magnitude()
        replay_relative_rotation = (
            Rotation.from_rotvec(targets[0, 3:]).inv()
            * Rotation.from_rotvec(targets[-1, 3:])
        ).magnitude()
        self.assertAlmostEqual(recorded_relative_rotation, replay_relative_rotation)

    def test_speed_scale_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = load_trajectory(self.make_episode(Path(temporary)))
        targets = anchor_trajectory(trajectory, np.asarray([0, 0, 0.5, 0, 0, 0]))
        metrics = trajectory_metrics(trajectory, targets, speed_scale=0.1)
        self.assertAlmostEqual(metrics["wall_duration_at_speed_scale_s"], 2.0)
        enforce_limits(metrics, 0.05, 0.4, 0.6, 0.05, 0.5, 0.25)
        with self.assertRaisesRegex(ValueError, "distance from start"):
            enforce_limits(metrics, 0.005, 0.4, 0.6, 0.05, 0.5, 0.25)


if __name__ == "__main__":
    unittest.main()
