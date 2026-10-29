"""Regression checks for Stray synchronization and uniform training exports."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from estimate_sync_by_shake import load_motion_signal
from export_umi_zarr import load_episode_segments, resample_observations


class DatasetPreparationTest(unittest.TestCase):
    def test_stray_rotation_rate_preferred_over_gravity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'imu.csv'
            path.write_text('timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n'
                            '0,0,0,1,3,4,0\n0.1,0,0,1,0,0,0\n0.2,0,0,1,0,0,2\n')
            signal = load_motion_signal(path, False, smooth_window_seconds=0)
            self.assertEqual(signal.signal_kind, 'gyro')
            np.testing.assert_allclose(signal.raw_signal, [5, 0, 2])

    def test_uniform_grid_and_rotation_wrap(self):
        times = np.array([0, .12, .2])
        rgb = np.arange(3, dtype=np.uint8).reshape(3, 1, 1, 1)
        pos = np.repeat(times[:, None], 3, axis=1)
        rot = np.zeros((3, 3))
        rot[:, 2] = np.deg2rad([170, 182, 190])
        images, positions, rotations, widths = resample_observations(
            times, rgb, pos, rot, times[:, None], 10)
        np.testing.assert_allclose(positions[:, 0], [0, .1, .2])
        np.testing.assert_allclose(widths[:, 0], [0, .1, .2])
        np.testing.assert_array_equal(images[:, 0, 0, 0], [0, 1, 2])
        np.testing.assert_allclose(Rotation.from_rotvec(rotations[1]).as_matrix(),
                                   Rotation.from_euler('z', 180, degrees=True).as_matrix(), atol=1e-6)

    def test_long_gaps_split_and_invalid_frames_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'metadata').mkdir()
            (root / 'metadata/source.json').write_text(json.dumps({'source_session_name': '001'}))
            arrays = {
                'd435i_timestamps': np.array([0, .1, .2, 1., 1.1, 1.2, 1.3]),
                'valid_mask': np.array([True] * 6 + [False]),
                'camera0_rgb': np.zeros((7, 224, 224, 3), dtype=np.uint8),
                'robot0_eef_pos': np.zeros((7, 3)),
                'robot0_eef_rot_axis_angle': np.zeros((7, 3)),
                'robot0_gripper_width': np.zeros((7, 1)),
            }
            for name, array in arrays.items():
                np.save(root / 'data' / f'{name}.npy', array)
            segments = load_episode_segments(root, 2, resample_fps=10, max_gap_seconds=.3)
            self.assertEqual([(s['source_start'], s['source_end']) for s in segments], [(0, 3), (3, 6)])


if __name__ == '__main__':
    unittest.main()
