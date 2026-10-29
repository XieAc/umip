import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_robot_tcp import solve_position, transform, spread_report


class TcpCalibrationTest(unittest.TestCase):
    def test_recovers_offset_after_removing_nonzero_old_tcp(self):
        rng = np.random.default_rng(31)
        rotations = Rotation.random(12, random_state=rng).as_matrix()
        offset = np.array([0.04, -0.02, 0.18])
        fixed = np.array([0.3, -0.4, 0.25])
        old_tcp = transform([0.01, 0.02, 0.09, 0.2, -0.1, 0.3])
        samples = []
        for rotation in rotations:
            flange = np.eye(4)
            flange[:3, :3] = rotation
            flange[:3, 3] = fixed - rotation @ offset
            measured_tcp = flange @ old_tcp
            samples.append(measured_tcp @ np.linalg.inv(old_tcp))
        estimated, report = solve_position(np.stack(samples))
        np.testing.assert_allclose(estimated, offset, atol=1e-12)
        self.assertLess(report['max_mm'], 1e-9)

    def test_single_axis_is_unobservable(self):
        samples = np.stack([transform([0, 0, 0, 0, 0, angle]) for angle in np.linspace(-1, 1, 8)])
        with self.assertRaisesRegex(ValueError, '姿态变化不足'):
            solve_position(samples)

    def test_independent_points_expose_wrong_offset(self):
        rotations = Rotation.from_euler('xyz', [[0, 0, 0], [50, 0, 0], [0, 50, 0], [-40, 20, 30]], degrees=True).as_matrix()
        error = np.array([0, 0, 0.02])
        report = spread_report(np.einsum('nij,j->ni', rotations, error))
        self.assertGreater(report['max_mm'], 2)


if __name__ == '__main__':
    unittest.main()
