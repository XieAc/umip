import unittest
from types import SimpleNamespace
import numpy as np
from _tig_solver_bridge import load_validate_solver


class ContactVarianceTest(unittest.TestCase):
    def test_variance_not_distance_and_rotation_invariant(self):
        solver = load_validate_solver()
        segments = [SimpleNamespace(start_index=i*3, end_index_exclusive=i*3+3,
                                    contact_index=i+1) for i in range(4)]
        points = np.array([[5,0,0]]*3 + [[0,0,0],[.01,0,0],[-.01,0,0]] +
                          [[0,0,0],[.001,0,0],[-.001,0,0]] + [[0,0,0]]*3)
        kept, discarded, variances = solver.discard_highest_variance_contact(segments, points)
        self.assertEqual(discarded.contact_index, 2)
        self.assertEqual([s.contact_index for s in kept], [1,3,4])
        rotated = points @ np.array([[0,1,0],[-1,0,0],[0,0,1]]) + 7
        _, discarded2, variance2 = solver.discard_highest_variance_contact(segments, rotated)
        self.assertEqual(discarded2.contact_index, 2)
        np.testing.assert_allclose(variances, variance2, atol=1e-9)
        with self.assertRaises(ValueError):
            solver.discard_highest_variance_contact(segments[:3], points)


if __name__ == '__main__':
    unittest.main()
