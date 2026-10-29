"""Offline tests of motor direction, temporal contract and session isolation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import zarr
from omegaconf import OmegaConf
from xac_config import load_width_calibration, positions_to_width
from driver.xc330 import XC330Controller
from diffusion_policy.dataset.xac_dataset import XacDataset


class XacTrainingTests(unittest.TestCase):
    def test_opposing_directions_and_width_roundtrip(self):
        calibration = load_width_calibration()
        rows = sorted(calibration['samples'], key=lambda r: r['width_m'])
        controller = object.__new__(XC330Controller)
        controller.ids = (1, 2)
        controller.widths = np.array([r['width_m'] for r in rows])
        controller.m1 = np.array([r['m1'] for r in rows])
        controller.m2 = np.array([r['m2'] for r in rows])
        closed = controller.width_to_targets(0)
        opened = controller.width_to_targets(.1)
        self.assertLess(opened[1], closed[1])
        self.assertGreater(opened[2], closed[2])
        for width in np.linspace(0, .1, 21):
            targets = controller.width_to_targets(width)
            measured = positions_to_width(targets[1], targets[2], calibration)[0]
            self.assertAlmostEqual(measured, width, delta=.001)
            wrapped = positions_to_width(targets[1]-4096, targets[2]+4096, calibration)[0]
            self.assertAlmostEqual(wrapped, measured)
        self.assertTrue(np.isnan(positions_to_width(4000, 100, calibration)[0]))

    def make_data(self, path, bad_timing=False):
        with zarr.ZipStore(str(path), mode='w') as store:
            root = zarr.group(store)
            data = root.create_group('data'); meta = root.create_group('meta')
            meta.array('episode_ends', np.array([20, 40, 60]))
            meta.array('source_session', np.array(['001', '001', '002']))
            times = np.tile(np.arange(20)/10, 3)
            if bad_timing: times[8] += .05
            data.array('timestamps', times)
            data.array('camera0_rgb', np.zeros((60,224,224,3), dtype=np.uint8), chunks=(1,224,224,3))
            pos = np.zeros((60,3), dtype=np.float32)
            rot = pos.copy(); width = np.linspace(0,.1,60,dtype=np.float32)[:,None]
            data.array('robot0_eef_pos', pos); data.array('robot0_eef_rot_axis_angle', rot)
            data.array('robot0_gripper_width', width)
            data.array('action', np.concatenate([pos,rot,width], axis=-1))

    def dataset(self, path):
        cfg = OmegaConf.load('diffusion_policy/config/task/xac.yaml')
        # Resolve the shape interpolation without Hydra.
        wrapper = OmegaConf.create({'task': cfg})
        shape = OmegaConf.to_container(wrapper.task.shape_meta, resolve=True)
        return XacDataset(str(path), shape_meta=shape, pose_repr={'obs_pose_repr':'relative','action_pose_repr':'relative'}, val_ratio=.5)

    def test_source_session_split_and_action_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.zarr.zip'; self.make_data(path)
            dataset = self.dataset(path)
            self.assertEqual(dataset.val_mask[0], dataset.val_mask[1])
            self.assertNotEqual(dataset.val_mask[0], dataset.val_mask[2])
            item = dataset[0]
            self.assertEqual(tuple(item['action'].shape), (16,10))
            idx = dataset.sampler.indices[0][0]
            np.testing.assert_allclose(item['action'][:,9], dataset.replay_buffer['action'][idx:idx+16,6])
            normalizer = dataset.get_normalizer()
            action = item['action']
            np.testing.assert_allclose(normalizer['action'].unnormalize(normalizer['action'].normalize(action)).detach().numpy(), action, atol=1e-6)

    def test_reject_time_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'gap.zarr.zip'; self.make_data(path, True)
            with self.assertRaisesRegex(ValueError, 'frame rate'): self.dataset(path)


if __name__ == '__main__': unittest.main()
