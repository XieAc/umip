"""XAC input contract: aligned RGB/pose/width, uniform time, session split."""
import numpy as np
import zarr
from diffusion_policy.dataset.umi_dataset import UmiDataset
from driver.xc330 import load_width_calibration


class XacDataset(UmiDataset):
    def __init__(self, dataset_path, frequency_hz=10.0, timing_tolerance=0.25,
                 gripper_calibration='calibration/gripper_width.json', **kwargs):
        samples = load_width_calibration(__import__('pathlib').Path(gripper_calibration))
        lo, hi = samples[0]['width_m'], samples[-1]['width_m']
        with zarr.ZipStore(dataset_path, mode='r') as store:
            root = zarr.open_group(store, mode='r')
            if 'source_session' not in root['meta'] or 'timestamps' not in root['data']:
                raise ValueError('Re-export with current export_umi_zarr.py: timestamps/session metadata required')
            groups = root['meta/source_session'][:]
            ends = root['meta/episode_ends'][:]
            times = root['data/timestamps'][:]
            width = root['data/robot0_gripper_width'][:]
            pos = root['data/robot0_eef_pos'][:]
            rot = root['data/robot0_eef_rot_axis_angle'][:]
            action = root['data/action'][:]
            if not all(np.isfinite(a).all() for a in (width, pos, rot, action, times)):
                raise ValueError('Nonfinite training labels/timestamps')
            if width.shape != (len(times), 1) or np.any(width < lo-1e-6) or np.any(width > hi+1e-6):
                raise ValueError('Gripper widths outside calibration range (metres expected)')
            expected = np.concatenate([pos, rot, width], axis=-1)
            if action.shape != expected.shape or not np.allclose(action, expected, atol=1e-6):
                raise ValueError('Zarr action must be absolute [position, axis-angle, width], not episode action.npy')
            if root['data/camera0_rgb'].shape != (len(times), 224, 224, 3) or root['data/camera0_rgb'].dtype != np.uint8:
                raise ValueError('Expected RGB uint8 224x224 letterbox')
            if frequency_hz <= 0 or not 0 < timing_tolerance < 1:
                raise ValueError('Invalid frequency/timing tolerance')
            start = 0
            for end in ends:
                dt = np.diff(times[start:end])
                if len(dt) == 0 or np.any(np.abs(dt*frequency_hz-1) > timing_tolerance):
                    raise ValueError(f'Nonuniform/wrong frame rate in segment at {start}: expected {frequency_hz} Hz; repair/re-export data')
                start = int(end)
            if start != len(times):
                raise ValueError('episode_ends does not match frame count')
        super().__init__(dataset_path=dataset_path, episode_groups=groups, **kwargs)
        if len(self) == 0 or (self.val_mask.any() and len(self.get_validation_dataset()) == 0):
            raise ValueError('No training/validation windows; collect longer demonstrations')
        self.source_sessions = [str(g) for g in groups]
        self.gripper_samples = samples
