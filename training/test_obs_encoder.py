"""Encoder behaviour that the removed pipeline got wrong, plus checkpoint loading rules."""
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from diffusion_policy.common.checkpoint_compat import ObsoleteCheckpointError, policy_cfg_for_loading


def make_cfg(*overrides):
    with initialize_config_dir(config_dir=str(Path('diffusion_policy/config').resolve()), version_base=None):
        return compose(config_name='train_xac_v2', overrides=[
            'policy.obs_encoder.pretrained=False', 'policy.down_dims=[16,32,64]', *overrides])


def obs(cfg, batch=2):
    torch.manual_seed(0)
    return {k: torch.rand(batch, v['horizon'], *v['shape']) for k, v in cfg.shape_meta.obs.items()}


class EncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_freezes_only_backbone(self):
        cfg = make_cfg('policy.obs_encoder.frozen=True')
        enc = instantiate(cfg.policy).obs_encoder
        self.assertFalse(any(p.requires_grad for p in enc.key_model_map.parameters()))
        self.assertTrue(enc.spatial_embedding.requires_grad)
        enc.train()
        self.assertFalse(any(m.training for m in enc.key_model_map.modules()))

    def test_imagenet_normalization_applied(self):
        cfg = make_cfg('policy.obs_encoder.transforms=null')
        enc = instantiate(cfg.policy).obs_encoder.eval()
        o = obs(cfg)
        img = o['camera0_rgb'].flatten(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            manual = enc.aggregate_feature(enc.key_model_map['camera0_rgb']((img - mean) / std))
            out = enc(o)
        torch.testing.assert_close(out[:, :manual.numel() // 2], manual.reshape(2, -1))

    def test_augmentation_only_in_training(self):
        cfg = make_cfg()
        enc = instantiate(cfg.policy).obs_encoder
        o = obs(cfg)
        with torch.no_grad():
            enc.eval()
            a, b = enc(o), enc(o)
            torch.testing.assert_close(a, b)
            enc.train()
            self.assertFalse(torch.allclose(enc(o), a))


class CheckpointCompatTest(unittest.TestCase):
    def test_rules(self):
        enc = dict(_target_='x', frozen=True, transforms=None)
        with self.assertRaises(ObsoleteCheckpointError):   # removed pipeline
            policy_cfg_for_loading({'obs_encoder': {**enc, 'imagenet_norm': True}})
        with self.assertRaises(ObsoleteCheckpointError):   # frozen random pooling
            policy_cfg_for_loading({'obs_encoder': {**enc, 'imagenet_norm': True, 'input_norm': 'imagenet'}})
        ok = policy_cfg_for_loading({'obs_encoder': {**enc, 'imagenet_norm': True, 'input_norm': 'imagenet',
                                                     'freeze_scope': 'backbone'}})
        self.assertEqual(ok['obs_encoder'], enc)
        self.assertEqual(policy_cfg_for_loading({'obs_encoder': enc})['obs_encoder'], enc)

    def test_current_config_instantiates_after_roundtrip(self):
        cfg = make_cfg()
        saved = OmegaConf.to_container(cfg.policy, resolve=True)
        instantiate(OmegaConf.create(policy_cfg_for_loading(saved)))


if __name__ == '__main__':
    unittest.main()
