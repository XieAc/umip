"""Check that offline checkpoint loading preserves the trained encoder."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import timm
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "xac_model_inference", PROJECT_ROOT / "test/test_model_inference.py"
)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


class PolicyLoadingTest(unittest.TestCase):
    def test_encoder_weights_and_outputs_survive_offline_loading(self):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, original_threads)
        create_model = timm.create_model

        def create_without_download(**kwargs):
            return create_model(
                **{**kwargs, "pretrained": False, "pretrained_cfg_overlay": None}
            )

        for pretrained, use_group_norm in ((True, True), (True, False),
                                           (False, True), (False, False)):
            with self.subTest(pretrained=pretrained, use_group_norm=use_group_norm):
                with initialize_config_dir(
                    config_dir=str(PROJECT_ROOT / "diffusion_policy/config"),
                    version_base=None,
                ):
                    cfg = compose(config_name="train_xac_v2", overrides=[
                        f"policy.obs_encoder.pretrained={pretrained}",
                        f"policy.obs_encoder.use_group_norm={use_group_norm}",
                        "policy.obs_encoder.frozen=True",
                        "policy.obs_encoder.transforms=null",
                        "policy.down_dims=[16,32,64]",
                        "+policy.obs_encoder.pretrained_weights_path=/nonexistent/resnet18.pth",
                    ])
                # Keep the training constructor's normalization decision, while
                # avoiding an ImageNet download for this synthetic checkpoint.
                with patch("timm.create_model", side_effect=create_without_download):
                    reference = instantiate(cfg.policy).eval()
                with torch.no_grad():
                    for module in reference.obs_encoder.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            module.running_mean.uniform_(-0.5, 0.5)
                            module.running_var.uniform_(0.5, 1.5)
                            module.num_batches_tracked.fill_(7)
                saved_cfg = OmegaConf.to_container(cfg, resolve=True)
                obs = {
                    key: torch.rand(1, value["horizon"], *value["shape"])
                    for key, value in saved_cfg["shape_meta"]["obs"].items()
                }
                with torch.inference_mode():
                    expected = reference.obs_encoder(obs)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "best.ckpt"
                    torch.save({"cfg": saved_cfg, "ema_model": reference.state_dict()}, checkpoint)
                    with patch("timm.models._builder.load_pretrained",
                               side_effect=AssertionError("Checkpoint loading requested ImageNet weights")):
                        loaded, payload = inference.load_policy(checkpoint, "cpu")
                self.assertEqual(payload["cfg"], saved_cfg)
                self.assertEqual(
                    [type(module) for module in reference.obs_encoder.modules()],
                    [type(module) for module in loaded.obs_encoder.modules()],
                )
                for key, value in reference.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
                with torch.inference_mode():
                    actual = loaded.obs_encoder(obs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
