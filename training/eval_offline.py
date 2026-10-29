"""Offline action-error evaluation of an XAC checkpoint on held-out demos.

Reports physical errors (mm / deg / gripper mm) of the sampled EMA policy
against the recorded demo, and the same errors for a "hold still" baseline
(predict no motion, keep current width).  A useful policy must clearly beat
the baseline; ``score`` < 3.0 means better than standing still.

    python eval_offline.py data/outputs/<run>/checkpoints/best.ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.action_metrics import evaluate_policy, score
from diffusion_policy.common.checkpoint_compat import policy_cfg_for_loading

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(payload, device):
    cfg = OmegaConf.create(payload["cfg"])
    policy_cfg = OmegaConf.create(policy_cfg_for_loading(
        OmegaConf.to_container(cfg.policy, resolve=True)))
    enc = policy_cfg.obs_encoder
    # Weights come from the checkpoint; skip downloading the ImageNet init.
    if enc.pretrained and enc.get("use_group_norm", False):
        enc.use_group_norm = False
    enc.pretrained = False
    policy = hydra.utils.instantiate(policy_cfg)
    policy.load_state_dict(payload["ema_model"], strict=True)
    return policy.to(device).eval(), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--dataset", default="dataset_v3/replay.zarr.zip")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-windows", type=int, default=400,
                   help="also score this many training windows to show the overfitting gap (0 = skip)")
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy, cfg = load_policy(payload, args.device)
    ds_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    ds_cfg["dataset_path"] = str(Path(args.dataset).resolve())
    dataset = hydra.utils.instantiate(ds_cfg)
    val = dataset.get_validation_dataset()
    n_exec = int(cfg.get("n_action_steps", 8))

    res = {"checkpoint": str(args.checkpoint), "epoch": payload.get("epoch")}
    model_val, hold_val, n = evaluate_policy(policy, val, args.device, n_exec)
    res.update(val_windows=n, val_score=score(model_val, hold_val),
               val=model_val, val_hold_baseline=hold_val)
    if args.train_windows:
        model_tr, hold_tr, n_tr = evaluate_policy(policy, dataset, args.device, n_exec,
                                                  max_windows=args.train_windows)
        res.update(train_windows=n_tr, train_score=score(model_tr, hold_tr), train=model_tr)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
