"""Physical action errors for relative [pos3, rot6d, width] action chunks."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.pose_util import rot6d_to_mat


def action_errors(pred, gt):
    """pred/gt: [N, H, 10]. Returns per-step (pos_mm, rot_deg, width_mm), each [N, H]."""
    pos_mm = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1) * 1000
    lead = pred.shape[:-1]
    # rot6d_to_mat only supports [N, 6]; flatten the batch/horizon dims.
    rp = rot6d_to_mat(pred[..., 3:9].reshape(-1, 6))
    rg = rot6d_to_mat(gt[..., 3:9].reshape(-1, 6))
    rel = np.swapaxes(rp, -1, -2) @ rg
    cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    rot_deg = np.degrees(np.arccos(cos)).reshape(lead)
    width_mm = np.abs(pred[..., 9] - gt[..., 9]) * 1000
    return pos_mm, rot_deg, width_mm


def summarize(errors, n_exec):
    pos, rot, width = errors
    out = {}
    for name, sl in (("exec", slice(0, n_exec)), ("horizon", slice(None))):
        out[f"pos_mm_{name}"] = float(pos[:, sl].mean())
        out[f"rot_deg_{name}"] = float(rot[:, sl].mean())
        out[f"width_mm_{name}"] = float(width[:, sl].mean())
    out["pos_mm_last"] = float(pos[:, -1].mean())
    return out


def hold_still(gt, current_width):
    """Baseline that predicts no motion and keeps the current width."""
    h = np.zeros_like(gt)
    h[..., 3] = 1.0
    h[..., 7] = 1.0  # rot6d of identity: rows [1,0,0], [0,1,0]
    h[..., 9] = current_width[:, None]
    return h


def score(model, hold):
    """Error relative to the hold-still baseline, summed over pos/rot/width
    on the executed steps.  3.0 = no better than standing still; lower is better."""
    return sum(model[k] / max(hold[k], 1e-6)
               for k in ("pos_mm_exec", "rot_deg_exec", "width_mm_exec"))


@torch.no_grad()
def evaluate_policy(policy, dataset_or_loader, device, n_exec, batch_size=64,
                    seed=0, max_windows=None, num_workers=4):
    """Sample the policy on a dataset; returns (model, hold_baseline, n_windows)."""
    loader = dataset_or_loader
    if not isinstance(loader, DataLoader):
        loader = DataLoader(loader, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    was_training = policy.training
    policy.eval()
    preds, gts, holds = [], [], []
    seen = 0
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            batch = dict_apply(batch, lambda x: x.to(device))
            pred = policy.predict_action(batch["obs"])["action"].float().cpu().numpy()
            gt = batch["action"].float().cpu().numpy()
            width = batch["obs"]["robot0_gripper_width"][:, -1, 0].float().cpu().numpy()
            preds.append(pred); gts.append(gt); holds.append(hold_still(gt, width))
            seen += len(gt)
            if max_windows and seen >= max_windows:
                break
    policy.train(was_training)
    pred, gt, hold = map(np.concatenate, (preds, gts, holds))
    return (summarize(action_errors(pred, gt), n_exec),
            summarize(action_errors(hold, gt), n_exec), len(gt))
