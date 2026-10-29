#!/usr/bin/env bash
set -euo pipefail
DATASET_DIR="${DATASET_DIR:-dataset_v3}"

test -f ${DATASET_DIR}/replay.zarr.zip || { echo "Missing ${DATASET_DIR}/replay.zarr.zip" >&2; exit 2; }
test -f ${DATASET_DIR}/replay_manifest.json || { echo "Missing ${DATASET_DIR}/replay_manifest.json" >&2; exit 2; }
test -f calibration/gripper_width.json || { echo "Missing calibration/gripper_width.json" >&2; exit 2; }
DATASET_PATH="$(pwd)/${DATASET_DIR}/replay.zarr.zip"

DATASET_PATH="${DATASET_PATH}" python - <<'PY'
import os
from omegaconf import OmegaConf
from diffusion_policy.dataset.xac_dataset import XacDataset

cfg = OmegaConf.load("diffusion_policy/config/task/xac.yaml")
root = OmegaConf.create({"task": cfg})
shape_meta = OmegaConf.to_container(root.task.shape_meta, resolve=True)
dataset = XacDataset(
    dataset_path=os.environ["DATASET_PATH"],
    shape_meta=shape_meta,
    pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
    val_ratio=0.1,
    frequency_hz=10.0,
    gripper_calibration="calibration/gripper_width.json",
)
print("train_windows:", len(dataset))
print("validation_episodes:", int(dataset.val_mask.sum()))
print("sample_obs:", {k: tuple(v.shape) for k, v in dataset[0]["obs"].items()})
print("sample_action:", tuple(dataset[0]["action"].shape))
if len(dataset) == 0 or int(dataset.val_mask.sum()) == 0:
    raise SystemExit("Training or validation split is empty")
PY

python -m unittest discover -s . -p 'test_xac_training.py' -v
