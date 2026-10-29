#!/usr/bin/env bash
# Start a new v2 experiment.  Extra Hydra overrides are passed through, e.g.
#   bash train_v2.sh data/outputs/my_run training.num_epochs=600
# Another dataset (its replay_manifest.csv must sit next to it):
#   DATASET=dataset_v2/replay.zarr.zip bash train_v2.sh data/outputs/my_run
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

OUTPUT="${1:-data/outputs/xac_v2_$(date +%Y%m%d_%H%M%S)}"
shift || true
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists; choose a new directory: $OUTPUT" >&2
  exit 2
fi
TRAIN_PYTHON="${TRAIN_PYTHON:-$HOME/miniconda3/envs/umip/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || TRAIN_PYTHON=python
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}"
if [[ ! -f "$PRETRAINED_WEIGHTS" ]]; then
  echo "Missing ResNet18 weights: $PRETRAINED_WEIGHTS" >&2
  exit 2
fi
mkdir -p -- "$OUTPUT"
"$TRAIN_PYTHON" -u train.py --config-name=train_xac_v2 \
  task.dataset_path="$(realpath "${DATASET:-dataset_v3/replay.zarr.zip}")" \
  ++policy.obs_encoder.pretrained_weights_path="$PRETRAINED_WEIGHTS" \
  hydra.run.dir="$OUTPUT" "$@" \
  2>&1 | tee "$OUTPUT/console.log"
