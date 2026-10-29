#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-xac_train}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.10 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements-train.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and PyTorch wheel.")
print("gpu:", torch.cuda.get_device_name(0))
print("arch:", torch.cuda.get_arch_list())
x = torch.randn(64, 64, device="cuda")
print("cuda_matmul:", float((x @ x).mean()))
PY

echo "Environment ready: conda activate ${ENV_NAME}"
