#!/usr/bin/env bash
# 1-GPU smoke: verify AFS cnn env + CUDA on a job worker.
set -euo pipefail

CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"

echo "=== smoke_cuda start $(date -Is) ==="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "which python=$(command -v python)"
echo "CNN_ENV=${CNN_ENV}"

echo "=== nvidia-smi -L ==="
nvidia-smi -L || { echo "nvidia-smi -L failed"; exit 2; }

echo "=== nvidia-smi (summary) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

echo "=== torch cuda ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() >= 1, "no CUDA device"
name = torch.cuda.get_device_name(0)
print("device0", name)
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("matmul_ok", float(y[0, 0]))
print("SMOKE_OK")
PY

echo "=== smoke_cuda done $(date -Is) ==="
