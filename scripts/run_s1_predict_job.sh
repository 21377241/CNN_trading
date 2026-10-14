#!/usr/bin/env bash
# S1：复用 R2014 权重，预测 2014–2026。
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"
MODEL="${1:?model chart|oned|mlp}"
echo "=== S1 predict model=${MODEL} $(date -Is) ==="
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
python ashare_train/run_s1_predict.py --model "$MODEL"
echo "=== done $(date -Is) ==="
