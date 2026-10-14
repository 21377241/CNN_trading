#!/usr/bin/env bash
# Job worker: symmetric chart-CNN HP search (factorial phase1 → phase2 → phase3).
set -euo pipefail

CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"

PHASE="${1:-all}"
CFG="${2:-experiments/E1_image_baseline/config_cnn_hp_symmetric.yaml}"

echo "=== chart HP symmetric phase=${PHASE} $(date -Is) ==="
echo "hostname=$(hostname) which_python=$(command -v python)"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"

python ashare_train/3b_cnn_hp_search.py --config "$CFG" --phase "$PHASE"
echo "=== done $(date -Is) ==="
