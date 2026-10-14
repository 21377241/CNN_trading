#!/usr/bin/env bash
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"
export E7_USE_EW=1
echo "=== E10 $(date -Is) ==="
python experiments/E10_combination/run_e10.py
echo "=== E10 done $(date -Is) ==="
