#!/usr/bin/env bash
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"
export E7_USE_EW=1
echo "=== E9 $(date -Is) ==="
python experiments/E9_tradable/run_e9.py
echo "=== E9 done $(date -Is) ==="
