#!/usr/bin/env bash
# E7 §15：S0/S1 得分稳健性（依赖 E7_incremental_v2 已有 sample/controls/returns）
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"
export E7_USE_EW=1
echo "=== E7 §15 S0/S1 $(date -Is) ==="
python experiments/E7_incremental/run_s0_s1.py
echo "=== E7 §15 done $(date -Is) ==="
