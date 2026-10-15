#!/usr/bin/env bash
# 提交 E2G ChartCNN-S / M（各 1 GPU；干净数据 R2008）
set -euo pipefail
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
JOB="${JOB_CLI:-/mnt/afs/users/wangyl/job}"
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
WORKER="$ROOT/scripts/run_e2g_job.sh"
chmod +x "$WORKER"

for arch in s m; do
  name="e2g-${arch}-r2008"
  echo "[submit] $name"
  "$JOB" submit "$WORKER" -g 1 -N "$name" \
    --project-root "$ROOT" --output-root "$ROOT/output" \
    -e "CNN_ENV=$CNN_ENV" -e "PROJECT_ROOT=$ROOT" \
    -- "$arch" all
done
"$JOB" list -L 6 2>&1 | head -12
