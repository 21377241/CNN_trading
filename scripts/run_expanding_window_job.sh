#!/usr/bin/env bash
# Job worker: one expanding-window (model, oos-year) task.
set -euo pipefail

CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"

MODEL="${1:?model required: chart|oned|mlp}"
YEAR="${2:?oos-year required}"

echo "=== expanding_window model=${MODEL} oos_year=${YEAR} $(date -Is) ==="
echo "hostname=$(hostname) which_python=$(command -v python)"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"

python ashare_train/run_expanding_window.py --model "$MODEL" --oos-year "$YEAR"
echo "=== done $(date -Is) ==="
