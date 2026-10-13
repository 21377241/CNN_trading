#!/usr/bin/env bash
# Submit expanding-window jobs (18 OOS years × models).
# Default: chart+oned on 1 GPU each; mlp on CPU (-g 0).
# Usage:
#   bash scripts/submit_expanding_window.sh
#   MODELS=chart YEARS=2008,2009 bash scripts/submit_expanding_window.sh
#   SPOT=1 bash scripts/submit_expanding_window.sh
#   DRY_RUN=1 bash scripts/submit_expanding_window.sh
set -euo pipefail

JOB_BIN="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output}"
WORKER="$ROOT/scripts/run_expanding_window_job.sh"
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"

MODELS="${MODELS:-chart,oned,mlp}"
YEARS="${YEARS:-$(seq -s, 2008 2025)}"
SPOT="${SPOT:-0}"
DRY_RUN="${DRY_RUN:-0}"

chmod +x "$WORKER" "$ROOT/scripts/submit_expanding_window.sh"
cd "$ROOT"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
IFS=',' read -r -a YEAR_ARR <<< "$YEARS"

echo "submit expanding_window models=${MODELS} years=${YEARS} spot=${SPOT} dry_run=${DRY_RUN}"

for model in "${MODEL_ARR[@]}"; do
  gpus=1
  [[ "$model" == "mlp" ]] && gpus=0
  for year in "${YEAR_ARR[@]}"; do
    name="ew-${model}-${year}"
    cmd=( "$JOB_BIN" submit "$WORKER" -g "$gpus" -N "$name"
      --project-root "$ROOT" --output-root "$OUT_ROOT"
      -e "CNN_ENV=$CNN_ENV" -e "PROJECT_ROOT=$ROOT" )
    [[ "$SPOT" == "1" ]] && cmd+=(-s)
    [[ "$DRY_RUN" == "1" ]] && cmd+=(--dry-run)
    cmd+=(-- "$model" "$year")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
  done
done
