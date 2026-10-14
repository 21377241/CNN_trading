#!/usr/bin/env bash
# Job worker: 主分析 EW 单窗对称搜参（model × R）。
# 用法: run_ew_main_job.sh <chart|oned|mlp> <R> [phase]
set -euo pipefail

CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"

MODEL="${1:?model required: chart|oned|mlp}"
R="${2:?retrain-year R required}"
PHASE="${3:-all}"
SYM_CFG="${SYM_CFG:-experiments/E1_image_baseline/config_cnn_hp_symmetric.yaml}"

echo "=== ew_main model=${MODEL} R=${R} phase=${PHASE} $(date -Is) ==="
echo "hostname=$(hostname) which_python=$(command -v python)"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"

case "$MODEL" in
  chart)
    python ashare_train/3b_cnn_hp_search.py \
      --config "$SYM_CFG" \
      --retrain-year "$R" \
      --phase "$PHASE"
    ;;
  oned|mlp)
    PRED_FLAG=()
    if [[ "$PHASE" == "all" || "$PHASE" == "2" || "$PHASE" == "none" ]]; then
      PRED_FLAG=(--predict)
    fi
    python ashare_train/3d_numeric_hp_search.py \
      --model "$MODEL" \
      --retrain-year "$R" \
      --phase "$PHASE" \
      "${PRED_FLAG[@]}"
    ;;
  *)
    echo "unknown model: $MODEL" >&2
    exit 2
    ;;
esac

echo "=== done $(date -Is) ==="
