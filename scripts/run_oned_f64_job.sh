#!/usr/bin/env bash
# Job worker: 1D-CNN 标准化 float32 饱和修正后的单窗完整重搜参（clean_spec 2026-10-15 记录；方案 A）。
# 输出到 ew_main_f64/R<R>/oned，不覆盖 ew_main/R<R>/oned。
# 用法: run_oned_f64_job.sh <R>
set -euo pipefail

CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"

R="${1:?retrain-year R required}"
case "$R" in 2017|2020|2023) ;; *) echo "R must be 2017, 2020 or 2023" >&2; exit 2 ;; esac

echo "=== oned f64 R=${R} $(date -Is) commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA) ==="
echo "hostname=$(hostname) which_python=$(command -v python)"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"
python tests/test_oned_channel_stats.py

python ashare_train/3d_numeric_hp_search.py \
  --model oned \
  --retrain-year "$R" \
  --phase all \
  --predict \
  --out-root "WORK_SPACE/paper/ew_main_f64/R${R}/oned"

echo "=== done $(date -Is) ==="
