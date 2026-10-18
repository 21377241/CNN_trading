#!/usr/bin/env bash
# 1D float64 标准化修正后：拼接 onedcnn_140_ew_v2，并重跑主表 / E7 / E9 / E10 / E11 / E12。
# 全部写入 *_f64 目录，不覆盖已验收的 ew_main/eval、E7_incremental_v2、E9_execution 等。
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"
PAPER="$ROOT/WORK_SPACE/paper"

export E7_USE_EW=1
export E7_ONED_KEY=onedcnn_140_ew_v2
export E7_OUT_DIR="$PAPER/E7_incremental_v2_f64"
export E7_DATA_DIR="$E7_OUT_DIR/data"
export E7_SAMPLE="$E7_DATA_DIR/e7_sample.feather"
export E9_OUT_DIR="$PAPER/E9_execution_f64"
export E10_OUT_DIR="$PAPER/E10_combination_f64"
export E11_OUT_DIR="$PAPER/E11_attribution_f64/stage1"
export E12_STAGE1_OUT="$PAPER/E12_signal_stability_f64/stage1"
export E12_STAGE2_OUT="$PAPER/E12_signal_stability_f64/stage2"
export E12_STAGE3_OUT="$PAPER/E12_signal_stability_f64/stage3"
export E9_WEEKLY="$E9_OUT_DIR/portfolio_weekly.parquet"
export MAIN_SUMMARY="$PAPER/ew_main_f64/eval/summary.csv"

echo "=== 1. stitch + main eval $(date -Is) ==="
python ashare_train/eval_oned_ew_v2.py

echo "=== 2. E7 sample (reuse returns/controls) $(date -Is) ==="
mkdir -p "$E7_DATA_DIR"
OLD_E7="$PAPER/E7_incremental_v2/data"
for f in weekly_returns.feather weekly_controls.feather weekly_returns_check.json weekly_controls_check.json; do
  if [[ -e "$OLD_E7/$f" && ! -e "$E7_DATA_DIR/$f" ]]; then
    ln -sfn "$OLD_E7/$f" "$E7_DATA_DIR/$f"
  fi
done
python -c "from experiments.E7_incremental.sample import build_sample; build_sample()"
python experiments/E7_incremental/run_fm.py
python experiments/E7_incremental/run_resid_ic.py
python experiments/E7_incremental/run_double_sort.py

echo "=== 3. E9 $(date -Is) ==="
python experiments/E9_tradable/run_e9.py

echo "=== 4. E10 $(date -Is) ==="
python experiments/E10_combination/run_e10.py

echo "=== 5. E11 stage1 $(date -Is) ==="
python experiments/E11_attribution/run_e11_stage1.py

echo "=== 6. E12 stages 1–3 $(date -Is) ==="
python experiments/E12_signal_stability/run_e12_stage1.py
python experiments/E12_signal_stability/run_e12_stage2.py
python experiments/E12_signal_stability/run_e12_stage3.py

echo "=== done $(date -Is) ==="
echo "outputs:"
echo "  $PAPER/ew_main_f64/eval"
echo "  $E7_OUT_DIR"
echo "  $E9_OUT_DIR"
echo "  $E10_OUT_DIR"
echo "  $E11_OUT_DIR"
echo "  $PAPER/E12_signal_stability_f64"
