#!/usr/bin/env bash
# E7 主分析：EW 拼接预测 + 干净控制变量 / 执行收益。
# 依赖：ashare_ret.feather、ashare_stock_info.feather（migration 2.3：未默认同迁 GPU 机）。
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"
export E7_USE_EW=1

PANEL="$ROOT/WORK_SPACE/data/ashare_processed_data/ashare_ret.feather"
INFO="$ROOT/WORK_SPACE/data/ashare_processed_data/ashare_stock_info.feather"
if [[ ! -f "$PANEL" || ! -f "$INFO" ]]; then
  echo "ERROR: E7 需要日面板与股票信息，当前缺失：" >&2
  echo "  $PANEL" >&2
  echo "  $INFO" >&2
  echo "按 docs/migration_gpu_server.md：可将 pred 拉回本机跑 E7，或把上述文件拷到本机 WORK_SPACE。" >&2
  exit 2
fi

echo "=== E7 EW build returns $(date -Is) ==="
python experiments/E7_incremental/build_returns.py
echo "=== E7 EW build controls $(date -Is) ==="
python experiments/E7_incremental/build_controls.py
echo "=== E7 EW Fama-MacBeth $(date -Is) ==="
python experiments/E7_incremental/run_fm.py
echo "=== E7 EW resid Rank IC $(date -Is) ==="
python experiments/E7_incremental/run_resid_ic.py
echo "=== E7 EW double sort $(date -Is) ==="
python experiments/E7_incremental/run_double_sort.py
echo "=== E7 EW done $(date -Is) ===  out=WORK_SPACE/paper/E7_incremental_v2"
