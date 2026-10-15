#!/usr/bin/env bash
# E2G worker: ChartCNN-S / ChartCNN-M（干净图像、对称 factorial、R2008/S0 切分）
# 用法: run_e2g_job.sh <s|m> [phase]
set -euo pipefail
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
export PATH="${CNN_ENV}/bin:${PATH}"
ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
cd "$ROOT"

ARCH="${1:?arch required: s|m}"
PHASE="${2:-all}"
case "$ARCH" in
  s|S) CFG=experiments/E2G_chart_capacity/config_e2g_s_r2008.yaml ;;
  m|M) CFG=experiments/E2G_chart_capacity/config_e2g_m_r2008.yaml ;;
  *) echo "unknown arch: $ARCH (use s or m)" >&2; exit 2 ;;
esac

echo "=== E2G arch=${ARCH} phase=${PHASE} $(date -Is) ==="
echo "hostname=$(hostname) cfg=$CFG"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"
# 启动前核对参数量
python - <<PY
from experiments.E2G_chart_capacity.architecture import parameter_breakdown
import yaml
cfg=yaml.safe_load(open("$CFG"))
ch=tuple(cfg["channels"])
k=ch[0]
got=parameter_breakdown(k)["total"]
exp=int(cfg["e2g"]["expected_parameters"])
assert ch==(k,2*k,4*k), ch
assert got==exp, (got, exp)
print("channels", ch, "params", got, "target", cfg["e2g"]["target_parameters"])
PY

python ashare_train/3b_cnn_hp_search.py --config "$CFG" --phase "$PHASE"
echo "=== E2G ${ARCH} done $(date -Is) ==="
