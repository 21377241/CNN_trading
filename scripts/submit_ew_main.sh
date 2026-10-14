#!/usr/bin/env bash
# 提交主分析 EW：三年一窗 × 三模型对称搜参。
# 用法:
#   scripts/submit_ew_main.sh                  # 全部 R×模型（跳过已有 RUNNING/同名产出）
#   scripts/submit_ew_main.sh --dry-run
#   scripts/submit_ew_main.sh --years 2011,2014 --models chart,oned,mlp
#   scripts/submit_ew_main.sh --link-r2008      # 仅把 R=2008 接到既有 S0 对称产出
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/afs/users/wangyl/cnn_trading/CNN_trading}"
JOB="${JOB_CLI:-/mnt/afs/users/wangyl/job}"
CNN_ENV="${CNN_ENV:-/mnt/afs/users/wangyl/conda_envs/cnn}"
WORKER="$ROOT/scripts/run_ew_main_job.sh"
EW_ROOT="$ROOT/WORK_SPACE/paper/ew_main"
SYM_CHART="$ROOT/WORK_SPACE/paper/E1c_cnn_hp_I20R5_clean_sym"
NUM_ROOT="$ROOT/WORK_SPACE/paper/hp_numeric_140"

YEARS=(2008 2011 2014 2017 2020 2023)
MODELS=(chart oned mlp)
DRY=0
DO_LINK=0
FORCE_SUBMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --link-r2008) DO_LINK=1; shift ;;
    --force) FORCE_SUBMIT=1; shift ;;
    --years) IFS=',' read -r -a YEARS <<< "$2"; shift 2 ;;
    --models) IFS=',' read -r -a MODELS <<< "$2"; shift 2 ;;
    -h|--help) sed -n '1,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

link_r2008() {
  mkdir -p "$EW_ROOT/R2008"
  # 数值：既有对称两阶段即 R=2008
  for m in mlp oned; do
    dst="$EW_ROOT/R2008/$m"
    src="$NUM_ROOT/$m"
    if [[ -e "$dst" || -L "$dst" ]]; then
      echo "[skip] $dst exists"
    elif [[ -d "$src" && -f "$src/selection.json" ]]; then
      ln -sfn "$src" "$dst"
      echo "[link] $dst -> $src"
      python - <<PY
import json, os
p = os.path.join("$dst", "split.json")
if not os.path.isfile(p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"train_years": list(range(2000, 2006)), "val_years": [2006, 2007],
                   "pred_years": [2008, 2009, 2010], "retrain_year": 2008,
                   "note": "linked from hp_numeric_140 (S0 ≡ R2008)"}, f, indent=2)
PY
    else
      echo "[warn] missing $src/selection.json" >&2
    fi
  done
  # chart：对称 factorial 进行中/完成后接到 R2008
  dst="$EW_ROOT/R2008/chart"
  if [[ -e "$dst" || -L "$dst" ]]; then
    echo "[skip] $dst exists"
  elif [[ -d "$SYM_CHART" ]]; then
    ln -sfn "$SYM_CHART" "$dst"
    echo "[link] $dst -> $SYM_CHART"
  else
    echo "[warn] missing $SYM_CHART" >&2
  fi
}

if [[ "$DO_LINK" -eq 1 ]]; then
  link_r2008
  exit 0
fi

# 默认：先衔接 R2008，再提交其余窗（及未衔接的模型）
link_r2008

chmod +x "$WORKER"

submit_one() {
  local model="$1" r="$2"
  local name="ewm-${model}-${r}"
  local out="$EW_ROOT/R${r}/${model}"
  # 已有 selection / phase3 则跳过（除非 --force）
  if [[ "$FORCE_SUBMIT" -eq 0 ]]; then
    if [[ "$model" == "chart" && -f "$out/phase3_full_train.csv" ]]; then
      echo "[skip] $name already has phase3"
      return 0
    fi
    if [[ "$model" != "chart" && -f "$out/selection.json" ]]; then
      echo "[skip] $name already has selection"
      return 0
    fi
    # R2008 chart 由 clean_sym 承接且仍在跑
    if [[ "$model" == "chart" && "$r" == "2008" ]]; then
      echo "[skip] $name → use E1c_cnn_hp_I20R5_clean_sym (job chart-hp-sym)"
      return 0
    fi
    if [[ "$model" != "chart" && "$r" == "2008" && -f "$NUM_ROOT/$model/selection.json" ]]; then
      echo "[skip] $name → linked hp_numeric_140/$model"
      return 0
    fi
  fi
  if [[ "$DRY" -eq 1 ]]; then
    echo "[dry] $JOB submit $WORKER -g 1 -N $name -- $model $r all"
    return 0
  fi
  echo "[submit] $name"
  "$JOB" submit "$WORKER" -g 1 -N "$name" \
    --project-root "$ROOT" --output-root "$ROOT/output" \
    -e "CNN_ENV=$CNN_ENV" -e "PROJECT_ROOT=$ROOT" \
    -- "$model" "$r" all
}

for r in "${YEARS[@]}"; do
  for m in "${MODELS[@]}"; do
    submit_one "$m" "$r"
  done
done

echo "==== quota ===="
"$JOB" quota 2>&1 | head -8
echo "==== recent ewm ===="
"$JOB" list -L 0 2>&1 | grep -E 'ewm-|chart-hp-sym' | head -40 || true
