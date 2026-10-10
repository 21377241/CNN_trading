#!/usr/bin/env bash
# =============================================================================
# run_csi800.sh  - CSI800 (A股中证800) 完整流水线
#
# 步骤：
#   1. 数据预处理        (1_preprocess_csi800_data.py)
#   2. 图表数据生成      (2_generate_csi800_chart_data.py)
#   3. 模型训练 + OOS   (3_train_csi800_model.py)
#   4. 组合分析（含TopN）(4_analyze_csi800_portfolio.py)
#
# 用法：
#   bash scripts/run_csi800.sh              # 从头完整运行
#   bash scripts/run_csi800.sh --from 3     # 从步骤 3 开始
#   bash scripts/run_csi800.sh --only 1     # 只运行步骤 1
#   bash scripts/run_csi800.sh --dry-run    # 仅打印命令，不执行
#
#   步骤 1 额外参数（传递给预处理脚本）：
#   bash scripts/run_csi800.sh --preprocess-args "--years 2015-2025"
#   bash scripts/run_csi800.sh --preprocess-args "--force-from 5"
#
# 日志：
#   WORK_SPACE/logs/csi800/run_YYYYMMDD_HHMMSS.log
# =============================================================================
set -euo pipefail

# ── 颜色定义 ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── 路径 ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_BASE="$PROJECT_ROOT/WORK_SPACE/logs/csi800"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_BASE/run_${TIMESTAMP}.log"

mkdir -p "$LOG_BASE"

# ── 参数解析 ──────────────────────────────────────────────
FROM_STEP=1
ONLY_STEPS=""
DRY_RUN=false
PREPROCESS_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)            FROM_STEP="$2"; shift 2 ;;
        --only)            ONLY_STEPS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --preprocess-args) PREPROCESS_ARGS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 步骤定义 ──────────────────────────────────────────────
declare -A STEP_NAME=(
    [1]="数据预处理"
    [2]="图表数据生成"
    [3]="模型训练 + OOS 预测"
    [4]="组合分析（十分位 + TopN）"
)
STEP_CMD_1="python csi800_train/1_preprocess_csi800_data.py $PREPROCESS_ARGS"
declare -A STEP_CMD=(
    [1]="$STEP_CMD_1"
    [2]="python csi800_train/2_generate_csi800_chart_data.py"
    [3]="python csi800_train/3_train_csi800_model.py"
    [4]="python csi800_train/4_analyze_csi800_portfolio.py"
)
TOTAL_STEPS=4

# ── 工具函数 ──────────────────────────────────────────────
should_run() {
    local step=$1
    [[ $step -lt $FROM_STEP ]] && return 1
    if [[ -n "$ONLY_STEPS" ]]; then
        IFS=',' read -ra only_arr <<< "$ONLY_STEPS"
        for s in "${only_arr[@]}"; do
            [[ "$s" == "$step" ]] && return 0
        done
        return 1
    fi
    return 0
}

log_and_print() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

run_step() {
    local step=$1
    local name="${STEP_NAME[$step]}"
    local cmd="${STEP_CMD[$step]}"
    local start_ts elapsed

    log_and_print "\n${CYAN}${BOLD}══════════════════════════════════════════════════${NC}"
    log_and_print "${CYAN}${BOLD}  [步骤 $step/$TOTAL_STEPS] $name${NC}"
    log_and_print "${CYAN}${BOLD}══════════════════════════════════════════════════${NC}"
    log_and_print "  命令: $cmd"
    log_and_print "  时间: $(date '+%Y-%m-%d %H:%M:%S')"

    if $DRY_RUN; then
        log_and_print "  ${YELLOW}[DRY-RUN] 跳过执行${NC}"
        return 0
    fi

    start_ts=$(date +%s)
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        elapsed=$(( $(date +%s) - start_ts ))
        log_and_print "${GREEN}  [OK] 步骤 $step 完成，耗时 ${elapsed}s${NC}"
    else
        local exit_code=$?
        elapsed=$(( $(date +%s) - start_ts ))
        log_and_print "${RED}  [FAIL] 步骤 $step 失败（exit=$exit_code），耗时 ${elapsed}s${NC}"
        log_and_print "${RED}  日志文件: $LOG_FILE${NC}"
        exit $exit_code
    fi
}

# ── 主流程 ────────────────────────────────────────────────
cd "$PROJECT_ROOT"

log_and_print "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
log_and_print "${BOLD}║       CNN 交易模型 - CSI800 完整流水线           ║${NC}"
log_and_print "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
log_and_print "  项目目录 : $PROJECT_ROOT"
log_and_print "  日志文件 : $LOG_FILE"
log_and_print "  起始步骤 : $FROM_STEP"
[[ -n "$ONLY_STEPS" ]] && log_and_print "  仅运行步骤: $ONLY_STEPS"
[[ -n "$PREPROCESS_ARGS" ]] && log_and_print "  预处理参数: $PREPROCESS_ARGS"
$DRY_RUN && log_and_print "  ${YELLOW}[DRY-RUN 模式]${NC}"
log_and_print "  开始时间 : $(date '+%Y-%m-%d %H:%M:%S')"

PIPELINE_START=$(date +%s)

for step in 1 2 3 4; do
    if should_run "$step"; then
        run_step "$step"
    else
        log_and_print "\n${YELLOW}  [跳过] 步骤 $step: ${STEP_NAME[$step]}${NC}"
    fi
done

PIPELINE_ELAPSED=$(( $(date +%s) - PIPELINE_START ))
PIPELINE_MIN=$(( PIPELINE_ELAPSED / 60 ))
PIPELINE_SEC=$(( PIPELINE_ELAPSED % 60 ))

log_and_print "\n${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
log_and_print "${GREEN}${BOLD}║  CSI800 流水线全部完成！总耗时 ${PIPELINE_MIN}m${PIPELINE_SEC}s${NC}${GREEN}${BOLD}              ║${NC}"
log_and_print "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
log_and_print "  结束时间 : $(date '+%Y-%m-%d %H:%M:%S')"
log_and_print "  完整日志 : $LOG_FILE"
log_and_print ""
log_and_print "  输出目录："
log_and_print "    预处理数据 : WORK_SPACE/data/csi800_processed_data/"
log_and_print "    图表数据集 : WORK_SPACE/data/csi800_stocks_dataset/stocks_CHN/"
log_and_print "    模型结果   : WORK_SPACE/csi800_model_res/"
log_and_print "    十分位分析 : WORK_SPACE/csi800_model_res/portfolio_simplified/"
log_and_print "    TopN 分析  : WORK_SPACE/csi800_model_res/topN_portfolio/"
