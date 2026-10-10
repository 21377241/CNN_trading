#!/usr/bin/env bash
# =============================================================================
# run_all.sh  - 同时运行 SPX + CSI800 完整流水线
#
# 用法：
#   bash scripts/run_all.sh               # 顺序运行 SPX → CSI800
#   bash scripts/run_all.sh --spx-only    # 只运行 SPX
#   bash scripts/run_all.sh --csi-only    # 只运行 CSI800
#   bash scripts/run_all.sh --from 3      # 两条流水线都从步骤 3 开始
#   bash scripts/run_all.sh --dry-run     # 仅打印命令
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

RUN_SPX=true
RUN_CSI=true
FROM_ARG=""
DRY_RUN_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --spx-only) RUN_CSI=false; shift ;;
        --csi-only) RUN_SPX=false; shift ;;
        --from)     FROM_ARG="--from $2"; shift 2 ;;
        --dry-run)  DRY_RUN_ARG="--dry-run"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║       CNN 交易模型 - 全量流水线调度              ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

TOTAL_START=$(date +%s)

if $RUN_SPX; then
    echo -e "${BOLD}▶ 启动 SPX 流水线...${NC}"
    bash "$SCRIPT_DIR/run_spx.sh" $FROM_ARG $DRY_RUN_ARG
    echo -e "${GREEN}✓ SPX 流水线完成${NC}\n"
fi

if $RUN_CSI; then
    echo -e "${BOLD}▶ 启动 CSI800 流水线...${NC}"
    bash "$SCRIPT_DIR/run_csi800.sh" $FROM_ARG $DRY_RUN_ARG
    echo -e "${GREEN}✓ CSI800 流水线完成${NC}\n"
fi

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo -e "${GREEN}${BOLD}全部流水线完成！总耗时 $(($TOTAL_ELAPSED/60))m$(($TOTAL_ELAPSED%60))s${NC}"
