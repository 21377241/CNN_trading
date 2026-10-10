# -*- coding: utf-8 -*-
"""
6_analyze_model_res_0_topn.py - 基于 model_res_0 的 TopN 组合分析

从 WORK_SPACE/model_res_0/model_res_0_predictions.csv 读取全量预测，
过滤到 2020-01-01 之后，执行 Top5/Top10/Top20 组合回测。

输出目录：
  WORK_SPACE/model_res_0/topN_portfolio/
    topN_portfolio_summary.txt
    topN_ew_portfolio.csv / topN_vw_portfolio.csv
    topN_ew_monthly_returns.csv / topN_vw_monthly_returns.csv
    topN_selections.csv
    topN_*.png
"""
import sys
import os
import logging
import pandas as pd

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
MODEL_RES_0_DIR  = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'model_res_0')
PRED_CSV         = os.path.join(MODEL_RES_0_DIR, 'model_res_0_predictions.csv')
OUTPUT_DIR       = os.path.join(MODEL_RES_0_DIR, 'topN_portfolio')

START_DATE = '2020-01-01'   # 仅分析该日期之后的数据
TOP_N_LIST = [5, 10, 20]

from Portfolio.portfolio import run_topn_portfolio_analysis


def main():
    logger.info("=" * 60)
    logger.info("  model_res_0 TopN 组合分析（2020-01 起）")
    logger.info("=" * 60)
    logger.info(f"预测文件 : {PRED_CSV}")
    logger.info(f"截止日期 : {START_DATE} 之后")
    logger.info(f"TopN 配置: {TOP_N_LIST}")
    logger.info(f"输出目录 : {OUTPUT_DIR}")

    if not os.path.exists(PRED_CSV):
        logger.error(f"找不到预测文件: {PRED_CSV}")
        return 1

    # 加载并过滤
    df = pd.read_csv(PRED_CSV)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df_filtered = df[df['ending_date'] >= START_DATE].copy()

    logger.info(f"原始行数: {len(df)}，过滤后: {len(df_filtered)}")
    logger.info(f"日期范围: {df_filtered['ending_date'].min().date()} "
                f"~ {df_filtered['ending_date'].max().date()}")
    logger.info(f"期末日期数: {df_filtered['ending_date'].nunique()}")

    if df_filtered.empty:
        logger.error("过滤后无数据，请检查 START_DATE 设置。")
        return 1

    # 执行 TopN 分析
    run_topn_portfolio_analysis(
        df=df_filtered,
        output_dir=OUTPUT_DIR,
        top_n_list=TOP_N_LIST
    )

    # 验证输出
    logger.info(f"\n{'='*60}")
    logger.info("输出文件：")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, fn)) / 1024
        logger.info(f"  {fn}  ({size_kb:.1f} KB)")

    report_path = os.path.join(OUTPUT_DIR, 'topN_portfolio_summary.txt')
    if os.path.exists(report_path):
        logger.info(f"\n{'─'*60}")
        with open(report_path, 'r', encoding='utf-8') as f:
            print(f.read())

    return 0


if __name__ == '__main__':
    sys.exit(main())
