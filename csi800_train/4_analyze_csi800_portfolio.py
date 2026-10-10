# -*- coding: utf-8 -*-
"""
4_analyze_csi800_portfolio.py - CSI800 投资组合分析

从步骤 3 生成的 ensem5_res_{year}.csv 读取预测结果，
执行十分位分组、等权/市值加权/H-L 分析，输出报告与图表。

输出目录：
  WORK_SPACE/csi800_model_res/portfolio_simplified/
    portfolio_summary.txt
    ew_portfolio.csv
    vw_portfolio.csv
    1_decile_returns.png
    2_sharpe_ratios.png
    3_hl_comparison.png
    4_monthly_returns_timeseries.png
"""
import sys
import os
import glob
import logging

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
from Misc.config import (
    CSI800_EXP_DIR, CSI800_OOS_YEARS, CSI800_PORTFOLIO_DIR, N_ENSEMBLES
)

# ──────────────────────────────────────────────
os.makedirs(CSI800_PORTFOLIO_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from Portfolio.portfolio import run_portfolio_analysis, run_topn_portfolio_analysis


def find_result_files(test_years: list = None,
                      n_ensembles: int = N_ENSEMBLES) -> list:
    """在 CSI800_EXP_DIR 下递归查找 ensem{n}_res_{year}.csv 文件"""
    test_years = test_years or CSI800_OOS_YEARS
    found = []

    pattern = os.path.join(CSI800_EXP_DIR, '**', 'ensem_res',
                            f'ensem{n_ensembles}_res_*.csv')
    all_files = glob.glob(pattern, recursive=True)

    for year in test_years:
        year_files = [f for f in all_files if f'_res_{year}.csv' in f]
        if year_files:
            year_files.sort(key=os.path.getmtime, reverse=True)
            found.append(year_files[0])
            logger.info(f"年份 {year}: {os.path.relpath(year_files[0], PROJECT_ROOT)}")
        else:
            logger.warning(f"年份 {year}: 未找到预测文件")

    return found


def main():
    logger.info("=" * 60)
    logger.info("  步骤 4（CSI800）：投资组合分析")
    logger.info("=" * 60)
    logger.info(f"测试年份 : {CSI800_OOS_YEARS}")
    logger.info(f"输出目录 : {CSI800_PORTFOLIO_DIR}")

    result_files = find_result_files(test_years=CSI800_OOS_YEARS, n_ensembles=N_ENSEMBLES)

    if not result_files:
        logger.error("未找到任何 OOS 预测文件！请先运行步骤 3（CSI800）。")
        logger.info(f"搜索目录: {CSI800_EXP_DIR}")
        return 1

    logger.info(f"\n共找到 {len(result_files)} 个预测文件")

    # ── 十分位分析 ──
    logger.info("\n[1/2] 十分位分组分析...")
    run_portfolio_analysis(
        result_files=result_files,
        output_dir=CSI800_PORTFOLIO_DIR,
        n_ensembles=N_ENSEMBLES,
        n_decile=10
    )

    # ── TopN 分析 ──
    TOP_N_LIST = [5, 10, 20]
    topn_dir = os.path.join(CSI800_EXP_DIR, 'topN_portfolio')
    os.makedirs(topn_dir, exist_ok=True)
    logger.info(f"\n[2/2] TopN 组合分析（TopN = {TOP_N_LIST}）...")
    run_topn_portfolio_analysis(
        result_files=result_files,
        output_dir=topn_dir,
        n_ensembles=N_ENSEMBLES,
        top_n_list=TOP_N_LIST
    )

    # ── 验证输出 ──
    logger.info(f"\n{'='*60}")
    logger.info("输出文件（十分位）：")
    expected_decile = [
        'portfolio_summary.txt', 'ew_portfolio.csv', 'vw_portfolio.csv',
        '1_decile_returns.png', '2_sharpe_ratios.png',
        '3_hl_comparison.png', '4_monthly_returns_timeseries.png',
    ]
    for fn in expected_decile:
        fp = os.path.join(CSI800_PORTFOLIO_DIR, fn)
        status = f"({os.path.getsize(fp)/1024:.1f} KB)" if os.path.exists(fp) else "缺失"
        tag = "[OK]" if os.path.exists(fp) else "[XX]"
        logger.info(f"  {tag} {fn} {status}")

    logger.info("输出文件（TopN）：")
    expected_topn = [
        'topN_portfolio_summary.txt', 'topN_ew_portfolio.csv',
        'topN_1_returns.png', 'topN_3_cumulative.png',
    ]
    for fn in expected_topn:
        fp = os.path.join(topn_dir, fn)
        status = f"({os.path.getsize(fp)/1024:.1f} KB)" if os.path.exists(fp) else "缺失"
        tag = "[OK]" if os.path.exists(fp) else "[XX]"
        logger.info(f"  {tag} {fn} {status}")

    # ── 打印摘要 ──
    for report_path, label in [
        (os.path.join(CSI800_PORTFOLIO_DIR, 'portfolio_summary.txt'), '十分位'),
        (os.path.join(topn_dir, 'topN_portfolio_summary.txt'), 'TopN'),
    ]:
        if os.path.exists(report_path):
            logger.info(f"\n{'─'*60}")
            logger.info(f"{label} 报告摘要：")
            with open(report_path, 'r', encoding='utf-8') as f:
                print(f.read())

    logger.info(f"\n[OK] 步骤 4（CSI800）完成！")
    logger.info(f"  十分位结果: {CSI800_PORTFOLIO_DIR}")
    logger.info(f"  TopN 结果: {topn_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
