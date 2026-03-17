# -*- coding: utf-8 -*-
"""
4_analyze_spx_portfolio.py - 投资组合分析

从步骤 3 生成的 ensem5_res_{year}.csv 读取预测结果，
执行十分位分组、等权/市值加权/H-L 分析，输出报告与图表。

输出目录：
  WORK_SPACE/spx_model_res/portfolio_simplified/
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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
from Misc.config import (
    EXP_DIR, PORTFOLIO_DIR, OOS_YEARS, N_ENSEMBLES,
    BENCHMARK_MODEL_WINDOW, FREQ
)

# ──────────────────────────────────────────────
os.makedirs(PORTFOLIO_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from Portfolio.portfolio import run_portfolio_analysis


def find_result_files(test_years: list = None,
                      n_ensembles: int = N_ENSEMBLES) -> list:
    """
    在 EXP_DIR 下递归查找 ensem{n}_res_{year}.csv 文件。

    Returns:
        文件路径列表（按年份排序）
    """
    test_years = test_years or OOS_YEARS
    found = []

    # 递归搜索
    pattern = os.path.join(EXP_DIR, '**', 'ensem_res',
                            f'ensem{n_ensembles}_res_*.csv')
    all_files = glob.glob(pattern, recursive=True)

    for year in test_years:
        year_files = [f for f in all_files if f'_res_{year}.csv' in f]
        if year_files:
            # 若有多个（不同实验），取最新的
            year_files.sort(key=os.path.getmtime, reverse=True)
            found.append(year_files[0])
            logger.info(f"年份 {year}: {os.path.relpath(year_files[0], PROJECT_ROOT)}")
        else:
            logger.warning(f"年份 {year}: 未找到预测文件")

    return found


def main():
    logger.info("=" * 60)
    logger.info("  步骤 4：投资组合分析")
    logger.info("=" * 60)
    logger.info(f"测试年份: {OOS_YEARS}")
    logger.info(f"输出目录: {PORTFOLIO_DIR}")

    # 查找预测文件
    result_files = find_result_files(test_years=OOS_YEARS, n_ensembles=N_ENSEMBLES)

    if not result_files:
        logger.error("未找到任何 OOS 预测文件！请先运行步骤 3。")
        logger.info(f"搜索目录: {EXP_DIR}")
        return 1

    logger.info(f"\n共找到 {len(result_files)} 个预测文件")

    # 执行组合分析
    run_portfolio_analysis(
        result_files=result_files,
        output_dir=PORTFOLIO_DIR,
        n_ensembles=N_ENSEMBLES,
        n_decile=10
    )

    # 验证输出
    logger.info(f"\n{'='*60}")
    logger.info("输出文件：")
    expected_outputs = [
        'portfolio_summary.txt',
        'ew_portfolio.csv',
        'vw_portfolio.csv',
        '1_decile_returns.png',
        '2_sharpe_ratios.png',
        '3_hl_comparison.png',
        '4_monthly_returns_timeseries.png',
    ]
    for fn in expected_outputs:
        fp = os.path.join(PORTFOLIO_DIR, fn)
        if os.path.exists(fp):
            size_kb = os.path.getsize(fp) / 1024
            logger.info(f"  [OK] {fn} ({size_kb:.1f} KB)")
        else:
            logger.warning(f"  [XX] {fn}（缺失）")

    # 打印报告摘要
    report_path = os.path.join(PORTFOLIO_DIR, 'portfolio_summary.txt')
    if os.path.exists(report_path):
        logger.info(f"\n{'─'*60}")
        logger.info("报告摘要：")
        with open(report_path, 'r', encoding='utf-8') as f:
            print(f.read())

    logger.info(f"\n[OK] 步骤 4 完成！所有结果已保存至: {PORTFOLIO_DIR}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

