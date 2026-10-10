# -*- coding: utf-8 -*-
"""
5_analyze_spx_topn_portfolio.py - TopN投资组合分析

从步骤3生成的ensem5_res_{year}.csv读取预测结果，
执行TopN分组（前5、10、20只股票）、等权/市值加权分析，
输出报告与图表。

输出目录：
  WORK_SPACE/spx_model_res/topN_portfolio/
    topN_portfolio_summary.txt   - 文本报告
    topN_ew_portfolio.csv        - 等权组合统计
    topN_vw_portfolio.csv        - 市值加权组合统计
    topN_ew_monthly_returns.csv  - 等权月度收益时间序列
    topN_vw_monthly_returns.csv  - 市值加权月度收益时间序列
    topN_1_returns.png           - 年化收益图
    topN_2_sharpe.png            - Sharpe比率图
    topN_3_cumulative.png        - 累积净值图
    topN_4_timeseries.png        - 月度收益时间序列图
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
    EXP_DIR, OOS_YEARS, N_ENSEMBLES,
    BENCHMARK_MODEL_WINDOW, FREQ
)

# TopN 输出目录
TOPN_PORTFOLIO_DIR = os.path.join(
    os.path.dirname(EXP_DIR),
    'topN_portfolio'
)

# ──────────────────────────────────────────────
os.makedirs(TOPN_PORTFOLIO_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from Portfolio.portfolio import run_topn_portfolio_analysis


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
    # TopN 配置
    TOP_N_LIST = [5, 10, 20]  # 可以修改为其他值，如 [3, 5, 10, 30]

    logger.info("=" * 60)
    logger.info("  步骤 5：TopN 投资组合分析")
    logger.info("=" * 60)
    logger.info(f"测试年份: {OOS_YEARS}")
    logger.info(f"TopN 配置: {TOP_N_LIST}")
    logger.info(f"输出目录: {TOPN_PORTFOLIO_DIR}")

    # 查找预测文件
    result_files = find_result_files(test_years=OOS_YEARS, n_ensembles=N_ENSEMBLES)

    if not result_files:
        logger.error("未找到任何 OOS 预测文件！请先运行步骤 3。")
        logger.info(f"搜索目录: {EXP_DIR}")
        return 1

    logger.info(f"\n共找到 {len(result_files)} 个预测文件")

    # 执行TopN组合分析
    run_topn_portfolio_analysis(
        result_files=result_files,
        output_dir=TOPN_PORTFOLIO_DIR,
        n_ensembles=N_ENSEMBLES,
        top_n_list=TOP_N_LIST
    )

    # 验证输出
    logger.info(f"\n{'='*60}")
    logger.info("输出文件：")
    expected_outputs = [
        'topN_portfolio_summary.txt',
        'topN_ew_portfolio.csv',
        'topN_vw_portfolio.csv',
        'topN_ew_monthly_returns.csv',
        'topN_vw_monthly_returns.csv',
        'topN_selections.csv',
        'topN_1_returns.png',
        'topN_2_sharpe.png',
        'topN_3_cumulative.png',
        'topN_4_timeseries.png',
    ]
    for fn in expected_outputs:
        fp = os.path.join(TOPN_PORTFOLIO_DIR, fn)
        if os.path.exists(fp):
            size_kb = os.path.getsize(fp) / 1024
            logger.info(f"  [OK] {fn} ({size_kb:.1f} KB)")
        else:
            logger.warning(f"  [XX] {fn}（缺失）")

    # 打印报告摘要
    report_path = os.path.join(TOPN_PORTFOLIO_DIR, 'topN_portfolio_summary.txt')
    if os.path.exists(report_path):
        logger.info(f"\n{'─'*60}")
        logger.info("报告摘要：")
        with open(report_path, 'r', encoding='utf-8') as f:
            print(f.read())

    logger.info(f"\n[OK] TopN 分析完成！所有结果已保存至: {TOPN_PORTFOLIO_DIR}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
