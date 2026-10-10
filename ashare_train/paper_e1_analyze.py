# -*- coding: utf-8 -*-
"""
paper_e1_analyze.py — 论文 E1：Test 期十分位 / H-L / TopN

用法:
  python ashare_train/paper_e1_analyze.py
"""
import sys
import os
import glob
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import PAPER_EXP_DIR_E1, PAPER_TEST_YEARS, N_ENSEMBLES, PAPER_DOCS_DIR
from Portfolio.portfolio import run_portfolio_analysis, run_topn_portfolio_analysis

ANN_FACTOR = 52


def find_files(exp_dir, years):
    pattern = os.path.join(exp_dir, '**', 'ensem_res', f'ensem{N_ENSEMBLES}_res_*.csv')
    all_f = glob.glob(pattern, recursive=True)
    found = []
    for y in years:
        ys = [f for f in all_f if f'_res_{y}.csv' in f]
        if ys:
            ys.sort(key=os.path.getmtime, reverse=True)
            found.append(ys[0])
    return found


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    files = find_files(PAPER_EXP_DIR_E1, PAPER_TEST_YEARS)
    logger.info("=" * 60)
    logger.info(f"  论文 E1 回测  files={len(files)}  years={PAPER_TEST_YEARS}")
    logger.info("=" * 60)
    if len(files) < len(PAPER_TEST_YEARS):
        logger.warning(f"预测文件不齐: 找到 {len(files)}/{len(PAPER_TEST_YEARS)}")
    if not files:
        logger.error("无预测文件，请先运行 paper_e1_train_i20r5.py")
        return 1

    note = (
        "论文 E1 I20/R5；chronological Train2005–2014 / Val2015–2017 / Test2018–2025；"
        "ann_factor=52；未剔涨停"
    )
    run_portfolio_analysis(
        result_files=files,
        output_dir=os.path.join(PAPER_EXP_DIR_E1, 'portfolio'),
        n_ensembles=N_ENSEMBLES,
        ann_factor=ANN_FACTOR,
    )
    run_topn_portfolio_analysis(
        result_files=files,
        output_dir=os.path.join(PAPER_EXP_DIR_E1, 'topN_portfolio'),
        n_ensembles=N_ENSEMBLES,
        top_n_list=[5, 10, 20],
        ann_factor=ANN_FACTOR,
        note=note,
    )

    # 同步摘要到 docs
    src = os.path.join(PAPER_EXP_DIR_E1, 'portfolio', 'portfolio_summary.txt')
    dst = os.path.join(PAPER_DOCS_DIR, 'paper_results_E1.md')
    os.makedirs(PAPER_DOCS_DIR, exist_ok=True)
    if os.path.exists(src):
        with open(src, 'r', encoding='utf-8') as f:
            body = f.read()
        topn = os.path.join(PAPER_EXP_DIR_E1, 'topN_portfolio', 'topN_portfolio_summary.txt')
        topn_body = ''
        if os.path.exists(topn):
            with open(topn, 'r', encoding='utf-8') as f:
                topn_body = f.read()
        with open(dst, 'w', encoding='utf-8') as f:
            f.write("# 论文 E1 结果：I20/R5（chronological）\n\n")
            f.write(f"- 切分：Train 2005–2014 / Val 2015–2017 / Test 2018–2025\n")
            f.write(f"- 预测文件数：{len(files)}\n")
            f.write(f"- 目录：`{PAPER_EXP_DIR_E1}`\n\n")
            f.write("## 十分位 / H-L\n\n```\n")
            f.write(body)
            f.write("\n```\n\n## TopN\n\n```\n")
            f.write(topn_body)
            f.write("\n```\n")
        logger.info(f"结果摘要: {dst}")

    logger.info("[OK] E1 回测完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())
