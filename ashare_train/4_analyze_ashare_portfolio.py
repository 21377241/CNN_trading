# -*- coding: utf-8 -*-
"""
4_analyze_ashare_portfolio.py - A股全市场组合分析（十分位 + H-L + TopN）

用法:
  python ashare_train/4_analyze_ashare_portfolio.py --spec I20R20
  python ashare_train/4_analyze_ashare_portfolio.py --spec I5R5
  python ashare_train/4_analyze_ashare_portfolio.py --spec I20R5 --exp-dir WORK_SPACE/paper/E1b_cnn_hp_I20R5
"""
import sys
import os
import glob
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_EXP_DIR_I20, ASHARE_EXP_DIR_I5, ASHARE_EXP_DIR_I20R5, ASHARE_OOS_YEARS, N_ENSEMBLES
)
from Portfolio.portfolio import run_portfolio_analysis, run_topn_portfolio_analysis

SPECS = {
    'I20R20': dict(exp_dir=ASHARE_EXP_DIR_I20,    ann_factor=12),
    'I5R5':   dict(exp_dir=ASHARE_EXP_DIR_I5,     ann_factor=52),
    'I20R5':  dict(exp_dir=ASHARE_EXP_DIR_I20R5,  ann_factor=52),
}


def find_files(exp_dir, years, n_ensembles):
    pattern = os.path.join(exp_dir, '**', 'ensem_res', f'ensem{n_ensembles}_res_*.csv')
    all_f = glob.glob(pattern, recursive=True)
    found = []
    for y in years:
        ys = [f for f in all_f if f'_res_{y}.csv' in f]
        if ys:
            ys.sort(key=os.path.getmtime, reverse=True)
            found.append(ys[0])
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', choices=list(SPECS.keys()), default='I20R20')
    parser.add_argument('--exp-dir', type=str, default=None, help='覆盖默认 exp 根目录')
    parser.add_argument('--n-ensembles', type=int, default=None)
    args = parser.parse_args()
    spec = SPECS[args.spec]
    if args.exp_dir:
        exp_dir = args.exp_dir
        if not os.path.isabs(exp_dir):
            exp_dir = os.path.join(PROJECT_ROOT, exp_dir.replace('/', os.sep))
    else:
        exp_dir = spec['exp_dir']
    ann = spec['ann_factor']
    n_ensem = args.n_ensembles if args.n_ensembles is not None else N_ENSEMBLES

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    port_dir = os.path.join(exp_dir, 'portfolio')
    topn_dir = os.path.join(exp_dir, 'topN_portfolio')
    os.makedirs(port_dir, exist_ok=True)
    os.makedirs(topn_dir, exist_ok=True)

    files = find_files(exp_dir, ASHARE_OOS_YEARS, n_ensem)
    logger.info("=" * 60)
    logger.info(f"  组合分析 [{args.spec}]  ann_factor={ann}  files={len(files)}  exp={exp_dir}")
    logger.info("=" * 60)
    if not files:
        logger.error(f"未找到预测文件于 {exp_dir}")
        return 1

    run_portfolio_analysis(
        result_files=files,
        output_dir=port_dir,
        n_ensembles=n_ensem,
        n_decile=10,
        ann_factor=ann,
    )
    run_topn_portfolio_analysis(
        result_files=files,
        output_dir=topn_dir,
        n_ensembles=n_ensem,
        top_n_list=[5, 10, 20],
        ann_factor=ann,
    )

    for d in (port_dir, topn_dir):
        logger.info(f"输出 {d}:")
        for fn in sorted(os.listdir(d)):
            logger.info(f"  {fn}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
