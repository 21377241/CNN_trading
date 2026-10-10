# -*- coding: utf-8 -*-
"""Evaluate Shared / Specific: Rank IC + H-L portfolio."""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from experiments.E3_shared_specific.paths import ANN, OUT_ROOT
from experiments.E3_shared_specific.utils import newey_west_t
from Portfolio.portfolio import run_portfolio_analysis, write_port_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_eval')

SCORE_COLS = ['S_sh', 'S_cnn_sp', 'S_mlp_sp', 'S_hgb_sp', 'S_cnn', 'S_mlp', 'S_hgb']


def run_eval(scores_path: str | None = None) -> pd.DataFrame:
    scores_path = scores_path or os.path.join(OUT_ROOT, 'scores.parquet')
    df0 = pd.read_parquet(scores_path)
    df0['ending_date'] = pd.to_datetime(df0['ending_date'])
    port_root = os.path.join(OUT_ROOT, 'e30_portfolio')
    os.makedirs(port_root, exist_ok=True)
    rows = []
    for col in SCORE_COLS:
        if col not in df0.columns:
            continue
        df = df0[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
        df['up_prob'] = df0.groupby('ending_date')[col].rank(pct=True).values
        ic_df = cross_section_ic(df, 'up_prob')
        ic = summarize_ic(ic_df)
        t_ic, p_ic = newey_west_t(ic_df['rank_ic'].values, lag=4)
        mdir = os.path.join(port_root, col)
        os.makedirs(mdir, exist_ok=True)
        tmp = write_port_input(df, mdir)
        run_portfolio_analysis(
            result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=ANN,
        )
        hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
        rows.append({
            'score': col,
            **ic,
            'rank_ic_nw_t': t_ic,
            'rank_ic_nw_p': p_ic,
            **hl,
            'n': len(df),
        })
        logger.info(f'{col}: rank_ic={ic["rank_ic_mean"]:.4f} ew_sharpe={hl.get("ew_hl_sharpe")}')
    summary = pd.DataFrame(rows)
    out = os.path.join(OUT_ROOT, 'e30_score_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return summary


if __name__ == '__main__':
    run_eval()
