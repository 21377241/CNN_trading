# -*- coding: utf-8 -*-
"""Evaluate specific scores: IC + portfolio + deltas vs baseline."""
from __future__ import annotations

import logging
import os

import pandas as pd

from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from experiments.E4_recent_path.paths import ANN, OUT_ROOT
from Portfolio.portfolio import run_portfolio_analysis

logger = logging.getLogger('e4_eval')


def eval_specific_df(
    sp_df: pd.DataFrame,
    tag: str,
    out_dir: str,
    sp_col: str = 'S_sp',
) -> dict:
    df = sp_df[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    df['up_prob'] = sp_df.groupby('ending_date')[sp_col].rank(pct=True).values
    ic = summarize_ic(cross_section_ic(df, 'up_prob'))
    mdir = os.path.join(out_dir, tag)
    os.makedirs(mdir, exist_ok=True)
    tmp = os.path.join(mdir, 'pred_for_port.csv')
    df.to_csv(tmp, index=False)
    run_portfolio_analysis(result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=ANN)
    hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
    return {'tag': tag, **ic, **hl, 'n': len(df)}


def attach_deltas(summary: pd.DataFrame, baseline_tag: str = 'baseline') -> pd.DataFrame:
    s = summary.copy()
    base = s.loc[s['tag'] == baseline_tag].iloc[0]
    s['delta_rank_ic'] = float(base['rank_ic_mean']) - s['rank_ic_mean']
    s['delta_ew_sharpe'] = float(base['ew_hl_sharpe']) - s['ew_hl_sharpe']
    s['sharpe_drop_pct'] = s['delta_ew_sharpe'] / abs(float(base['ew_hl_sharpe'])) * 100.0
    s.loc[s['tag'] == baseline_tag, 'sharpe_drop_pct'] = 0.0
    s.loc[s['tag'] == baseline_tag, 'delta_rank_ic'] = 0.0
    s.loc[s['tag'] == baseline_tag, 'delta_ew_sharpe'] = 0.0
    return s


def ensure_out(*parts: str) -> str:
    p = os.path.join(OUT_ROOT, *parts)
    os.makedirs(p, exist_ok=True)
    return p
