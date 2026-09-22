# -*- coding: utf-8 -*-
"""E4-5: Disagreement × Recent-5 path features only (CNN vs MLP)."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import newey_west_t, norm_stock_id
from experiments.E4_recent_path.build_pathshape import PATH_COLS, build_pathshape
from experiments.E4_recent_path.paths import MIN_CS_N, NW_LAG, OUT_ROOT
from experiments.E4_recent_path.scores_pairwise import build_baseline_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e4_e45')

DISAGREE_HI = 0.8
DISAGREE_LO = 0.2
DISAGREE_MIN_N = 5

X_COLS = PATH_COLS  # recent-5 only


def run_e45() -> pd.DataFrame:
    out_dir = os.path.join(OUT_ROOT, 'e45_disagreement')
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(OUT_ROOT, 'baseline_scores.parquet')
    if not os.path.exists(scores_path):
        build_baseline_scores(scores_path)
    path_path = os.path.join(OUT_ROOT, 'pathshape.parquet')
    if not os.path.exists(path_path):
        build_pathshape(path_path)
    scores = pd.read_parquet(scores_path)
    path = pd.read_parquet(path_path)
    scores['StockID'] = norm_stock_id(scores['StockID'])
    path['StockID'] = norm_stock_id(path['StockID'])
    scores['ending_date'] = pd.to_datetime(scores['ending_date'])
    path['ending_date'] = pd.to_datetime(path['ending_date'])
    df = scores.merge(path, on=['StockID', 'ending_date'], how='inner')

    rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        rk_c = g['S_cnn'].rank(pct=True)
        rk_m = g['S_mlp'].rank(pct=True)
        pos = (rk_c > DISAGREE_HI) & (rk_m < DISAGREE_LO)  # C+ M-
        neg = (rk_c < DISAGREE_LO) & (rk_m > DISAGREE_HI)  # C- M+
        if pos.sum() < DISAGREE_MIN_N or neg.sum() < DISAGREE_MIN_N:
            continue
        for x in X_COLS:
            if x not in g.columns:
                continue
            xa = g.loc[pos, x].mean()
            xb = g.loc[neg, x].mean()
            if not (np.isfinite(xa) and np.isfinite(xb)):
                continue
            rows.append({
                'ending_date': dt, 'feature': x,
                'delta': float(xa - xb),
                'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
            })
    daily = pd.DataFrame(rows)
    daily.to_csv(os.path.join(out_dir, 'e45_daily_delta.csv'), index=False, float_format='%.6f')

    summ = []
    for feat, g in daily.groupby('feature'):
        t, p = newey_west_t(g['delta'].values, lag=NW_LAG)
        summ.append({
            'feature': feat,
            'delta_mean': float(g['delta'].mean()),
            'nw_t': t, 'nw_p': p,
            'n_days': int(len(g)),
            'mean_n_pos': float(g['n_pos'].mean()),
            'mean_n_neg': float(g['n_neg'].mean()),
        })
    summary = pd.DataFrame(summ).sort_values('nw_t', key=lambda s: s.abs(), ascending=False)
    out = os.path.join(out_dir, 'e45_delta_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return summary


if __name__ == '__main__':
    run_e45()
