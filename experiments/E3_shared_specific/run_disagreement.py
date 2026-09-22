# -*- coding: utf-8 -*-
"""E3-1 Disagreement sample characterization."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.paths import (
    DISAGREE_HI, DISAGREE_LO, DISAGREE_MIN_N, NW_LAG, OUT_ROOT,
)
from experiments.E3_shared_specific.utils import newey_west_t, norm_stock_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_dis')

PAIRS = [
    ('S_cnn', 'S_mlp', 'cnn_mlp'),
    ('S_cnn', 'S_hgb', 'cnn_hgb'),
    ('S_mlp', 'S_hgb', 'mlp_hgb'),
]

X_COLS = [
    'r_1', 'r_5', 'r_10', 'r_20',
    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'range_rel',
    'v_end', 'v_end_over_mean', 'ret_std', 'overnight', 'log_mcap', 'hit_limit',
]


def run_disagreement() -> pd.DataFrame:
    scores = pd.read_parquet(os.path.join(OUT_ROOT, 'scores.parquet'))
    geom = pd.read_parquet(os.path.join(OUT_ROOT, 'geometry.parquet'))
    scores['StockID'] = norm_stock_id(scores['StockID'])
    geom['StockID'] = norm_stock_id(geom['StockID'])
    scores['ending_date'] = pd.to_datetime(scores['ending_date'])
    geom['ending_date'] = pd.to_datetime(geom['ending_date'])
    df = scores.merge(geom, on=['StockID', 'ending_date'], how='inner')
    logger.info(f'disagree panel rows={len(df):,}')

    rows = []
    for sa, sb, name in PAIRS:
        for dt, g in df.groupby('ending_date'):
            rk_a = g[sa].rank(pct=True)
            rk_b = g[sb].rank(pct=True)
            pos = (rk_a > DISAGREE_HI) & (rk_b < DISAGREE_LO)
            neg = (rk_a < DISAGREE_LO) & (rk_b > DISAGREE_HI)
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
                    'pair': name, 'ending_date': dt, 'feature': x,
                    'delta': float(xa - xb),
                    'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
                })
    daily = pd.DataFrame(rows)
    daily.to_csv(os.path.join(OUT_ROOT, 'e31_daily_delta.csv'), index=False, float_format='%.6f')

    summ = []
    for (pair, feat), g in daily.groupby(['pair', 'feature']):
        t, p = newey_west_t(g['delta'].values, lag=NW_LAG)
        summ.append({
            'pair': pair, 'feature': feat,
            'delta_mean': float(g['delta'].mean()),
            'nw_t': t, 'nw_p': p,
            'n_days': int(len(g)),
            'mean_n_pos': float(g['n_pos'].mean()),
            'mean_n_neg': float(g['n_neg'].mean()),
        })
    summary = pd.DataFrame(summ).sort_values(['pair', 'feature'])
    out = os.path.join(OUT_ROOT, 'e31_delta_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.head(20).to_string(index=False)}')
    return summary


if __name__ == '__main__':
    run_disagreement()
