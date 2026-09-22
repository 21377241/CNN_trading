# -*- coding: utf-8 -*-
"""E3-2 Spec × feature-block cross-sectional R²."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.paths import MIN_CS_N, OUT_ROOT
from experiments.E3_shared_specific.utils import cs_zscore, norm_stock_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_r2')

BLOCKS = {
    'Recent': ['r_1', 'r_5', 'r_10', 'r_20'],
    'Geometry': ['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'range_rel'],
    'Volume': ['v_end', 'v_end_over_mean'],
    'Volatility': ['ret_std'],
    'Gap': ['overnight'],
    'Size': ['log_mcap'],
    'Limit': ['hit_limit'],
}

Y_COLS = ['S_sh', 'S_cnn_sp', 'S_mlp_sp', 'S_hgb_sp']


def _block_r2(y: np.ndarray, X: np.ndarray) -> float:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < MIN_CS_N or X.shape[1] == 0:
        return np.nan
    yy, XX = y[m], X[m]
    if XX.shape[1] == 1 and np.nanstd(XX[:, 0]) < 1e-12:
        return np.nan
    pred = LinearRegression().fit(XX, yy).predict(XX)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - np.sum((yy - pred) ** 2) / ss_tot)


def run_spec_vs_features() -> pd.DataFrame:
    scores = pd.read_parquet(os.path.join(OUT_ROOT, 'scores.parquet'))
    geom = pd.read_parquet(os.path.join(OUT_ROOT, 'geometry.parquet'))
    scores['StockID'] = norm_stock_id(scores['StockID'])
    geom['StockID'] = norm_stock_id(geom['StockID'])
    scores['ending_date'] = pd.to_datetime(scores['ending_date'])
    geom['ending_date'] = pd.to_datetime(geom['ending_date'])
    df = scores.merge(geom, on=['StockID', 'ending_date'], how='inner')

    daily_rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        for ycol in Y_COLS:
            y = cs_zscore(g[ycol]).values
            for bname, cols in BLOCKS.items():
                cols = [c for c in cols if c in g.columns]
                if not cols:
                    continue
                X = np.column_stack([cs_zscore(g[c]).values for c in cols])
                r2 = _block_r2(y, X)
                daily_rows.append({'ending_date': dt, 'score': ycol, 'block': bname, 'r2': r2})
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(os.path.join(OUT_ROOT, 'e32_daily_r2.csv'), index=False, float_format='%.6f')

    summary = (
        daily.groupby(['score', 'block'], as_index=False)['r2']
        .agg(r2_mean='mean', r2_median='median', n_days='count')
    )
    out = os.path.join(OUT_ROOT, 'e32_block_r2.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return summary


if __name__ == '__main__':
    run_spec_vs_features()
