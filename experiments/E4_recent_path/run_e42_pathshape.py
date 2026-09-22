# -*- coding: utf-8 -*-
"""E4-2: CumRet5 + PathShape nested R² (ΔR²) on pairwise specific scores."""
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

from experiments.E3_shared_specific.utils import cs_zscore, newey_west_t, norm_stock_id
from experiments.E4_recent_path.build_pathshape import PATH_COLS, build_pathshape
from experiments.E4_recent_path.paths import MIN_CS_N, NW_LAG, OUT_ROOT
from experiments.E4_recent_path.scores_pairwise import build_baseline_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e4_e42')

SHAPE_COLS = ['LateEarly', 'PathVol', 'UpCount5', 'MaxDD5', 'Reversal5', 'Monotonicity']
Y_COLS = ['S_cnn_sp', 'S_mlp_sp']


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < MIN_CS_N:
        return np.nan
    yy, XX = y[m], X[m]
    if XX.shape[1] == 0:
        return np.nan
    if XX.shape[1] == 1 and np.nanstd(XX[:, 0]) < 1e-12:
        return np.nan
    pred = LinearRegression().fit(XX, yy).predict(XX)
    ss = np.sum((yy - yy.mean()) ** 2)
    if ss < 1e-12:
        return np.nan
    return float(1.0 - np.sum((yy - pred) ** 2) / ss)


def run_e42() -> pd.DataFrame:
    out_dir = os.path.join(OUT_ROOT, 'e42_pathshape')
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(OUT_ROOT, 'baseline_scores.parquet')
    if not os.path.exists(scores_path):
        build_baseline_scores(scores_path)
    scores = pd.read_parquet(scores_path)
    path_path = os.path.join(OUT_ROOT, 'pathshape.parquet')
    if not os.path.exists(path_path):
        build_pathshape(path_path)
    path = pd.read_parquet(path_path)
    scores['StockID'] = norm_stock_id(scores['StockID'])
    path['StockID'] = norm_stock_id(path['StockID'])
    scores['ending_date'] = pd.to_datetime(scores['ending_date'])
    path['ending_date'] = pd.to_datetime(path['ending_date'])
    df = scores.merge(path, on=['StockID', 'ending_date'], how='inner')
    logger.info(f'e42 panel rows={len(df):,}')

    daily_rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        for ycol in Y_COLS:
            y = cs_zscore(g[ycol]).values
            x_cum = cs_zscore(g['CumRet5']).values.reshape(-1, 1)
            x_shape = np.column_stack([cs_zscore(g[c]).values for c in SHAPE_COLS if c in g.columns])
            r2_c = _r2(y, x_cum)
            r2_f = _r2(y, np.column_stack([x_cum, x_shape]))
            # single shape features incremental (optional detail)
            for sc in SHAPE_COLS:
                if sc not in g.columns:
                    continue
                xs = np.column_stack([x_cum, cs_zscore(g[sc]).values.reshape(-1, 1)])
                r2_s = _r2(y, xs)
                daily_rows.append({
                    'ending_date': dt, 'score': ycol, 'spec': sc,
                    'r2_cum': r2_c, 'r2_full_or_single': r2_s,
                    'delta_r2': (r2_s - r2_c) if np.isfinite(r2_s) and np.isfinite(r2_c) else np.nan,
                    'kind': 'single',
                })
            daily_rows.append({
                'ending_date': dt, 'score': ycol, 'spec': 'ALL_SHAPE',
                'r2_cum': r2_c, 'r2_full_or_single': r2_f,
                'delta_r2': (r2_f - r2_c) if np.isfinite(r2_f) and np.isfinite(r2_c) else np.nan,
                'kind': 'nested',
            })

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(os.path.join(out_dir, 'e42_daily_delta_r2.csv'), index=False, float_format='%.6f')

    summ = []
    for (score, spec, kind), g in daily.groupby(['score', 'spec', 'kind']):
        t, p = newey_west_t(g['delta_r2'].values, lag=NW_LAG)
        summ.append({
            'score': score, 'spec': spec, 'kind': kind,
            'r2_cum_mean': float(g['r2_cum'].mean()),
            'r2_aug_mean': float(g['r2_full_or_single'].mean()),
            'delta_r2_mean': float(g['delta_r2'].mean()),
            'nw_t': t, 'nw_p': p, 'n_days': int(len(g)),
        })
    summary = pd.DataFrame(summ).sort_values(['score', 'kind', 'delta_r2_mean'], ascending=[True, True, False])
    out = os.path.join(out_dir, 'e42_delta_r2_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return summary


if __name__ == '__main__':
    run_e42()
