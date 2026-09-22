# -*- coding: utf-8 -*-
"""Build forward horizon / overnight / intraday labels + TurnoverProxy_5 for E5 panel."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import norm_stock_id
from experiments.E5_mechanism.paths import ASHARE_RET, OUT_ROOT, PATHSHAPE_PATH, SCORES_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_fwd')

EPS = 1e-12


def _compound(rets: np.ndarray) -> np.ndarray:
    """rets: (n, k) simple daily returns → cumulative simple return."""
    return np.expm1(np.nansum(np.log1p(rets), axis=1))


def build_daily_features() -> pd.DataFrame:
    cols = ['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Vol', 'MarketCap', 'Ret', 'Ret_5d', 'Ret_week']
    d = pd.read_feather(ASHARE_RET, columns=cols)
    d['StockID'] = norm_stock_id(d['StockID'])
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values(['StockID', 'Date']).reset_index(drop=True)
    g = d.groupby('StockID', sort=False)

    # forward daily simple returns r1..r10
    for h in range(1, 11):
        d[f'r{h}'] = g['Ret'].shift(-h)

    # cumulative windows
    rmat = np.column_stack([d[f'r{h}'].values for h in range(1, 11)])
    d['R_1_2'] = _compound(rmat[:, :2])
    d['R_3_5'] = _compound(rmat[:, 2:5])
    d['R_1_5'] = _compound(rmat[:, :5])
    d['R_1_10'] = _compound(rmat[:, :10])

    # overnight / intraday log returns for next 5 days
    o = d['Open'].values.astype(float)
    c = d['Close'].values.astype(float)
    # within-stock shifts
    o_f = {h: g['Open'].shift(-h).values.astype(float) for h in range(1, 6)}
    c_f = {h: g['Close'].shift(-h).values.astype(float) for h in range(1, 6)}
    c_prev = {h: g['Close'].shift(-(h - 1)).values.astype(float) for h in range(1, 6)}  # C_{t+h-1}; h=1 → C_t

    on_cols = []
    id_cols = []
    for h in range(1, 6):
        on = np.log(o_f[h] / (c_prev[h] + EPS))
        id_ = np.log(c_f[h] / (o_f[h] + EPS))
        bad_on = ~(np.isfinite(o_f[h]) & np.isfinite(c_prev[h]) & (o_f[h] > 0) & (c_prev[h] > 0))
        bad_id = ~(np.isfinite(c_f[h]) & np.isfinite(o_f[h]) & (c_f[h] > 0) & (o_f[h] > 0))
        on[bad_on] = np.nan
        id_[bad_id] = np.nan
        d[f'on{h}'] = on
        d[f'id{h}'] = id_
        on_cols.append(f'on{h}')
        id_cols.append(f'id{h}')

    d['ON_1_5'] = d[on_cols].sum(axis=1, min_count=1)
    d['ID_1_5'] = d[id_cols].sum(axis=1, min_count=1)

    # TurnoverProxy_5: past 5 days mean(Vol/MarketCap), inclusive of t
    mcap = d['MarketCap'].values.astype(float)
    vol = d['Vol'].values.astype(float)
    to = np.where(np.isfinite(mcap) & (mcap > 0) & np.isfinite(vol), vol / mcap, np.nan)
    d['_to'] = to
    d['TurnoverProxy_5'] = (
        d.groupby('StockID', sort=False)['_to']
        .transform(lambda s: s.rolling(5, min_periods=3).mean())
    )
    d.drop(columns=['_to'], inplace=True)

    keep = (
        ['Date', 'StockID', 'Ret_5d', 'Ret_week', 'MarketCap', 'TurnoverProxy_5']
        + [f'r{h}' for h in range(1, 11)]
        + ['R_1_2', 'R_3_5', 'R_1_5', 'R_1_10']
        + [f'on{h}' for h in range(1, 6)] + [f'id{h}' for h in range(1, 6)]
        + ['ON_1_5', 'ID_1_5']
    )
    return d[keep].rename(columns={'Date': 'ending_date'})


def build_forward_panel(out_path: str | None = None) -> pd.DataFrame:
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = out_path or os.path.join(OUT_ROOT, 'forward_labels.parquet')
    if os.path.exists(out_path):
        logger.info(f'resume {out_path}')
        return pd.read_parquet(out_path)

    logger.info('building daily forward features…')
    daily = build_daily_features()
    scores = pd.read_parquet(SCORES_PATH)
    scores['StockID'] = norm_stock_id(scores['StockID'])
    scores['ending_date'] = pd.to_datetime(scores['ending_date'])
    keys = ['StockID', 'ending_date']
    score_cols = keys + ['S_cnn', 'S_mlp', 'S_cnn_sp', 'S_mlp_sp', 'ret_val', 'binary_label']
    # MarketCap from scores if present
    if 'MarketCap' in scores.columns:
        score_cols.append('MarketCap')

    logger.info('merging panel…')
    panel = scores[score_cols].merge(daily, on=keys, how='left', suffixes=('', '_d'))
    if 'MarketCap_d' in panel.columns:
        panel['MarketCap'] = panel['MarketCap'].fillna(panel['MarketCap_d'])
        panel.drop(columns=['MarketCap_d'], inplace=True)

    # pathshape states
    if os.path.exists(PATHSHAPE_PATH):
        path = pd.read_parquet(PATHSHAPE_PATH)
        path['StockID'] = norm_stock_id(path['StockID'])
        path['ending_date'] = pd.to_datetime(path['ending_date'])
        use = [c for c in ['CumRet5', 'PathVol', 'MaxDD5', 'LateEarly', 'VolEndOverMean5'] if c in path.columns]
        panel = panel.merge(path[keys + use], on=keys, how='left')

    panel['log_mcap'] = np.where(
        np.isfinite(panel['MarketCap'].values) & (panel['MarketCap'].values > 0),
        np.log(panel['MarketCap'].values.astype(float)),
        np.nan,
    )
    panel.to_parquet(out_path, index=False)
    logger.info(f'forward panel rows={len(panel):,} -> {out_path}')
    return panel


if __name__ == '__main__':
    build_forward_panel()
