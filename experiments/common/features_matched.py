# -*- coding: utf-8 -*-
"""
Matched tabular features aligned with chart rendering:
  adjust_price (Close0=1 via Ret) + MA20 on adjusted close + window Vol min-max.
  No MA5.

Optional chart-orientation indicators (Bøjstrup A3, without MA5):
  δ^(20)  MA20 availability: 1{ enough closes for full lag }
  u^P     candle orientation: 1{C >= O}
  u^V     volume-bar orientation (paper A3.19)

  Base:     20 * 6 = 120
  + inds:   20 * 9 = 180
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.common.panel import norm_stock_id


def adjust_price_arrays(open_, high, low, close, ret) -> tuple:
    T = len(close)
    close_adj = np.empty(T, dtype=np.float64)
    close_adj[0] = 1.0
    for t in range(1, T):
        r = ret[t]
        if np.isnan(r):
            r = 0.0
        close_adj[t] = close_adj[t - 1] * (1.0 + r)
    ratio = np.where(np.abs(close) > 1e-10, close_adj / close, 1.0)
    o = open_ * ratio
    h = high * ratio
    l = low * ratio
    c = close_adj
    h = np.maximum(np.maximum(h, o), c)
    l = np.minimum(np.minimum(l, o), c)
    return o, h, l, c


def candle_and_volume_orientation(o, c) -> tuple:
    """Bøjstrup (A3.18)–(A3.19): u^P and u^V."""
    T = len(c)
    u_p = (c >= o).astype(np.float64)
    u_v = np.empty(T, dtype=np.float64)
    for j in range(T):
        if c[j] > o[j]:
            u_v[j] = 1.0
        elif c[j] < o[j]:
            u_v[j] = 0.0
        elif j == 0:
            u_v[j] = 1.0
        else:
            u_v[j] = 1.0 if c[j] >= c[j - 1] else 0.0
    return u_p, u_v


def window_to_features(
    o, h, l, c, v,
    ma_lag: int = 20,
    include_indicators: bool = False,
) -> np.ndarray:
    T = len(c)
    ma = np.empty(T, dtype=np.float64)
    delta = np.empty(T, dtype=np.float64)
    for t in range(T):
        left = max(0, t + 1 - ma_lag)
        n_obs = t + 1 - left
        # Match Jiang chart: rolling(..., min_periods=1)
        ma[t] = c[left:t + 1].mean()
        # Full-lag availability (paper δ^(k)); partial MA still kept to match chart
        delta[t] = 1.0 if n_obs >= ma_lag else 0.0

    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        v_n = np.zeros(T, dtype=np.float64)
    else:
        v_n = (v - vmin) / (vmax - vmin)
        v_n = np.nan_to_num(v_n, nan=0.0)

    if not include_indicators:
        feats = np.stack([o, h, l, c, ma, v_n], axis=1)
    else:
        u_p, u_v = candle_and_volume_orientation(o, c)
        feats = np.stack([o, h, l, c, ma, delta, v_n, u_p, u_v], axis=1)
    return feats.reshape(-1).astype(np.float32)


def feature_dim(window_size: int = 20, include_indicators: bool = False) -> int:
    per_day = 9 if include_indicators else 6
    return window_size * per_day


def build_feature_matrix(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    window_size: int = 20,
    ma_lag: int = 20,
    include_indicators: bool = False,
    show_progress: bool = True,
) -> tuple:
    daily = daily.copy()
    daily['StockID'] = daily['StockID'].map(norm_stock_id)
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values(['StockID', 'Date'])

    panel = panel.copy().reset_index(drop=True)
    panel['StockID'] = panel['StockID'].map(norm_stock_id)
    panel['ending_date'] = pd.to_datetime(panel['ending_date'])

    grouped = {}
    for sid, g in daily.groupby('StockID', sort=False):
        grouped[sid] = {
            'dates': g['Date'].values.astype('datetime64[ns]'),
            'Open': g['Open'].values.astype(np.float64),
            'High': g['High'].values.astype(np.float64),
            'Low': g['Low'].values.astype(np.float64),
            'Close': g['Close'].values.astype(np.float64),
            'Ret': g['Ret'].values.astype(np.float64),
            'Vol': g['Vol'].values.astype(np.float64),
        }

    xs = []
    keep_pos = []
    stock_groups = list(panel.groupby('StockID', sort=False))
    if show_progress:
        stock_groups = tqdm(stock_groups, desc='features_by_stock')

    expected_dim = feature_dim(window_size, include_indicators)
    for sid, psub in stock_groups:
        g = grouped.get(sid)
        if g is None:
            continue
        dates = g['dates']
        ends = psub['ending_date'].values.astype('datetime64[ns]')
        pos = np.searchsorted(dates, ends, side='right') - 1
        for local_i, (p, ed) in enumerate(zip(pos, ends)):
            if p < window_size - 1:
                continue
            if dates[p] != ed:
                continue
            sl = slice(p - window_size + 1, p + 1)
            o, h, l, c = adjust_price_arrays(
                g['Open'][sl], g['High'][sl], g['Low'][sl], g['Close'][sl], g['Ret'][sl]
            )
            v = g['Vol'][sl]
            if np.any(~np.isfinite(c)) or np.any(~np.isfinite(o)):
                continue
            feat = window_to_features(
                o, h, l, c, v, ma_lag=ma_lag, include_indicators=include_indicators
            )
            if feat.shape[0] != expected_dim or not np.all(np.isfinite(feat)):
                continue
            xs.append(feat)
            keep_pos.append(psub.index[local_i])

    if not xs:
        return np.zeros((0, expected_dim), dtype=np.float32), panel.iloc[0:0].copy()

    X = np.stack(xs, axis=0)
    meta = panel.loc[keep_pos].reset_index(drop=True)
    return X, meta
