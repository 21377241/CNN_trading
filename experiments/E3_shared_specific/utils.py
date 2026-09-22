# -*- coding: utf-8 -*-
"""Shared helpers for E3."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def norm_stock_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)


def cs_zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return (s - mu) / sd


def newey_west_t(x: np.ndarray, lag: int = 4) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return float('nan'), float('nan')
    mu = x.mean()
    gamma0 = np.mean((x - mu) ** 2)
    var = gamma0
    for L in range(1, min(lag, T - 1) + 1):
        w = 1.0 - L / (lag + 1)
        gamma = np.mean((x[L:] - mu) * (x[:-L] - mu))
        var += 2 * w * gamma
    se = np.sqrt(max(var, 0.0) / T)
    if se < 1e-15:
        return float('nan'), float('nan')
    t = mu / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=T - 1))
    return float(t), float(p)


def daily_triple_resid(sc: np.ndarray, sm: np.ndarray, sh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS residuals: each score on the other two."""
    Xmh = np.column_stack([sm, sh])
    Xch = np.column_stack([sc, sh])
    Xcm = np.column_stack([sc, sm])
    u_c = sc - LinearRegression().fit(Xmh, sc).predict(Xmh)
    u_m = sm - LinearRegression().fit(Xch, sm).predict(Xch)
    u_h = sh - LinearRegression().fit(Xcm, sh).predict(Xcm)
    return u_c, u_m, u_h


def limit_pct_series(stock_ids: pd.Series, dates: pd.Series) -> np.ndarray:
    sid = stock_ids.astype(str).str.zfill(6)
    d = pd.to_datetime(dates)
    limit = np.full(len(sid), 0.10, dtype=float)
    s688 = sid.str.startswith('688')
    chi = sid.str.startswith('300') | sid.str.startswith('301')
    bj = sid.str.startswith('8') | sid.str.startswith('4')
    limit = np.where(s688, 0.20, limit)
    limit = np.where(chi & (d >= pd.Timestamp('2020-08-24')), 0.20, limit)
    limit = np.where(bj & (d >= pd.Timestamp('2023-09-04')), 0.30, limit)
    return limit
