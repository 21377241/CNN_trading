# -*- coding: utf-8 -*-
"""Shared IC / H-L helpers for E5."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.E3_shared_specific.utils import newey_west_t
from experiments.E5_mechanism.paths import MIN_CS_N, NW_LAG


def daily_rank_ic(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby('ending_date'):
        s = g[score_col].values.astype(float)
        r = g[ret_col].values.astype(float)
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < MIN_CS_N:
            continue
        sr, _ = spearmanr(s[m], r[m])
        if not np.isfinite(sr):
            continue
        rows.append({'ending_date': dt, 'rank_ic': float(sr), 'n': int(m.sum())})
    return pd.DataFrame(rows)


def summarize_daily_ic(daily: pd.DataFrame, lag: int = NW_LAG) -> dict:
    if daily.empty:
        return {'rank_ic_mean': np.nan, 'nw_t': np.nan, 'nw_p': np.nan, 'n_days': 0}
    t, p = newey_west_t(daily['rank_ic'].values, lag=lag)
    return {
        'rank_ic_mean': float(daily['rank_ic'].mean()),
        'nw_t': t,
        'nw_p': p,
        'n_days': int(len(daily)),
    }


def daily_hl_spread(df: pd.DataFrame, score_col: str, ret_col: str, n_decile: int = 10) -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby('ending_date'):
        g = g[[score_col, ret_col]].dropna()
        if len(g) < MIN_CS_N:
            continue
        rnk = g[score_col].rank(method='first', pct=True)
        d = np.ceil(rnk * n_decile).clip(1, n_decile).astype(int)
        hi = g.loc[d == n_decile, ret_col]
        lo = g.loc[d == 1, ret_col]
        if len(hi) < 2 or len(lo) < 2:
            continue
        rows.append({'ending_date': dt, 'hl': float(hi.mean() - lo.mean())})
    return pd.DataFrame(rows)


def summarize_hl(daily: pd.DataFrame, lag: int = NW_LAG) -> dict:
    if daily.empty:
        return {'hl_mean': np.nan, 'hl_nw_t': np.nan, 'hl_nw_p': np.nan}
    t, p = newey_west_t(daily['hl'].values, lag=lag)
    return {'hl_mean': float(daily['hl'].mean()), 'hl_nw_t': t, 'hl_nw_p': p}
