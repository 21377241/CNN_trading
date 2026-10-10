# -*- coding: utf-8 -*-
"""IC helpers and portfolio wrappers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr


def newey_west_t(x: np.ndarray, lag: int = 4) -> tuple[float, float]:
    """Newey-West (Bartlett) t-statistic and two-sided p-value for the mean of a time series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return float('nan'), float('nan')
    mu = x.mean()
    var = np.mean((x - mu) ** 2)
    for L in range(1, min(lag, T - 1) + 1):
        var += 2 * (1.0 - L / (lag + 1)) * np.mean((x[L:] - mu) * (x[:-L] - mu))
    se = np.sqrt(max(var, 0.0) / T)
    if se < 1e-15:
        return float('nan'), float('nan')
    t = mu / se
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=T - 1)))


def annualized_sharpe(r: np.ndarray, periods: int = 52) -> float:
    """Frozen convention (docs/protocol/sharpe_convention.md): mean * sqrt(periods) / std (ddof=1)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) < 1e-15:
        return float('nan')
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def cross_section_ic(df: pd.DataFrame, score_col: str, ret_col: str = 'ret_val') -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < 10:
            continue
        s = g[score_col].values
        r = g[ret_col].values
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < 10:
            continue
        pr, _ = pearsonr(s[m], r[m])
        sr, _ = spearmanr(s[m], r[m])
        rows.append({'ending_date': dt, 'ic': pr, 'rank_ic': sr})
    return pd.DataFrame(rows)


def summarize_ic(ic_df: pd.DataFrame) -> dict:
    if ic_df.empty:
        return {'ic_mean': np.nan, 'rank_ic_mean': np.nan, 'n_dates': 0}
    return {
        'ic_mean': float(ic_df['ic'].mean()),
        'rank_ic_mean': float(ic_df['rank_ic'].mean()),
        'n_dates': int(len(ic_df)),
    }


def yearly_rank_ic_mean(df: pd.DataFrame, score_col: str, ret_col: str = 'ret_val') -> pd.Series:
    ic_df = cross_section_ic(df, score_col, ret_col)
    if ic_df.empty:
        return pd.Series(dtype=float)
    ic_df['year'] = pd.to_datetime(ic_df['ending_date']).dt.year
    return ic_df.groupby('year')['rank_ic'].mean()
