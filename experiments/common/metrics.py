# -*- coding: utf-8 -*-
"""IC helpers and portfolio wrappers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


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
