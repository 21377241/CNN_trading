# -*- coding: utf-8 -*-
"""E7 conditional double sorts (protocol section 6.3).

Each week: quintiles on a conditioning control, then score deciles within each quintile; equal-weighted decile 10
minus decile 1 next-week return (ret_oo). Reported per quintile and as the average across the five quintiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import (  # noqa: E402
    OUT, SCORES, annualized_sharpe, newey_west_t, write_manifest,
)
from experiments.E7_incremental.sample import load_sample  # noqa: E402

COND = ['rev_1w', 'ret_20d', 'max_20d', 'vol_20d', 'turn_20d', 'size', 'illiq_20d']


def bucket(df: pd.DataFrame, col: str, by: list[str], n: int) -> pd.Series:
    pct = df.groupby(by, sort=False)[col].rank(pct=True, method='first')
    return np.minimum(np.ceil(pct * n) - 1, n - 1).astype(int)


def stats(x: pd.Series) -> dict:
    t, p = newey_west_t(x.to_numpy())
    return {'hl_mean_wk': float(x.mean()), 'hl_ann': float(x.mean() * 52), 't_nw': t, 'p': p,
            'sharpe_ann': annualized_sharpe(x.to_numpy()), 'n_weeks': int(x.notna().sum())}


def double_sort(df: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    df = df[['ending_date', 'ret_oo', *COND, *scores]].reset_index(drop=True)
    rows = []
    for cond in COND:
        df['q'] = bucket(df, cond, ['ending_date'], 5)
        for k in scores:
            df['dec'] = bucket(df, k, ['ending_date', 'q'], 10)
            ext = df[df['dec'].isin([0, 9])].groupby(['ending_date', 'q', 'dec'])['ret_oo'].mean().unstack('dec')
            hl = (ext[9] - ext[0]).rename('hl')
            avg = hl.groupby('ending_date').mean()
            rows.append({'score': k, 'cond': cond, 'quintile': 'avg', **stats(avg)})
            for q, s in hl.groupby('q'):
                rows.append({'score': k, 'cond': cond, 'quintile': int(q) + 1, **stats(s.droplevel('q'))})
    return pd.DataFrame(rows)


def main() -> int:
    df = load_sample()
    out = double_sort(df[df['in_main']], SCORES)
    out.to_csv(OUT / 'double_sort.csv', index=False)
    write_manifest('run_double_sort', {'cond': COND})
    avg = out[out['quintile'] == 'avg']
    print(avg.pivot_table(index='cond', columns='score', values='t_nw', sort=False).round(2).to_string())
    print(avg.pivot_table(index='cond', columns='score', values='sharpe_ann', sort=False).round(2).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
