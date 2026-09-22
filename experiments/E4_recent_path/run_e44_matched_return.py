# -*- coding: utf-8 -*-
"""E4-4: Matched-Return path double sort (CumRet5 × LateEarly) on S^sp."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import newey_west_t, norm_stock_id
from experiments.E4_recent_path.build_pathshape import build_pathshape
from experiments.E4_recent_path.paths import MIN_CS_N, NW_LAG, OUT_ROOT
from experiments.E4_recent_path.scores_pairwise import build_baseline_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e4_e44')

SCORE_COLS = ['S_cnn_sp', 'S_mlp_sp']


def _decile(s: pd.Series, n: int = 10) -> pd.Series:
    r = s.rank(method='first', pct=True)
    return np.ceil(r * n).clip(1, n).astype(int)


def run_e44() -> pd.DataFrame:
    out_dir = os.path.join(OUT_ROOT, 'e44_matched_return')
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(OUT_ROOT, 'baseline_scores.parquet')
    if not os.path.exists(scores_path):
        build_baseline_scores(scores_path)
    path_path = os.path.join(OUT_ROOT, 'pathshape.parquet')
    if not os.path.exists(path_path):
        build_pathshape(path_path)
    scores = pd.read_parquet(scores_path)
    path = pd.read_parquet(path_path)
    scores['StockID'] = norm_stock_id(scores['StockID'])
    path['StockID'] = norm_stock_id(path['StockID'])
    scores['ending_date'] = pd.to_datetime(scores['ending_date'])
    path['ending_date'] = pd.to_datetime(path['ending_date'])
    df = scores.merge(path[['StockID', 'ending_date', 'CumRet5', 'LateEarly']], on=['StockID', 'ending_date'])

    daily_rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        g = g.copy()
        g['d_ret'] = _decile(g['CumRet5'], 10)
        for dret, gg in g.groupby('d_ret'):
            if len(gg) < 20:
                continue
            gg = gg.copy()
            gg['d_path'] = _decile(gg['LateEarly'], 5)
            for scol in SCORE_COLS:
                hi = gg.loc[gg['d_path'] == 5, scol]
                lo = gg.loc[gg['d_path'] == 1, scol]
                if len(hi) < 2 or len(lo) < 2:
                    continue
                row = {
                    'ending_date': dt,
                    'cumret_decile': int(dret),
                    'score': scol,
                    'spread': float(hi.mean() - lo.mean()),
                    'n_hi': int(len(hi)),
                    'n_lo': int(len(lo)),
                }
                if 'ret_val' in gg.columns:
                    r_hi = gg.loc[gg['d_path'] == 5, 'ret_val']
                    r_lo = gg.loc[gg['d_path'] == 1, 'ret_val']
                    row['ret_spread'] = float(r_hi.mean() - r_lo.mean())
                daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(os.path.join(out_dir, 'e44_daily_spreads.csv'), index=False, float_format='%.6f')

    summ = []
    for (score, dret), g in daily.groupby(['score', 'cumret_decile']):
        t, p = newey_west_t(g['spread'].values, lag=NW_LAG)
        row = {
            'score': score, 'cumret_decile': int(dret),
            'sp_spread_mean': float(g['spread'].mean()),
            'nw_t': t, 'nw_p': p, 'n_days': int(len(g)),
        }
        if 'ret_spread' in g.columns:
            tr, pr = newey_west_t(g['ret_spread'].values, lag=NW_LAG)
            row.update({'ret_spread_mean': float(g['ret_spread'].mean()), 'ret_nw_t': tr, 'ret_nw_p': pr})
        summ.append(row)

    pool = []
    for (score, dt), g in daily.groupby(['score', 'ending_date']):
        pool.append({
            'score': score, 'ending_date': dt,
            'spread': g['spread'].mean(),
            'ret_spread': g['ret_spread'].mean() if 'ret_spread' in g else np.nan,
        })
    pool_df = pd.DataFrame(pool)
    for score, g in pool_df.groupby('score'):
        t, p = newey_west_t(g['spread'].values, lag=NW_LAG)
        row = {
            'score': score, 'cumret_decile': 0,
            'sp_spread_mean': float(g['spread'].mean()),
            'nw_t': t, 'nw_p': p, 'n_days': int(len(g)),
        }
        if g['ret_spread'].notna().any():
            tr, pr = newey_west_t(g['ret_spread'].values, lag=NW_LAG)
            row.update({'ret_spread_mean': float(g['ret_spread'].mean()), 'ret_nw_t': tr, 'ret_nw_p': pr})
        summ.append(row)

    summary = pd.DataFrame(summ).sort_values(['score', 'cumret_decile'])
    out = os.path.join(out_dir, 'e44_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return summary


if __name__ == '__main__':
    run_e44()
