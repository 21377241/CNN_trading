# -*- coding: utf-8 -*-
"""E5-6: Continuation vs reversal — CumRet5 × S^sp double sorts."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import newey_west_t
from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.paths import MIN_CS_N, NW_LAG, OUT_ROOT
from experiments.E5_mechanism.run_e54_pressure import attach_pressure

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e56')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RETS = ['R_1_5', 'r1']


def _tercile(s: pd.Series) -> pd.Series:
    r = s.rank(method='average', pct=True)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[r <= 1 / 3] = 1
    out[(r > 1 / 3) & (r <= 2 / 3)] = 2
    out[r > 2 / 3] = 3
    return out


def run_e56(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    if 'Pressure' not in panel.columns:
        panel = attach_pressure(panel)

    # --- 3×3 CumRet5 × S^sp: cell mean future return ---
    cell_daily = []
    for ret in RETS:
        for score in SCORES:
            for dt, g in panel.groupby('ending_date'):
                g = g.dropna(subset=['CumRet5', score, ret])
                if len(g) < MIN_CS_N:
                    continue
                g = g.copy()
                g['q_ret'] = _tercile(g['CumRet5'])
                g['q_sp'] = _tercile(g[score])
                for qr in (1, 2, 3):
                    for qs in (1, 2, 3):
                        cell = g.loc[(g['q_ret'] == qr) & (g['q_sp'] == qs), ret]
                        if len(cell) < 5:
                            continue
                        cell_daily.append({
                            'ending_date': dt, 'ret': ret, 'score': score,
                            'q_cumret': int(qr), 'q_sp': int(qs),
                            'mean_ret': float(cell.mean()), 'n': int(len(cell)),
                        })
    cells = pd.DataFrame(cell_daily)
    cells.to_csv(os.path.join(OUT_ROOT, 'e56_daily_33_cells.csv'), index=False, float_format='%.6f')

    cell_summ = (
        cells.groupby(['ret', 'score', 'q_cumret', 'q_sp'], as_index=False)
        .agg(mean_ret=('mean_ret', 'mean'), n_days=('ending_date', 'nunique'))
    )
    # NW on each cell's daily series
    nw_rows = []
    for keys, g in cells.groupby(['ret', 'score', 'q_cumret', 'q_sp']):
        t, p = newey_west_t(g['mean_ret'].values, lag=NW_LAG)
        nw_rows.append({
            'ret': keys[0], 'score': keys[1], 'q_cumret': keys[2], 'q_sp': keys[3],
            'mean_ret': float(g['mean_ret'].mean()), 'nw_t': t, 'nw_p': p, 'n_days': int(len(g)),
        })
    cell_out = pd.DataFrame(nw_rows)
    cell_out.to_csv(os.path.join(OUT_ROOT, 'e56_33_summary.csv'), index=False, float_format='%.6f')

    # --- Spreads: within CumRet tercile, High S^sp − Low S^sp ---
    spread_daily = []
    for ret in RETS:
        for score in SCORES:
            for dt, g in panel.groupby('ending_date'):
                g = g.dropna(subset=['CumRet5', score, ret])
                if len(g) < MIN_CS_N:
                    continue
                g = g.copy()
                g['q_ret'] = _tercile(g['CumRet5'])
                g['q_sp'] = _tercile(g[score])
                for qr in (1, 2, 3):
                    gg = g.loc[g['q_ret'] == qr]
                    hi = gg.loc[gg['q_sp'] == 3, ret]
                    lo = gg.loc[gg['q_sp'] == 1, ret]
                    if len(hi) < 3 or len(lo) < 3:
                        continue
                    spread_daily.append({
                        'ending_date': dt, 'ret': ret, 'score': score,
                        'q_cumret': int(qr),
                        'spread': float(hi.mean() - lo.mean()),
                    })
    spreads = pd.DataFrame(spread_daily)
    spreads.to_csv(os.path.join(OUT_ROOT, 'e56_daily_spreads.csv'), index=False, float_format='%.6f')

    spread_summ = []
    for (ret, score, qr), g in spreads.groupby(['ret', 'score', 'q_cumret']):
        t, p = newey_west_t(g['spread'].values, lag=NW_LAG)
        # label: qr=3 positive recent → if spread>0 continuation; qr=1 negative recent → spread>0 means high S predicts rebound = reversal-like
        label = {1: 'neg_recent', 2: 'mid_recent', 3: 'pos_recent'}[int(qr)]
        spread_summ.append({
            'ret': ret, 'score': score, 'q_cumret': int(qr), 'recent_label': label,
            'spread_mean': float(g['spread'].mean()), 'nw_t': t, 'nw_p': p, 'n_days': int(len(g)),
        })
    # pooled across cumret terciles
    for (ret, score), g in spreads.groupby(['ret', 'score']):
        by_day = g.groupby('ending_date')['spread'].mean()
        t, p = newey_west_t(by_day.values, lag=NW_LAG)
        spread_summ.append({
            'ret': ret, 'score': score, 'q_cumret': 0, 'recent_label': 'pooled',
            'spread_mean': float(by_day.mean()), 'nw_t': t, 'nw_p': p, 'n_days': int(len(by_day)),
        })
    spread_out = pd.DataFrame(spread_summ)
    path = os.path.join(OUT_ROOT, 'e56_cont_rev_spreads.csv')
    spread_out.to_csv(path, index=False, float_format='%.6f')
    logger.info(f'wrote {path}\n{spread_out.to_string(index=False)}')

    # --- High-pressure subset: same spreads when Pressure tercile=High ---
    hp_rows = []
    for ret in RETS:
        for score in SCORES:
            for dt, g in panel.groupby('ending_date'):
                g = g.dropna(subset=['CumRet5', score, ret, 'Pressure'])
                if len(g) < MIN_CS_N:
                    continue
                g = g.copy()
                g['q_p'] = _tercile(g['Pressure'])
                g = g.loc[g['q_p'] == 3]
                if len(g) < 30:
                    continue
                g['q_ret'] = _tercile(g['CumRet5'])
                g['q_sp'] = _tercile(g[score])
                for qr in (1, 3):
                    gg = g.loc[g['q_ret'] == qr]
                    hi = gg.loc[gg['q_sp'] == 3, ret]
                    lo = gg.loc[gg['q_sp'] == 1, ret]
                    if len(hi) < 2 or len(lo) < 2:
                        continue
                    hp_rows.append({
                        'ending_date': dt, 'ret': ret, 'score': score,
                        'q_cumret': int(qr),
                        'spread': float(hi.mean() - lo.mean()),
                    })
    hp = pd.DataFrame(hp_rows)
    hp_summ = []
    if len(hp):
        for (ret, score, qr), g in hp.groupby(['ret', 'score', 'q_cumret']):
            t, p = newey_west_t(g['spread'].values, lag=NW_LAG)
            hp_summ.append({
                'ret': ret, 'score': score, 'q_cumret': int(qr),
                'recent_label': 'neg_recent' if qr == 1 else 'pos_recent',
                'subset': 'HighPressure',
                'spread_mean': float(g['spread'].mean()), 'nw_t': t, 'nw_p': p, 'n_days': int(len(g)),
            })
    hp_out = pd.DataFrame(hp_summ)
    hp_path = os.path.join(OUT_ROOT, 'e56_high_pressure_spreads.csv')
    hp_out.to_csv(hp_path, index=False, float_format='%.6f')
    logger.info(f'wrote {hp_path}\n{hp_out.to_string(index=False)}')
    return spread_out


if __name__ == '__main__':
    run_e56()
