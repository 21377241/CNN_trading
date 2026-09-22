# -*- coding: utf-8 -*-
"""E5-3: Market-state dependence of specific Rank IC (on R_1_5)."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import newey_west_t
from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.paths import MIN_CS_N, NW_LAG, OUT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e53')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RET = 'R_1_5'
STATES = {
    'TurnoverProxy_5': 'TurnoverProxy_5',
    'Volatility': 'PathVol',
    'Size': 'log_mcap',
    'CumRet5': 'CumRet5',
}


def _tercile(s: pd.Series) -> pd.Series:
    r = s.rank(method='average', pct=True)
    out = pd.Series(np.nan, index=s.index)
    out = out.astype(float)
    out[r <= 1 / 3] = 1
    out[(r > 1 / 3) & (r <= 2 / 3)] = 2
    out[r > 2 / 3] = 3
    return out


def _ic(s: np.ndarray, r: np.ndarray) -> float:
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < max(10, MIN_CS_N // 3):
        return np.nan
    sr, _ = spearmanr(s[m], r[m])
    return float(sr) if np.isfinite(sr) else np.nan


def run_e53(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    daily_rows = []
    for state_name, col in STATES.items():
        if col not in panel.columns:
            logger.warning(f'missing state col {col}')
            continue
        for dt, g in panel.groupby('ending_date'):
            if len(g) < MIN_CS_N or g[col].notna().sum() < MIN_CS_N:
                continue
            g = g.copy()
            g['_bin'] = _tercile(g[col])
            for score in SCORES:
                for b, lab in [(1, 'Low'), (3, 'High')]:
                    gb = g.loc[g['_bin'] == b]
                    ic = _ic(gb[score].values, gb[RET].values)
                    if not np.isfinite(ic):
                        continue
                    daily_rows.append({
                        'ending_date': dt, 'state': state_name, 'score': score,
                        'bucket': lab, 'rank_ic': ic, 'n': int(len(gb)),
                    })
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(os.path.join(OUT_ROOT, 'e53_daily_state_ic.csv'), index=False, float_format='%.6f')

    # High - Low deltas
    summ = []
    for (state, score), g in daily.groupby(['state', 'score']):
        wide = g.pivot_table(index='ending_date', columns='bucket', values='rank_ic', aggfunc='mean')
        if 'High' not in wide.columns or 'Low' not in wide.columns:
            continue
        delta = (wide['High'] - wide['Low']).dropna()
        t, p = newey_west_t(delta.values, lag=NW_LAG)
        summ.append({
            'state': state, 'score': score,
            'ic_low': float(wide['Low'].mean()),
            'ic_high': float(wide['High'].mean()),
            'delta_high_minus_low': float(delta.mean()),
            'nw_t': t, 'nw_p': p, 'n_days': int(len(delta)),
        })
    out = pd.DataFrame(summ).sort_values(['state', 'score'])
    path = os.path.join(OUT_ROOT, 'e53_state_delta.csv')
    out.to_csv(path, index=False, float_format='%.6f')
    logger.info(f'wrote {path}\n{out.to_string(index=False)}')
    return out


if __name__ == '__main__':
    run_e53()
