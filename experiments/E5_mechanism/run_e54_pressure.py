# -*- coding: utf-8 -*-
"""E5-4: Pre-registered Pressure index (no future-return tuning).

Pressure_i,t = z(TurnoverProxy_5) + z(-MaxDD5) + z(VolEndOverMean5)

Sign of MaxDD5: MaxDD5 is negative for deeper drawdowns; we use -MaxDD5 so
larger values = more path stress / pressure. Fixed before any Phase-2 outcome look.
"""
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

from experiments.E3_shared_specific.utils import cs_zscore, newey_west_t
from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.paths import MIN_CS_N, NW_LAG, OUT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e54')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RET = 'R_1_5'
# Also report r1 for CNN short-horizon narrative
RETS = ['R_1_5', 'r1']


def attach_pressure(panel: pd.DataFrame) -> pd.DataFrame:
    need = ['TurnoverProxy_5', 'MaxDD5', 'VolEndOverMean5']
    for c in need:
        if c not in panel.columns:
            raise KeyError(f'missing {c} for Pressure')
    df = panel.copy()
    # Pre-registered: deeper drawdown (more negative MaxDD5) → higher pressure
    df['_dd_stress'] = -df['MaxDD5'].astype(float)
    pieces = []
    for dt, g in df.groupby('ending_date', sort=False):
        g = g.copy()
        z_to = cs_zscore(g['TurnoverProxy_5'])
        z_dd = cs_zscore(g['_dd_stress'])
        z_v = cs_zscore(g['VolEndOverMean5'])
        g['Pressure'] = (z_to + z_dd + z_v) / 3.0
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out.drop(columns=['_dd_stress'], inplace=True)
    return out


def _tercile(s: pd.Series) -> pd.Series:
    r = s.rank(method='average', pct=True)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[r <= 1 / 3] = 1
    out[(r > 1 / 3) & (r <= 2 / 3)] = 2
    out[r > 2 / 3] = 3
    return out


def _ic(s, r) -> float:
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < max(10, MIN_CS_N // 3):
        return np.nan
    sr, _ = spearmanr(s[m], r[m])
    return float(sr) if np.isfinite(sr) else np.nan


def run_e54(panel: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    if 'Pressure' not in panel.columns:
        panel = attach_pressure(panel)
    press_path = os.path.join(OUT_ROOT, 'pressure_panel.parquet')
    slim_cols = [
        c for c in [
            'StockID', 'ending_date', 'S_cnn_sp', 'S_mlp_sp', 'Pressure', 'CumRet5',
            'log_mcap', 'TurnoverProxy_5', 'PathVol', 'MaxDD5', 'VolEndOverMean5',
            'R_1_5', 'r1', 'ret_val',
        ] if c in panel.columns
    ]
    panel[slim_cols].to_parquet(press_path, index=False)
    logger.info(f'pressure panel -> {press_path}')

    daily_rows = []
    for ret in RETS:
        for dt, g in panel.groupby('ending_date'):
            if g['Pressure'].notna().sum() < MIN_CS_N:
                continue
            g = g.copy()
            g['_bin'] = _tercile(g['Pressure'])
            for score in SCORES:
                for b, lab in [(1, 'Low'), (2, 'Mid'), (3, 'High')]:
                    gb = g.loc[g['_bin'] == b]
                    ic = _ic(gb[score].values, gb[ret].values)
                    if not np.isfinite(ic):
                        continue
                    daily_rows.append({
                        'ending_date': dt, 'ret': ret, 'score': score,
                        'bucket': lab, 'rank_ic': ic, 'n': int(len(gb)),
                    })
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(os.path.join(OUT_ROOT, 'e54_daily_pressure_ic.csv'), index=False, float_format='%.6f')

    summ = []
    for (ret, score), g in daily.groupby(['ret', 'score']):
        wide = g.pivot_table(index='ending_date', columns='bucket', values='rank_ic', aggfunc='mean')
        for col in ('Low', 'Mid', 'High'):
            if col not in wide.columns:
                wide[col] = np.nan
        delta = (wide['High'] - wide['Low']).dropna()
        t, p = newey_west_t(delta.values, lag=NW_LAG)
        summ.append({
            'ret': ret, 'score': score,
            'ic_low': float(wide['Low'].mean()),
            'ic_mid': float(wide['Mid'].mean()),
            'ic_high': float(wide['High'].mean()),
            'delta_high_minus_low': float(delta.mean()) if len(delta) else np.nan,
            'nw_t': t, 'nw_p': p, 'n_days': int(len(delta)),
        })
    out = pd.DataFrame(summ)
    path = os.path.join(OUT_ROOT, 'e54_pressure_ic.csv')
    out.to_csv(path, index=False, float_format='%.6f')
    logger.info(f'wrote {path}\n{out.to_string(index=False)}')
    return out, panel


if __name__ == '__main__':
    run_e54()
