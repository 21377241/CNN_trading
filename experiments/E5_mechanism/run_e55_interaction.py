# -*- coding: utf-8 -*-
"""E5-5: Fama–MacBeth interaction S^sp × Pressure on R_1_5 (and r1)."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import cs_zscore, newey_west_t
from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.paths import MIN_CS_N, NW_LAG, OUT_ROOT
from experiments.E5_mechanism.run_e54_pressure import attach_pressure

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e55')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RETS = ['R_1_5', 'r1']


def _cs_reg(y, X) -> np.ndarray | None:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < MIN_CS_N:
        return None
    coef = LinearRegression(fit_intercept=True).fit(X[m], y[m]).coef_
    return coef


def run_e55(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    if 'Pressure' not in panel.columns:
        panel = attach_pressure(panel)

    daily_rows = []
    for ret in RETS:
        for score in SCORES:
            for dt, g in panel.groupby('ending_date'):
                g = g.copy()
                # Controls: Size, CumRet5 (avoid stacking Turnover which is inside Pressure)
                need = [score, 'Pressure', 'log_mcap', 'CumRet5', ret]
                if any(c not in g.columns for c in need):
                    continue
                y = g[ret].values.astype(float)
                sp = cs_zscore(g[score]).values
                pr = cs_zscore(g['Pressure']).values
                sz = cs_zscore(g['log_mcap']).values
                cr = cs_zscore(g['CumRet5']).values
                inter = sp * pr
                X = np.column_stack([sp, pr, inter, sz, cr])
                coef = _cs_reg(y, X)
                if coef is None:
                    continue
                daily_rows.append({
                    'ending_date': dt, 'ret': ret, 'score': score,
                    'b_sp': float(coef[0]),
                    'b_pressure': float(coef[1]),
                    'b_interact': float(coef[2]),
                    'b_size': float(coef[3]),
                    'b_cumret5': float(coef[4]),
                })
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(os.path.join(OUT_ROOT, 'e55_daily_fm.csv'), index=False, float_format='%.8f')

    summ = []
    for (ret, score), g in daily.groupby(['ret', 'score']):
        row = {'ret': ret, 'score': score, 'n_days': int(len(g))}
        for col in ['b_sp', 'b_pressure', 'b_interact', 'b_size', 'b_cumret5']:
            t, p = newey_west_t(g[col].values, lag=NW_LAG)
            row[f'{col}_mean'] = float(g[col].mean())
            row[f'{col}_nw_t'] = t
            row[f'{col}_nw_p'] = p
        summ.append(row)
    out = pd.DataFrame(summ)
    path = os.path.join(OUT_ROOT, 'e55_interaction_fm.csv')
    out.to_csv(path, index=False, float_format='%.6f')
    logger.info(f'wrote {path}\n{out.to_string(index=False)}')
    return out


if __name__ == '__main__':
    run_e55()
