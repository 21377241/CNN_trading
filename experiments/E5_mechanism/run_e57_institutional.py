# -*- coding: utf-8 -*-
"""E5-7: Institutional-adjacent tests (T+1-related state & price limits).

Caution: these are state-dependence / screening tests, NOT causal proof of T+1
or limit rules. Report with that framing.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.utils import cs_zscore, newey_west_t
from experiments.E7_incremental.common import limit_flags_at
from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.paths import MIN_CS_N, NW_LAG, OUT_ROOT
from experiments.E5_mechanism.run_e54_pressure import attach_pressure

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e57')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RETS = ['r1', 'R_1_5']


def _tercile(s: pd.Series) -> pd.Series:
    r = s.rank(method='average', pct=True)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[r <= 1 / 3] = 1
    out[(r > 1 / 3) & (r <= 2 / 3)] = 2
    out[r > 2 / 3] = 3
    return out


def _ic(s, r) -> float:
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < max(15, MIN_CS_N // 3):
        return np.nan
    sr, _ = spearmanr(s[m], r[m])
    return float(sr) if np.isfinite(sr) else np.nan


def attach_limit_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """LimitHit / FailedLimit on the signal date from common.limit_state (CSMAR limit prices, exact to the cent).
    Days without a price limit are NaN."""
    ls = limit_flags_at(panel)
    has = ls['no_limit'].notna() & ~ls['no_limit'].astype(bool)
    flags = {
        'LimitHit': ls['close_up'].astype(bool) | ls['close_dn'].astype(bool),
        'LimitUp': ls['close_up'].astype(bool),
        'LimitDn': ls['close_dn'].astype(bool),
        'FailedLimit': (ls['touch_up'].astype(bool) & ~ls['close_up'].astype(bool))
        | (ls['touch_dn'].astype(bool) & ~ls['close_dn'].astype(bool)),
    }
    out = panel.copy()
    for k, v in flags.items():
        out[k] = np.where(has, v.astype(float), np.nan)
    # Normal = not hit and not failed (clear free trading day)
    out['LimitState'] = 'Normal'
    out.loc[out['FailedLimit'] == 1, 'LimitState'] = 'FailedLimit'
    out.loc[out['LimitHit'] == 1, 'LimitState'] = 'LimitHit'  # hit overrides failed
    out.loc[~np.isfinite(out['LimitHit'].values), 'LimitState'] = np.nan
    return out


def attach_t1_state(panel: pd.DataFrame) -> pd.DataFrame:
    """T+1-adjacent state: Neg CumRet5 × High TurnoverProxy (pre-registered screen)."""
    df = panel.copy()
    df['T1_state'] = 0.0
    for dt, g in df.groupby('ending_date', sort=False):
        if g['CumRet5'].notna().sum() < MIN_CS_N or g['TurnoverProxy_5'].notna().sum() < MIN_CS_N:
            continue
        q_c = _tercile(g['CumRet5'])
        q_t = _tercile(g['TurnoverProxy_5'])
        df.loc[g.index, 'T1_state'] = ((q_c == 1) & (q_t == 3)).astype(float).values
    return df


def _state_ic_table(panel: pd.DataFrame, state_col: str, in_value, out_label='in', out_value=None) -> pd.DataFrame:
    rows = []
    for ret in RETS:
        for score in SCORES:
            daily_in, daily_out, daily_delta = [], [], []
            for dt, g in panel.groupby('ending_date'):
                if state_col == 'LimitState':
                    gin = g.loc[g[state_col] == in_value]
                    gout = g.loc[g[state_col] == 'Normal']
                else:
                    gin = g.loc[g[state_col] == in_value]
                    gout = g.loc[g[state_col] != in_value]
                ic_in = _ic(gin[score].values, gin[ret].values)
                ic_out = _ic(gout[score].values, gout[ret].values)
                if np.isfinite(ic_in):
                    daily_in.append({'ending_date': dt, 'rank_ic': ic_in})
                if np.isfinite(ic_out):
                    daily_out.append({'ending_date': dt, 'rank_ic': ic_out})
                if np.isfinite(ic_in) and np.isfinite(ic_out):
                    daily_delta.append({'ending_date': dt, 'delta': ic_in - ic_out})
            di, do, dd = pd.DataFrame(daily_in), pd.DataFrame(daily_out), pd.DataFrame(daily_delta)
            t_in, p_in = newey_west_t(di['rank_ic'].values, lag=NW_LAG) if len(di) else (np.nan, np.nan)
            t_out, p_out = newey_west_t(do['rank_ic'].values, lag=NW_LAG) if len(do) else (np.nan, np.nan)
            t_d, p_d = newey_west_t(dd['delta'].values, lag=NW_LAG) if len(dd) else (np.nan, np.nan)
            rows.append({
                'state': f'{state_col}={in_value}',
                'ret': ret, 'score': score,
                'ic_in': float(di['rank_ic'].mean()) if len(di) else np.nan,
                'ic_out_or_normal': float(do['rank_ic'].mean()) if len(do) else np.nan,
                'delta_in_minus_out': float(dd['delta'].mean()) if len(dd) else np.nan,
                'nw_t_delta': t_d, 'nw_p_delta': p_d,
                'nw_t_in': t_in, 'n_days_in': int(len(di)), 'n_days_delta': int(len(dd)),
            })
    return pd.DataFrame(rows)


def _fm_interaction(panel: pd.DataFrame, state_col: str, ret: str, score: str) -> dict:
    daily = []
    for dt, g in panel.groupby('ending_date'):
        g = g.dropna(subset=[score, state_col, ret, 'log_mcap', 'CumRet5'])
        if len(g) < MIN_CS_N:
            continue
        y = g[ret].values.astype(float)
        sp = cs_zscore(g[score]).values
        st = g[state_col].values.astype(float)
        # if LimitState string, skip FM here
        if not np.issubdtype(g[state_col].dtype, np.number):
            return {}
        inter = sp * st
        sz = cs_zscore(g['log_mcap']).values
        cr = cs_zscore(g['CumRet5']).values
        X = np.column_stack([sp, st, inter, sz, cr])
        m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if m.sum() < MIN_CS_N:
            continue
        coef = LinearRegression(fit_intercept=True).fit(X[m], y[m]).coef_
        daily.append({'b_sp': coef[0], 'b_state': coef[1], 'b_interact': coef[2]})
    if not daily:
        return {'ret': ret, 'score': score, 'state': state_col, 'n_days': 0}
    d = pd.DataFrame(daily)
    out = {'ret': ret, 'score': score, 'state': state_col, 'n_days': int(len(d))}
    for col in ['b_sp', 'b_state', 'b_interact']:
        t, p = newey_west_t(d[col].values, lag=NW_LAG)
        out[f'{col}_mean'] = float(d[col].mean())
        out[f'{col}_nw_t'] = t
        out[f'{col}_nw_p'] = p
    return out


def run_e57(panel: pd.DataFrame | None = None) -> dict:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    if 'Pressure' not in panel.columns:
        panel = attach_pressure(panel)
    logger.info('attaching T+1-adjacent & limit flags…')
    panel = attach_t1_state(panel)
    panel = attach_limit_flags(panel)

    # --- A. T+1-adjacent ---
    t1_ic = _state_ic_table(panel, 'T1_state', 1.0)
    t1_ic.to_csv(os.path.join(OUT_ROOT, 'e57_t1_state_ic.csv'), index=False, float_format='%.6f')
    logger.info(f'T1 IC:\n{t1_ic.to_string(index=False)}')

    fm_rows = []
    for ret in RETS:
        for score in SCORES:
            fm_rows.append(_fm_interaction(panel, 'T1_state', ret, score))
    # also HighPressure × already have E5-5; add NegCumRet alone binary
    panel['NegCumRet'] = np.nan
    for dt, g in panel.groupby('ending_date'):
        idx = g.index
        q = _tercile(g['CumRet5'])
        panel.loc[idx, 'NegCumRet'] = (q == 1).astype(float)
    for ret in RETS:
        for score in SCORES:
            fm_rows.append(_fm_interaction(panel, 'NegCumRet', ret, score))
    t1_fm = pd.DataFrame([r for r in fm_rows if r])
    t1_fm.to_csv(os.path.join(OUT_ROOT, 'e57_t1_interaction_fm.csv'), index=False, float_format='%.6f')
    logger.info(f'T1 FM:\n{t1_fm.to_string(index=False)}')

    # --- B. Price limits ---
    lim_parts = []
    for st in ('LimitHit', 'FailedLimit', 'LimitUp', 'LimitDn'):
        # binary columns already 0/1
        lim_parts.append(_state_ic_table(panel, st, 1.0))
    lim_ic = pd.concat(lim_parts, ignore_index=True)
    lim_ic.to_csv(os.path.join(OUT_ROOT, 'e57_limit_state_ic.csv'), index=False, float_format='%.6f')
    logger.info(f'Limit IC:\n{lim_ic.to_string(index=False)}')

    # LimitState categorical vs Normal
    cat_rows = []
    for st_name in ('LimitHit', 'FailedLimit'):
        cat_rows.append(_state_ic_table(panel, 'LimitState', st_name))
    cat_ic = pd.concat(cat_rows, ignore_index=True)
    cat_ic.to_csv(os.path.join(OUT_ROOT, 'e57_limit_vs_normal_ic.csv'), index=False, float_format='%.6f')

    # coverage
    cov = {
        'pct_T1_state': float(panel['T1_state'].mean()),
        'pct_LimitHit': float(panel['LimitHit'].mean()),
        'pct_FailedLimit': float(panel['FailedLimit'].mean()),
        'pct_LimitUp': float(panel['LimitUp'].mean()),
        'pct_LimitDn': float(panel['LimitDn'].mean()),
        'n': int(len(panel)),
    }
    pd.DataFrame([cov]).to_csv(os.path.join(OUT_ROOT, 'e57_coverage.csv'), index=False, float_format='%.6f')
    logger.info(f'coverage {cov}')
    return {'t1_ic': t1_ic, 't1_fm': t1_fm, 'lim_ic': lim_ic, 'coverage': cov}


if __name__ == '__main__':
    run_e57()
