# -*- coding: utf-8 -*-
"""E7 sample construction and the shared weekly transformation (protocol sections 2 and 4.3)."""
from __future__ import annotations

import pandas as pd

from experiments.E7_incremental.common import (
    CTRL_A, CTRL_B, KEYS, OUT, OUT_DATA, SCORES, WEEKLY_CONTROLS, WEEKLY_RETURNS, cs_rank, cs_winsor, cs_winsor_z,
    industry_demean, load_canonical_preds,
)

SAMPLE = OUT_DATA / 'e7_sample.feather'
RET_COLS = ['ret_oo', 'ret_cc', 'ret_oo_d50', 'entry_lock_up', 'not_enterable', 'oo_stale', 'delisted_oo']
FLAG_COLS = ['FloatMV', 'limit_up_sig', 'limit_dn_sig', 'age_days', 'industry']


def _flow_row(step: str, df: pd.DataFrame) -> dict:
    return {'step': step, 'rows': len(df), 'weeks': int(df['ending_date'].nunique()),
            'stocks': int(df['StockID'].nunique())}


def build_sample() -> pd.DataFrame:
    """Base sample (canonical prediction keys) merged with returns and controls; writes sample_flow.csv."""
    df = load_canonical_preds()
    rets = pd.read_feather(WEEKLY_RETURNS, columns=KEYS + RET_COLS)
    ctrls = pd.read_feather(WEEKLY_CONTROLS, columns=KEYS + CTRL_A + CTRL_B + FLAG_COLS)
    df = df.merge(rets, on=KEYS, how='left').merge(ctrls, on=KEYS, how='left')
    flow = [_flow_row('base: canonical prediction keys', df)]
    s = df[df['ret_oo'].notna()]
    flow.append(_flow_row('ret_oo available (enterable at t+1 open)', s))
    s = s[s[CTRL_A].notna().all(axis=1)]
    flow.append(_flow_row('layer A complete', s))
    s = s[s[[c for c in CTRL_B if c != 'mom_12_1']].notna().all(axis=1) & s['industry'].notna()]
    flow.append(_flow_row('layer B complete except mom_12_1 (+ industry)', s))
    s = s[s['mom_12_1'].notna()]
    flow.append(_flow_row('mom_12_1 available = main sample', s))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flow).to_csv(OUT / 'sample_flow.csv', index=False)
    df['in_main'] = df.index.isin(s.index)
    df['industry'] = df['industry'].astype(str)
    df = df.reset_index(drop=True)
    df.to_feather(SAMPLE)
    return df


def load_sample() -> pd.DataFrame:
    return pd.read_feather(SAMPLE)


def prepare(df: pd.DataFrame, x_cols: list[str], y: str = 'ret_oo', transform: str = 'rank', fe: bool = True,
            y_winsor: bool = False, keep: tuple = ()) -> pd.DataFrame:
    """Weekly cross-sectional transformation of regressors, optional industry demeaning of y and X."""
    out = df[['ending_date', 'industry', *keep]].copy()
    if transform == 'rank':
        out[x_cols] = cs_rank(df, x_cols)
    elif transform == 'wz':
        out[x_cols] = cs_winsor_z(df, x_cols)
    else:
        raise ValueError(transform)
    out['y'] = cs_winsor(df[y], df['ending_date']) if y_winsor else df[y]
    out = out.dropna(subset=x_cols + ['y'])
    if fe:
        out[x_cols + ['y']] = industry_demean(out, x_cols + ['y'])
    return out


__all__ = ['SAMPLE', 'build_sample', 'load_sample', 'prepare', 'SCORES']
