# -*- coding: utf-8 -*-
"""Layer A / Layer B control variables at every weekly signal date (protocol section 4), from the extended daily panel.

All windows run over each stock's own trading days using unclipped close-to-close returns; values only use
information up to the signal-date close. Rolling statistics are computed on the full (StockID, Date)-sorted
arrays and masked where the window would reach into the previous stock.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import (  # noqa: E402
    CTRL_A, CTRL_B, EXT_PANEL, LIMIT_TOL, OUT, OUT_DATA, STOCK_INFO, WEEKLY_CONTROLS, limit_pct, week_ends,
)

COLS = ['Date', 'StockID', 'High', 'Low', 'Close', 'Vol', 'Amount', 'Turnover', 'FloatMV', 'PE_TTM', 'PB', 'is_ST']


def main() -> int:
    t0 = time.time()
    p = pd.read_feather(EXT_PANEL, columns=COLS)
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    we = week_ends(cal)
    code = p['StockID'].cat.codes.to_numpy()
    pos = pd.Series(code).groupby(code).cumcount().to_numpy()
    same = np.r_[False, code[1:] == code[:-1]]
    sig = np.flatnonzero(p['Date'].isin(we).to_numpy())

    C = p['Close'].astype('float64')
    H = p['High'].astype('float64')
    L = p['Low'].astype('float64')
    V = p['Vol'].astype('float64')
    r = C.pct_change().where(same)

    w_prev = p['FloatMV'].astype('float64').shift(1).where(same)
    ok = r.notna() & w_prev.notna() & (w_prev > 0)
    mkt = (r[ok] * w_prev[ok]).groupby(p.loc[ok, 'Date']).sum() / w_prev[ok].groupby(p.loc[ok, 'Date']).sum()
    m = p['Date'].map(mkt).astype('float64').where(r.notna())

    out = pd.DataFrame({'StockID': p['StockID'].astype(str).to_numpy()[sig], 'ending_date': p['Date'].to_numpy()[sig]})

    def take(series: pd.Series, min_pos: int) -> np.ndarray:
        v = series.to_numpy()[sig].astype('float64')
        v[pos[sig] < min_pos] = np.nan
        return v

    def roll(s: pd.Series, w: int, fn: str, minp: int | None = None) -> pd.Series:
        return getattr(s.rolling(w, min_periods=minp or w), fn)()

    # Layer A: same 20-day window as the model inputs
    out['rev_1w'] = take(C / C.shift(5) - 1, 5)
    out['ret_20d'] = take(C / C.shift(20) - 1, 20)
    out['max_20d'] = take(roll(r, 20, 'max'), 20)
    out['vol_20d'] = take(roll(r, 20, 'std'), 20)
    out['vtrend'] = take(np.log(roll(V, 5, 'mean') / roll(V, 20, 'mean')), 19)
    out['range_20d'] = take(roll((H - L) / C, 20, 'mean'), 19)
    lo20, hi20 = roll(L, 20, 'min'), roll(H, 20, 'max')
    out['clv_20d'] = take(((C - lo20) / (hi20 - lo20)).where(hi20 > lo20), 19)
    out['ma_gap'] = take(C / roll(C, 20, 'mean') - 1, 19)

    # Layer B: outside the window / firm characteristics
    out['mom_12_1'] = take(C.shift(20) / C.shift(250) - 1, 250)
    out['size'] = take(np.log(p['FloatMV'].astype('float64').where(p['FloatMV'] > 0)), 0)
    out['turn_20d'] = take(roll(p['Turnover'].astype('float64'), 20, 'mean'), 19)
    amt = p['Amount'].astype('float64').where(p['Amount'] > 0)
    out['illiq_20d'] = take(np.log1p(roll(r.abs() / amt, 20, 'mean', 15) * 1e8), 19)
    cov = r.rolling(20, min_periods=15).cov(m)
    var_m = roll(m, 20, 'var', 15)
    var_r = roll(r, 20, 'var', 15)
    resid_var = (var_r - cov ** 2 / var_m.where(var_m > 0)).clip(lower=0)
    out['ivol_20d'] = take(np.sqrt(resid_var), 19)
    pb = p['PB'].astype('float64')
    pe = p['PE_TTM'].astype('float64')
    out['bm'] = take((1 / pb).where(pb > 0), 0)
    out['ep'] = take((1 / pe).where(pe != 0), 0)

    # flags for robustness filters
    out['FloatMV'] = take(p['FloatMV'].astype('float64'), 0)
    out['is_ST'] = p['is_ST'].to_numpy()[sig]
    out['ret_1d'] = take(r, 1)
    lim = limit_pct(out['StockID'], out['ending_date'])
    out['limit_up_sig'] = out['ret_1d'] >= lim - LIMIT_TOL
    out['limit_dn_sig'] = out['ret_1d'] <= -(lim - LIMIT_TOL)
    out['n_hist'] = pos[sig] + 1

    info = pd.read_feather(STOCK_INFO, columns=['StockID', 'industry', 'list_date'])
    out = out.merge(info, on='StockID', how='left')
    first_trade = p.groupby('StockID', observed=True)['Date'].min()
    first_trade.index = first_trade.index.astype(str)
    start = out['list_date'].fillna(out['StockID'].map(first_trade))
    out['age_days'] = (out['ending_date'] - start).dt.days
    out = out.drop(columns=['list_date'])

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_feather(WEEKLY_CONTROLS)
    oos = out[(out['ending_date'] >= '2008-01-01') & (out['ending_date'] <= '2025-12-31') & (out['is_ST'] == 0)]
    chk = {'rows': len(out), 'weeks': int(out['ending_date'].nunique()),
           'oos_nonST_rows': len(oos),
           'oos_missing_share': {c: float(oos[c].isna().mean()) for c in CTRL_A + CTRL_B + ['industry']},
           'oos_describe': oos[CTRL_A + CTRL_B].describe(percentiles=[0.01, 0.5, 0.99]).round(4).to_dict(),
           'seconds': round(time.time() - t0)}
    (OUT / 'weekly_controls_check.json').write_text(json.dumps(chk, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in chk.items() if k != 'oos_describe'}, indent=2, default=str))
    print(oos[CTRL_A + CTRL_B].describe(percentiles=[0.01, 0.5, 0.99]).T.round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
