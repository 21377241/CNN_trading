# -*- coding: utf-8 -*-
"""Layer A / Layer B control variables for every universe row (E7 protocol v2, section 4).

Every 20-day variable uses the same window as the model inputs: the 20 most recent *market* trading days up to the
signal date t (clean_window.gather_windows), with the same missing-day rules. Daily returns inside the window are
ratios of consecutive valid closes within the window (a return spanning a missing day is booked on the resumption
day, as in the image), so a fully traded window has 19 returns. Values only use information up to the close of t.
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
    CTRL_A, CTRL_B, EXT_PANEL, OUT, OUT_DATA, STOCK_INFO, WEEKLY_CONTROLS, hit_limit, limit_pct,
)
from Misc.config import ASHARE_UNIVERSE_PATH  # noqa: E402
from Data.clean_window import WINDOW, _ffill_rows  # noqa: E402

PANEL_COLS = ['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'MA20', 'Vol', 'Amount', 'Turnover', 'FloatMV',
              'PE_TTM', 'PB', 'is_ST']
MOM_FROM, MOM_TO = 250, 20
MIN_RET_VOL = 5
MIN_DAYS_LIQ = 15


def dense_grid(p: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Index, dict]:
    dates = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    stocks = pd.Index(p['StockID'].cat.categories.astype(str))
    di = dates.get_indexer(p['Date'])
    si = p['StockID'].cat.codes.to_numpy()
    g = {}
    for col in ('Open', 'High', 'Low', 'Close', 'MA20', 'Vol', 'Amount', 'Turnover', 'FloatMV', 'PE_TTM', 'PB',
                'is_ST'):
        dt = np.float64 if col == 'Close' else np.float32
        a = np.full((len(dates), len(stocks)), np.nan, dtype=dt)
        a[di, si] = p[col].to_numpy(dtype=dt)
        g[col] = a
    g['MA20'] = pd.DataFrame(g['MA20']).ffill().to_numpy()
    return dates, stocks, g


def market_returns(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily stock returns vs the last valid close (spanning suspensions), cap-weighted market return, and
    the market index used to compound market returns over the same span as a stock return."""
    C = g['Close']
    Cf = pd.DataFrame(C).ffill().to_numpy()
    prev = np.vstack([np.full((1, C.shape[1]), np.nan), Cf[:-1]])
    r = C / prev - 1
    w = np.vstack([np.full((1, C.shape[1]), np.nan), pd.DataFrame(g['FloatMV']).ffill().to_numpy()[:-1]])
    ok = np.isfinite(r) & np.isfinite(w) & (w > 0)
    m = np.where(ok, r * w, 0).sum(axis=1) / np.where(ok, w, 0).sum(axis=1)
    m = np.where(ok.any(axis=1), m, 0.0)
    return r, m, np.cumprod(1 + m)


def window_controls(g: dict, Cf_full: np.ndarray, M: np.ndarray, t: np.ndarray, s: np.ndarray) -> dict:
    rows = t[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    take = lambda a: a[rows, s[:, None]]  # noqa: E731
    O, H, L, C, V = (take(g[k]) for k in ('Open', 'High', 'Low', 'Close', 'Vol'))
    amt, turn = take(g['Amount']), take(g['Turnover'])
    valid = np.isfinite(C)
    P = _ffill_rows(C)
    last = WINDOW - 1
    prevP = np.concatenate([np.full((len(t), 1), np.nan), P[:, :-1]], axis=1)
    first = valid.argmax(axis=1)
    has_prev = valid & (np.arange(WINDOW)[None, :] > first[:, None])
    r = np.where(has_prev, C / prevP - 1, np.nan)

    # market return over the same span as each in-window stock return
    Mw = M[rows]
    Mf = _ffill_rows(np.where(valid, Mw, np.nan))
    prevM = np.concatenate([np.full((len(t), 1), np.nan), Mf[:, :-1]], axis=1)
    mr = np.where(has_prev, Mw / prevM - 1, np.nan)

    n_ret = has_prev.sum(axis=1)
    out = {}
    with np.errstate(all='ignore'):
        out['rev_1w'] = P[:, last] / P[:, last - 5] - 1
        out['ret_20d'] = P[:, last] / P[:, 0] - 1
        out['max_20d'] = np.where(n_ret >= 1, np.nanmax(np.where(has_prev, r, -np.inf), axis=1), np.nan)
        out['vol_20d'] = np.where(n_ret >= MIN_RET_VOL, np.nanstd(r, axis=1, ddof=1), np.nan)
        v5 = np.nanmean(np.where(valid[:, -5:], V[:, -5:], np.nan), axis=1)
        v20 = np.nanmean(np.where(valid, V, np.nan), axis=1)
        out['vtrend'] = np.where((v5 > 0) & (v20 > 0), np.log(v5 / v20), np.nan)
        out['range_20d'] = np.nanmean(np.where(valid, (H - L) / C, np.nan), axis=1)
        lo = np.nanmin(np.where(valid, L, np.nan), axis=1)
        hi = np.nanmax(np.where(valid, H, np.nan), axis=1)
        out['clv_20d'] = np.where(hi > lo, (C[:, last] - lo) / (hi - lo), np.nan)
        out['ma_gap'] = C[:, last] / g['MA20'][t, s] - 1

        out['turn_20d'] = np.nanmean(np.where(valid, turn, np.nan), axis=1)
        il = np.where(has_prev & (amt > 0), np.abs(r) / amt, np.nan)
        n_il = np.isfinite(il).sum(axis=1)
        out['illiq_20d'] = np.where(n_il >= MIN_DAYS_LIQ, np.log1p(np.nanmean(il, axis=1) * 1e8), np.nan)
        ok = np.isfinite(r) & np.isfinite(mr)
        n = ok.sum(axis=1)
        rm, mm = np.where(ok, r, np.nan), np.where(ok, mr, np.nan)
        dr, dm = rm - np.nanmean(rm, axis=1, keepdims=True), mm - np.nanmean(mm, axis=1, keepdims=True)
        cov = np.nansum(dr * dm, axis=1) / (n - 1)
        var_m = np.nansum(dm ** 2, axis=1) / (n - 1)
        var_r = np.nansum(dr ** 2, axis=1) / (n - 1)
        resid = np.clip(var_r - cov ** 2 / np.where(var_m > 0, var_m, np.nan), 0, None)
        out['ivol_20d'] = np.where(n >= MIN_DAYS_LIQ, np.sqrt(resid), np.nan)

        c_to = Cf_full[t - MOM_TO, s]
        c_from = Cf_full[np.maximum(t - MOM_FROM, 0), s]
        out['mom_12_1'] = np.where(t >= MOM_FROM, c_to / c_from - 1, np.nan)
    out['has_st_in_win'] = (np.nan_to_num(take(g['is_ST'])) > 0).any(axis=1)
    return out


def main() -> int:
    t0 = time.time()
    p = pd.read_feather(EXT_PANEL, columns=PANEL_COLS)
    p['is_ST'] = p['is_ST'].astype(np.float64)
    dates, stocks, g = dense_grid(p)
    del p
    r_full, _, M = market_returns(g)
    Cf_full = pd.DataFrame(g['Close']).ffill().to_numpy()
    traded = np.isfinite(g['Close'])
    n_hist_grid = np.cumsum(traded, axis=0, dtype=np.int32)

    u = pd.read_feather(ASHARE_UNIVERSE_PATH, columns=['StockID', 'ending_date', 't_idx', 'split', 'n_obs_in_win',
                                                       'board', 'MarketCap'])
    u['StockID'] = u['StockID'].astype(str)
    t = dates.get_indexer(pd.DatetimeIndex(u['ending_date']))
    s = stocks.get_indexer(u['StockID'])
    if (t < 0).any() or (s < 0).any() or (t != u['t_idx'].to_numpy()).any():
        raise RuntimeError('universe 与面板的日历或股票不一致')

    parts = []
    for a in range(0, len(u), 200_000):
        b = min(a + 200_000, len(u))
        parts.append(pd.DataFrame(window_controls(g, Cf_full, M, t[a:b], s[a:b])))
        print(f'  {b:,}/{len(u):,}  {time.time() - t0:.0f}s', flush=True)
    out = pd.concat([u[['StockID', 'ending_date', 'split', 'n_obs_in_win', 'board']].reset_index(drop=True),
                     pd.concat(parts, ignore_index=True)], axis=1)

    fmv = g['FloatMV'][t, s]
    out['size'] = np.log(np.where(fmv > 0, fmv, np.nan))
    pb, pe = g['PB'][t, s], g['PE_TTM'][t, s]
    with np.errstate(divide='ignore'):
        out['bm'] = np.where(pb > 0, 1 / pb, np.nan)
        out['ep'] = np.where(pe != 0, 1 / pe, np.nan)
    out['FloatMV'] = fmv
    out['is_ST'] = g['is_ST'][t, s] > 0
    out['ret_1d'] = r_full[t, s]
    age = n_hist_grid[t, s] - 1
    first_idx = traded.argmax(axis=0)
    first_trade = pd.Series(dates[first_idx], index=stocks)
    lim = limit_pct(out['StockID'], out['ending_date'], is_st=out['is_ST'], age=age,
                    first_trade=out['StockID'].map(first_trade), data_start=dates[0])
    out['limit_up_sig'] = hit_limit(out['ret_1d'], lim, 'up')
    out['limit_dn_sig'] = hit_limit(out['ret_1d'], lim, 'dn')
    out['n_hist'] = age + 1

    info = pd.read_feather(STOCK_INFO, columns=['StockID', 'industry', 'list_date'])
    info['StockID'] = info['StockID'].astype(str)
    out = out.merge(info, on='StockID', how='left')
    start = out['list_date'].fillna(out['StockID'].map(first_trade))
    out['age_days'] = (out['ending_date'] - start).dt.days
    out = out.drop(columns=['list_date'])

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_feather(WEEKLY_CONTROLS)
    oos = out[out['split'] == 'oos']
    chk = {'rows': len(out), 'weeks': int(out['ending_date'].nunique()), 'oos_rows': len(oos),
           'window': 'market-day (E7 protocol v2, section 4)',
           'oos_missing_share': {c: float(oos[c].isna().mean()) for c in CTRL_A + CTRL_B + ['industry']},
           'oos_share_has_st_in_win': float(oos['has_st_in_win'].mean()),
           'oos_describe': oos[CTRL_A + CTRL_B].describe(percentiles=[0.01, 0.5, 0.99]).round(4).to_dict(),
           'seconds': round(time.time() - t0)}
    (OUT / 'weekly_controls_check.json').write_text(json.dumps(chk, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in chk.items() if k != 'oos_describe'}, indent=2, default=str))
    print(oos[CTRL_A + CTRL_B].describe(percentiles=[0.01, 0.5, 0.99]).T.round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
