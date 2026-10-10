# -*- coding: utf-8 -*-
"""Execution-consistent, unclipped weekly returns for every stock trading on a weekly signal date (protocol section 3).

ret_oo : Open(t+1) -> Open(first trading day after the next signal date T)
ret_cc : Close(t)  -> Close(T)
Exit on a suspended day is deferred to the stock's next trading day (at most MAX_DEFER market days); beyond that the
last available close is used (`*_stale`). A stock with no trade after the holding period, ending before the data end,
is treated as delisted at its last close (`delisted`; `*_d50` versions apply a -50% delisting return).
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
    EXT_PANEL, KEYS, LIMIT_TOL, OUT, OUT_DATA, WEEKLY_RETURNS, limit_pct, load_canonical_preds, week_ends,
)

MAX_DEFER = 20
CLIP = 0.11


def exit_price(key, code, d, px_exit, px_close, sig_rows, target, n_days, last_idx_of_code):
    """Exit at px_exit on the first trading day >= target (within MAX_DEFER), else last close before target."""
    N = len(key)
    big = np.int64(n_days + 100)
    tgt_key = code[sig_rows].astype(np.int64) * big + target
    j = np.searchsorted(key, tgt_key, side='left')
    jc = np.minimum(j, N - 1)
    has_after = (j < N) & (code[jc] == code[sig_rows])
    on_time = has_after & (d[jc] <= target + MAX_DEFER)
    stale = has_after & ~on_time
    gone = ~has_after
    delisted = gone & (last_idx_of_code[code[sig_rows]] < n_days - 1 - MAX_DEFER)
    px = np.full(len(sig_rows), np.nan)
    px[on_time] = px_exit[jc[on_time]]
    prev = np.maximum(j - 1, 0)
    use_prev = stale | delisted
    px[use_prev] = px_close[prev[use_prev]]
    delay = np.where(on_time, d[jc] - target, -1)
    return px, delay, stale, delisted


def main() -> int:
    t0 = time.time()
    p = pd.read_feather(EXT_PANEL, columns=['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'is_ST'])
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    n_days = len(cal)
    we = week_ends(cal)
    we_idx = cal.searchsorted(we)
    next_we = np.full(n_days, -1, dtype=np.int64)
    next_we[we_idx[:-1]] = we_idx[1:]

    code = p['StockID'].cat.codes.to_numpy().astype(np.int64)
    d = cal.searchsorted(p['Date'].to_numpy()).astype(np.int64)
    key = code * np.int64(n_days + 100) + d
    assert np.all(np.diff(key) > 0), 'panel must be sorted by StockID, Date without duplicates'
    O = p['Open'].to_numpy(np.float64)
    H = p['High'].to_numpy(np.float64)
    L = p['Low'].to_numpy(np.float64)
    C = p['Close'].to_numpy(np.float64)
    N = len(p)
    last_idx_of_code = pd.Series(d).groupby(code).max().reindex(range(code.max() + 1)).fillna(-1).to_numpy()

    sig = np.flatnonzero(np.isin(d, we_idx) & (next_we[d] >= 0))
    T = next_we[d[sig]]
    out = pd.DataFrame({'StockID': p['StockID'].to_numpy()[sig], 'ending_date': cal[d[sig]], 'next_signal': cal[T]})

    # close-to-close
    px, delay, stale, dl = exit_price(key, code, d, C, C, sig, T, n_days, last_idx_of_code)
    out['ret_cc'] = px / C[sig] - 1
    out['cc_exit_delay'], out['cc_stale'], out['delisted'] = delay, stale, dl

    # next-open entry to the open after T
    nxt = np.minimum(sig + 1, N - 1)
    enterable = (sig + 1 < N) & (code[nxt] == code[sig]) & (d[nxt] == d[sig] + 1)
    e = T + 1
    valid_e = e < n_days
    px_o, delay_o, stale_o, dl_o = exit_price(key, code, d, O, C, sig, np.minimum(e, n_days - 1), n_days,
                                              last_idx_of_code)
    entry = np.where(enterable, O[nxt], np.nan)
    out['ret_oo'] = np.where(valid_e, px_o / entry - 1, np.nan)
    out['not_enterable'] = ~enterable
    out['oo_exit_delay'], out['oo_stale'], out['delisted_oo'] = delay_o, stale_o, dl_o
    pos = pd.Series(code).groupby(code).cumcount().to_numpy()
    first_trade = cal[pd.Series(d).groupby(code).transform('min').to_numpy()]
    lim = limit_pct(out['StockID'].astype(str), cal[d[nxt]], is_st=p['is_ST'].to_numpy()[nxt], age=pos[nxt],
                    first_trade=first_trade[nxt], data_start=cal[0])
    one_price = (O[nxt] == H[nxt]) & (H[nxt] == L[nxt]) & (L[nxt] == C[nxt])
    out['entry_lock_up'] = enterable & one_price & (C[nxt] / C[sig] - 1 >= lim - LIMIT_TOL)
    out['ret_cc_d50'] = np.where(out['delisted'], (1 + out['ret_cc']) * 0.5 - 1, out['ret_cc'])
    out['ret_oo_d50'] = np.where(out['delisted_oo'], (1 + out['ret_oo']) * 0.5 - 1, out['ret_oo'])

    # clipped close-to-close on the same exit day, for reconciliation with the original label
    r = np.full(N, 0.0)
    same = np.r_[False, code[1:] == code[:-1]]
    r[same] = C[1:][same[1:]] / C[:-1][same[1:]] - 1
    cum = np.cumsum(np.log1p(np.clip(r, -CLIP, CLIP)))
    jT = np.searchsorted(key, code[sig] * np.int64(n_days + 100) + T)
    jT = np.minimum(jT, N - 1)
    exact = (code[jT] == code[sig]) & (d[jT] == T)
    out['ret_cc_clip_exact'] = np.where(exact, np.exp(cum[jT] - cum[sig]) - 1, np.nan)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out['StockID'] = out['StockID'].astype(str)
    out.reset_index(drop=True).to_feather(WEEKLY_RETURNS)
    print(f'{len(out):,} stock-weeks, {out["ending_date"].nunique()} weeks, {time.time() - t0:.0f}s')

    # ---- validation against the canonical OOS labels ----
    try:
        pr = load_canonical_preds()[KEYS + ['ret_val']]
    except (FileNotFoundError, RuntimeError) as e:
        print(f'跳过与规范预测的对账：{e}')
        return 0
    m = pr.merge(out, on=KEYS, how='left')
    ex = m['ret_cc_clip_exact'].notna()
    chk = {
        'weekly_rows': len(out),
        'pred_rows': len(pr), 'pred_rows_matched': int(m['ret_cc'].notna().sum()),
        'pred_rows_ret_oo_available': int(m['ret_oo'].notna().sum()),
        'pred_share_not_enterable': float(m['not_enterable'].mean()),
        'pred_share_entry_lock_up': float(m['entry_lock_up'].mean()),
        'pred_share_oo_exit_deferred': float((m['oo_exit_delay'] > 0).mean()),
        'pred_share_oo_stale': float(m['oo_stale'].mean()),
        'pred_share_delisted_oo': float(m['delisted_oo'].mean()),
        'clip_recompute_matches_ret_val_1e-5': float(((m.loc[ex, 'ret_cc_clip_exact'] - m.loc[ex, 'ret_val']).abs()
                                                       < 1e-5).mean()),
        'share_cc_exit_on_T': float(ex.mean()),
        'ret_cc_equals_ret_val_1e-5': float(((m['ret_cc'] - m['ret_val']).abs() < 1e-5).mean()),
        'corr_ret_oo_ret_cc': float(m[['ret_oo', 'ret_cc']].corr().iloc[0, 1]),
        'ret_oo_quantiles': m['ret_oo'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(4).to_dict(),
        'ret_cc_quantiles': m['ret_cc'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(4).to_dict(),
    }
    (OUT / 'weekly_returns_check.json').write_text(json.dumps(chk, indent=2, default=str), encoding='utf-8')
    print(json.dumps(chk, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
