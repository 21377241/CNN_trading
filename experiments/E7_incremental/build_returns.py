# -*- coding: utf-8 -*-
"""Execution-consistent, unclipped weekly returns for every stock trading on a weekly signal date (protocol section 3).

ret_oo : Open(t+1) -> Open(first trading day after the next signal date T)
ret_cc : Close(t)  -> Close(T)
Exit on a suspended day is deferred to the stock's next trading day (at most MAX_DEFER market days); beyond that the
last available close is used (`*_stale`). For ret_oo a one-price limit-down day also blocks the exit (E9 protocol 3.2;
`oo_exit_lock_dn`). One-price limit days come from common.limit_state (raw prices at the limit price, to the cent).
A stock with no trade after the holding period, ending before the data end, is treated as delisted at its last close
(`delisted`; `*_d50` versions apply a -50% delisting return).
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
    EXT_PANEL, KEYS, OUT, OUT_DATA, WEEKLY_RETURNS, limit_state, week_ends,
)
from Misc.config import ASHARE_UNIVERSE_PATH  # noqa: E402

MAX_DEFER = 20


def exit_price(key, code, d, px_exit, px_close, sig_rows, target, n_days, last_idx_of_code, can_exit=None):
    """Exit at px_exit on the first trading day >= target on which the position can be sold (can_exit; default every
    trading day), within MAX_DEFER market days; otherwise the last close within the deferral window (`stale`)."""
    N = len(key)
    big = np.int64(n_days + 100)
    tgt_key = code[sig_rows].astype(np.int64) * big + target
    j = np.searchsorted(key, tgt_key, side='left')
    jc = np.minimum(j, N - 1)
    has_after = (j < N) & (code[jc] == code[sig_rows])
    if can_exit is None:
        je = jc
    else:
        nxt_ok = np.where(can_exit, np.arange(N), N)
        nxt_ok = np.minimum.accumulate(nxt_ok[::-1])[::-1]
        je = np.minimum(nxt_ok[jc], N - 1)
        has_after_ok = has_after & (nxt_ok[jc] < N) & (code[je] == code[sig_rows])
    ok = has_after if can_exit is None else has_after_ok
    on_time = ok & (d[je] <= target + MAX_DEFER)
    stale = has_after & ~on_time
    gone = ~has_after
    delisted = gone & (last_idx_of_code[code[sig_rows]] < n_days - 1 - MAX_DEFER)
    px = np.full(len(sig_rows), np.nan)
    px[on_time] = px_exit[je[on_time]]
    last_in_win = np.searchsorted(key, code[sig_rows].astype(np.int64) * big + target + MAX_DEFER, side='right') - 1
    prev = np.where(stale, last_in_win, np.maximum(j - 1, 0))
    use_prev = stale | delisted
    px[use_prev] = px_close[prev[use_prev]]
    delay = np.where(on_time, d[je] - target, -1)
    locked = on_time & (je != jc) if can_exit is not None else np.zeros(len(sig_rows), dtype=bool)
    return px, delay, stale, delisted, locked


def main() -> int:
    t0 = time.time()
    p = pd.read_feather(EXT_PANEL, columns=['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Close_raw', 'adj_factor',
                                            'PreClose', 'LimitUp', 'LimitDown', 'LimitStatus'])
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

    # daily limit state of every panel row
    ls = limit_state(p)
    lock_up_all = ls['one_price_up'].to_numpy()
    lock_dn_all = ls['one_price_dn'].to_numpy()
    status = p['LimitStatus'].to_numpy()
    has_lim = ~ls['no_limit'].to_numpy()
    limit_check = {
        'rows_with_limit_price': int(has_lim.sum()),
        'close_up_vs_LimitStatus_1_agreement': float((ls['close_up'].to_numpy() == (status == 1))[has_lim].mean()),
        'close_dn_vs_LimitStatus_-1_agreement': float((ls['close_dn'].to_numpy() == (status == -1))[has_lim].mean()),
        'one_price_lock_up_rows': int(lock_up_all.sum()), 'one_price_lock_dn_rows': int(lock_dn_all.sum()),
    }
    del ls, status
    print('limit check:', json.dumps(limit_check), flush=True)

    # close-to-close
    px, delay, stale, dl, _ = exit_price(key, code, d, C, C, sig, T, n_days, last_idx_of_code)
    out['ret_cc'] = px / C[sig] - 1
    out['cc_exit_delay'], out['cc_stale'], out['delisted'] = delay, stale, dl

    # next-open entry to the open after T; exit deferred past suspensions and one-price limit-down days
    nxt = np.minimum(sig + 1, N - 1)
    enterable = (sig + 1 < N) & (code[nxt] == code[sig]) & (d[nxt] == d[sig] + 1)
    e = T + 1
    valid_e = e < n_days
    px_o, delay_o, stale_o, dl_o, locked_o = exit_price(key, code, d, O, C, sig, np.minimum(e, n_days - 1), n_days,
                                                        last_idx_of_code, can_exit=~lock_dn_all)
    entry = np.where(enterable, O[nxt], np.nan)
    out['ret_oo'] = np.where(valid_e, px_o / entry - 1, np.nan)
    out['not_enterable'] = ~enterable
    out['oo_exit_delay'], out['oo_stale'], out['delisted_oo'] = delay_o, stale_o, dl_o
    out['oo_exit_lock_dn'] = locked_o
    out['entry_lock_up'] = enterable & lock_up_all[nxt]
    out['ret_cc_d50'] = np.where(out['delisted'], (1 + out['ret_cc']) * 0.5 - 1, out['ret_cc'])
    out['ret_oo_d50'] = np.where(out['delisted_oo'], (1 + out['ret_oo']) * 0.5 - 1, out['ret_oo'])

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out['StockID'] = out['StockID'].astype(str)
    out.reset_index(drop=True).to_feather(WEEKLY_RETURNS)
    print(f'{len(out):,} stock-weeks, {out["ending_date"].nunique()} weeks, {time.time() - t0:.0f}s')
    limit_check['weekly_share_oo_exit_lock_dn'] = float(out['oo_exit_lock_dn'].mean())
    limit_check['weekly_share_entry_lock_up'] = float(out['entry_lock_up'].mean())
    (OUT / 'limit_flag_check.json').write_text(json.dumps(limit_check, indent=2), encoding='utf-8')

    # ---- validation against the universe labels (Close(T) / Close(t) - 1 when the stock trades on T) ----
    u = pd.read_feather(ASHARE_UNIVERSE_PATH, columns=KEYS + ['split', 'ret_label', 'is_bj'])
    u['StockID'] = u['StockID'].astype(str)
    m = u.merge(out, on=KEYS, how='left', validate='one_to_one', indicator=True)
    on_t = m['cc_exit_delay'] == 0
    censored = m['ret_cc'].isna() & ~m['delisted'].astype(bool) & ~m['cc_stale'].astype(bool)
    oos = (m['split'] == 'oos') & (m['is_bj'] == 0)
    mo = m[oos]
    chk = {
        'weekly_rows': len(out),
        'universe_rows': len(u), 'universe_rows_matched': int((m['_merge'] == 'both').sum()),
        'universe_rows_ret_cc_nan': int(m['ret_cc'].isna().sum()),
        'ret_cc_nan_suspended_to_sample_end': int((censored & (m['next_signal'] > cal[-1 - MAX_DEFER])).sum()),
        'universe_label_rows_exit_on_T': int((on_t & m['ret_label'].notna()).sum()),
        'ret_cc_equals_ret_label_on_T_1e-9': float(((m.loc[on_t, 'ret_cc'] - m.loc[on_t, 'ret_label']).abs()
                                                     < 1e-9).mean()),
        'label_nan_but_cc_exit_on_T': int((on_t & m['ret_label'].isna()).sum()),
        'oos_main_rows': len(mo),
        'oos_share_ret_oo_available': float(mo['ret_oo'].notna().mean()),
        'oos_share_not_enterable': float(mo['not_enterable'].mean()),
        'oos_share_entry_lock_up': float(mo['entry_lock_up'].mean()),
        'oos_share_oo_exit_deferred': float((mo['oo_exit_delay'] > 0).mean()),
        'oos_share_oo_exit_lock_dn': float(mo['oo_exit_lock_dn'].mean()),
        'oos_share_oo_stale': float(mo['oo_stale'].mean()),
        'oos_share_delisted_oo': float(mo['delisted_oo'].mean()),
        'oos_corr_ret_oo_ret_cc': float(mo[['ret_oo', 'ret_cc']].corr().iloc[0, 1]),
        'oos_ret_oo_quantiles': mo['ret_oo'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(4).to_dict(),
        'oos_ret_cc_quantiles': mo['ret_cc'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(4).to_dict(),
    }
    (OUT / 'weekly_returns_check.json').write_text(json.dumps(chk, indent=2, default=str), encoding='utf-8')
    print(json.dumps(chk, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
