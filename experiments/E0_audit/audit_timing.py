# -*- coding: utf-8 -*-
"""E0-T3: data dictionary and timing audit for the weekly OOS panel.

Quantifies sampling frequency, label construction (clipping, suspensions), sample selection,
train/validation/OOS boundaries, universe filters, market-cap fallback, price adjustment and
price-limit / suspension frictions at the signal date and next trading day.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.E2C_temporal.eval_e2c import nw_t_mean

DATA = PROJECT_ROOT / 'WORK_SPACE' / 'data' / 'ashare_processed_data'
LABELS = PROJECT_ROOT / 'WORK_SPACE' / 'data' / 'ashare_stocks_dataset' / 'I20R5' / 'stocks_CHN' / 'dataset_all'
OUT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E0_audit' / 'timing'
PREDS = {
    'chartcnn_e1b_ric': 'WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/preds/pred_cnn_oos.csv',
    'onedcnn_120': 'WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv',
    'mlp4_120_s5': 'WORK_SPACE/paper/E2F_mlp_depth/preds/pred_mlp4_oos.csv',
}
KEYS = ['StockID', 'ending_date']
CLIP = 0.11
LIMIT_TOL = 0.005


def limit_pct(stock: pd.Series, date: pd.Series) -> np.ndarray:
    s = stock.astype(str)
    lim = np.full(len(s), 0.10)
    chinext = s.str.startswith(('300', '301')) & (date >= pd.Timestamp('2020-08-24'))
    star = s.str.startswith(('688', '689'))
    bse = s.str.startswith(('4', '8', '92'))
    lim[chinext.values | star.values] = 0.20
    lim[bse.values] = 0.30
    return lim


def load_panel() -> pd.DataFrame:
    cols = ['Date', 'StockID', 'Open', 'Close', 'Vol', 'MarketCap', 'sec_name', 'Ret', 'EWMA_vol']
    p = pd.read_feather(DATA / 'ashare_ret.feather', columns=cols)
    raw = pd.read_feather(DATA / 'ashare_daily_ret_raw.feather')
    p['StockID'] = p['StockID'].astype(str).str.zfill(6)
    raw['StockID'] = raw['StockID'].astype(str).str.zfill(6)
    p = p.merge(raw, on=['Date', 'StockID'], how='left')
    return p.sort_values(['StockID', 'Date']).reset_index(drop=True)


def week_ends(dates: pd.Series) -> pd.DatetimeIndex:
    d = pd.DataFrame({'Date': pd.DatetimeIndex(dates.unique()).sort_values()})
    iso = d['Date'].dt.isocalendar()
    return pd.DatetimeIndex(d.groupby(iso['year'] * 100 + iso['week'])['Date'].max().sort_values().values)


def weekly_returns(p: pd.DataFrame, ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Close(t) -> Close(next market week end), requiring the stock to trade on both dates (original rule)."""
    g = p.groupby('StockID', sort=False)
    p['c_clip'] = g['Ret'].transform(lambda x: np.log1p(x).cumsum())
    p['c_raw'] = np.log1p(p['daily_ret_raw'].fillna(0.0))
    p['c_raw'] = p.groupby('StockID', sort=False)['c_raw'].cumsum()
    nxt = pd.Series(ends[1:], index=ends[:-1])
    e = p[p['Date'].isin(ends)][['StockID', 'Date', 'c_clip', 'c_raw']].copy()
    e['next_end'] = e['Date'].map(nxt)
    m = e.merge(e[['StockID', 'Date', 'c_clip', 'c_raw']].rename(columns={'Date': 'next_end'}),
                on=['StockID', 'next_end'], how='left', suffixes=('', '_n'))
    m['ret_clip'] = np.exp(m['c_clip_n'] - m['c_clip']) - 1
    m['ret_raw'] = np.exp(m['c_raw_n'] - m['c_raw']) - 1
    return m.rename(columns={'Date': 'ending_date'})[KEYS + ['next_end', 'ret_clip', 'ret_raw']]


def hl_sharpe(df: pd.DataFrame, score: str, ret: str) -> dict:
    rows = []
    for dt, g in df.dropna(subset=[score, ret]).groupby('ending_date'):
        dec = pd.qcut(g[score].rank(method='first'), 10, labels=False)
        hi, lo = g[dec == 9], g[dec == 0]
        rows.append({'ew': hi[ret].mean() - lo[ret].mean(),
                     'vw': np.average(hi[ret], weights=hi['MarketCap'].clip(lower=0))
                     - np.average(lo[ret], weights=lo['MarketCap'].clip(lower=0))})
    r = pd.DataFrame(rows)
    return {f'{k}_sharpe': float(r[k].mean() / r[k].std() * np.sqrt(52)) for k in ('ew', 'vw')} | {
        f'{k}_mean_wk': float(r[k].mean()) for k in ('ew', 'vw')} | {
        f'{k}_nw_t': nw_t_mean(r[k].values) for k in ('ew', 'vw')}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    p = load_panel()
    cal = pd.DatetimeIndex(p['Date'].unique()).sort_values()
    ends = week_ends(p['Date'])
    res['panel'] = {
        'rows': len(p), 'stocks': int(p['StockID'].nunique()),
        'date_min': str(cal.min().date()), 'date_max': str(cal.max().date()), 'trading_days': len(cal),
    }

    # ---- price adjustment and market cap ----
    s1 = p[p['StockID'] == '000001'].set_index('Date')['Close']
    res['price_adjustment'] = {
        '000001_close_first': float(s1.iloc[0]), '000001_close_last': float(s1.iloc[-1]),
        '000001_first_date': str(s1.index[0].date()), '000001_last_date': str(s1.index[-1].date()),
        'min_close_all': float(p['Close'].min()),
    }
    fallback = np.isclose(p['MarketCap'], p['Vol'] * p['Close'], rtol=1e-9) & (p['MarketCap'] > 0)
    res['market_cap'] = {
        'share_le_zero': float((p['MarketCap'] <= 0).mean()),
        'share_equal_vol_x_close': float(fallback.mean()),
        'median': float(p['MarketCap'].median()),
        'p01': float(p['MarketCap'].quantile(0.01)), 'p99': float(p['MarketCap'].quantile(0.99)),
    }

    # ---- daily returns: clipping, gaps, ST ----
    g = p.groupby('StockID', sort=False)
    p['prev_date'] = g['Date'].shift(1)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    p['gap_days'] = p['Date'].map(pos) - p['prev_date'].map(pos)
    raw = p['daily_ret_raw']
    res['daily_returns'] = {
        'share_raw_abs_gt_clip': float((raw.abs() > CLIP + 1e-9).mean()),
        'share_rows_after_gap_gt1': float((p['gap_days'] > 1).mean()),
        'share_rows_after_gap_gt20': float((p['gap_days'] > 20).mean()),
        'share_clipped_among_gap_gt1': float((raw[p['gap_days'] > 1].abs() > CLIP).mean()),
        'raw_ret_p999': float(raw.quantile(0.999)), 'raw_ret_max': float(raw.max()),
        'rows_with_ST_in_name': int(p['sec_name'].astype(str).str.upper().str.contains('ST').sum()),
        'zero_volume_rows_share': float((p['Vol'] <= 0).mean()),
    }
    last = g['Date'].max()
    first = g['Date'].min()
    res['universe'] = {
        'stocks_last_date_before_2025_06': int((last < pd.Timestamp('2025-06-01')).sum()),
        'stocks_first_date_after_2000_03': int((first > pd.Timestamp('2000-03-01')).sum()),
    }

    # ---- weekly labels ----
    w = weekly_returns(p, ends)
    pred = pd.read_csv(PROJECT_ROOT / PREDS['chartcnn_e1b_ric'])
    pred['StockID'] = pred['StockID'].astype(str).str.zfill(6)
    pred['ending_date'] = pd.to_datetime(pred['ending_date'])
    for name, f in PREDS.items():
        if name == 'chartcnn_e1b_ric':
            continue
        q = pd.read_csv(PROJECT_ROOT / f, usecols=KEYS + ['up_prob'])
        q['StockID'] = q['StockID'].astype(str).str.zfill(6)
        q['ending_date'] = pd.to_datetime(q['ending_date'])
        pred = pred.merge(q.rename(columns={'up_prob': name}), on=KEYS, how='left')
    pred = pred.rename(columns={'up_prob': 'chartcnn_e1b_ric'}).merge(w, on=KEYS, how='left')

    wd = pred['ending_date'].dt.dayofweek.value_counts().sort_index()
    oos_ends = ends[(ends >= pred['ending_date'].min()) & (ends <= pred['ending_date'].max())]
    tdays = pd.Series(np.diff(np.searchsorted(cal, oos_ends)))
    res['sampling'] = {
        'oos_weeks': int(pred['ending_date'].nunique()),
        'market_week_ends_in_oos_range': len(oos_ends),
        'all_ending_dates_are_week_ends': bool(pred['ending_date'].isin(ends).all()),
        'weekday_counts': {int(k): int(v) for k, v in wd.items()},
        'trading_days_between_signals': tdays.value_counts().sort_index().to_dict(),
        'last_oos_week': str(pred['ending_date'].max().date()),
    }
    d = (pred['ret_val'] - pred['ret_clip']).abs()
    dr = (pred['ret_raw'] - pred['ret_clip']).abs()
    res['label'] = {
        'ret_val_reproduced_by_clipped_close_to_close': float((d < 1e-6).mean()),
        'ret_val_max_abs_diff_vs_recomputed': float(d.max()),
        'share_labels_affected_by_clip': float((dr > 1e-6).mean()),
        'mean_abs_raw_minus_clipped_when_affected': float(dr[dr > 1e-6].mean()),
        'n_oos_rows': len(pred), 'nan_ret_val': int(pred['ret_val'].isna().sum()),
    }
    pred.loc[pred['ret_raw'].isna(), 'ret_raw'] = pred['ret_val']
    impact = []
    for name in PREDS:
        for ret in ('ret_val', 'ret_raw'):
            impact.append({'model': name, 'label': ret, **hl_sharpe(pred, name, ret)})
    pd.DataFrame(impact).to_csv(OUT / 'label_clip_impact.csv', index=False)

    # ---- selection: eligible at t but no label because not trading at next week end ----
    p['n_obs'] = g.cumcount() + 1
    e = p[p['Date'].isin(oos_ends) & (p['n_obs'] >= 20) & (p['EWMA_vol'] > 0)][['StockID', 'Date']]
    e = e.rename(columns={'Date': 'ending_date'}).merge(w, on=KEYS, how='left')
    e = e[e['ending_date'] < oos_ends.max()]
    no_label = e['ret_clip'].isna()
    in_pred = e.merge(pred[KEYS].assign(in_pred=1), on=KEYS, how='left')['in_pred'].fillna(0).astype(bool).values
    res['selection'] = {
        'eligible_rows': len(e),
        'share_eligible_without_next_week_label': float(no_label.mean()),
        'share_eligible_in_pred': float(in_pred.mean()),
        'pred_rows_not_eligible_by_rule': int(len(pred) - in_pred.sum()),
    }

    # ---- train / validation / OOS boundaries ----
    lab = []
    for y in range(2000, 2026):
        f = LABELS / f'20d_week_has_vb_[20]_ma_{y}_labels.feather'
        if f.exists():
            lab.append(pd.read_feather(f, columns=['StockID', 'ending_date', 'ret_val']))
    lab = pd.concat(lab, ignore_index=True)
    lab['ending_date'] = pd.to_datetime(lab['ending_date'])
    nxt = pd.Series(ends[1:], index=ends[:-1])
    lab['label_end'] = lab['ending_date'].map(nxt)
    res['boundaries'] = {}
    for name, last_year in (('train_2000_2005', 2005), ('val_2006_2007', 2007)):
        s = lab[lab['ending_date'].dt.year == last_year]
        cross = s[s['label_end'].dt.year > last_year]
        res['boundaries'][name] = {
            'last_signal_week': str(s['ending_date'].max().date()),
            'rows_label_crossing_into_next_period': int(len(cross)),
            'share_of_period_rows': float(len(cross) / len(s)) if len(s) else np.nan,
            'weeks_crossing': int(cross['ending_date'].nunique()),
        }

    # ---- signal-date and next-day frictions ----
    p['next_open'] = g['Open'].shift(-1)
    p['next_date'] = g['Date'].shift(-1)
    p['next_mkt_date'] = p['Date'].map(pd.Series(cal[1:], index=cal[:-1]))
    p['limit'] = limit_pct(p['StockID'], p['Date'])
    sig = p[p['Date'].isin(oos_ends)][['StockID', 'Date', 'daily_ret_raw', 'Close', 'next_open', 'next_date',
                                        'next_mkt_date', 'limit', 'Vol']]
    sig = sig.rename(columns={'Date': 'ending_date'})
    f = pred[KEYS + ['chartcnn_e1b_ric', 'onedcnn_120']].merge(sig, on=KEYS, how='left')
    f['limit_up_close'] = f['daily_ret_raw'] >= f['limit'] - LIMIT_TOL
    f['limit_dn_close'] = f['daily_ret_raw'] <= -(f['limit'] - LIMIT_TOL)
    f['next_open_ret'] = f['next_open'] / f['Close'] - 1
    f['suspended_next_day'] = f['next_date'] != f['next_mkt_date']
    f['limit_up_next_open'] = (f['next_open_ret'] >= f['limit'] - LIMIT_TOL) & ~f['suspended_next_day']
    f['limit_dn_next_open'] = (f['next_open_ret'] <= -(f['limit'] - LIMIT_TOL)) & ~f['suspended_next_day']
    flags = ['limit_up_close', 'limit_dn_close', 'limit_up_next_open', 'limit_dn_next_open', 'suspended_next_day']
    fr = []
    for name in ('chartcnn_e1b_ric', 'onedcnn_120'):
        dec = f.groupby('ending_date')[name].transform(lambda x: pd.qcut(x.rank(method='first'), 10, labels=False))
        for label, mask in (('all', np.ones(len(f), bool)), ('top_decile', dec == 9), ('bottom_decile', dec == 0)):
            fr.append({'model': name, 'group': label, **{k: float(f.loc[mask, k].mean()) for k in flags}})
    pd.DataFrame(fr).to_csv(OUT / 'execution_frictions.csv', index=False)
    res['signal_row_present_in_panel'] = float(f['Close'].notna().mean())

    # ---- listing age ----
    late = first[first > pd.Timestamp('2000-03-01')]
    age = (pred['ending_date'] - pred['StockID'].map(late)).dt.days
    res['listing_age'] = {
        'share_oos_rows_within_365d_of_first_trade': float((age < 365).mean()),
        'share_oos_rows_within_60d_of_first_trade': float((age < 60).mean()),
    }

    (OUT / 'audit_summary.json').write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(pd.DataFrame(impact).to_string(index=False))
    print(pd.DataFrame(fr).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
