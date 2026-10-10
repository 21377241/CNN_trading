# -*- coding: utf-8 -*-
"""
build_universe.py - 周度样本键、标签、切分（clean_spec 第 3.1 节、第 4 节）

输入：ashare_ret.feather（1_preprocess_ashare_data.py 输出）
输出：WORK_SPACE/data/ashare_processed_data/universe_weekly.feather
      WORK_SPACE/data/ashare_processed_data/universe_check.json

列：
  StockID, ending_date        信号日 t（ISO 周最后一个市场交易日）
  t_idx                       t 在市场交易日历中的序号
  year, split                 信号日年份；train 2000-2005 / val 2006-2007 / oos 2008-2025
  embargo                     2005、2007 年最后一个信号周 = 1（训练/验证时剔除）
  next_date                   下一个信号日
  ret_label                   Close(next_date) / Close(t) - 1（未裁剪；next_date 无交易为 NaN）
  label                       1(ret_label > 0)；ret_label 为 NaN 时 -1
  n_obs_in_win                窗口（t 往前 20 个市场日）内有交易的天数
  board                       main / chinext / star / bse
  MarketCap, Close            t 日流通市值、后复权收盘价

样本条件：t 日有交易；t 日非 ST；首个交易日序号 <= t_idx - 39。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, ASHARE_UNIVERSE_PATH, PRE_CLEAN_WORK_DIR  # noqa: E402

WINDOW = 20
MA_LOOKBACK = 19
TRAIN_YEARS = range(2000, 2006)
VAL_YEARS = range(2006, 2008)
OOS_YEARS = range(2008, 2026)


def week_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    d = pd.DataFrame({'Date': pd.DatetimeIndex(calendar).sort_values()})
    iso = d['Date'].dt.isocalendar()
    return pd.DatetimeIndex(d.groupby(iso['year'] * 100 + iso['week'])['Date'].max().sort_values().values)


def board_of(sid: pd.Series) -> np.ndarray:
    s = sid.astype(str)
    out = np.full(len(s), 'main', dtype=object)
    out[s.str.startswith(('300', '301')).values] = 'chinext'
    out[s.str.startswith(('688', '689')).values] = 'star'
    out[s.str.startswith(('4', '8', '92')).values] = 'bse'
    return out


def build(panel: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DatetimeIndex(np.sort(panel['Date'].unique()))
    midx = pd.Series(np.arange(len(cal)), index=cal)
    panel = panel.assign(m=midx.reindex(panel['Date']).values)
    code = panel['StockID'].cat.codes.to_numpy().astype(np.int64)
    m = panel['m'].to_numpy().astype(np.int64)
    key = code * 100_000 + m
    assert np.all(np.diff(key) > 0), '面板须按 StockID、Date 排序且无重复'

    first_m = panel.groupby('StockID', observed=True)['m'].transform('min').to_numpy()
    lo = np.searchsorted(key, key - (WINDOW - 1), side='left')
    n_obs = np.arange(len(key)) - lo + 1

    we = week_end_dates(cal)
    we_m = midx.reindex(we).to_numpy()
    is_we = np.isin(m, we_m)
    nxt = pd.Series(np.r_[we_m[1:], -1], index=we_m)
    next_m = nxt.reindex(m[is_we]).to_numpy()

    u = panel.loc[is_we, ['StockID', 'Date', 'Close', 'MarketCap', 'is_ST']].copy()
    u['t_idx'] = m[is_we]
    u['first_idx'] = first_m[is_we]
    u['n_obs_in_win'] = n_obs[is_we].astype(np.int16)
    u['next_m'] = next_m
    keep = (u['is_ST'] == 0) & (u['first_idx'] <= u['t_idx'] - (WINDOW + MA_LOOKBACK)) & (u['next_m'] >= 0)
    u = u[keep.values].copy()

    nkey = u['StockID'].cat.codes.to_numpy().astype(np.int64) * 100_000 + u['next_m'].to_numpy()
    pos = np.searchsorted(key, nkey)
    pos_c = np.minimum(pos, len(key) - 1)
    hit = key[pos_c] == nkey
    close_next = np.where(hit, panel['Close'].to_numpy()[pos_c], np.nan)
    u['ret_label'] = close_next / u['Close'].to_numpy() - 1.0
    u['label'] = np.where(np.isnan(u['ret_label']), -1, (u['ret_label'] > 0).astype(int)).astype(np.int8)
    u['next_date'] = cal[u['next_m'].to_numpy()]
    u = u.rename(columns={'Date': 'ending_date'})
    u['year'] = u['ending_date'].dt.year.astype(np.int16)
    u['split'] = np.select(
        [u['year'].isin(TRAIN_YEARS), u['year'].isin(VAL_YEARS), u['year'].isin(OOS_YEARS)],
        ['train', 'val', 'oos'], default='none')
    last_we = {y: we[we.year == y].max() for y in (2005, 2007)}
    u['embargo'] = u['ending_date'].isin(list(last_we.values())).astype(np.int8)
    u['board'] = board_of(u['StockID'])
    u = u[u['split'] != 'none']
    cols = ['StockID', 'ending_date', 't_idx', 'year', 'split', 'embargo', 'next_date', 'ret_label', 'label',
            'n_obs_in_win', 'board', 'MarketCap', 'Close']
    return u[cols].sort_values(['ending_date', 'StockID']).reset_index(drop=True)


def reconcile_old(u: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """与旧 I20R5 样本键（labels.feather，按旧 EquityDataset 规则过滤）对账。"""
    old_dir = Path(PRE_CLEAN_WORK_DIR, 'data', 'ashare_stocks_dataset', 'I20R5', 'stocks_CHN', 'dataset_all')
    if not old_dir.exists():
        return {'skipped': str(old_dir)}
    frames = []
    for y in range(2000, 2026):
        f = old_dir / f'20d_week_has_vb_[20]_ma_{y}_labels.feather'
        if f.exists():
            frames.append(pd.read_feather(f, columns=['StockID', 'ending_date', 'ret_val', 'EWMA_vol']))
    old = pd.concat(frames, ignore_index=True)
    old['StockID'] = old['StockID'].astype(str).str.zfill(6)
    old['ending_date'] = pd.to_datetime(old['ending_date'])
    old_all = len(old)
    old_lab = old[(old['ret_val'].notna()) & (old['EWMA_vol'] > 0)]
    new = u[['StockID', 'ending_date', 'split', 'ret_label', 'n_obs_in_win']].copy()
    new['StockID'] = new['StockID'].astype(str)
    mg = old_lab.merge(new, on=['StockID', 'ending_date'], how='outer', indicator=True)
    both = mg['_merge'] == 'both'
    only_old = mg[mg['_merge'] == 'left_only'][['StockID', 'ending_date']]
    p = panel[['StockID', 'Date', 'is_ST']].copy()
    p['StockID'] = p['StockID'].astype(str)
    oo = only_old.merge(p, left_on=['StockID', 'ending_date'], right_on=['StockID', 'Date'], how='left')
    only_new = mg[mg['_merge'] == 'right_only']
    d = (mg.loc[both, 'ret_val'] - mg.loc[both, 'ret_label']).abs()
    oos_new = u[u['split'] == 'oos']
    return {
        'old_rows_all': old_all,
        'old_rows_labeled_ewma_pos': len(old_lab),
        'new_rows': len(u),
        'both': int(both.sum()),
        'only_old': int(len(only_old)),
        'only_old_no_trade_on_t': int(oo['Date'].isna().sum()),
        'only_old_st_on_t': int((oo['is_ST'] == 1).sum()),
        'only_new': int(len(only_new)),
        'only_new_label_nan': int(only_new['ret_label'].isna().sum()),
        'only_new_n_obs_lt_20': int((only_new['n_obs_in_win'] < 20).sum()),
        'label_absdiff_le_1e-9_share': float((d <= 1e-9).mean()),
        'label_absdiff_gt_1e-3_share': float((d > 1e-3).mean()),
        'oos_new_rows': len(oos_new),
        'oos_old_rows_all': int((old['ending_date'].dt.year >= 2008).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-reconcile', action='store_true')
    args = ap.parse_args(argv)
    t0 = time.time()
    panel = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Close', 'MarketCap', 'is_ST'])
    u = build(panel)
    u.to_feather(ASHARE_UNIVERSE_PATH)
    chk = {
        'rows': len(u),
        'by_split': u['split'].value_counts().to_dict(),
        'embargo_rows': int(u['embargo'].sum()),
        'n_weeks_oos': int(u.loc[u['split'] == 'oos', 'ending_date'].nunique()),
        'label_nan_share_by_split': u.assign(nan=u['label'] < 0).groupby('split')['nan'].mean().to_dict(),
        'up_share_by_split': u[u['label'] >= 0].groupby('split')['label'].mean().to_dict(),
        'n_obs_lt_20_share': float((u['n_obs_in_win'] < 20).mean()),
        'n_obs_quantiles': u['n_obs_in_win'].quantile([0.01, 0.05, 0.5]).to_dict(),
        'board_oos': u.loc[u['split'] == 'oos', 'board'].value_counts().to_dict(),
        'ret_label_abs_max': float(u['ret_label'].abs().max()),
    }
    if not args.no_reconcile:
        chk['vs_pre_clean_I20R5'] = reconcile_old(u, pd.read_feather(ASHARE_PANEL_PATH,
                                                                      columns=['Date', 'StockID', 'is_ST']))
    Path(ASHARE_PROCESSED_DATA_DIR, 'universe_check.json').write_text(
        json.dumps(chk, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps(chk, indent=2, ensure_ascii=False, default=str))
    print(f'完成，用时 {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
