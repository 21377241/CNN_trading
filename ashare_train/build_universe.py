# -*- coding: utf-8 -*-
"""
build_universe.py - 周度样本键、标签、切分（clean_spec 第 3.1 节、第 4 节）

输入：ashare_ret.feather（1_preprocess_ashare_data.py 输出）
输出：WORK_SPACE/data/ashare_processed_data/universe_weekly.feather
      WORK_SPACE/data/ashare_processed_data/universe_check.json

列：
  StockID, ending_date        信号日 t（ISO 周最后一个市场交易日）
  t_idx                       t 在市场交易日历中的序号
  year, split                 信号日年份；train 2000-2005 / val 2006-2007 / oos 2008-2026（截至 SAMPLE_END）
  embargo                     2005、2007 年最后一个信号周 = 1（训练/验证时剔除）
  next_date                   下一个信号日
  ret_label                   Close(next_date) / Close(t) - 1（未裁剪；next_date 无交易为 NaN）
  label                       1(ret_label > 0)；ret_label 为 NaN 时 -1
  n_obs_in_win                窗口（t 往前 20 个市场日）内有交易的天数
  board                       main / chinext / star / bse（t 日的 Markettype：1、4 / 16 / 32 / 64）
  is_bj                       board == 'bse'（北交所；主样本剔除，只用于稳健性检验）
  MarketCap, Close            t 日流通市值、后复权收盘价

样本条件：t 日有交易且收盘价为正；t 日非 ST；t 日不在退市整理期；首个交易日序号 <= t_idx - 39；
窗口（t 往前 20 个市场日）内没有成交量为 0 的交易行（CSMAR Filling = 0 但 Dnshrtrd = 0，价格只是前收盘的延续）。

退市整理期：CSMAR Trdsta 没有整理期编码（整理期首日标 *ST，其后标正常交易；2020 年前的创业板股票没有 *ST），
按公司文件的证券简称识别：简称含“退”的股票，其最后一段连续交易（与前一交易行相隔超过 1 个市场日即断开，
整理期前有停牌）即整理期，长度应为 30 天（2020 年改革前）或 15 天。
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

from Misc.config import (  # noqa: E402
    ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, ASHARE_STOCK_INFO_PATH, ASHARE_UNIVERSE_PATH,
)

WINDOW = 20
MA_LOOKBACK = 19
MAX_CONSOLIDATION_DAYS = 30
TRAIN_YEARS = range(2000, 2006)
VAL_YEARS = range(2006, 2008)
OOS_YEARS = range(2008, 2027)
BOARD_OF_MARKETTYPE = {1: 'main', 4: 'main', 16: 'chinext', 32: 'star', 64: 'bse'}


def week_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    d = pd.DataFrame({'Date': pd.DatetimeIndex(calendar).sort_values()})
    iso = d['Date'].dt.isocalendar()
    return pd.DatetimeIndex(d.groupby(iso['year'] * 100 + iso['week'])['Date'].max().sort_values().values)


def board_of(markettype: pd.Series) -> np.ndarray:
    b = markettype.astype(int).map(BOARD_OF_MARKETTYPE)
    if b.isna().any():
        raise ValueError(f'unknown Markettype: {sorted(markettype[b.isna()].unique())}')
    return b.to_numpy(dtype=object)


def board_prefix_mismatch(u: pd.DataFrame) -> dict:
    s = u['StockID'].astype(str)
    pre = np.full(len(s), 'main', dtype=object)
    pre[s.str.startswith(('300', '301', '302')).to_numpy()] = 'chinext'
    pre[s.str.startswith(('688', '689')).to_numpy()] = 'star'
    pre[s.str.startswith(('4', '8', '92')).to_numpy()] = 'bse'
    bad = u.loc[pre != u['board'].to_numpy(), ['StockID', 'board']].astype(str)
    return {'rows': len(bad), 'stocks': bad.drop_duplicates().head(20).values.tolist()}


def delist_consolidation(code: np.ndarray, m: np.ndarray, delisting_name: np.ndarray) -> tuple[np.ndarray, dict]:
    """code、m 按 (StockID, Date) 排序；delisting_name[code] 为简称是否含“退”。返回每行是否在退市整理期。"""
    start = np.r_[True, (code[1:] != code[:-1]) | (np.diff(m) > 1)]
    run = np.cumsum(start)
    last_row = np.r_[code[1:] != code[:-1], True]
    final_run = np.zeros(code.max() + 1, dtype=np.int64)
    final_run[code[last_row]] = run[last_row]
    run_len = np.bincount(run)
    cand = delisting_name[code] & (run == final_run[code])
    flag = cand & (run_len[run] <= MAX_CONSOLIDATION_DAYS)
    lens = pd.Series(run_len[run[last_row]], index=code[last_row])[delisting_name[code[last_row]]]
    return flag, {'stocks': int(len(lens)), 'final_run_length_counts': lens.value_counts().sort_index().to_dict(),
                  'rows': int(flag.sum())}


def build(panel: pd.DataFrame, flow: dict | None = None, delisting_name: np.ndarray | None = None) -> pd.DataFrame:
    flow = {} if flow is None else flow
    if not (panel['Close'] > 0).all():
        raise ValueError('面板中存在非正收盘价')
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
    cz = np.r_[0, np.cumsum(panel['Vol_raw'].to_numpy() <= 0)]
    n_zero = cz[np.arange(len(key)) + 1] - cz[lo]
    if delisting_name is None:
        delisting_name = np.zeros(code.max() + 1, dtype=bool)
    consol, flow['delist_consolidation'] = delist_consolidation(code, m, delisting_name)

    we = week_end_dates(cal)
    we_m = midx.reindex(we).to_numpy()
    is_we = np.isin(m, we_m)
    nxt = pd.Series(np.r_[we_m[1:], -1], index=we_m)
    next_m = nxt.reindex(m[is_we]).to_numpy()

    u = panel.loc[is_we, ['StockID', 'Date', 'Close', 'MarketCap', 'is_ST', 'Markettype']].copy()
    u['t_idx'] = m[is_we]
    u['first_idx'] = first_m[is_we]
    u['n_obs_in_win'] = n_obs[is_we].astype(np.int16)
    u['n_zero_vol'] = n_zero[is_we]
    u['consol'] = consol[is_we]
    u['next_m'] = next_m
    u = u[(u['next_m'] >= 0).values]
    yr = u['Date'].dt.year
    in_years = yr.isin(TRAIN_YEARS) | yr.isin(VAL_YEARS) | yr.isin(OOS_YEARS)
    flow['signal_rows_outside_split_years'] = int((~in_years).sum())
    u = u[in_years.values]
    flow['signal_rows_with_next_signal_date'] = len(u)
    conds = [('st_on_t', u['is_ST'] != 0),
             ('delist_consolidation_on_t', u['consol']),
             ('ipo_within_39_market_days', u['first_idx'] > u['t_idx'] - (WINDOW + MA_LOOKBACK)),
             ('zero_volume_row_in_window', u['n_zero_vol'] > 0)]
    drop = np.zeros(len(u), dtype=bool)
    for name, c in conds:
        c = c.to_numpy()
        flow[f'drop_{name}'] = int((c & ~drop).sum())
        drop |= c
    u = u[~drop].copy()

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
    u['board'] = board_of(u['Markettype'])
    u['is_bj'] = (u['board'] == 'bse').astype(np.int8)
    if (u['split'] == 'none').any():
        raise RuntimeError('split 未覆盖的年份')
    flow['universe_rows'] = len(u)
    cols = ['StockID', 'ending_date', 't_idx', 'year', 'split', 'embargo', 'next_date', 'ret_label', 'label',
            'n_obs_in_win', 'board', 'is_bj', 'MarketCap', 'Close']
    return u[cols].sort_values(['ending_date', 'StockID']).reset_index(drop=True)


def main(argv=None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    t0 = time.time()
    panel = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Close', 'MarketCap', 'is_ST', 'Vol_raw',
                                                        'Markettype'])
    info = pd.read_feather(ASHARE_STOCK_INFO_PATH, columns=['StockID', 'name'])
    names = info.set_index(info['StockID'].astype(str))['name'].reindex(panel['StockID'].cat.categories.astype(str))
    flow = {}
    u = build(panel, flow, names.fillna('').str.contains('退').to_numpy())
    del panel
    u.to_feather(ASHARE_UNIVERSE_PATH)
    chk = {
        'flow': flow,
        'rows': len(u),
        'last_signal_date': str(u['ending_date'].max().date()),
        'board_by_split': u.groupby(['split', 'board']).size().unstack(fill_value=0).to_dict('index'),
        'board_vs_code_prefix_mismatch': board_prefix_mismatch(u),
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
    Path(ASHARE_PROCESSED_DATA_DIR, 'universe_check.json').write_text(
        json.dumps(chk, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps(chk, indent=2, ensure_ascii=False, default=str))
    print(f'完成，用时 {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
