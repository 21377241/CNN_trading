# -*- coding: utf-8 -*-
"""Extract the extended daily panel (all rows, including ST days) from the raw back-adjusted CSV files.

Outputs
  ashare_extended.feather   daily: prices, volume, amount, turnover, shares, float MV, PE/PB, ST / limit-up flags
  ashare_stock_info.feather static: current name, industry (current classification), listing / delisting dates

The raw `前收盘价` and moving-average fields are not used: `前收盘价` equals the day's low in the raw files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import DATA, EXT_PANEL, OUT, RAW_DIR, STOCK_INFO  # noqa: E402

DAILY = {
    '日期': 'Date', '开盘价': 'Open', '最高价': 'High', '最低价': 'Low', '收盘价': 'Close',
    '成交量（股）': 'Vol', '成交额（元）': 'Amount', '换手率': 'Turnover', '涨幅%': 'pct_chg',
    '流通股本（股）': 'FloatShares', '总股本（股）': 'TotalShares', '流通市值（元）': 'FloatMV',
    '滚动市盈率': 'PE_TTM', '市净率': 'PB', '是否ST': 'is_ST', '是否涨停': 'is_limit_up',
}
STATIC = {'代码': 'code', '名称': 'name', '所属行业': 'industry', '上市时间': 'list_date', '退市时间': 'delist_date'}
FLOATS = ['Open', 'High', 'Low', 'Close', 'Vol', 'Amount', 'Turnover', 'pct_chg', 'FloatShares', 'TotalShares',
          'FloatMV', 'PE_TTM', 'PB']


def read_one(path: Path) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(path, encoding='utf-8-sig', usecols=list(DAILY) + list(STATIC))
    sid = path.stem.zfill(6)
    info = {
        'StockID': sid,
        'name': d['名称'].iloc[-1] if len(d) else None,
        'industry': d['所属行业'].mode().iloc[0] if d['所属行业'].notna().any() else None,
        'n_industry': int(d['所属行业'].nunique()),
        'list_date': pd.to_datetime(d['上市时间'].astype(str), errors='coerce').min(),
        'delist_date': pd.to_datetime(d['退市时间'].astype(str), errors='coerce').min(),
        'rows': len(d),
    }
    d = d[list(DAILY)].rename(columns=DAILY)
    d['Date'] = pd.to_datetime(d['Date'])
    for c in FLOATS:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
    d['is_ST'] = (d['is_ST'] == '是').astype('int8')
    d['is_limit_up'] = (d['is_limit_up'] == '是').astype('int8')
    d.insert(1, 'StockID', sid)
    d = d.sort_values('Date').drop_duplicates('Date', keep='last')
    return d, info


def main() -> int:
    t0 = time.time()
    files = sorted(RAW_DIR.glob('*.csv'))
    print(f'{len(files)} raw files')
    res = Parallel(n_jobs=8, batch_size=16, verbose=0)(delayed(read_one)(f) for f in files)
    info = pd.DataFrame([r[1] for r in res])
    panel = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    del res
    panel['StockID'] = panel['StockID'].astype('category')
    panel = panel.sort_values(['StockID', 'Date'], kind='stable').reset_index(drop=True)
    print(f'read {len(panel):,} rows in {time.time() - t0:.0f}s')

    DATA.mkdir(parents=True, exist_ok=True)
    panel.to_feather(EXT_PANEL)
    info['industry'] = info['industry'].astype('category')
    info.to_feather(STOCK_INFO)

    # ---- validation ----
    chk: dict = {'rows': len(panel), 'stocks': int(panel['StockID'].nunique()),
                 'date_min': str(panel['Date'].min().date()), 'date_max': str(panel['Date'].max().date()),
                 'st_row_share': float(panel['is_ST'].mean()), 'limit_up_share': float(panel['is_limit_up'].mean()),
                 'stocks_multi_industry': int((info['n_industry'] > 1).sum()),
                 'stocks_with_delist_date': int(info['delist_date'].notna().sum()),
                 'missing_share': {c: float(panel[c].isna().mean()) for c in FLOATS}}
    proc = pd.read_feather(DATA / 'ashare_ret.feather', columns=['Date', 'StockID', 'Close'])
    proc['StockID'] = proc['StockID'].astype(str).str.zfill(6)
    rng = np.random.default_rng(0)
    sample = rng.choice(proc['StockID'].unique(), 400, replace=False)
    a = proc[proc['StockID'].isin(sample)]
    b = panel[panel['StockID'].isin(sample) & (panel['is_ST'] == 0)][['Date', 'StockID', 'Close']].copy()
    b['StockID'] = b['StockID'].astype(str)
    m = a.merge(b, on=['Date', 'StockID'], how='outer', suffixes=('_proc', '_ext'), indicator=True)
    both = m[m['_merge'] == 'both']
    chk['vs_processed_400_stocks'] = {
        'rows_proc': len(a), 'rows_ext_nonST': len(b),
        'only_in_proc': int((m['_merge'] == 'left_only').sum()), 'only_in_ext': int((m['_merge'] == 'right_only').sum()),
        'close_rel_diff_max': float((both['Close_ext'] / both['Close_proc'] - 1).abs().max()),
        'close_match_share_1e-5': float(((both['Close_ext'] / both['Close_proc'] - 1).abs() < 1e-5).mean()),
    }
    del proc, a, b, m, both
    g = panel.groupby('StockID', sort=False, observed=True)['Close']
    r = g.pct_change() * 100
    ok = r.notna() & panel['pct_chg'].notna()
    diff = (r[ok] - panel.loc[ok, 'pct_chg']).abs()
    chk['close_return_vs_pct_chg'] = {'rows': int(ok.sum()), 'share_within_0.05pp': float((diff < 0.05).mean()),
                                      'share_within_0.5pp': float((diff < 0.5).mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'extended_panel_check.json').write_text(json.dumps(chk, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(chk, indent=2, ensure_ascii=False))
    print(f'done in {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
