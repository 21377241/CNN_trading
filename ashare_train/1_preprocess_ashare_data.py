# -*- coding: utf-8 -*-
"""
1_preprocess_ashare_data.py - A 股全市场统一日面板（clean_spec 第 2 节 D1、D2、D5、D8）

读取每只股票的后复权 CSV 与不复权 CSV，输出：
  WORK_SPACE/data/ashare_processed_data/ashare_ret.feather        统一日面板（含 ST 行）
  WORK_SPACE/data/ashare_processed_data/ashare_stock_info.feather  静态信息（行业、上市/退市日期）
  WORK_SPACE/data/ashare_processed_data/panel_check.json           验收检查

日面板列：
  Date, StockID(category)
  Open, High, Low, Close        后复权价格（float64）
  Close_raw, adj_factor         不复权收盘价；复权因子 = Close / Close_raw
  Vol                           送转复权后的成交量 = Vol_raw / adj_factor（上市时股本口径）
  Vol_raw                       原始“成交量（股）”
  Amount, Turnover, pct_chg, FloatShares, TotalShares, FloatMV, MarketCap(=FloatMV), PE_TTM, PB
  is_ST, is_limit_up            原始逐日标记
  Ret                           Close / 前一行 Close - 1，不裁剪；每只股票第一行为 NaN
  MA20                          最近 20 个有效收盘价的均值（按行滚动，min_periods=20）
  EWMA_vol                      Ret² 的 EWMA（alpha=0.05），滞后一期；只作参考，不用于样本筛选

用法:
  python ashare_train/1_preprocess_ashare_data.py [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (  # noqa: E402
    ASHARE_RAW_ADJ_DIR, ASHARE_RAW_UNADJ_DIR, ASHARE_PROCESSED_DATA_DIR,
    ASHARE_PANEL_PATH, ASHARE_STOCK_INFO_PATH, PRE_CLEAN_WORK_DIR,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('preprocess')

DAILY = {
    '日期': 'Date', '开盘价': 'Open', '最高价': 'High', '最低价': 'Low', '收盘价': 'Close',
    '成交量（股）': 'Vol_raw', '成交额（元）': 'Amount', '换手率': 'Turnover', '涨幅%': 'pct_chg',
    '流通股本（股）': 'FloatShares', '总股本（股）': 'TotalShares', '流通市值（元）': 'FloatMV',
    '滚动市盈率': 'PE_TTM', '市净率': 'PB', '是否ST': 'is_ST', '是否涨停': 'is_limit_up',
}
STATIC = {'代码': 'code', '名称': 'name', '所属行业': 'industry', '上市时间': 'list_date', '退市时间': 'delist_date'}
PRICE64 = ['Open', 'High', 'Low', 'Close', 'Vol_raw', 'Amount', 'FloatShares', 'TotalShares', 'FloatMV']
FLOAT32 = ['Turnover', 'pct_chg', 'PE_TTM', 'PB']
MA_LAG = 20


def read_one(adj_path: Path, unadj_dir: Path) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(adj_path, encoding='utf-8-sig', usecols=list(DAILY) + list(STATIC))
    sid = adj_path.stem.zfill(6)
    info = {
        'StockID': sid,
        'name': d['名称'].iloc[-1] if len(d) else None,
        'industry': d['所属行业'].mode().iloc[0] if d['所属行业'].notna().any() else None,
        'n_industry': int(d['所属行业'].nunique()),
        'list_date': pd.to_datetime(d['上市时间'], errors='coerce', format='mixed').min(),
        'delist_date': pd.to_datetime(d['退市时间'], errors='coerce', format='mixed').min(),
        'rows_raw': len(d),
    }
    d = d[list(DAILY)].rename(columns=DAILY)
    d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
    for c in PRICE64:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype('float64')
    for c in FLOAT32:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
    d['is_ST'] = (d['is_ST'].astype(str).str.strip() == '是').astype('int8')
    d['is_limit_up'] = (d['is_limit_up'].astype(str).str.strip() == '是').astype('int8')
    d = d.dropna(subset=['Date', 'Close'])
    d = d[(d['Close'] > 0) & (d['Open'] > 0)]
    d = d.sort_values('Date').drop_duplicates('Date', keep='last')

    unadj_path = unadj_dir / adj_path.name
    info['has_unadj'] = unadj_path.exists()
    if info['has_unadj']:
        u = pd.read_csv(unadj_path, encoding='utf-8-sig', usecols=['日期', '收盘价'])
        u = u.rename(columns={'日期': 'Date', '收盘价': 'Close_raw'})
        u['Date'] = pd.to_datetime(u['Date'], errors='coerce')
        u['Close_raw'] = pd.to_numeric(u['Close_raw'], errors='coerce')
        u = u.dropna().drop_duplicates('Date', keep='last')
        u = u[u['Close_raw'] > 0]
        d = d.merge(u, on='Date', how='left')
    else:
        d['Close_raw'] = np.nan
    fac = d['Close'] / d['Close_raw']
    info['adj_factor_missing_rows'] = int(fac.isna().sum())
    fac = fac.ffill().bfill()
    d['adj_factor'] = fac.astype('float64')
    d['Close_raw'] = d['Close_raw'].astype('float32')
    vol_adj = d['Vol_raw'].astype('float64') / d['adj_factor']
    d['Vol'] = vol_adj.where(d['adj_factor'].notna(), d['Vol_raw'].astype('float64'))
    d.insert(1, 'StockID', sid)
    info['rows'] = len(d)
    return d.reset_index(drop=True), info


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby('StockID', sort=False, observed=True)
    prev = g['Close'].shift(1)
    panel['Ret'] = panel['Close'] / prev - 1.0
    panel['MA20'] = g['Close'].transform(lambda s: s.rolling(MA_LAG, min_periods=MA_LAG).mean())
    r2 = panel['Ret'].fillna(0.0) ** 2
    panel['EWMA_vol'] = r2.groupby(panel['StockID'], sort=False, observed=True).transform(
        lambda x: x.ewm(alpha=0.05, adjust=False).mean().shift(1))
    panel['MarketCap'] = panel['FloatMV'].astype('float64')
    return panel


def split_volume_check(panel: pd.DataFrame, rng_seed: int = 0, n_stocks: int = 400) -> dict:
    """送转除权（复权因子跳升 > 5%）前后 5 日成交量均值之比的中位数。"""
    rng = np.random.default_rng(rng_seed)
    ids = rng.choice(panel['StockID'].cat.categories, n_stocks, replace=False)
    sub = panel[panel['StockID'].isin(ids)]
    ratios = {'Vol': [], 'Vol_raw': [], 'Turnover': []}
    n_events = 0
    for _, s in sub.groupby('StockID', sort=False, observed=True):
        f = s['adj_factor'].to_numpy()
        jump = np.flatnonzero(f[1:] / f[:-1] - 1 > 0.05) + 1
        for i in jump:
            if i < 5 or i + 5 > len(s):
                continue
            n_events += 1
            for c in ratios:
                v = s[c].to_numpy(dtype=float)
                pre, post = v[i - 5:i].mean(), v[i:i + 5].mean()
                if pre > 0:
                    ratios[c].append(post / pre)
    return {'n_events': n_events, **{f'median_post_pre_{c}': float(np.median(v)) for c, v in ratios.items() if v}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=8)
    args = ap.parse_args(argv)
    if os.path.exists(ASHARE_PANEL_PATH) and not args.force:
        logger.info(f'已存在: {ASHARE_PANEL_PATH}（--force 重建）')
        return 0

    t0 = time.time()
    adj_dir, unadj_dir = Path(ASHARE_RAW_ADJ_DIR), Path(ASHARE_RAW_UNADJ_DIR)
    files = sorted(f for f in adj_dir.glob('*.csv') if not f.stem.endswith('(1)'))
    logger.info(f'后复权文件 {len(files)} 个；不复权目录 {unadj_dir}')
    res = Parallel(n_jobs=args.n_jobs, batch_size=16)(delayed(read_one)(f, unadj_dir) for f in files)
    info = pd.DataFrame([r[1] for r in res])
    panel = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    del res
    panel['StockID'] = panel['StockID'].astype('category')
    panel = panel.sort_values(['StockID', 'Date'], kind='stable').reset_index(drop=True)
    logger.info(f'读入 {len(panel):,} 行，{panel["StockID"].nunique()} 只，用时 {time.time() - t0:.0f}s')

    panel = add_derived(panel)
    cols = ['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Close_raw', 'adj_factor', 'Vol', 'Vol_raw',
            'Amount', 'Turnover', 'pct_chg', 'FloatShares', 'TotalShares', 'FloatMV', 'MarketCap', 'PE_TTM', 'PB',
            'is_ST', 'is_limit_up', 'Ret', 'MA20', 'EWMA_vol']
    panel = panel[cols]
    os.makedirs(ASHARE_PROCESSED_DATA_DIR, exist_ok=True)
    panel.to_feather(ASHARE_PANEL_PATH)
    info['industry'] = info['industry'].astype('category')
    info.to_feather(ASHARE_STOCK_INFO_PATH)
    logger.info(f'写出 {ASHARE_PANEL_PATH}')

    chk: dict = {
        'rows': len(panel), 'stocks': int(panel['StockID'].nunique()),
        'date_min': str(panel['Date'].min().date()), 'date_max': str(panel['Date'].max().date()),
        'n_market_days': int(panel['Date'].nunique()),
        'st_row_share': float(panel['is_ST'].mean()),
        'ret_abs_max': float(panel['Ret'].abs().max()),
        'ret_share_abs_gt_0.11': float((panel['Ret'].abs() > 0.11).mean()),
        'ret_top10_abs': panel.loc[panel['Ret'].abs().nlargest(10).index, ['Date', 'StockID', 'Ret']]
                              .astype(str).values.tolist(),
        'stocks_empty': info.loc[info['rows'] == 0, 'StockID'].tolist(),
        'stocks_without_unadj_file': int((~info['has_unadj']).sum()),
        'adj_factor_missing_rows_before_fill': int(info['adj_factor_missing_rows'].sum()),
        'adj_factor_nan_after_fill': int(panel['adj_factor'].isna().sum()),
        'split_volume_check': split_volume_check(panel),
    }
    old_panel = os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_processed_data', 'ashare_ret.feather')
    if os.path.exists(old_panel):
        old = pd.read_feather(old_panel, columns=['Date', 'StockID', 'Close', 'Vol'])
        old['StockID'] = old['StockID'].astype(str).str.zfill(6)
        new = panel.loc[panel['is_ST'] == 0, ['Date', 'StockID', 'Close', 'Vol_raw']].copy()
        new['StockID'] = new['StockID'].astype(str)
        m = old.merge(new, on=['Date', 'StockID'], how='outer', indicator=True, suffixes=('_old', '_new'))
        both = m['_merge'] == 'both'
        chk['vs_pre_clean_panel_nonST'] = {
            'rows_old': len(old), 'rows_new_nonST': len(new),
            'only_old': int((m['_merge'] == 'left_only').sum()),
            'only_new': int((m['_merge'] == 'right_only').sum()),
            'close_match_share_1e-9': float((np.abs(m.loc[both, 'Close_new'] / m.loc[both, 'Close_old'] - 1)
                                             < 1e-9).mean()),
            'vol_raw_match_share': float((m.loc[both, 'Vol_raw'] == m.loc[both, 'Vol'].astype(float)).mean()),
        }
    Path(ASHARE_PROCESSED_DATA_DIR, 'panel_check.json').write_text(
        json.dumps(chk, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    logger.info(json.dumps(chk, indent=2, ensure_ascii=False, default=str))
    logger.info(f'完成，用时 {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
