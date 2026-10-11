# -*- coding: utf-8 -*-
"""
1_preprocess_ashare_data.py - A 股统一日面板（CSMAR 版，clean_spec 第 9 节“数据源更换为 CSMAR”）

读取 CSMAR_ROOT 下的日个股回报率文件（TRDNEW_Dalyr，多个分段 zip）与公司文件（TRDNEW_Co），输出：
  WORK_SPACE/data/ashare_processed_data/ashare_ret.feather        统一日面板（只含交易行，含 ST 行与北交所）
  WORK_SPACE/data/ashare_processed_data/ashare_stock_info.feather  静态信息（上市/退市日期、行业、板块）
  WORK_SPACE/data/ashare_processed_data/suspension_days.parquet    停牌日（Filling = 1）的 (StockID, Date)
  WORK_SPACE/data/ashare_processed_data/panel_flow.csv             每一步剔除的行数
  WORK_SPACE/data/ashare_processed_data/panel_check.json           构建时的计数与断言结果

日面板列：
  Date, StockID(category, 6 位代码)
  Open, High, Low, Close        后复权价格 = 原始价 × adj_factor
  Close_raw                     原始收盘价 Clsprc
  adj_factor, adj_factor_nd     Adjprcwd / Clsprc（含现金红利）、Adjprcnd / Clsprc（不含现金红利）
  Vol                           Dnshrtrd / adj_factor_nd（上市时股本口径，只按送转、配股等股本变动复权）
  Vol_raw, Amount               Dnshrtrd（股）、Dnvaltrd（元）
  FloatMV, MarketCap            Dsmvosd × 1000（元）；MarketCap = FloatMV
  TotalMV                       Dsmvtl3 × 1000（含 B 股、H 股的公司总市值，用于估值比率）
  TotalMV_A                     Dsmvtll × 1000（A 股总市值，含非流通 A 股；CH 因子的规模与权重）
  FloatShares, TotalShares      FloatMV / Clsprc、Dsmvtl2 × 1000 / Clsprc
  Turnover                      Dnshrtrd / FloatShares（小数）
  is_ST                         Trdsta ∈ {2,3,5,6,8,9,11,12,14,15,16}
  Ret, Ret_nd                   Dretwd、Dretnd；每只股票第一行为 NaN；不裁剪
  MA20                          最近 20 个有效后复权收盘价的均值（按行滚动，min_periods=20）
  EWMA_vol                      Ret² 的 EWMA（alpha=0.05），滞后一期；只作参考，不用于样本筛选
  PreClose, LimitUp, LimitDown, LimitStatus, Markettype, Trdsta   CSMAR 原字段

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
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (  # noqa: E402
    ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, ASHARE_STOCK_INFO_PATH, CSMAR_ROOT, SAMPLE_END,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('preprocess')

DAILY_ZIP_GLOB = '日个股回报率文件*.zip'
COMPANY_ZIP_GLOB = '公司文件*.zip'
B_SHARE_MARKETS = (2, 8)
ST_CODES = (2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16)
KEY = ['Stkcd', 'Trddt']

F64 = ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd', 'Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd',
       'PreClosePrice', 'LimitDown', 'LimitUp']
F32 = ['Dnvaltrd', 'Dsmvosd', 'Dsmvtll', 'Dsmvtl2', 'Dsmvtl3', 'LimitStatus']
COL_TYPES = {'Stkcd': pa.int32(), 'Trddt': pa.date32(), 'Markettype': pa.int8(), 'Trdsta': pa.int8(),
             'Filling': pa.int8(), **{c: pa.float64() for c in F64}, **{c: pa.float32() for c in F32}}
MA_LAG = 20
# Same legal entity renamed with a new code (old code, new code, old last trading day, new first trading day);
# clean_spec 第 9 节“构建中发现的两类缺失”及“审计发现的问题与处理”
CODE_CHANGES = [(300114, 302132, '2025-02-14', '2025-02-17'), (22, 1872, '2018-12-20', '2018-12-26'),
                (43, 1914, '2019-12-12', '2019-12-16'), (601313, 601360, '2018-02-14', '2018-02-28'),
                (600849, 601607, '2010-02-03', '2010-03-09')]
FIRST_RET_TOL = 1e-3
# (code, listing date of the current company); rows before it belong to an absorbed predecessor listed under the same code
CODE_REUSE = [(600018, '2006-10-26')]
LISTDT_LAG_TOL = 5
RETURN_FIELDS = ['Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd', 'PreClosePrice']
MAX_ALL_RETURN_FIELDS_MISSING = 10


def read_daily_zip(path: Path, end: pd.Timestamp, flow: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one segment zip; return (trading rows, suspension keys). Updates the row-flow counters."""
    trade, susp = [], []
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.startswith('TRDNEW_Dalyr') and n.endswith('.csv'))
        if not names:
            raise RuntimeError(f'{path.name}: no TRDNEW_Dalyr csv')
        for n in names:
            with z.open(n) as fh:
                t = pacsv.read_csv(fh, convert_options=pacsv.ConvertOptions(
                    column_types=COL_TYPES, include_columns=list(COL_TYPES)))
            d = t.to_pandas()
            d['Trddt'] = pd.to_datetime(d['Trddt'])
            flow['raw'] += len(d)
            m = d['Markettype'].isin(B_SHARE_MARKETS)
            flow['drop_b_share'] += int(m.sum())
            d = d[~m]
            m = d['Trddt'] > end
            flow['drop_after_sample_end'] += int(m.sum())
            d = d[~m]
            m = d['Filling'] == 2
            flow['drop_filling2_holiday'] += int(m.sum())
            d = d[~m]
            m = d['Filling'] == 1
            flow['drop_filling1_suspension'] += int(m.sum())
            susp.append(d.loc[m, KEY])
            d = d[~m]
            bad = ~d['Filling'].eq(0)
            if bad.any():
                raise ValueError(f'{path.name}/{n}: unexpected Filling values {d.loc[bad, "Filling"].unique()}')
            trade.append(d)
    return pd.concat(trade, ignore_index=True), pd.concat(susp, ignore_index=True)


def dedup(d: pd.DataFrame, what: str, chk: dict) -> pd.DataFrame:
    """Drop exact duplicate rows; raise if the same key carries different content."""
    n0 = len(d)
    d = d.drop_duplicates()
    chk[f'{what}_exact_duplicates_dropped'] = n0 - len(d)
    dup = d.duplicated(KEY, keep=False)
    if dup.any():
        sample = d.loc[dup].sort_values(KEY).head(10).to_string()
        raise ValueError(f'{what}: {int(dup.sum())} rows share a key with different content\n{sample}')
    return d


def drop_rows_without_return_fields(d: pd.DataFrame, flow: dict, chk: dict) -> pd.DataFrame:
    m = d[RETURN_FIELDS].isna().all(axis=1)
    flow['drop_all_return_fields_missing'] = int(m.sum())
    chk['all_return_fields_missing_rows'] = d.loc[m, KEY].astype(str).values.tolist()
    if m.sum() > MAX_ALL_RETURN_FIELDS_MISSING:
        raise ValueError(f'{int(m.sum())} trading rows miss all return fields (limit {MAX_ALL_RETURN_FIELDS_MISSING})')
    return d[~m]


def apply_code_changes(d: pd.DataFrame, chk: dict) -> pd.DataFrame:
    """Merge a renamed security into one series under the new code.

    First-day return of the new code: CSMAR's Dretwd/Dretnd if it equals close over the old code's last close;
    otherwise (CSMAR treated the new code as an IPO) close over PreClosePrice, allowed only when PreClosePrice equals
    the old last close. Comparable prices of the new code: chained from the old code's last comparable price with
    the daily returns when missing; otherwise rescaled by a constant so that the first day continues the old series."""
    log = []
    for old, new, d0, d1 in CODE_CHANGES:
        o, n = d['Stkcd'] == old, d['Stkcd'] == new
        if o.any() and not n.any() and d['Trddt'].max() < pd.Timestamp(d1):
            log.append({'old': f'{old:06d}', 'new': f'{new:06d}', 'skipped': 'sample ends before the new code starts'})
            continue
        if not o.any() or not n.any():
            raise ValueError(f'code change {old}->{new}: one side missing')
        last_o, first_n = d.loc[o, 'Trddt'].max(), d.loc[n, 'Trddt'].min()
        if last_o != pd.Timestamp(d0) or first_n != pd.Timestamp(d1):
            raise ValueError(f'code change {old}->{new}: dates {last_o.date()} / {first_n.date()} != {d0} / {d1}')
        row_o = d.loc[o & (d['Trddt'] == last_o)].iloc[0]
        idx_n = d.loc[n].sort_values('Trddt').index
        i0 = idx_n[0]
        if d.loc[idx_n, ['Dretwd', 'Dretnd']].isna().any().any():
            raise ValueError(f'code change {old}->{new}: new code has missing daily returns')
        r_cc = d.at[i0, 'Clsprc'] / row_o['Clsprc'] - 1
        entry = {'old': f'{old:06d}', 'new': f'{new:06d}', 'old_last_day': d0, 'new_first_day': d1,
                 'old_rows': int(o.sum()), 'new_rows': len(idx_n), 'csmar_first_dretwd': float(d.at[i0, 'Dretwd'])}
        for ret in ('Dretwd', 'Dretnd'):
            if abs(d.at[i0, ret] - r_cc) > FIRST_RET_TOL:
                if abs(d.at[i0, 'PreClosePrice'] - row_o['Clsprc']) >= 0.005:
                    raise ValueError(f'code change {old}->{new}: first-day {ret} {d.at[i0, ret]} != close/old close '
                                     f'{r_cc} and PreClose != old close; needs manual review')
                d.at[i0, ret] = d.at[i0, 'Clsprc'] / d.at[i0, 'PreClosePrice'] - 1
                entry[f'first_{ret}_overridden'] = float(d.at[i0, ret])
        for adj, ret in (('Adjprcwd', 'Dretwd'), ('Adjprcnd', 'Dretnd')):
            a = d.loc[idx_n, adj]
            if a.isna().all():
                d.loc[idx_n, adj] = row_o[adj] * np.cumprod(1.0 + d.loc[idx_n, ret].to_numpy())
                entry[f'{adj}_rule'] = 'chained'
            elif a.notna().all():
                c = row_o[adj] * (1.0 + d.at[i0, ret]) / a.iloc[0]
                d.loc[idx_n, adj] = a * c
                entry[f'{adj}_rule'] = f'rescaled x{c:.6f}'
            else:
                raise ValueError(f'code change {old}->{new}: {adj} partially missing')
        d.loc[o, 'Stkcd'] = new
        log.append(entry)
    chk['code_changes'] = log
    return d


def read_company(root: Path) -> pd.DataFrame:
    path = sorted(root.glob(COMPANY_ZIP_GLOB))
    if len(path) != 1:
        raise RuntimeError(f'expected one company zip, found {path}')
    with zipfile.ZipFile(path[0]) as z:
        co = pd.read_csv(z.open('TRDNEW_Co.csv'), dtype=str)
    if co['Stkcd'].duplicated().any():
        raise ValueError('TRDNEW_Co has duplicate Stkcd')
    return co


def drop_code_reuse_predecessors(d: pd.DataFrame, s: pd.DataFrame, co: pd.DataFrame, flow: dict,
                                 chk: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A code whose company-file listing date is more than LISTDT_LAG_TOL market days after its first trading row
    was reused by a new listing; the earlier rows belong to the predecessor and are dropped. Detected cases must
    equal CODE_REUSE."""
    cal = np.sort(d.loc[d['Markettype'].isin([1, 4]), 'Trddt'].unique())
    first = d.groupby('Stkcd')['Trddt'].min()
    listdt = pd.to_datetime(co.set_index(co['Stkcd'].astype(int))['Listdt'], errors='coerce').reindex(first.index)
    targets = {new for _, new, _, _ in CODE_CHANGES}
    ok = listdt.notna() & ~first.index.isin(list(targets))
    lag = np.searchsorted(cal, first[ok].to_numpy()) - np.searchsorted(cal, listdt[ok].to_numpy())
    found = {(int(c), str(listdt[c].date())) for c in first[ok].index[lag < -LISTDT_LAG_TOL]}
    if found != {(c, dt) for c, dt in CODE_REUSE}:
        raise ValueError(f'code reuse detected {sorted(found)} != registered {CODE_REUSE}')
    log = []
    for code, dt in CODE_REUSE:
        m = (d['Stkcd'] == code) & (d['Trddt'] < pd.Timestamp(dt))
        ms = (s['Stkcd'] == code) & (s['Trddt'] < pd.Timestamp(dt))
        log.append({'code': f'{code:06d}', 'listdt': dt, 'trading_rows_dropped': int(m.sum()),
                    'suspension_rows_dropped': int(ms.sum()),
                    'predecessor_span': [str(d.loc[m, 'Trddt'].min().date()), str(d.loc[m, 'Trddt'].max().date())]})
        d, s = d[~m], s[~ms]
    flow['drop_code_reuse_predecessor'] = sum(x['trading_rows_dropped'] for x in log)
    chk['code_reuse'] = log
    return d, s


def build_panel(d: pd.DataFrame, chk: dict) -> pd.DataFrame:
    for c in ['Clsprc', 'Adjprcwd', 'Adjprcnd']:
        bad = ~(d[c] > 0)
        chk[f'nonpositive_or_missing_{c}'] = int(bad.sum())
        if bad.any():
            raise ValueError(f'{int(bad.sum())} trading rows with non-positive or missing {c}:\n'
                             f'{d.loc[bad, KEY + [c]].head(10).to_string()}')
    order = np.lexsort((d['Trddt'].to_numpy(), d['Stkcd'].to_numpy()))
    d = d.reset_index(drop=True)
    for col in list(d.columns):
        d[col] = d[col].to_numpy()[order]
    del order

    o, h, l, c = (d[x].to_numpy() for x in ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc'])
    chk['ohlc_inconsistent_rows'] = int(((l > np.minimum(o, c) + 1e-9) | (h < np.maximum(o, c) - 1e-9)
                                         | (o <= 0) | (h <= 0) | (l <= 0)).sum())
    chk['zero_volume_trading_rows'] = int((d['Dnshrtrd'] <= 0).sum())
    del o, h, l, c

    # in-place construction (memory): new columns are added, consumed raw columns dropped
    d.rename(columns={'Trddt': 'Date', 'Stkcd': 'StockID', 'Clsprc': 'Close_raw', 'Dnshrtrd': 'Vol_raw',
                      'Dnvaltrd': 'Amount', 'PreClosePrice': 'PreClose', 'Dretwd': 'Ret', 'Dretnd': 'Ret_nd'},
             inplace=True)
    out = d
    del d
    out['adj_factor'] = out['Adjprcwd'] / out['Close_raw']
    out['adj_factor_nd'] = out['Adjprcnd'] / out['Close_raw']
    out['Open'] = out.pop('Opnprc') * out['adj_factor']
    out['High'] = out.pop('Hiprc') * out['adj_factor']
    out['Low'] = out.pop('Loprc') * out['adj_factor']
    out['Close'] = out.pop('Adjprcwd')
    out.pop('Adjprcnd')
    out['Vol'] = out['Vol_raw'] / out['adj_factor_nd']
    out['FloatMV'] = out.pop('Dsmvosd').astype('float64') * 1000.0
    out['MarketCap'] = out['FloatMV']
    out['TotalMV'] = out.pop('Dsmvtl3').astype('float64') * 1000.0
    out['TotalMV_A'] = out.pop('Dsmvtll').astype('float64') * 1000.0
    out['FloatShares'] = out['FloatMV'] / out['Close_raw']
    out['TotalShares'] = out.pop('Dsmvtl2').astype('float64') * 1000.0 / out['Close_raw']
    out['Turnover'] = (out['Vol_raw'] / out['FloatShares']).astype('float32')
    out['is_ST'] = out['Trdsta'].isin(ST_CODES).astype('int8')
    out.pop('Filling')
    first = out['StockID'].ne(out['StockID'].shift(1))
    out['Ret'] = out['Ret'].mask(first)
    out['Ret_nd'] = out['Ret_nd'].mask(first)
    for col in ['Ret', 'FloatMV', 'TotalMV', 'TotalMV_A', 'TotalShares']:
        chk[f'missing_{col}'] = int(out[col].isna().sum()) - (int(first.sum()) if col == 'Ret' else 0)

    g = out.groupby('StockID', sort=False)
    out['MA20'] = g['Close'].transform(lambda s: s.rolling(MA_LAG, min_periods=MA_LAG).mean())
    r2 = out['Ret'].fillna(0.0) ** 2
    out['EWMA_vol'] = r2.groupby(out['StockID'], sort=False).transform(
        lambda x: x.ewm(alpha=0.05, adjust=False).mean().shift(1)).astype('float32')
    codes = np.unique(out['StockID'].to_numpy())
    out['StockID'] = pd.Categorical.from_codes(np.searchsorted(codes, out['StockID'].to_numpy()),
                                               categories=[f'{x:06d}' for x in codes])
    return out


PANEL_COLS = ['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Close_raw', 'adj_factor', 'adj_factor_nd',
              'Vol', 'Vol_raw', 'Amount', 'Turnover', 'FloatShares', 'TotalShares', 'FloatMV', 'MarketCap',
              'TotalMV', 'TotalMV_A', 'is_ST', 'Ret', 'Ret_nd', 'MA20', 'EWMA_vol', 'PreClose', 'LimitUp', 'LimitDown',
              'LimitStatus', 'Markettype', 'Trdsta']


def write_feather_consuming(df: pd.DataFrame, cols: list[str], path: str) -> None:
    """Convert column by column, dropping each pandas column once converted, so peak memory stays ~1x."""
    import pyarrow.feather as feather
    if set(df.columns) != set(cols):
        raise ValueError(f'column mismatch: {set(df.columns) ^ set(cols)}')
    arrays = []
    for c in cols:
        arrays.append(pa.Array.from_pandas(df.pop(c)))
    feather.write_feather(pa.Table.from_arrays(arrays, names=cols), path)


def build_stock_info(co: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    co = co.rename(columns={'Stkcd': 'StockID', 'Stknme': 'name', 'Listdt': 'list_date',
                            'DelistedDate': 'delist_date', 'Nnindcd': 'industry', 'Nnindnme': 'industry_name',
                            'Markettype': 'markettype_co'})
    co['StockID'] = co['StockID'].str.zfill(6)
    for c in ['list_date', 'delist_date']:
        co[c] = pd.to_datetime(co[c], errors='coerce')
    span = panel.groupby('StockID', observed=True)['Date'].agg(first_trade='min', last_trade='max', n_rows='size')
    span = span.reset_index()
    span['StockID'] = span['StockID'].astype(str)
    info = span.merge(co[['StockID', 'name', 'list_date', 'delist_date', 'industry', 'industry_name',
                          'markettype_co']], on='StockID', how='left', validate='one_to_one')
    last_mt = panel.groupby('StockID', observed=True)['Markettype'].last().rename('markettype_last').reset_index()
    last_mt['StockID'] = last_mt['StockID'].astype(str)
    info = info.merge(last_mt, on='StockID', how='left', validate='one_to_one')
    info['in_company_file'] = info['name'].notna()
    info['code_history'] = ''
    for old, new, _, _ in CODE_CHANGES:
        old_s, new_s = f'{old:06d}', f'{new:06d}'
        old_list = co.loc[co['StockID'] == old_s, 'list_date']
        if len(old_list) != 1:
            raise ValueError(f'code change {old_s}->{new_s}: old code not unique in TRDNEW_Co')
        k = info['StockID'] == new_s
        info.loc[k, 'list_date'] = old_list.iloc[0]
        info.loc[k, 'code_history'] = f'{old_s}->{new_s}'
    for c in ['industry', 'industry_name']:
        info[c] = info[c].astype('category')
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    if os.path.exists(ASHARE_PANEL_PATH) and not args.force:
        logger.info(f'已存在: {ASHARE_PANEL_PATH}（--force 重建）')
        return 0

    t0 = time.time()
    root = Path(CSMAR_ROOT)
    end = pd.Timestamp(SAMPLE_END)
    zips = sorted(root.glob(DAILY_ZIP_GLOB))
    logger.info(f'{len(zips)} 个日行情分段: {[z.name for z in zips]}')
    flow = dict.fromkeys(['raw', 'drop_b_share', 'drop_after_sample_end', 'drop_filling2_holiday',
                          'drop_filling1_suspension'], 0)
    chk: dict = {'zips': [z.name for z in zips], 'sample_end': SAMPLE_END}
    trade, susp = [], []
    for zp in zips:
        a, b = read_daily_zip(zp, end, flow)
        logger.info(f'  {zp.name}: 交易行 {len(a):,}，停牌行 {len(b):,}')
        trade.append(a)
        susp.append(b)
    d = pd.concat(trade, ignore_index=True)
    del trade
    s = pd.concat(susp, ignore_index=True)
    d = dedup(d, 'trading', chk)
    s = s.drop_duplicates()
    overlap = d[KEY].merge(s, on=KEY, how='inner')
    chk['keys_both_trading_and_suspension'] = len(overlap)
    if len(overlap):
        raise ValueError(f'{len(overlap)} keys are both trading and suspension rows:\n{overlap.head(10)}')
    flow['trading_rows_after_dedup'] = len(d)
    d = drop_rows_without_return_fields(d, flow, chk)
    d = apply_code_changes(d, chk)
    co = read_company(root)
    d, s = drop_code_reuse_predecessors(d, s, co, flow, chk)
    flow['trading_rows_final'] = len(d)
    logger.info(f'交易行 {len(d):,}，{d["Stkcd"].nunique()} 只，用时 {time.time() - t0:.0f}s')

    panel = build_panel(d, chk)
    del d
    logger.info(f'面板构建完成，用时 {time.time() - t0:.0f}s')
    os.makedirs(ASHARE_PROCESSED_DATA_DIR, exist_ok=True)

    for c in chk['code_changes']:
        if 'skipped' not in c:
            s.loc[s['Stkcd'] == int(c['old']), 'Stkcd'] = int(c['new'])
    s = s.rename(columns={'Stkcd': 'StockID', 'Trddt': 'Date'})
    s['StockID'] = s['StockID'].map(lambda x: f'{x:06d}')
    s.sort_values(['StockID', 'Date']).to_parquet(
        os.path.join(ASHARE_PROCESSED_DATA_DIR, 'suspension_days.parquet'), index=False)

    info = build_stock_info(co, panel)
    info.to_feather(ASHARE_STOCK_INFO_PATH)

    pd.Series(flow, name='rows').rename_axis('step').to_csv(
        os.path.join(ASHARE_PROCESSED_DATA_DIR, 'panel_flow.csv'))
    chk.update({
        'flow': flow, 'rows': len(panel), 'stocks': int(panel['StockID'].nunique()),
        'stocks_by_last_markettype': info['markettype_last'].value_counts().sort_index().to_dict(),
        'stocks_not_in_company_file': info.loc[~info['in_company_file'], 'StockID'].tolist(),
        'date_min': str(panel['Date'].min().date()), 'date_max': str(panel['Date'].max().date()),
        'n_market_days': int(panel['Date'].nunique()),
        'st_row_share': float(panel['is_ST'].mean()),
        'ret_abs_max': float(panel['Ret'].abs().max()),
        'ret_share_abs_gt_0.11': float((panel['Ret'].abs() > 0.11).mean()),
    })
    n_rows = len(panel)
    write_feather_consuming(panel, PANEL_COLS, ASHARE_PANEL_PATH)
    del panel
    logger.info(f'写出 {ASHARE_PANEL_PATH}（{n_rows:,} 行）')
    Path(ASHARE_PROCESSED_DATA_DIR, 'panel_check.json').write_text(
        json.dumps(chk, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    logger.info(json.dumps(chk, indent=2, ensure_ascii=False, default=str))
    logger.info(f'完成，用时 {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
