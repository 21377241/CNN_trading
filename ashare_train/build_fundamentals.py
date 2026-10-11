# -*- coding: utf-8 -*-
"""
build_fundamentals.py - 财务时点表（clean_spec 第 9 节“数据源更换为 CSMAR”第 4 条）

读取 CSMAR 资产负债表（FS_Combas）与利润表（FS_Comins），只用合并报表、季末会计期间，输出：
  WORK_SPACE/data/ashare_processed_data/fundamentals_pit.parquet
    StockID, Accper, avail_date          会计期末；生效日 = 法定截止日之后的第一个市场交易日
    BE, BE_source                        账面权益 = 归母权益 − 优先股 − 永续债（缺失时用 所有者权益合计 − 少数股东权益）
    NI_TTM, NI_TTM_source                归属于母公司所有者的净利润 TTM（缺失时用净利润）
    REV_TTM, REV_TTM_source              营业总收入 TTM（缺失时用营业收入）
    NI_cum, REV_cum                      本期累计值（原值）
  WORK_SPACE/data/ashare_processed_data/fundamentals_check.json

下游用法：对每个 (StockID, Date)，取 avail_date <= Date 的最近一期（按 avail_date，再按 Accper）。

用法:
  python ashare_train/build_fundamentals.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import ASHARE_FUNDAMENTALS_PATH, ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, CSMAR_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('fundamentals')

QUARTER_ENDS = {'03-31': 3, '06-30': 6, '09-30': 9, '12-31': 12}
# statutory deadline (month, day, year offset) by fiscal quarter
DEADLINE = {3: (4, 30, 0), 6: (8, 31, 0), 9: (10, 31, 0), 12: (4, 30, 1)}
CODE_CHANGES = [('300114', '302132'), ('000022', '001872'), ('000043', '001914'), ('601313', '601360'),
                ('600849', '601607')]
# code reused by a new listing (1_preprocess_ashare_data.CODE_REUSE): statements whose statutory deadline precedes
# the new listing date belong to the absorbed predecessor and are dropped before TTM differencing
CODE_REUSE = [('600018', '2006-10-26')]
BS_COLS = {'A003100000': 'eq_parent', 'A003000000': 'eq_total', 'A003200000': 'minority',
           'A003112101': 'pref', 'A003112201': 'perp', 'A001000000': 'assets', 'A002000000': 'liab'}
IS_COLS = {'B002000101': 'ni_parent', 'B002000000': 'ni_total', 'B001100000': 'rev_total', 'B001101000': 'rev_op'}


def read_statement(glob_pat: str, member: str, cols: dict) -> pd.DataFrame:
    path = sorted(Path(CSMAR_ROOT).glob(glob_pat))
    if len(path) != 1:
        raise RuntimeError(f'expected one {glob_pat}, found {path}')
    with zipfile.ZipFile(path[0]) as z:
        d = pd.read_csv(z.open(member), usecols=['Stkcd', 'Accper', 'Typrep'] + list(cols), dtype={
            'Stkcd': str, 'Accper': str, 'Typrep': str, **{c: 'float64' for c in cols}})
    d = d.rename(columns={'Stkcd': 'StockID', **cols})
    d['StockID'] = d['StockID'].str.zfill(6)
    return d


def keep_consolidated_quarters(d: pd.DataFrame, what: str, chk: dict) -> pd.DataFrame:
    n0 = len(d)
    d = d[d['Typrep'] == 'A']
    n1 = len(d)
    md = d['Accper'].str[5:]
    d = d[md.isin(list(QUARTER_ENDS))].copy()
    chk[f'{what}_rows'] = {'raw': n0, 'consolidated': n1, 'quarter_end': len(d),
                           'dropped_non_quarter_end_by_mmdd': md[~md.isin(list(QUARTER_ENDS))].value_counts().to_dict()}
    if d.duplicated(['StockID', 'Accper']).any():
        raise ValueError(f'{what}: duplicate (StockID, Accper) among consolidated quarter-end rows')
    d['Accper'] = pd.to_datetime(d['Accper'])
    return d.drop(columns='Typrep')


def merge_code_changes(d: pd.DataFrame, what: str, chk: dict) -> pd.DataFrame:
    """Records of the new code are kept; periods only reported under the old code are added with seg = 1 so that
    TTM differencing never mixes the two segments (e.g. a backdoor listing whose pre-change statements are the shell's)."""
    log = []
    d['seg'] = np.int8(0)
    for old, new in CODE_CHANGES:
        o, n = d['StockID'] == old, d['StockID'] == new
        extra = o & ~d['Accper'].isin(d.loc[n, 'Accper'])
        d.loc[extra, 'StockID'] = new
        d.loc[extra, 'seg'] = np.int8(1)
        d = d[~(d['StockID'] == old)]
        log.append({'old': old, 'new': new, 'old_rows_added_to_new': int(extra.sum()), 'old_rows_dropped': int(o.sum() - extra.sum())})
    chk[f'{what}_code_changes'] = log
    return d


def statutory_deadline(accper: pd.Series) -> pd.DatetimeIndex:
    q, y = accper.dt.month.to_numpy(), accper.dt.year.to_numpy()
    return pd.to_datetime([f'{yy + DEADLINE[m][2]}-{DEADLINE[m][0]:02d}-{DEADLINE[m][1]:02d}' for yy, m in zip(y, q)])


def drop_code_reuse_predecessors(d: pd.DataFrame, what: str, chk: dict) -> pd.DataFrame:
    log = []
    for code, listdt in CODE_REUSE:
        m = (d['StockID'] == code) & (statutory_deadline(d['Accper']) < pd.Timestamp(listdt))
        log.append({'code': code, 'listdt': listdt, 'rows_dropped': int(m.sum()),
                    'last_dropped_accper': str(d.loc[m, 'Accper'].max().date()) if m.any() else None})
        d = d[~m]
    chk[f'{what}_code_reuse'] = log
    return d


def ttm(d: pd.DataFrame, cum_col: str) -> tuple[pd.Series, pd.Series]:
    """TTM = cum(y,q) + annual(y-1) - cum(y-1,q); Q4 = annual; fallback = annual(y-1) when a piece is missing."""
    y, q = d['Accper'].dt.year, d['Accper'].dt.month
    key = d[['StockID', 'seg']].assign(y=y, q=q, v=d[cum_col])
    lut = key.set_index(['StockID', 'seg', 'y', 'q'])['v']
    annual_prev = lut.reindex(pd.MultiIndex.from_arrays([key['StockID'], key['seg'], y - 1, np.full(len(d), 12)])).to_numpy()
    same_q_prev = lut.reindex(pd.MultiIndex.from_arrays([key['StockID'], key['seg'], y - 1, q])).to_numpy()
    cur = d[cum_col].to_numpy()
    full = cur + annual_prev - same_q_prev
    out = np.where(q.to_numpy() == 12, cur, full)
    src = np.where(q.to_numpy() == 12, 'annual', 'cum+prev_annual-prev_same_q')
    miss = np.isnan(out)
    out = np.where(miss, annual_prev, out)
    src = np.where(miss & ~np.isnan(annual_prev), 'fallback_prev_annual', src)
    src = np.where(np.isnan(out), 'missing', src)
    return pd.Series(out, index=d.index), pd.Series(src, index=d.index)


def main() -> int:
    chk: dict = {}
    bs = keep_consolidated_quarters(read_statement('资产负债表*.zip', 'FS_Combas.csv', BS_COLS), 'balance_sheet', chk)
    is_ = keep_consolidated_quarters(read_statement('利润表*.zip', 'FS_Comins.csv', IS_COLS), 'income_statement', chk)
    bs = merge_code_changes(bs, 'balance_sheet', chk)
    is_ = merge_code_changes(is_, 'income_statement', chk)
    bs = drop_code_reuse_predecessors(bs, 'balance_sheet', chk)
    is_ = drop_code_reuse_predecessors(is_, 'income_statement', chk)

    be = bs['eq_parent'] - bs['pref'].fillna(0.0) - bs['perp'].fillna(0.0)
    fb = bs['eq_total'] - bs['minority'].fillna(0.0) - bs['pref'].fillna(0.0) - bs['perp'].fillna(0.0)
    bs['BE'] = be.where(bs['eq_parent'].notna(), fb)
    bs['BE_source'] = np.where(bs['eq_parent'].notna(), 'eq_parent',
                               np.where(bs['eq_total'].notna(), 'eq_total_minus_minority', 'missing'))

    is_['NI_cum'] = is_['ni_parent'].where(is_['ni_parent'].notna(), is_['ni_total'])
    is_['REV_cum'] = is_['rev_total'].where(is_['rev_total'].notna(), is_['rev_op'])
    is_['NI_cum_source'] = np.where(is_['ni_parent'].notna(), 'ni_parent', np.where(is_['ni_total'].notna(), 'ni_total', 'missing'))
    is_ = is_.sort_values(['StockID', 'Accper']).reset_index(drop=True)
    is_['NI_TTM'], is_['NI_TTM_source'] = ttm(is_, 'NI_cum')
    is_['REV_TTM'], is_['REV_TTM_source'] = ttm(is_, 'REV_cum')

    f = bs[['StockID', 'Accper', 'BE', 'BE_source', 'eq_parent', 'eq_total', 'minority', 'assets', 'liab']].merge(
        is_[['StockID', 'Accper', 'NI_cum', 'NI_cum_source', 'NI_TTM', 'NI_TTM_source', 'REV_cum', 'REV_TTM',
             'REV_TTM_source']], on=['StockID', 'Accper'], how='outer', validate='one_to_one')
    for c in ['BE_source', 'NI_cum_source', 'NI_TTM_source', 'REV_TTM_source']:
        f[c] = f[c].fillna('no_statement')

    cal = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'Markettype'])
    cal = np.sort(cal.loc[cal['Markettype'].isin([1, 4]), 'Date'].unique())
    deadline = statutory_deadline(f['Accper'])
    pos = np.searchsorted(cal, deadline.to_numpy(), side='right')
    f['deadline'] = deadline
    f['avail_date'] = pd.NaT
    ok = pos < len(cal)
    f.loc[ok, 'avail_date'] = cal[pos[ok]]
    f['avail_date'] = pd.to_datetime(f['avail_date'])
    chk['records'] = len(f)
    chk['records_not_yet_available_by_sample_end'] = f.loc[~ok, ['StockID', 'Accper']].astype(str).to_dict('records')
    f = f[ok].sort_values(['StockID', 'Accper']).reset_index(drop=True)
    if not (f['avail_date'] > f['Accper']).all():
        raise ValueError('avail_date not after Accper')
    if not (f['avail_date'] > f['deadline']).all():
        raise ValueError('avail_date not after statutory deadline')

    chk['stocks'] = int(f['StockID'].nunique())
    chk['accper_range'] = [str(f['Accper'].min().date()), str(f['Accper'].max().date())]
    for c in ['BE_source', 'NI_TTM_source', 'REV_TTM_source']:
        chk[c] = f[c].value_counts(dropna=False).to_dict()
    chk['BE_nonpositive_share'] = float((f['BE'] <= 0).mean())
    f.to_parquet(ASHARE_FUNDAMENTALS_PATH, index=False)
    Path(ASHARE_PROCESSED_DATA_DIR, 'fundamentals_check.json').write_text(
        json.dumps(chk, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    logger.info(json.dumps(chk, indent=2, ensure_ascii=False, default=str))
    logger.info(f'写出 {ASHARE_FUNDAMENTALS_PATH}（{len(f):,} 行）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
