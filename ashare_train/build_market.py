# -*- coding: utf-8 -*-
"""
build_market.py - 市场日度表（clean_spec 第 9 节“数据源更换为 CSMAR”第 5 条）

输出 WORK_SPACE/data/ashare_processed_data/market_daily.parquet，每个市场交易日一行：
  Date                      面板中沪深主板（Markettype 1/4）有成交的日期
  rf_annual                 一年期定期存款利率（%，TRDNEW_Nrrate.Nrrdata，取上一交易日的值）
  rf                        持有期 [t-1, t] 的无风险收益 = (1 + rf_annual/100)^(自然日数/365) - 1
  mkt_vw / mkt_ew / mkt_tv  Cndalym Markettype 53（沪深 A + 创业板 + 科创板）的流通市值加权 / 等权 / 总市值加权收益（含红利再投资）
  mkt_bj_vw / mkt_bj_ew     Markettype 117（另含北交所），北交所稳健性用
  mkt_n                     Markettype 53 的有效公司数
  idx_<code>_close / idx_<code>_ret   指数收盘与日收益；000001/399001/000300 来自 TRDNEW_Index，
                                      000905/000852/399006/000688 来自 index_supplement（新浪）
另写 market_check.json（日历与指数的对账、缺失计数）。

用法:
  python ashare_train/build_market.py
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

from Misc.config import ASHARE_MARKET_PATH, ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, CSMAR_ROOT, SAMPLE_END  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('market')

CSMAR_INDICES = ['000001', '399001', '000300']
SUPPLEMENT_INDICES = ['000905', '000852', '399006', '000688']
MKT_MAIN, MKT_BJ = 53, 117


def zip_csv(glob_pat: str, member: str, **kw) -> pd.DataFrame:
    path = sorted(Path(CSMAR_ROOT).glob(glob_pat))
    if len(path) != 1:
        raise RuntimeError(f'expected one {glob_pat}, found {path}')
    with zipfile.ZipFile(path[0]) as z:
        return pd.read_csv(z.open(member), **kw)


def main() -> int:
    chk: dict = {}
    end = pd.Timestamp(SAMPLE_END)
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'Markettype'])
    cal = pd.DatetimeIndex(np.sort(p.loc[p['Markettype'].isin([1, 4]), 'Date'].unique()), name='Date')
    del p
    m = pd.DataFrame(index=cal)

    rf = zip_csv('无风险利率文件*.zip', 'TRDNEW_Nrrate.csv', usecols=['Clsdt', 'Nrr1', 'Nrrdata'], parse_dates=['Clsdt'])
    if set(rf['Nrr1']) != {'NRI01'} or rf['Clsdt'].duplicated().any():
        raise ValueError('TRDNEW_Nrrate: unexpected benchmark or duplicate dates')
    rf = rf.set_index('Clsdt')['Nrrdata'].sort_index()
    rate_at = rf.reindex(cal, method='ffill')
    m['rf_annual'] = rate_at.shift(1).to_numpy()
    days = pd.Series(cal, index=cal).diff().dt.days
    m['rf'] = (1 + m['rf_annual'] / 100) ** (days / 365) - 1
    chk['rf_dates_missing_rate'] = int(rate_at.isna().sum())

    mk = zip_csv('综合日*.zip', 'TRDNEW_Cndalym.csv', usecols=['Markettype', 'Trddt', 'Cdretwdos', 'Cdretwdeq', 'Cdretwdtl3', 'Cdnstkcal'],
                 parse_dates=['Trddt'])
    mk = mk[mk['Trddt'] <= end]
    for mt, cols in [(MKT_MAIN, {'Cdretwdos': 'mkt_vw', 'Cdretwdeq': 'mkt_ew', 'Cdretwdtl3': 'mkt_tv', 'Cdnstkcal': 'mkt_n'}),
                     (MKT_BJ, {'Cdretwdos': 'mkt_bj_vw', 'Cdretwdeq': 'mkt_bj_ew'})]:
        s = mk[mk['Markettype'] == mt].set_index('Trddt')
        if s.index.duplicated().any():
            raise ValueError(f'Cndalym Markettype {mt}: duplicate dates')
        chk[f'cndalym_{mt}_dates_not_in_calendar'] = [str(x.date()) for x in s.index.difference(cal)]
        for c, new in cols.items():
            m[new] = s[c].reindex(cal).to_numpy()
        after = cal[cal >= s.index.min()]
        chk[f'cndalym_{mt}'] = {'first': str(s.index.min().date()), 'n': len(s),
                                'calendar_days_missing_after_first': [str(x.date()) for x in after[~after.isin(s.index)]]}

    ix = zip_csv('指数文件*.zip', 'TRDNEW_Index.csv', usecols=['Indexcd', 'Trddt', 'Clsindex', 'Retindex'],
                 dtype={'Indexcd': str}, parse_dates=['Trddt'])
    ix = ix[ix['Trddt'] <= end]
    for code in CSMAR_INDICES:
        s = ix[ix['Indexcd'] == code].set_index('Trddt').sort_index()
        if s.index.duplicated().any():
            raise ValueError(f'index {code}: duplicate dates')
        chk[f'idx_{code}'] = {'source': 'TRDNEW_Index', 'first': str(s.index.min().date()), 'n': len(s),
                              'dates_not_in_calendar': [str(x.date()) for x in s.index.difference(cal)],
                              'calendar_days_missing_after_first': int((~cal[cal >= s.index.min()].isin(s.index)).sum())}
        m[f'idx_{code}_close'] = s['Clsindex'].reindex(cal).to_numpy()
        m[f'idx_{code}_ret'] = s['Retindex'].reindex(cal).to_numpy()
    for code in SUPPLEMENT_INDICES:
        s = pd.read_csv(Path(CSMAR_ROOT, 'index_supplement', f'{code}.csv'), parse_dates=['date']).set_index('date').sort_index()
        s = s[s.index <= end]
        if s.index.duplicated().any() or (s['close'] <= 0).any():
            raise ValueError(f'index {code}: duplicate dates or non-positive close')
        chk[f'idx_{code}'] = {'source': 'index_supplement (sina)', 'first': str(s.index.min().date()), 'n': len(s),
                              'dates_not_in_calendar': [str(x.date()) for x in s.index.difference(cal)],
                              'calendar_days_missing_after_first': int((~cal[cal >= s.index.min()].isin(s.index)).sum())}
        close = s['close'].reindex(cal)
        m[f'idx_{code}_close'] = close.to_numpy()
        m[f'idx_{code}_ret'] = (s['close'] / s['close'].shift(1) - 1).reindex(cal).to_numpy()

    sh = ix[ix['Indexcd'] == '000001'].set_index('Trddt').index
    chk['calendar'] = {'n': len(cal), 'first': str(cal.min().date()), 'last': str(cal.max().date()),
                       'not_in_000001': [str(x.date()) for x in cal.difference(sh)],
                       'in_000001_not_in_calendar': [str(x.date()) for x in sh.difference(cal)]}
    if cal.max() > end:
        raise ValueError('calendar beyond SAMPLE_END')

    m = m.reset_index()
    m.to_parquet(ASHARE_MARKET_PATH, index=False)
    Path(ASHARE_PROCESSED_DATA_DIR, 'market_check.json').write_text(json.dumps(chk, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(json.dumps({k: (v if not isinstance(v, list) or len(v) < 30 else f'{len(v)} items: {v[:10]} ... {v[-5:]}')
                            for k, v in chk.items()}, indent=2, ensure_ascii=False, default=str))
    logger.info(f'写出 {ASHARE_MARKET_PATH}（{len(m):,} 行）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
