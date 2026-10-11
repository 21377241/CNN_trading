# -*- coding: utf-8 -*-
"""
build_ch_factors.py - 自建 CH-3 / CH-4 因子（Liu, Stambaugh and Yuan 2019, JFE；clean_spec 第 9 节）

官方文件（Stambaugh 网站，F:\\csmar_ashare_daily\\stambaugh_ch）只到 2021-12。按原文方法在 CSMAR 面板上自建
2000-01 至 SAMPLE_END 的日度因子，在 2000-01 至 2021-12 上与官方文件核对。

每月末（形成日 f）的样本：沪深 A 股（Markettype 1、4、16、32，不含北交所），
  上市满 6 个月（首个交易日 <= f − 6 个月）、过去一年（250 个市场交易日）有交易 >= 120 天、
  最近一个月（20 个市场交易日）有交易 >= 15 天，剔除 A 股总市值（TotalMV_A）最小的 30%（壳价值）。
  “月”“年”按市场交易日计：春节所在的日历月可能不足 15 个交易日，按日历月会剔除全部股票。
排序（独立双重排序，断点取自剔除后的样本）：
  规模：中位数分为 S / B；
  EP = 盈利 / 公司总市值（TotalMV，含 B、H 股），30% / 70% 分位分为 G / M / V；
  异常换手率 = 最近 20 个市场交易日的日均换手率 / 最近 250 个市场交易日的日均换手率（只计有交易的日子），
  30% / 70% 分位分为 P（低）/ N / O（高）。
组合日收益：下一个月内按前一交易日 A 股总市值加权（`Ret` = Dretwd）。
  MKT = 样本内全部股票的市值加权收益；MKTRF = MKT − rf（market_daily.rf）
  SMB3 = (SV + SM + SG) / 3 − (BV + BM + BG) / 3；VMG = (SV + BV) / 2 − (SG + BG) / 2
  SMB_T = (SP + SN + SO) / 3 − (BP + BN + BO) / 3；SMB4 = (SMB3 + SMB_T) / 2；PMO = (SP + BP) / 2 − (SO + BO) / 2

原文未写明或无法完全复现的细节，事先列出候选，按与官方文件 2000-2021 日度相关系数的均值
（SMB3、VMG、SMB4、PMO 四个因子）选一组，之后固定：
  earnings ∈ {ttm: 归母净利润 TTM, quarter: 最近一个单季, ytd: 本年累计}（官方用扣除非经常性损益的净利润，CSMAR 未购买）
  neg_ep ∈ {include: 负 EP 参与分组（落入 G）, exclude: 负 EP 不进入 EP 分组}
  shell_cut ∈ {after_filters: 先做上市、交易天数筛选再剔除最小 30%, before_filters: 先剔除再筛选}
财报可得日用法定截止日（build_fundamentals.pit_asof），比官方用的实际披露日保守。

输出（WORK_SPACE/data/ashare_processed_data/）：
  ch_factors_daily.parquet     Date, mkt, rf, mktrf, SMB3, VMG, SMB4, PMO, n_stocks（小数，非百分比）
  ch_factors_monthly.parquet   月度（组合日收益按月复利后再相减）
  ch_portfolios_daily.parquet  12 个排序组合的日收益
  ch_factors_check.json；docs/protocol/ch_factor_check.md

用法:
  python ashare_train/build_ch_factors.py
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
for _p in (PROJECT_ROOT, CODE_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Misc.config import (  # noqa: E402
    ASHARE_FUNDAMENTALS_PATH, ASHARE_MARKET_PATH, ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, CSMAR_ROOT, SAMPLE_END,
)
from ashare_train.build_fundamentals import pit_asof  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('ch_factors')

HIST_START = pd.Timestamp('1998-12-01')
FIRST_FORMATION = pd.Timestamp('1999-12-01')
OFFICIAL_END = pd.Timestamp('2021-12-31')
CH_MARKETS = (1, 4, 16, 32)
SHELL_CUT = 0.30
MONTH_DAYS, MIN_DAYS_MONTH = 20, 15
YEAR_DAYS, MIN_DAYS_YEAR = 250, 120
MIN_LISTED_MONTHS = 6
PORTS = [f'{s}{v}' for s in 'SB' for v in 'GMV'] + [f'{s}{v}' for s in 'SB' for v in 'PNO']
VARIANTS = {'earnings': ['ttm', 'quarter', 'ytd'], 'neg_ep': ['include', 'exclude'],
            'shell_cut': ['after_filters', 'before_filters']}
SELECT_ON = ['SMB3', 'VMG', 'SMB4', 'PMO']
OFFICIAL_DIR = Path(CSMAR_ROOT, 'stambaugh_ch')
DOC = Path(PROJECT_ROOT, 'docs', 'protocol', 'ch_factor_check.md')


def load_dense():
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Ret', 'TotalMV_A', 'TotalMV', 'Turnover',
                                                    'Markettype'])
    first_trade = p.groupby('StockID', observed=True)['Date'].min()
    p = p[(p['Date'] >= HIST_START) & p['Markettype'].isin(CH_MARKETS)]
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    stocks = pd.Index(p['StockID'].astype(str).unique()).sort_values()
    di = cal.get_indexer(p['Date'])
    si = stocks.get_indexer(p['StockID'].astype(str))
    shape = (len(cal), len(stocks))

    def dense(col):
        a = np.full(shape, np.nan)
        a[di, si] = p[col].to_numpy(np.float64)
        return a

    d = {k: dense(k) for k in ('Ret', 'TotalMV_A', 'TotalMV', 'Turnover')}
    d['traded'] = np.zeros(shape, dtype=bool)
    d['traded'][di, si] = True
    ft = first_trade.copy()
    ft.index = ft.index.astype(str)
    return cal, stocks, d, ft.reindex(stocks).to_numpy()


def single_quarter(fund: pd.DataFrame) -> pd.Series:
    """Average quarterly earnings over the most recent reporting interval: (cum − cum of the previous reported period
    of the same fiscal year) / quarters in between. Before 2002 there are no Q1/Q3 reports, so an interim report covers
    two quarters and an annual report two (or four) quarters."""
    f = fund.loc[fund['NI_cum'].notna(), ['StockID', 'Accper', 'NI_cum']].sort_values(['StockID', 'Accper'])
    q = f['Accper'].dt.month.to_numpy() // 3
    same_year = (f['StockID'].eq(f['StockID'].shift()) & f['Accper'].dt.year.eq(f['Accper'].dt.year.shift())).to_numpy()
    prev_cum = np.where(same_year, f['NI_cum'].shift().to_numpy(), 0.0)
    prev_q = np.where(same_year, np.r_[0, q[:-1]], 0)
    out = pd.Series(np.nan, index=fund.index)
    out.loc[f.index] = (f['NI_cum'].to_numpy() - prev_cum) / (q - prev_q)
    return out


def formation_inputs(cal, stocks, d, first_trade, fund) -> list[dict]:
    """Per formation date: characteristics of every stock (variant-independent part)."""
    T = len(cal)
    cs_tr = np.vstack([np.zeros((1, len(stocks))), np.cumsum(d['traded'], axis=0)])
    cs_to = np.vstack([np.zeros((1, len(stocks))), np.cumsum(np.nan_to_num(d['Turnover']), axis=0)])
    mva_f = d['mva_f'] = pd.DataFrame(d['TotalMV_A']).ffill().to_numpy()
    mv_f = pd.DataFrame(d['TotalMV']).ffill().to_numpy()
    ym = cal.year * 100 + cal.month
    last_of_month = np.flatnonzero(np.r_[ym[1:] != ym[:-1], True])
    fis = [i for i in last_of_month if cal[i] >= FIRST_FORMATION and i < T - 1]
    keys = pd.DataFrame([(s, cal[i]) for i in fis for s in stocks], columns=['StockID', 'ending_date'])
    fund = fund.assign(NI_q=single_quarter(fund))
    e = pit_asof(keys, fund, ['NI_TTM', 'NI_q', 'NI_cum'])
    out = []
    for j, fi in enumerate(fis):
        f = cal[fi]
        ms = max(fi + 1 - MONTH_DAYS, 0)
        ys = max(fi + 1 - YEAR_DAYS, 0)
        n_m = cs_tr[fi + 1] - cs_tr[ms]
        n_y = cs_tr[fi + 1] - cs_tr[ys]
        with np.errstate(invalid='ignore', divide='ignore'):
            abn_to = ((cs_to[fi + 1] - cs_to[ms]) / n_m) / ((cs_to[fi + 1] - cs_to[ys]) / n_y)
        sl = slice(j * len(stocks), (j + 1) * len(stocks))
        mv = mv_f[fi]
        with np.errstate(invalid='ignore', divide='ignore'):
            ep = {k: e[c].to_numpy()[sl] / np.where(mv > 0, mv, np.nan)
                  for k, c in (('ttm', 'NI_TTM'), ('quarter', 'NI_q'), ('ytd', 'NI_cum'))}
        nxt = fis[j + 1] if j + 1 < len(fis) else T - 1
        out.append({'fi': fi, 'days': np.arange(fi + 1, nxt + 1), 'size': mva_f[fi], 'n_m': n_m, 'n_y': n_y,
                    'listed': first_trade <= (f - pd.DateOffset(months=MIN_LISTED_MONTHS)).to_datetime64(),
                    'abn_to': abn_to, 'ep': ep})
    return out


def tercile(x: np.ndarray, base: np.ndarray, labels: str) -> np.ndarray:
    g = np.full(len(x), '', dtype=object)
    ok = base & np.isfinite(x)
    if ok.sum() < 3:
        return g
    lo, hi = np.quantile(x[ok], [0.3, 0.7])
    g[ok & (x <= lo)] = labels[0]
    g[ok & (x > lo) & (x <= hi)] = labels[1]
    g[ok & (x > hi)] = labels[2]
    return g


def assign(fm: dict, earnings: str, neg_ep: str, shell_cut: str) -> tuple[np.ndarray, dict]:
    size = fm['size']
    has = np.isfinite(size) & (size > 0) & (fm['n_m'] >= 1)
    filt = fm['listed'] & (fm['n_m'] >= MIN_DAYS_MONTH) & (fm['n_y'] >= MIN_DAYS_YEAR)
    if shell_cut == 'before_filters':
        cut = np.quantile(size[has], SHELL_CUT)
        u = has & (size > cut) & filt
    else:
        base = has & filt
        cut = np.quantile(size[base], SHELL_CUT)
        u = base & (size > cut)
    med = np.median(size[u])
    sb = np.where(size <= med, 'S', 'B')
    ep = fm['ep'][earnings]
    ep_base = u & (ep > 0) if neg_ep == 'exclude' else u
    g_ep = tercile(ep, ep_base, 'GMV')
    g_to = tercile(fm['abn_to'], u, 'PNO')
    masks = {}
    for s in 'SB':
        for v in 'GMV':
            masks[f'{s}{v}'] = u & (sb == s) & (g_ep == v)
        for v in 'PNO':
            masks[f'{s}{v}'] = u & (sb == s) & (g_to == v)
    return u, masks


def portfolio_returns(d, forms, variant) -> pd.DataFrame:
    T = d['Ret'].shape[0]
    mva_f = d['mva_f']
    rows = np.full((T, 1 + len(PORTS)), np.nan)
    n_u = np.zeros(T, dtype=np.int32)
    for fm in forms:
        u, masks = assign(fm, **variant)
        M = np.stack([u] + [masks[k] for k in PORTS], axis=1).astype(np.float64)
        D = fm['days']
        R = d['Ret'][D]
        W = mva_f[D - 1]
        ok = np.isfinite(R) & np.isfinite(W) & (W > 0)
        Rw = np.where(ok, R * W, 0.0)
        Ww = np.where(ok, W, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            rows[D] = (Rw @ M) / (Ww @ M)
        n_u[D] = int(u.sum())
    df = pd.DataFrame(rows, columns=['mkt'] + PORTS)
    df['n_stocks'] = n_u
    return df


def factors(port: pd.DataFrame) -> pd.DataFrame:
    p = port
    out = pd.DataFrame(index=p.index)
    out['mkt'] = p['mkt']
    out['SMB3'] = p[['SG', 'SM', 'SV']].mean(axis=1, skipna=False) - p[['BG', 'BM', 'BV']].mean(axis=1, skipna=False)
    out['VMG'] = (p['SV'] + p['BV']) / 2 - (p['SG'] + p['BG']) / 2
    smb_t = p[['SP', 'SN', 'SO']].mean(axis=1, skipna=False) - p[['BP', 'BN', 'BO']].mean(axis=1, skipna=False)
    out['SMB4'] = (out['SMB3'] + smb_t) / 2
    out['PMO'] = (p['SP'] + p['BP']) / 2 - (p['SO'] + p['BO']) / 2
    return out


def read_official(name: str, date_col: str) -> pd.DataFrame:
    path = OFFICIAL_DIR / name
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    i = next(k for k, line in enumerate(lines) if line[:8].isdigit())
    o = pd.read_csv(path, skiprows=i - 1)
    o[date_col] = pd.to_datetime(o[date_col].astype(str), format='%Y%m%d')
    return o


def official_daily() -> pd.DataFrame:
    o3 = read_official('CH_3_fac_daily_update_20211231.csv', 'date').set_index('date')
    o4 = read_official('CH_4_fac_daily_update_20211231.csv', 'date').set_index('date')
    return pd.DataFrame({'mkt': (o3['mktrf'] + o3['rf_dly']) / 100, 'SMB3': o3['SMB'] / 100, 'VMG': o3['VMG'] / 100,
                         'SMB4': o4['SMB'] / 100, 'PMO': o4['PMO'] / 100})


def compare(fac: pd.DataFrame, days: pd.DatetimeIndex | None = None) -> dict:
    off = official_daily()
    if days is not None:
        off = off.loc[off.index.intersection(days)]
    f = fac.reindex(off.index)
    res = {'days_official': len(off), 'days_matched': int(f['mkt'].notna().sum())}
    for c in off.columns:
        ok = f[c].notna() & off[c].notna()
        a, b = f.loc[ok, c], off.loc[ok, c]
        res[c] = {'corr': float(a.corr(b)), 'mean_ann_ours': float(a.mean() * 250), 'mean_ann_official': float(b.mean() * 250),
                  'te_ann': float((a - b).std() * np.sqrt(250))}
    return res


def monthly(port: pd.DataFrame, dates: pd.DatetimeIndex, rf: pd.Series) -> pd.DataFrame:
    g = (1 + port.drop(columns='n_stocks').set_index(dates)).groupby(dates.to_period('M')).prod(min_count=1) - 1
    m = factors(g)
    m['rf'] = (1 + rf.reindex(dates).fillna(0)).groupby(dates.to_period('M')).prod().reindex(m.index) - 1
    m['mktrf'] = m['mkt'] - m['rf']
    return m


def compare_monthly(m: pd.DataFrame) -> dict:
    o = read_official('CH_4_fac_update_20211231.csv', 'mnthdt')
    o.index = o['mnthdt'].dt.to_period('M')
    res = {}
    for ours, theirs in (('mktrf', 'mktrf'), ('SMB4', 'SMB'), ('VMG', 'VMG'), ('PMO', 'PMO')):
        a = m[ours].reindex(o.index)
        b = o[theirs] / 100
        ok = a.notna() & b.notna()
        res[ours] = {'months': int(ok.sum()), 'corr': float(a[ok].corr(b[ok])),
                     'mean_ann_ours': float(a[ok].mean() * 12), 'mean_ann_official': float(b[ok].mean() * 12)}
    return res


def write_doc(chk: dict) -> None:
    sel = chk['selected']
    lines = ['# 自建 CH-3 / CH-4 因子与官方文件核对', '',
             '由 `ashare_train/build_ch_factors.py` 生成；方法见脚本说明与 `clean_spec.md` 第 9 节。'
             '官方文件：Stambaugh 网站 CH-3、CH-4 日度和月度因子（2000-01 至 2021-12）。', '',
             '## 候选设定（按 SMB3、VMG、SMB4、PMO 日度相关系数的均值排序）', '',
             '| earnings | neg_ep | shell_cut | MKT | SMB3 | VMG | SMB4 | PMO | 均值 |', '|---|---|---|---|---|---|---|---|---|']
    for v in chk['variants']:
        c = v['daily']
        lines.append(f"| {v['earnings']} | {v['neg_ep']} | {v['shell_cut']} | {c['mkt']['corr']:.4f} | "
                     f"{c['SMB3']['corr']:.4f} | {c['VMG']['corr']:.4f} | {c['SMB4']['corr']:.4f} | {c['PMO']['corr']:.4f} | "
                     f"{v['score']:.4f} |")
    lines += ['', f"选定：earnings = {sel['earnings']}，neg_ep = {sel['neg_ep']}，shell_cut = {sel['shell_cut']}。", '',
              '## 选定设定的核对', '', '| 因子 | 日度相关 | 年化均值（自建） | 年化均值（官方） | 年化跟踪误差 | 月度相关 | 月度年化均值（自建） | 月度年化均值（官方） |',
              '|---|---|---|---|---|---|---|---|']
    dly, mon = chk['selected_daily'], chk['selected_monthly']
    for c, mc in (('mkt', 'mktrf'), ('SMB3', None), ('VMG', 'VMG'), ('SMB4', 'SMB4'), ('PMO', 'PMO')):
        x = dly[c]
        mm = mon.get(mc, {}) if mc else {}
        lines.append(f"| {c} | {x['corr']:.4f} | {x['mean_ann_ours']:.2%} | {x['mean_ann_official']:.2%} | {x['te_ann']:.2%} | "
                     + (f"{mm['corr']:.4f} | {mm['mean_ann_ours']:.2%} | {mm['mean_ann_official']:.2%} |" if mm else '— | — | — |'))
    lines += ['', '说明：日度 MKT 与官方 mktrf + rf_dly 比较；月度 MKT 一列为 mktrf。官方数值以百分比保留两位小数。',
              f"每月样本股票数（中位数）：{chk['n_stocks_median_by_period']}。"]
    DOC.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    t0 = time.time()
    cal, stocks, d, first_trade = load_dense()
    logger.info(f'稠密矩阵 {d["Ret"].shape}，用时 {time.time() - t0:.0f}s')
    fund = pd.read_parquet(ASHARE_FUNDAMENTALS_PATH)
    forms = formation_inputs(cal, stocks, d, first_trade, fund)
    logger.info(f'形成日 {len(forms)} 个（{cal[forms[0]["fi"]].date()} 至 {cal[forms[-1]["fi"]].date()}），用时 {time.time() - t0:.0f}s')
    mk = pd.read_parquet(ASHARE_MARKET_PATH, columns=['Date', 'rf']).set_index('Date')['rf']

    variants, facs = [], []
    for combo in itertools.product(*VARIANTS.values()):
        variants.append(dict(zip(VARIANTS, combo)))
        facs.append(factors(portfolio_returns(d, forms, variants[-1])).set_index(cal))
    off = official_daily()
    common = off.index[off.notna().all(axis=1).to_numpy()]
    for f in facs:
        common = common.intersection(f.index[f.notna().all(axis=1).to_numpy()])
    for v, f in zip(variants, facs):
        cmp_ = compare(f, common)
        v['daily'] = cmp_
        v['score'] = float(np.mean([cmp_[c]['corr'] for c in SELECT_ON]))
        logger.info(f"{tuple(v[k] for k in VARIANTS)}: " + ', '.join(f"{c} {cmp_[c]['corr']:.4f}" for c in ['mkt'] + SELECT_ON))
    del facs
    variants.sort(key=lambda v: -v['score'])
    best = {k: variants[0][k] for k in VARIANTS}

    port = portfolio_returns(d, forms, best)
    port.index = cal
    first_day = cal[forms[0]['fi'] + 1]
    port = port[(port.index >= first_day) & (port.index <= pd.Timestamp(SAMPLE_END))]
    fac = factors(port)
    fac['rf'] = mk.reindex(fac.index).to_numpy()
    fac['mktrf'] = fac['mkt'] - fac['rf']
    fac['n_stocks'] = port['n_stocks']
    if fac[['mkt', 'SMB3', 'VMG', 'SMB4', 'PMO', 'rf']].isna().any().any():
        raise RuntimeError(f"因子有缺失日：{fac[fac.isna().any(axis=1)].index[:10].tolist()}")
    mon = monthly(port, port.index, mk)

    out_dir = Path(ASHARE_PROCESSED_DATA_DIR)
    fac.reset_index(names='Date')[['Date', 'mkt', 'rf', 'mktrf', 'SMB3', 'VMG', 'SMB4', 'PMO', 'n_stocks']].to_parquet(
        out_dir / 'ch_factors_daily.parquet', index=False)
    mon_out = mon.copy()
    mon_out.index = mon_out.index.to_timestamp('M')
    mon_out.reset_index(names='month').to_parquet(out_dir / 'ch_factors_monthly.parquet', index=False)
    port.drop(columns='n_stocks').reset_index(names='Date').to_parquet(out_dir / 'ch_portfolios_daily.parquet', index=False)

    ns = port['n_stocks'].groupby(port.index.year // 5 * 5).median()
    chk = {'variants': variants, 'variant_comparison_days': [len(common), str(common.min().date()), str(common.max().date())],
           'selected': best, 'selected_daily': compare(fac), 'selected_monthly': compare_monthly(mon),
           'days': len(fac), 'first_day': str(fac.index[0].date()), 'last_day': str(fac.index[-1].date()),
           'n_stocks_median_by_period': {f'{k}-{k + 4}': int(v) for k, v in ns.items()},
           'official_end': str(OFFICIAL_END.date()), 'seconds': round(time.time() - t0)}
    (out_dir / 'ch_factors_check.json').write_text(json.dumps(chk, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    write_doc(chk)
    logger.info(json.dumps({k: chk[k] for k in ('selected', 'selected_daily', 'selected_monthly', 'n_stocks_median_by_period')},
                           indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
