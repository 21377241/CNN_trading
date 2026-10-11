# -*- coding: utf-8 -*-
"""
reconcile_pre_clean.py - 新（CSMAR）数据与旧数据商数据的对账（只报告，不进入任何构建、模型或结果）

clean 分支中，数据流程（面板、universe、图像、特征、E7 数据、CH 因子）只读 CSMAR 原始文件和由它构建的产物；
本脚本是数据流程中唯一读取旧数据的地方，用于记录换源前后的差异。旧数据缺失时对应部分跳过。

阶段:
  panel     日面板：D1 重叠股票-日的日收益、D2 只在一方出现的股票-日、F2 BM/EP 与旧 PB/PE 倒数的秩相关
            （旧面板：WORK_DIR/_olddata，即旧数据上的干净版本面板）
  universe  周度样本键与旧 I20R5 样本键（旧工作区 PRE_CLEAN_WORK_DIR）
  inputs    样本外键与旧 chart-CNN（M0）预测键；窗口收盘价路径与旧面板；新旧图像对比图

输出：WORK_SPACE/data/ashare_processed_data/reconcile_pre_clean/<phase>.json，docs/protocol/pre_clean_reconciliation.md

用法:
  python ashare_train/reconcile_pre_clean.py [--phases panel universe inputs]
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
for _p in (PROJECT_ROOT, CODE_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Misc.config import (  # noqa: E402
    ASHARE_COUNTRY, ASHARE_FUNDAMENTALS_PATH, ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, ASHARE_STOCKS_SAVEPATH,
    ASHARE_UNIVERSE_PATH, PRE_CLEAN_WORK_DIR, SAMPLE_END, WORK_DIR,
)
from Data.dgp_config import get_dataset_dir, get_dataset_filename  # noqa: E402
from ashare_train.audit_csmar_panel import (  # noqa: E402
    AUDIT_DIR, CODE_MAP, CODE_REUSE, MAIN_MARKETS, SEED, day_int, day_str, panel_col, result,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('reconcile')

OUT_DIR = Path(ASHARE_PROCESSED_DATA_DIR, 'reconcile_pre_clean')
REPORT = Path(PROJECT_ROOT, 'docs', 'protocol', 'pre_clean_reconciliation.md')
FIG = Path(PROJECT_ROOT, 'docs', 'figures', 'clean_render_compare.png')
OLD_CLEAN_PANEL = Path(WORK_DIR, '_olddata', 'data', 'ashare_processed_data', 'ashare_ret.feather')
PRE_CLEAN_PANEL = Path(PRE_CLEAN_WORK_DIR, 'data', 'ashare_processed_data', 'ashare_ret.feather')
PRE_CLEAN_IMG_DIR = Path(PRE_CLEAN_WORK_DIR, 'data', 'ashare_stocks_dataset', 'I20R5', 'stocks_CHN', 'dataset_all')
M0_PRED = Path(PRE_CLEAN_WORK_DIR, 'paper', 'E2_matched_is_ind_chrono_ric', 'preds', 'pred_cnn_oos.csv')
NEW_IMG_DIR = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5_clean'))
OLD_YEARS = range(2000, 2026)


def fn(year, suffix):
    return get_dataset_filename(20, 'week', True, [20], year, suffix)


def save(phase: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), **payload}
    Path(OUT_DIR, f'{phase}.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                                              encoding='utf-8')


# --------------------------------------------------------------------------------------------- panel

def phase_panel() -> None:
    if not OLD_CLEAN_PANEL.exists():
        save('panel', {'skipped': str(OLD_CLEAN_PANEL)})
        return
    results, details = [], {}
    new = {c: panel_col(ASHARE_PANEL_PATH, c) for c in ['StockID', 'Date', 'Ret', 'Ret_nd', 'adj_factor', 'Markettype',
                                                         'TotalMV']}
    new['day'] = day_int(new.pop('Date'))
    kn = (new['StockID'] << 20) | new['day']
    old = {c: panel_col(OLD_CLEAN_PANEL, c) for c in ['StockID', 'Date', 'Ret', 'pct_chg', 'PB', 'PE_TTM']}
    old['day'] = day_int(old.pop('Date'))
    ko = (old['StockID'] << 20) | old['day']
    oo = np.argsort(ko, kind='stable')
    for c in old:
        old[c] = old[c][oo]
    ko = ko[oo]
    if (np.diff(kn) <= 0).any() or (np.diff(ko) <= 0).any():
        raise ValueError('panel keys not unique / not sorted')
    _, i_n, i_o = np.intersect1d(kn, ko, assume_unique=True, return_indices=True)
    details['overlap_rows'] = len(i_n)

    # D1
    rn, ro = new['Ret'][i_n], old['Ret'][i_o]
    both = np.isfinite(rn) & np.isfinite(ro)
    diff = np.abs(rn - ro)
    within = diff[both] <= 0.001
    d1 = float(within.mean())
    nn = len(new['day'])
    first_new = np.r_[True, new['StockID'][1:] != new['StockID'][:-1]]
    prev_n = np.r_[0, np.arange(nn - 1)]
    jump = new['adj_factor'] / new['adj_factor'][prev_n] - 1
    jump[first_new] = np.nan
    first_old = np.r_[True, old['StockID'][1:] != old['StockID'][:-1]]
    gap_new = np.where(first_new, -1, new['day'] - new['day'][prev_n])
    prev_o = np.r_[0, np.arange(len(ko) - 1)]
    gap_old = np.where(first_old, -1, old['day'] - old['day'][prev_o])
    bad = np.flatnonzero(both)[~within]
    rng = np.random.default_rng(SEED)
    samp = rng.choice(bad, size=min(100, len(bad)), replace=False)

    def why(k: int) -> str:
        jn, jo = i_n[k], i_o[k]
        if gap_new[jn] != gap_old[jo]:
            return '前一交易行不同（一方缺日）'
        if np.isfinite(jump[jn]) and abs(jump[jn]) > 1e-4:
            if np.isfinite(old['pct_chg'][jo]) and abs(old['pct_chg'][jo] / 100 - new['Ret_nd'][jn]) < 0.001:
                return '除权日：旧面板复权收益错误（旧涨跌幅与新 Ret_nd 一致）'
            return '除权日：复权方法差异'
        if np.isfinite(old['pct_chg'][jo]) and abs(old['pct_chg'][jo] / 100 - new['Ret_nd'][jn]) < 0.001:
            return '非除权日：旧面板复权价错误（旧涨跌幅与新一致）'
        return '其他'

    att = pd.Series([why(k) for k in samp]).value_counts()
    all_cat = pd.Series([why(k) for k in bad]).value_counts() if len(bad) <= 200000 else None
    details['D1_sample100_attribution'] = att.to_dict()
    if all_cat is not None:
        details['D1_all_attribution'] = all_cat.to_dict()
    details['D1_examples'] = [(f'{new["StockID"][i_n[k]]:06d}', day_str(new['day'][i_n[k]]), float(rn[k]), float(ro[k]),
                               why(k)) for k in samp[:30]]
    yb = pd.Series(new['day'][i_n[bad]].astype('datetime64[D]')).dt.year.value_counts().sort_index()
    details['D1_exceed_by_year'] = yb.to_dict()
    results.append(result('D1', '与旧面板重叠股票-日的日收益', '差异 ≤ 0.1 个百分点 ≥ 98%；超出的抽 100 条归因',
                          f'重叠 {both.sum():,} 行，{d1:.4%} 在 0.1pp 内；超出 {len(bad):,} 行；抽样归因 {att.to_dict()}',
                          d1 >= 0.98))

    # D2
    only_new = np.setdiff1d(np.arange(nn), i_n, assume_unique=True)
    only_old = np.setdiff1d(np.arange(len(ko)), i_o, assume_unique=True)
    old_max = old['day'].max()
    old_codes = np.unique(old['StockID'])
    first_old_day = pd.Series(old['day']).groupby(old['StockID']).min()
    cat_n = np.full(len(only_new), '其他', dtype=object)
    s_n, d_n = new['StockID'][only_new], new['day'][only_new]
    in_old_stock = np.isin(s_n, old_codes)
    f_old = first_old_day.reindex(s_n).to_numpy()
    cat_n[~in_old_stock] = '旧面板无此股票'
    cat_n[in_old_stock & (d_n < f_old)] = '早于旧面板该股首日'
    cat_n[d_n > old_max] = '晚于旧面板截止日'
    cat_n[new['Markettype'][only_new] == 64] = '北交所'
    fills = pd.read_parquet(Path(AUDIT_DIR, 'raw_fill_keys.parquet'))
    fills['Stkcd'] = fills['Stkcd'].replace(CODE_MAP)
    kf = (fills['Stkcd'].to_numpy(np.int64) << 20) | fills['day'].to_numpy()
    fmap = pd.Series(fills['Filling'].to_numpy(), index=kf)
    fmap = fmap[~fmap.index.duplicated()]
    k_oo = ko[only_old]
    fill_of = fmap.reindex(k_oo).to_numpy()
    cat_o = np.full(len(only_old), '其他', dtype=object)
    cat_o[~np.isin(old['StockID'][only_old], np.unique(new['StockID']))] = '新面板无此股票'
    s_o, d_o = old['StockID'][only_old], old['day'][only_old]
    merged_old = np.isin(s_o, list(CODE_MAP))
    cat_o[merged_old] = '已并入新代码的旧代码'
    dup_key = (pd.Series(s_o).map(CODE_MAP).fillna(-1).to_numpy(np.int64) << 20) | d_o.astype(np.int64)
    dup = merged_old & np.isin(dup_key, ko)
    cat_o[dup] = '已并入新代码的旧代码（旧面板在新代码下另有同日行，即重复历史）'
    details['D2_old_duplicate_history'] = pd.Series(s_o[dup]).map(lambda c: f'{c:06d}').value_counts().to_dict()
    for code, dt in CODE_REUSE.items():
        cat_o[(old['StockID'][only_old] == code) & (old['day'][only_old] < day_int([np.datetime64(dt)])[0])] = '代码复用的前身段'
    cat_o[fill_of == 1] = 'CSMAR 标为停牌（Filling = 1）'
    cat_o[fill_of == 2] = 'CSMAR 标为节假日（Filling = 2）'
    cat_o[old['day'][only_old] > day_int([np.datetime64(SAMPLE_END)])[0]] = '晚于样本截止日'
    cn, co_ = pd.Series(cat_n).value_counts(), pd.Series(cat_o).value_counts()
    details['D2_new_only'] = cn.to_dict()
    details['D2_old_only'] = co_.to_dict()
    oth_o = only_old[cat_o == '其他']
    details['D2_old_only_other_examples'] = [(f'{old["StockID"][j]:06d}', day_str(old['day'][j])) for j in oth_o[:30]]
    details['D2_old_only_other_by_year'] = pd.Series(old['day'][oth_o].astype('datetime64[D]')).dt.year.value_counts().sort_index().to_dict()
    oth_n = only_new[cat_n == '其他']
    details['D2_new_only_other_examples'] = [(f'{new["StockID"][j]:06d}', day_str(new['day'][j])) for j in oth_n[:30]]
    details['D2_new_only_other_by_year'] = pd.Series(new['day'][oth_n].astype('datetime64[D]')).dt.year.value_counts().sort_index().to_dict()
    results.append(result('D2', '只在新或旧面板出现的股票-日', '分类并解释',
                          f'只在新面板 {len(only_new):,}：{cn.to_dict()}；只在旧面板 {len(only_old):,}：{co_.to_dict()}', None,
                          '“其他”类见详情中的样例与年份分布'))

    # F2 valuation ratios vs old PB / PE
    fund = pd.read_parquet(ASHARE_FUNDAMENTALS_PATH, columns=['StockID', 'Accper', 'avail_date', 'BE', 'NI_TTM'])
    fund['code'] = fund['StockID'].astype(int)
    fund = fund.sort_values(['avail_date', 'Accper'])
    cal = np.unique(new['day'][np.isin(new['Markettype'], [1, 4])])
    me = pd.Series(cal).groupby(pd.Series(cal.astype('datetime64[D]')).dt.to_period('M')).max()
    me = me[(me.index >= pd.Period('2008-01', 'M')) & (me.index <= pd.Period('2025-12', 'M'))]
    me = me[me <= old_max]
    rows = []
    in_new = np.isin(new['Markettype'], MAIN_MARKETS)
    for per, d in me.items():
        an = np.flatnonzero((new['day'] == d) & in_new)
        ao = np.flatnonzero(old['day'] == d)
        nd = pd.DataFrame({'code': new['StockID'][an], 'mv': new['TotalMV'][an]})
        od = pd.DataFrame({'code': old['StockID'][ao], 'PB': old['PB'][ao], 'PE': old['PE_TTM'][ao]})
        ts = pd.Timestamp(np.datetime64(int(d), 'D'))
        fl = fund[fund['avail_date'] <= ts].groupby('code')[['BE', 'NI_TTM']].last()
        x = nd.merge(od, on='code').merge(fl, left_on='code', right_index=True, how='left')
        bm_n = np.where(x['BE'] > 0, x['BE'] / x['mv'], np.nan)
        bm_o = np.where(x['PB'] > 0, 1 / x['PB'], np.nan)
        ep_n = x['NI_TTM'] / x['mv']
        ep_o = np.where(x['PE'] != 0, 1 / x['PE'], np.nan)
        r1 = pd.Series(bm_n).corr(pd.Series(bm_o), method='spearman')
        r2 = pd.Series(ep_n).corr(pd.Series(ep_o), method='spearman')
        rows.append({'month': str(per), 'n': len(x), 'rho_bm': r1, 'rho_ep': r2})
    rf2 = pd.DataFrame(rows)
    details['F2_by_month'] = rf2.to_dict('records')
    mb, mp = float(rf2['rho_bm'].median()), float(rf2['rho_ep'].median())
    results.append(result('F2', 'BM、EP 与旧数据商 PB、PE 倒数的截面秩相关（月末，2008 起）', '中位数 ≥ 0.9',
                          f'{len(rf2)} 个月；BM 中位 {mb:.3f}（最小 {rf2["rho_bm"].min():.3f}）；EP 中位 {mp:.3f}（最小 {rf2["rho_ep"].min():.3f}）',
                          mb >= 0.9 and mp >= 0.9))
    save('panel', {'results': results, 'details': details})


# --------------------------------------------------------------------------------------------- universe

def phase_universe() -> None:
    """Weekly keys vs the pre-clean I20R5 keys (labels.feather, filtered by the old EquityDataset rule)."""
    if not PRE_CLEAN_IMG_DIR.exists():
        save('universe', {'skipped': str(PRE_CLEAN_IMG_DIR)})
        return
    u = pd.read_feather(ASHARE_UNIVERSE_PATH)
    frames = []
    for y in OLD_YEARS:
        f = PRE_CLEAN_IMG_DIR / fn(y, '_labels.feather')
        if f.exists():
            frames.append(pd.read_feather(f, columns=['StockID', 'ending_date', 'ret_val', 'EWMA_vol']))
    old = pd.concat(frames, ignore_index=True)
    old['StockID'] = old['StockID'].astype(str).str.zfill(6)
    old['ending_date'] = pd.to_datetime(old['ending_date'])
    old_lab = old[(old['ret_val'].notna()) & (old['EWMA_vol'] > 0)]
    new = u[['StockID', 'ending_date', 'split', 'ret_label', 'n_obs_in_win']].copy()
    new['StockID'] = new['StockID'].astype(str)
    mg = old_lab.merge(new, on=['StockID', 'ending_date'], how='outer', indicator=True)
    both = mg['_merge'] == 'both'
    only_old = mg[mg['_merge'] == 'left_only'][['StockID', 'ending_date']]
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['StockID', 'Date', 'is_ST'])
    p['StockID'] = p['StockID'].astype(str)
    oo = only_old.merge(p, left_on=['StockID', 'ending_date'], right_on=['StockID', 'Date'], how='left')
    only_new = mg[mg['_merge'] == 'right_only']
    d = (mg.loc[both, 'ret_val'] - mg.loc[both, 'ret_label']).abs()
    save('universe', {
        'old_rows_all': len(old), 'old_rows_labeled_ewma_pos': len(old_lab), 'new_rows': len(u),
        'both': int(both.sum()), 'only_old': int(len(only_old)),
        'only_old_no_trade_on_t': int(oo['Date'].isna().sum()), 'only_old_st_on_t': int((oo['is_ST'] == 1).sum()),
        'only_new': int(len(only_new)), 'only_new_label_nan': int(only_new['ret_label'].isna().sum()),
        'only_new_n_obs_lt_20': int((only_new['n_obs_in_win'] < 20).sum()),
        'label_absdiff_le_1e-9_share': float((d <= 1e-9).mean()),
        'label_absdiff_gt_1e-3_share': float((d > 1e-3).mean()),
        'oos_new_rows': int((u['split'] == 'oos').sum()),
        'oos_old_rows_all': int((old['ending_date'].dt.year >= 2008).sum()),
    })


# --------------------------------------------------------------------------------------------- inputs

def vs_m0(uni: pd.DataFrame, cal: pd.DatetimeIndex, first: pd.Series) -> dict:
    if not M0_PRED.exists():
        return {'skipped': str(M0_PRED)}
    m0 = pd.read_csv(M0_PRED, usecols=['StockID', 'ending_date'], dtype={'StockID': str})
    m0['StockID'] = m0['StockID'].str.zfill(6)
    m0['ending_date'] = pd.to_datetime(m0['ending_date'])
    u = uni[uni['split'] == 'oos'][['StockID', 'ending_date', 'ret_label', 'n_obs_in_win']]
    mg = m0.merge(u, on=['StockID', 'ending_date'], how='outer', indicator=True)
    only_m0 = mg[mg['_merge'] == 'left_only'].copy()
    only_new = mg[mg['_merge'] == 'right_only']
    t_idx = cal.get_indexer(only_m0['ending_date'])
    first_idx = cal.get_indexer(first.reindex(only_m0['StockID']).values)
    ipo = (t_idx - first_idx) < 39
    return {
        'm0_rows': len(m0), 'new_oos_rows': len(u), 'both': int((mg['_merge'] == 'both').sum()),
        'only_m0': len(only_m0), 'only_m0_ipo_rule': int(ipo.sum()), 'only_m0_other': int((~ipo).sum()),
        'only_new': len(only_new), 'only_new_no_label': int(only_new['ret_label'].isna().sum()),
        'only_new_with_missing_days_in_window': int((only_new['n_obs_in_win'] < 20).sum()),
        'only_new_share_of_new': float(len(only_new) / len(u)),
    }


def close_path_vs_old(uni: pd.DataFrame, p: pd.DataFrame, cal: pd.DatetimeIndex, n_sample=40_000, seed=0) -> dict:
    """Window close path normalised to the first close: new panel vs the pre-clean vendor panel (fully traded windows)."""
    if not PRE_CLEAN_PANEL.exists():
        return {'skipped': str(PRE_CLEAN_PANEL)}
    rng = np.random.default_rng(seed)
    s = uni.iloc[np.sort(rng.choice(len(uni), n_sample, replace=False))][['StockID', 'ending_date', 't_idx']]
    s = s.reset_index(drop=True)
    win = s.loc[s.index.repeat(20)].reset_index(names='k')
    win['pos'] = np.tile(np.arange(20), len(s))
    win['Date'] = cal[win['t_idx'].to_numpy() - 19 + win['pos'].to_numpy()]
    old = pd.read_feather(PRE_CLEAN_PANEL, columns=['Date', 'StockID', 'Close'])
    old['StockID'] = old['StockID'].astype(str).str.zfill(6)
    nc = win.merge(p, on=['StockID', 'Date'], how='left').sort_values(['k', 'pos'])['Close'].to_numpy().reshape(-1, 20)
    oc = win.merge(old, on=['StockID', 'Date'], how='left').sort_values(['k', 'pos'])['Close'].to_numpy().reshape(-1, 20)
    full = np.isfinite(nc).all(axis=1) & np.isfinite(oc).all(axis=1)
    dev = np.abs(oc[full] / oc[full][:, :1] - nc[full] / nc[full][:, :1]).max(axis=1)
    return {'windows': int(full.sum()), 'share_max_dev_lt_1e-3': float((dev < 1e-3).mean()),
            'median_max_dev': float(np.median(dev))}


def image_figure(seed=3) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    y = 2015
    old_lbl = pd.read_feather(PRE_CLEAN_IMG_DIR / fn(y, '_labels.feather'), columns=['StockID', 'ending_date'])
    old_lbl['StockID'] = old_lbl['StockID'].astype(str).str.zfill(6)
    new_lbl = pd.read_feather(os.path.join(NEW_IMG_DIR, fn(y, '_labels.feather')),
                              columns=['StockID', 'ending_date', 'n_obs_in_win'])
    mg = new_lbl.reset_index().merge(old_lbl.reset_index(), on=['StockID', 'ending_date'], suffixes=('_new', '_old'))
    pick = pd.concat([mg[mg['n_obs_in_win'] == 20].sample(3, random_state=seed),
                      mg[mg['n_obs_in_win'] < 20].sample(1, random_state=seed)])
    n_old = os.path.getsize(PRE_CLEAN_IMG_DIR / fn(y, '_images.dat')) // 3840
    oi = np.memmap(PRE_CLEAN_IMG_DIR / fn(y, '_images.dat'), dtype='uint8', mode='r', shape=(n_old, 64, 60))
    ni = np.memmap(os.path.join(NEW_IMG_DIR, fn(y, '_images.dat')), dtype='uint8', mode='r',
                   shape=(len(new_lbl), 64, 60))
    fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 7))
    for j, (_, r) in enumerate(pick.iterrows()):
        for i, (arr, title) in enumerate(((oi[r['index_old']], '旧图像（旧数据，无开收盘）'), (ni[r['index_new']], '干净版本（CSMAR）'))):
            ax = axes[i, j]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}\n{r['StockID']} {r['ending_date'].date()} n_obs={r['n_obs_in_win']}", fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    return FIG.relative_to(PROJECT_ROOT).as_posix()


def phase_inputs() -> None:
    uni = pd.read_feather(ASHARE_UNIVERSE_PATH)
    uni['StockID'] = uni['StockID'].astype(str)
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Close'])
    p['StockID'] = p['StockID'].astype(str)
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    first = p.groupby('StockID')['Date'].min()
    out = {'oos_keys_vs_m0': vs_m0(uni, cal, first), 'close_path_vs_pre_clean_vendor': close_path_vs_old(uni, p, cal)}
    del p
    out['figure'] = image_figure() if PRE_CLEAN_IMG_DIR.exists() else {'skipped': str(PRE_CLEAN_IMG_DIR)}
    save('inputs', out)


# --------------------------------------------------------------------------------------------- report

def write_report() -> None:
    lines = ['# 新旧数据对账（只报告）', '',
             '由 `ashare_train/reconcile_pre_clean.py` 生成。clean 分支的数据流程不读取任何旧数据；本文件只记录换源（旧数据商 → CSMAR）'
             '前后的差异，不作为任何构建、模型或结果的输入。详细结果见 '
             '`WORK_SPACE/data/ashare_processed_data/reconcile_pre_clean/<phase>.json`。', '']
    mark = {True: '符合预期', False: '**不符合预期**', None: '仅报告'}
    for ph in PHASES:
        f = Path(OUT_DIR, f'{ph}.json')
        if not f.exists():
            lines += [f'## {ph}', '', '未运行。', '']
            continue
        j = json.loads(f.read_text(encoding='utf-8'))
        lines += [f'## {ph}（{j["time"]}）', '']
        if 'results' in j:
            lines += ['| 编号 | 检查 | 标准 | 结果 | 结论 |', '|---|---|---|---|---|']
            lines += [f'| {r["id"]} | {r["check"]} | {r["criterion"]} | {r["value"]} | {mark[r["passed"]]} |'
                      for r in j['results']]
        else:
            body = {k: v for k, v in j.items() if k != 'time'}
            lines += ['```json', json.dumps(body, indent=2, ensure_ascii=False, default=str), '```']
        lines.append('')
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'写出 {REPORT}')


PHASES = {'panel': phase_panel, 'universe': phase_universe, 'inputs': phase_inputs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phases', nargs='*', default=list(PHASES), choices=list(PHASES))
    args = ap.parse_args(argv)
    for ph in args.phases:
        t0 = time.time()
        logger.info(f'== {ph}')
        PHASES[ph]()
        logger.info(f'== {ph} 完成（{time.time() - t0:.0f}s）')
    write_report()
    return 0


if __name__ == '__main__':
    sys.exit(main())
