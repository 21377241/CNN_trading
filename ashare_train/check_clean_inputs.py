# -*- coding: utf-8 -*-
"""
check_clean_inputs.py - 干净版本数据与输入的验收（plan 阶段 2、3）

  A. 回归：对既无裁剪日、也不跨 ST/停牌缺口的窗口，新 Close₀ 路径与旧工作区 adjust_price 逐位一致；
     新面板 Close/Close₀ 与 cumprod(1+Ret) 一致（收益未裁剪）
  B. 样本键：universe = 图像 labels（逐年拼接）= 特征 meta（行序相同）
  C. 像素一致：由 140 维特征推出的开/收盘刻度行、高低竖线、成交量柱高与图像像素一致
  D. 持有期内被标为 ST 的股票-周出现在 OOS universe 中且有标签
  E. 与旧规范样本（M0 预测键）对账，列出差异原因
  F. 新旧图像对比 PNG

输出：WORK_SPACE/paper/clean_inputs/checks/clean_inputs_check.json 与 docs/figures/clean_render_compare.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
for p in (PROJECT_ROOT, CODE_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from Misc.config import (  # noqa: E402
    ASHARE_COUNTRY, ASHARE_PANEL_PATH, ASHARE_STOCKS_SAVEPATH, ASHARE_UNIVERSE_PATH, CLEAN_INPUTS_DIR,
    PRE_CLEAN_WORK_DIR,
)
from Data.clean_window import close0_path, gather_windows, load_grid  # noqa: E402
from Data.dgp_config import get_dataset_dir, get_dataset_filename  # noqa: E402
from Data.ohlc_render import layout  # noqa: E402

OUT_DIR = os.path.join(CLEAN_INPUTS_DIR, 'checks')
FIG = os.path.join(PROJECT_ROOT, 'docs', 'figures', 'clean_render_compare.png')
OLD_PANEL = os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_processed_data', 'ashare_ret.feather')
OLD_IMG_DIR = os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_stocks_dataset', 'I20R5', 'stocks_CHN', 'dataset_all')
M0_PRED = os.path.join(PRE_CLEAN_WORK_DIR, 'paper', 'E2_matched_is_ind_chrono_ric', 'preds', 'pred_cnn_oos.csv')
NEW_IMG_DIR = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5_clean'))
LAY = layout(20)
YEARS = list(range(2000, 2026))


def fn(year, suffix):
    return get_dataset_filename(20, 'week', True, [20], year, suffix)


def old_adjust_price(o, h, l, c, r):
    ca = np.ones_like(c)
    ca[:, 1:] = np.cumprod(1.0 + np.nan_to_num(r[:, 1:]), axis=1)
    ratio = np.where(np.abs(c) > 1e-10, ca / c, 1.0)
    o2, h2, l2 = o * ratio, h * ratio, l * ratio
    return o2, np.maximum(np.maximum(h2, o2), ca), np.minimum(np.minimum(l2, o2), ca), ca


def check_regression(grid, uni, n_sample=60_000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    s = uni.iloc[np.sort(rng.choice(len(uni), n_sample, replace=False))].reset_index(drop=True)
    old = pd.read_feather(OLD_PANEL, columns=['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Ret'])
    old['StockID'] = old['StockID'].astype(str).str.zfill(6)
    old = old.sort_values(['StockID', 'Date']).reset_index(drop=True)
    sid_codes = {k: i for i, k in enumerate(grid.stocks)}
    oc = old['StockID'].map(sid_codes).to_numpy()
    od = grid.dates.get_indexer(old['Date'])
    okey = oc.astype(np.int64) * 100_000 + od
    skey = grid.stock_index(s['StockID']).astype(np.int64) * 100_000 + s['t_idx'].to_numpy()
    pos = np.searchsorted(okey, skey, side='right') - 1
    has_t = (pos >= 19) & (okey[np.maximum(pos, 0)] == skey)
    rows = pos[:, None] + np.arange(-19, 1)[None, :]
    rows = np.where(has_t[:, None], rows, 0)
    contiguous = has_t & (okey[rows[:, 0]] == skey - 19)
    newp = close0_path(gather_windows(grid, s['t_idx'].to_numpy(), grid.stock_index(s['StockID'])))
    full = newp['valid'].all(axis=1)
    o, h, l, c, r = (old[k].to_numpy()[rows] for k in ('Open', 'High', 'Low', 'Close', 'Ret'))
    r_true = np.full_like(r, np.nan)
    r_true[:, 1:] = c[:, 1:] / c[:, :-1] - 1
    unclipped = np.all(np.abs(r[:, 1:] - r_true[:, 1:]) < 1e-9, axis=1)
    elig = contiguous & full & unclipped
    oo, oh, ol, ocl = old_adjust_price(o[elig], h[elig], l[elig], c[elig], r[elig])
    diffs = {k: float(np.nanmax(np.abs(a - newp[k][elig]))) for k, a in
             (('O', oo), ('H', oh), ('L', ol), ('C', ocl))}
    new_ret_cum = np.cumprod(np.c_[np.ones(full.sum()), newp['C'][full][:, 1:] / newp['C'][full][:, :-1]], axis=1)
    return {
        'n_sample': n_sample, 'old_row_on_t': int(has_t.sum()), 'old_window_contiguous_market_days': int(contiguous.sum()),
        'eligible_clean_windows': int(elig.sum()), 'max_abs_diff_close0_path': diffs,
        'cumprod_ret_vs_close_ratio_max_abs_diff': float(np.abs(new_ret_cum - newp['C'][full]).max()),
    }


def check_keys(uni) -> dict:
    labels = pd.concat([pd.read_feather(os.path.join(NEW_IMG_DIR, fn(y, '_labels.feather')),
                                        columns=['StockID', 'ending_date']) for y in YEARS], ignore_index=True)
    fd = os.path.join(CLEAN_INPUTS_DIR, 'features140')
    meta = pd.concat([pd.read_feather(os.path.join(fd, f'meta_{t}.feather'), columns=['StockID', 'ending_date'])
                      for t in ('is', 'oos')], ignore_index=True)
    u = uni[['StockID', 'ending_date']].astype({'StockID': str}).reset_index(drop=True)
    sizes = {y: os.path.getsize(os.path.join(NEW_IMG_DIR, fn(y, '_images.dat'))) // (64 * 60) for y in YEARS}
    xs = sum(np.load(os.path.join(fd, f'X_{t}.npy'), mmap_mode='r').shape[0] for t in ('is', 'oos'))
    return {
        'universe_rows': len(u), 'image_rows': int(sum(sizes.values())), 'feature_rows': int(xs),
        'labels_equal_universe_in_order': bool(labels.equals(u)),
        'meta_equal_universe_in_order': bool(meta.equals(u)),
    }


def check_pixels(n_per_year=1500, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    fd = os.path.join(CLEAN_INPUTS_DIR, 'features140')
    X_all = {t: np.load(os.path.join(fd, f'X_{t}.npy'), mmap_mode='r') for t in ('is', 'oos')}
    offset, stats = {'is': 0, 'oos': 0}, {'open': [], 'close': [], 'hl': [], 'vol': [], 'blank_missing': []}
    ph, vt, vb = LAY['price_h'], LAY['vol_top'], LAY['vol_bottom']
    for y in YEARS:
        tag = 'is' if y < 2008 else 'oos'
        n = os.path.getsize(os.path.join(NEW_IMG_DIR, fn(y, '_images.dat'))) // (64 * 60)
        img = np.memmap(os.path.join(NEW_IMG_DIR, fn(y, '_images.dat')), dtype='uint8', mode='r', shape=(n, 64, 60))
        idx = np.sort(rng.choice(n, min(n, n_per_year), replace=False))
        X = np.asarray(X_all[tag][offset[tag] + idx], dtype=np.float64).reshape(len(idx), 20, 7)
        offset[tag] += n
        im = np.asarray(img[idx])
        yp = lambda v: np.floor((1 - v) * (ph - 1) + 0.5).astype(int)  # noqa: E731

        def tick_ok(ii, v, col):
            y0 = yp(v)
            hit = im[ii, y0, col] == 255
            near_half = np.abs(((1 - v) * (ph - 1)) % 1 - 0.5) < 1e-4
            alt = np.clip(np.where(((1 - v) * (ph - 1)) % 1 < 0.5, y0 + 1, y0 - 1), 0, ph - 1)
            return hit, hit | (near_half & (im[ii, alt, col] == 255))

        for t in range(20):
            ok = X[:, t, 6] > 0.5
            ii = np.flatnonzero(ok)
            for name, k, col in (('open', 0, 3 * t), ('close', 3, 3 * t + 2)):
                strict, loose = tick_ok(ii, X[ii, t, k], col)
                stats[name].append(strict)
                stats.setdefault(f'{name}_allowing_half_pixel_ties', []).append(loose)
            col = im[ii, :ph, 3 * t + 1] > 0
            top, bot = yp(X[ii, t, 1]), yp(X[ii, t, 2])
            stats['hl'].append(col[np.arange(len(ii)), top] & col[np.arange(len(ii)), bot])
            bh = np.maximum(1, (X[ii, t, 5] * (vb - vt)).astype(int))
            stats['vol'].append((im[ii, vt:, 3 * t + 1] > 0).sum(axis=1) == bh + 1)
            jj = np.flatnonzero(~ok)
            stats['blank_missing'].append(im[jj, :, 3 * t:3 * t + 3].reshape(len(jj), 64 * 3).max(axis=1) == 0)
    return {k: float(np.concatenate(v).mean()) for k, v in stats.items()}


def check_st_onset(uni) -> dict:
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'is_ST'])
    p['StockID'] = p['StockID'].astype(str)
    st = p[p['is_ST'] == 1][['Date', 'StockID']].rename(columns={'Date': 'next_date'})
    st['st_next'] = 1
    u = uni[uni['split'] == 'oos'][['StockID', 'next_date', 'ret_label']].astype({'StockID': str})
    u = u.merge(st, on=['StockID', 'next_date'], how='left')
    m = u['st_next'] == 1
    return {
        'oos_rows_st_at_next_signal_date': int(m.sum()),
        'with_label': int(u.loc[m, 'ret_label'].notna().sum()),
        'mean_ret_label_st_onset': float(u.loc[m, 'ret_label'].mean()),
        'mean_ret_label_rest': float(u.loc[~m, 'ret_label'].mean()),
    }


def check_vs_m0(uni, grid) -> dict:
    if not os.path.exists(M0_PRED):
        return {'skipped': M0_PRED}
    m0 = pd.read_csv(M0_PRED, usecols=['StockID', 'ending_date'], dtype={'StockID': str})
    m0['StockID'] = m0['StockID'].str.zfill(6)
    m0['ending_date'] = pd.to_datetime(m0['ending_date'])
    u = uni[uni['split'] == 'oos'][['StockID', 'ending_date', 'ret_label', 'n_obs_in_win', 't_idx']].astype(
        {'StockID': str})
    mg = m0.merge(u, on=['StockID', 'ending_date'], how='outer', indicator=True)
    only_m0 = mg[mg['_merge'] == 'left_only'].copy()
    only_new = mg[mg['_merge'] == 'right_only']
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'is_ST'])
    first = p.groupby('StockID', observed=True)['Date'].min()
    first.index = first.index.astype(str)
    only_m0['t_idx'] = grid.dates.get_indexer(only_m0['ending_date'])
    only_m0['first_idx'] = grid.dates.get_indexer(first.reindex(only_m0['StockID']).values)
    ipo = (only_m0['t_idx'] - only_m0['first_idx']) < 39
    return {
        'm0_rows': len(m0), 'new_oos_rows': len(u), 'both': int((mg['_merge'] == 'both').sum()),
        'only_m0': len(only_m0), 'only_m0_ipo_rule': int(ipo.sum()), 'only_m0_other': int((~ipo).sum()),
        'only_new': len(only_new),
        'only_new_no_label': int(only_new['ret_label'].isna().sum()),
        'only_new_with_missing_days_in_window': int((only_new['n_obs_in_win'] < 20).sum()),
        'only_new_share_of_new': float(len(only_new) / len(u)),
    }


def make_figure(seed=3) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y = 2015
    old_lbl = pd.read_feather(os.path.join(OLD_IMG_DIR, fn(y, '_labels.feather')), columns=['StockID', 'ending_date'])
    old_lbl['StockID'] = old_lbl['StockID'].astype(str).str.zfill(6)
    new_lbl = pd.read_feather(os.path.join(NEW_IMG_DIR, fn(y, '_labels.feather')),
                              columns=['StockID', 'ending_date', 'n_obs_in_win'])
    mg = new_lbl.reset_index().merge(old_lbl.reset_index(), on=['StockID', 'ending_date'], suffixes=('_new', '_old'))
    rng = np.random.default_rng(seed)
    pick = pd.concat([mg[mg['n_obs_in_win'] == 20].sample(3, random_state=seed),
                      mg[mg['n_obs_in_win'] < 20].sample(1, random_state=seed)])
    n_old = os.path.getsize(os.path.join(OLD_IMG_DIR, fn(y, '_images.dat'))) // 3840
    n_new = len(new_lbl)
    oi = np.memmap(os.path.join(OLD_IMG_DIR, fn(y, '_images.dat')), dtype='uint8', mode='r', shape=(n_old, 64, 60))
    ni = np.memmap(os.path.join(NEW_IMG_DIR, fn(y, '_images.dat')), dtype='uint8', mode='r', shape=(n_new, 64, 60))
    fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 7))
    for j, (_, r) in enumerate(pick.iterrows()):
        for i, (arr, title) in enumerate(((oi[r['index_old']], '旧图像（无开收盘）'), (ni[r['index_new']], '干净版本'))):
            ax = axes[i, j]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}\n{r['StockID']} {r['ending_date'].date()} n_obs={r['n_obs_in_win']}", fontsize=8)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    _ = rng
    return FIG


def main() -> int:
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    os.makedirs(OUT_DIR, exist_ok=True)
    uni = pd.read_feather(ASHARE_UNIVERSE_PATH)
    grid = load_grid(ASHARE_PANEL_PATH)
    res = {}
    res['A_regression'] = check_regression(grid, uni)
    print(json.dumps(res['A_regression'], indent=2))
    res['B_keys'] = check_keys(uni)
    print(json.dumps(res['B_keys'], indent=2))
    res['C_pixels'] = check_pixels()
    print(json.dumps(res['C_pixels'], indent=2))
    res['D_st_onset'] = check_st_onset(uni)
    print(json.dumps(res['D_st_onset'], indent=2))
    res['E_vs_m0_oos_keys'] = check_vs_m0(uni, grid)
    print(json.dumps(res['E_vs_m0_oos_keys'], indent=2))
    del grid
    res['F_figure'] = make_figure()
    with open(os.path.join(OUT_DIR, 'clean_inputs_check.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print('写出', os.path.join(OUT_DIR, 'clean_inputs_check.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
