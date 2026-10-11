# -*- coding: utf-8 -*-
"""
check_clean_inputs.py - 干净版本数据与输入的验收（plan 阶段 2、3）

  A0. 样本键：不调用 build_universe，用稠密网格独立重算 universe（键集合、n_obs_in_win、ret_label、is_bj）
  A. 特征：抽样 4 万行，直接由面板行（按窗口 20 个市场日合并）独立重算 140 维特征，与 X_is / X_oos 比对；
     另报告收盘价路径与旧数据商面板的差异（换了数据源，只报告）
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
    ASHARE_COUNTRY, ASHARE_PANEL_PATH, ASHARE_STOCK_INFO_PATH, ASHARE_STOCKS_SAVEPATH, ASHARE_UNIVERSE_PATH, CLEAN_INPUTS_DIR,
    PRE_CLEAN_WORK_DIR,
)
from Data.clean_window import FEATURE_NAMES, load_grid  # noqa: E402
from Data.dgp_config import get_dataset_dir, get_dataset_filename  # noqa: E402
from Data.ohlc_render import layout  # noqa: E402

OUT_DIR = os.path.join(CLEAN_INPUTS_DIR, 'checks')
FIG = os.path.join(PROJECT_ROOT, 'docs', 'figures', 'clean_render_compare.png')
OLD_PANEL = os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_processed_data', 'ashare_ret.feather')
OLD_IMG_DIR = os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_stocks_dataset', 'I20R5', 'stocks_CHN', 'dataset_all')
M0_PRED = os.path.join(PRE_CLEAN_WORK_DIR, 'paper', 'E2_matched_is_ind_chrono_ric', 'preds', 'pred_cnn_oos.csv')
NEW_IMG_DIR = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5_clean'))
LAY = layout(20)
YEARS = list(range(2000, 2027))


def fn(year, suffix):
    return get_dataset_filename(20, 'week', True, [20], year, suffix)


def check_universe_independent(uni) -> dict:
    """Recompute the universe from the panel with a dense (market day x stock) grid instead of build_universe's
    sorted-key search: signal dates, the five sample conditions, n_obs_in_win and ret_label. Delisting consolidation
    is folded into the ST grid: for stocks whose name contains 退, the trading days after the last untraded market
    day before the final trade, if at most 30."""
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Close', 'is_ST', 'Vol_raw', 'Markettype'])
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    stocks = pd.Index(p['StockID'].astype(str).unique()).sort_values()
    di, si = cal.get_indexer(p['Date']), stocks.get_indexer(p['StockID'].astype(str))
    shape = (len(cal), len(stocks))
    C = np.full(shape, np.nan)
    C[di, si] = p['Close'].to_numpy()
    st = np.zeros(shape, dtype=bool)
    st[di, si] = p['is_ST'].to_numpy() == 1
    zero = np.zeros(shape, dtype=bool)
    zero[di, si] = p['Vol_raw'].to_numpy() <= 0
    mt = np.zeros(shape, dtype=np.int8)
    mt[di, si] = p['Markettype'].to_numpy()
    del p
    traded = np.isfinite(C)
    first = np.where(traded.any(axis=0), traded.argmax(axis=0), len(cal))
    info = pd.read_feather(ASHARE_STOCK_INFO_PATH, columns=['StockID', 'name'])
    tui = set(info.loc[info['name'].fillna('').str.contains('退'), 'StockID'].astype(str))
    for j in np.flatnonzero(stocks.isin(tui)):
        col = traded[:, j]
        last = len(col) - 1 - col[::-1].argmax()
        gaps = np.flatnonzero(~col[:last + 1])
        start = gaps[-1] + 1 if len(gaps) else 0
        if last - start + 1 <= 30:
            st[start:last + 1, j] = True
    iso = cal.isocalendar()
    wk = (iso['year'] * 100 + iso['week']).to_numpy()
    sig = np.flatnonzero(np.r_[wk[1:] != wk[:-1], True])
    ctr = np.vstack([np.zeros((1, len(stocks)), dtype=np.int32), np.cumsum(traded, axis=0, dtype=np.int32)])
    cz = np.vstack([np.zeros((1, len(stocks)), dtype=np.int32), np.cumsum(zero, axis=0, dtype=np.int32)])
    rows = []
    for j, t in enumerate(sig[:-1]):
        y = cal[t].year
        if y < 2000:
            continue
        lo = max(t - 19, 0)
        n_obs = ctr[t + 1] - ctr[lo]
        nz = cz[t + 1] - cz[lo]
        ok = traded[t] & (C[t] > 0) & ~st[t] & (first <= t - 39) & (nz == 0)
        s_idx = np.flatnonzero(ok)
        nxt = sig[j + 1]
        rows.append(pd.DataFrame({'StockID': stocks[s_idx], 'ending_date': cal[t], 'n_obs_in_win': n_obs[s_idx],
                                  'ret_label': C[nxt, s_idx] / C[t, s_idx] - 1, 'mt': mt[t, s_idx]}))
    ind = pd.concat(rows, ignore_index=True)
    u = uni[['StockID', 'ending_date', 'n_obs_in_win', 'ret_label', 'is_bj']].astype({'StockID': str})
    mg = u.merge(ind, on=['StockID', 'ending_date'], how='outer', indicator=True, suffixes=('', '_ind'))
    both = mg['_merge'] == 'both'
    lab_eq = np.isclose(mg.loc[both, 'ret_label'], mg.loc[both, 'ret_label_ind'], rtol=0, atol=1e-12, equal_nan=True)
    return {
        'universe_rows': len(u), 'independent_rows': len(ind),
        'only_universe': int((mg['_merge'] == 'left_only').sum()), 'only_independent': int((mg['_merge'] == 'right_only').sum()),
        'n_obs_equal_share': float((mg.loc[both, 'n_obs_in_win'] == mg.loc[both, 'n_obs_in_win_ind']).mean()),
        'ret_label_equal_share': float(lab_eq.mean()),
        'is_bj_equal_markettype_64_share': float(((mg.loc[both, 'is_bj'] == 1) == (mg.loc[both, 'mt'] == 64)).mean()),
    }


def check_features_independent(uni, n_sample=40_000, seed=0) -> dict:
    """Recompute the 140 features of sampled universe rows directly from panel rows (pandas merge on the 20 market
    days of the window), then compare with X_is / X_oos. Also report how far the Close path is from the pre-clean
    vendor panel (not a pass criterion: different data source)."""
    rng = np.random.default_rng(seed)
    uni = uni.reset_index(drop=True)
    is_rows = np.flatnonzero(uni['year'].to_numpy() < 2008)
    pick = np.sort(rng.choice(len(uni), n_sample, replace=False))
    s = uni.iloc[pick][['StockID', 'ending_date', 't_idx']].astype({'StockID': str}).reset_index(drop=True)
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'MA20', 'Vol'])
    p['StockID'] = p['StockID'].astype(str)
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    win = s.loc[s.index.repeat(20)].reset_index(names='k')
    win['pos'] = np.tile(np.arange(20), len(s))
    win['Date'] = cal[win['t_idx'].to_numpy() - 19 + win['pos'].to_numpy()]
    win = win.merge(p, on=['StockID', 'Date'], how='left')
    del p
    win = win.sort_values(['k', 'pos'])
    a = {c: win[c].to_numpy().reshape(len(s), 20) for c in ('Open', 'High', 'Low', 'Close', 'MA20', 'Vol')}
    valid = np.isfinite(a['Close'])
    c0 = np.array([row[v][0] for row, v in zip(a['Close'], valid)])[:, None]
    O, H, L, Cc, MA = (a[c] / c0 for c in ('Open', 'High', 'Low', 'Close', 'MA20'))
    H, L = np.fmax(H, np.fmax(O, Cc)), np.fmin(L, np.fmin(O, Cc))
    lo = np.where(valid, np.fmin(L, O), np.inf).min(axis=1, keepdims=True)
    hi = np.where(valid, np.fmax(H, O), -np.inf).max(axis=1, keepdims=True)
    flat = (hi - lo) < 1e-10
    rng_ = np.maximum(hi - lo, 1e-10)
    n = lambda x: (x - lo) / rng_  # noqa: E731
    Cn = pd.DataFrame(np.where(valid, n(Cc), np.nan)).T.ffill().bfill().T.to_numpy()
    exp = np.stack([np.where(valid, n(O), Cn), np.where(valid, n(H), Cn), np.where(valid, n(L), Cn), Cn,
                    np.where(valid & np.isfinite(MA) & ~flat, n(MA), Cn),
                    np.where(valid, a['Vol'], 0) / np.nanmax(np.where(valid, a['Vol'], np.nan), axis=1, keepdims=True),
                    valid.astype(float)], axis=2)
    fd = os.path.join(CLEAN_INPUTS_DIR, 'features140')
    X_is = np.load(os.path.join(fd, 'X_is.npy'), mmap_mode='r')
    X_oos = np.load(os.path.join(fd, 'X_oos.npy'), mmap_mode='r')
    n_is = len(is_rows)
    got = np.stack([np.asarray(X_is[i] if i < n_is else X_oos[i - n_is]) for i in pick]).reshape(len(s), 20, 7)
    d = np.abs(got.astype(np.float64) - exp)
    scale = np.maximum(1.0, np.abs(exp))
    out = {'n_sample': n_sample, 'max_abs_diff_by_channel': dict(zip(FEATURE_NAMES, d.max(axis=(0, 1)).round(8).tolist())),
           'share_within_float32_tol': float((d <= 1e-6 * scale).mean()),
           'max_abs_value_by_channel': dict(zip(FEATURE_NAMES, np.abs(exp).max(axis=(0, 1)).round(4).tolist())),
           'flat_range_windows': int(flat.sum())}

    if os.path.exists(OLD_PANEL):
        old = pd.read_feather(OLD_PANEL, columns=['Date', 'StockID', 'Close'])
        old['StockID'] = old['StockID'].astype(str).str.zfill(6)
        w2 = win[['k', 'pos', 'StockID', 'Date']].merge(old, on=['StockID', 'Date'], how='left').sort_values(['k', 'pos'])
        oc = w2['Close'].to_numpy().reshape(len(s), 20)
        full = valid.all(axis=1) & np.isfinite(oc).all(axis=1)
        dev = np.abs(oc[full] / oc[full][:, :1] - Cc[full]).max(axis=1)
        out['vs_pre_clean_vendor_close_path'] = {'windows': int(full.sum()), 'share_max_dev_lt_1e-3': float((dev < 1e-3).mean()),
                                                 'median_max_dev': float(np.median(dev))}
    return out


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
    res = {}
    res['A0_universe_independent'] = check_universe_independent(uni)
    print(json.dumps(res['A0_universe_independent'], indent=2))
    res['A_features_independent'] = check_features_independent(uni)
    print(json.dumps(res['A_features_independent'], indent=2))
    grid = load_grid(ASHARE_PANEL_PATH)
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
