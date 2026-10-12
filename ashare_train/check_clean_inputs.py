# -*- coding: utf-8 -*-
"""
check_clean_inputs.py - 干净版本数据与输入的验收（plan 阶段 2、3）

  A0. 样本键：不调用 build_universe，用稠密网格独立重算 universe（键集合、n_obs_in_win、ret_label、is_bj）
  A. 特征：抽样 4 万行，直接由面板行（按窗口 20 个市场日合并）独立重算 140 维特征，与 X_is / X_oos 比对
  B. 样本键：universe = 图像 labels（逐年拼接）= 特征 meta（行序相同）
  C. 像素一致：由 140 维特征推出的开/收盘刻度行、高低竖线、成交量柱高与图像像素一致
  D. 持有期内被标为 ST 的股票-周出现在 OOS universe 中且有标签
  E. 官方渲染器：A 中独立重建的窗口交给 Jiang 官方 DrawOHLC（Data/jkx_chart_library.py）重画，与已存图像逐像素比对
只读 CSMAR 构建的产物；与旧数据的对账见 reconcile_pre_clean.py。

输出：WORK_SPACE/paper/clean_inputs/checks/clean_inputs_check.json
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
)
from Data.clean_window import FEATURE_NAMES  # noqa: E402
from Data.dgp_config import get_dataset_dir, get_dataset_filename  # noqa: E402
from Data.jkx_chart_library import DrawOHLC as JkxDrawOHLC  # noqa: E402
from Data.ohlc_render import layout  # noqa: E402

OUT_DIR = os.path.join(CLEAN_INPUTS_DIR, 'checks')
NEW_IMG_DIR = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5_clean'))
LAY = layout(20)
YEARS = list(range(2000, 2027))


def fn(year, suffix):
    return get_dataset_filename(20, 'week', True, [20], year, suffix)


def check_universe_independent(uni) -> dict:
    """Recompute the universe from the panel with a dense (market day x stock) grid instead of build_universe's
    sorted-key search: signal dates, the six sample conditions, n_obs_in_win and ret_label. Delisting consolidation
    is folded into the ST grid: for stocks whose name contains 退, the trading days after the last untraded market
    day before the final trade, if at most 30."""
    p = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Close', 'is_ST', 'Vol_raw', 'Markettype',
                                                    'Open', 'High', 'Low', 'MA20'])
    cal = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    stocks = pd.Index(p['StockID'].astype(str).unique()).sort_values()
    di, si = cal.get_indexer(p['Date']), stocks.get_indexer(p['StockID'].astype(str))
    shape = (len(cal), len(stocks))
    C = np.full(shape, np.nan)
    C[di, si] = p['Close'].to_numpy()
    px = p[['Open', 'High', 'Low', 'Close', 'MA20']]
    pmax, pmin = np.full(shape, np.nan), np.full(shape, np.nan)
    pmax[di, si] = px.max(axis=1).to_numpy()
    pmin[di, si] = px.min(axis=1).to_numpy()
    del px
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
        wmax, wmin = np.fmax.reduce(pmax[lo:t + 1], axis=0), np.fmin.reduce(pmin[lo:t + 1], axis=0)
        flat = (wmax - wmin) <= 1e-10 * wmax
        ok = traded[t] & (C[t] > 0) & ~st[t] & (first <= t - 39) & (nz == 0) & ~flat
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


def check_features_independent(uni, n_sample=40_000, n_official=10_000, seed=0) -> tuple[dict, dict]:
    """Recompute the 140 features of sampled universe rows directly from panel rows (pandas merge on the 20 market
    days of the window), then compare with X_is / X_oos. The first n_official rebuilt windows are also redrawn with
    the official DrawOHLC and compared with the stored images (check E)."""
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
    has_ma = valid & np.isfinite(MA)
    lo = np.fmin(np.where(valid, L, np.inf), np.where(has_ma, MA, np.inf)).min(axis=1, keepdims=True)
    hi = np.fmax(np.where(valid, H, -np.inf), np.where(has_ma, MA, -np.inf)).max(axis=1, keepdims=True)
    flat = ~(hi > lo)
    n = lambda x: (x - lo) / np.where(flat, 1.0, hi - lo)  # noqa: E731
    Cn = pd.DataFrame(np.where(valid, n(Cc), np.nan)).T.ffill().bfill().T.to_numpy()
    exp = np.stack([np.where(valid, n(O), Cn), np.where(valid, n(H), Cn), np.where(valid, n(L), Cn), Cn,
                    np.where(has_ma, n(MA), Cn),
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
           'flat_range_windows': int(flat.sum()),
           'ma_sets_price_bound_share': float(((np.where(has_ma, MA, -np.inf).max(axis=1) > np.where(valid, H, -np.inf).max(axis=1))
                                               | (np.where(has_ma, MA, np.inf).min(axis=1) < np.where(valid, L, np.inf).min(axis=1))).mean())}

    k = np.arange(min(n_official, n_sample))
    y_of = uni['year'].to_numpy()
    start = {y: int(np.flatnonzero(y_of == y)[0]) for y in np.unique(y_of[pick[k]])}
    imgs = {}
    same = np.zeros(len(k), dtype=bool)
    for j in k:
        y = int(y_of[pick[j]])
        if y not in imgs:
            m = int((y_of == y).sum())
            imgs[y] = np.memmap(os.path.join(NEW_IMG_DIR, fn(y, '_images.dat')), dtype='uint8', mode='r', shape=(m, 64, 60))
        df = pd.DataFrame({'Open': np.where(valid[j], O[j], np.nan), 'High': np.where(valid[j], H[j], np.nan),
                           'Low': np.where(valid[j], L[j], np.nan), 'Close': Cc[j], 'ma20': np.where(has_ma[j], MA[j], np.nan),
                           'Vol': np.where(valid[j] & (a['Vol'][j] > 0), a['Vol'][j], np.nan)})
        ref = JkxDrawOHLC(df, has_volume_bar=True, ma_lags=[20]).draw_image()
        same[j] = ref is not None and np.array_equal(np.asarray(ref, dtype=np.uint8), imgs[y][pick[j] - start[y]])
    official = {'n_compared': int(len(k)), 'pixel_identical_share': float(same.mean()),
                'mismatch_examples': s.loc[k[~same][:5], ['StockID', 'ending_date']].astype(str).values.tolist()}
    return out, official


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
        yp = lambda v: (ph - 1) - np.around(v * (ph - 1)).astype(int)  # noqa: E731

        def tick_ok(ii, v, col):
            y0 = yp(v)
            hit = im[ii, y0, col] == 255
            frac = (v * (ph - 1)) % 1
            near_half = np.abs(frac - 0.5) < 1e-4
            alt = np.clip(np.where(y0 == yp(v + 1e-3 / (ph - 1)), y0 + 1, y0 - 1), 0, ph - 1)
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
            vh = X[ii, t, 5] * (vb - vt + 1)
            got = (im[ii, vt:, 3 * t + 1] > 0).sum(axis=1)
            stats['vol'].append(got == np.maximum(1, np.around(vh)))
            stats.setdefault('vol_allowing_half_pixel_ties', []).append(
                (got == np.maximum(1, np.around(vh))) | ((np.abs(vh % 1 - 0.5) < 1e-4) & (np.abs(got - vh) < 0.51)))
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


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    uni = pd.read_feather(ASHARE_UNIVERSE_PATH)
    res = {}
    res['A0_universe_independent'] = check_universe_independent(uni)
    print(json.dumps(res['A0_universe_independent'], indent=2))
    res['A_features_independent'], res['E_official_renderer'] = check_features_independent(uni)
    print(json.dumps(res['A_features_independent'], indent=2))
    print(json.dumps(res['E_official_renderer'], indent=2, ensure_ascii=False))
    res['B_keys'] = check_keys(uni)
    print(json.dumps(res['B_keys'], indent=2))
    res['C_pixels'] = check_pixels()
    print(json.dumps(res['C_pixels'], indent=2))
    res['D_st_onset'] = check_st_onset(uni)
    print(json.dumps(res['D_st_onset'], indent=2))
    with open(os.path.join(OUT_DIR, 'clean_inputs_check.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print('写出', os.path.join(OUT_DIR, 'clean_inputs_check.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
