# -*- coding: utf-8 -*-
"""
2m1_rerender_legacy_windows.py — 阶段 4a 的 M1 输入：旧数据流程 + Jiang 渲染器。

按旧 labels 文件的样本键（StockID, ending_date）在旧面板上复原旧窗口：
  - 窗口 = 该股票 ending_date 当日或之前最近的 20 个交易行（旧代码 available.tail(20)）
  - 价格路径：首日 Close=1，按（裁剪过的）Ret 累乘；O/H/L 按 Close_adj/Close 缩放，H/L 包住 O/C
  - 均线：调整后 Close 的 rolling(20, min_periods=1)（窗口内累计均线）
  - 成交量：原始 Vol
先用旧画法（bar_style='legacy_candle'）抽样重画并与旧 .dat 逐字节比对，必须完全一致；
再用 render_ohlc_batch 重画全部样本，写入 I20R5_rendfix，labels 原样复制；同一抽样与 Jiang 官方 DrawOHLC
（Data/jkx_chart_library.py）逐像素比对，必须完全一致。

用法:
  python ashare_train/2m1_rerender_legacy_windows.py
  python ashare_train/2m1_rerender_legacy_windows.py --years 2001 2002 --force
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import ASHARE_COUNTRY, OLD_DATA_WORK_DIR, PRE_CLEAN_WORK_DIR, MA_LAGS, VOLUME_BAR
from Data.dgp_config import get_dataset_dir, get_dataset_filename
from Data.chart_library import DrawOHLC
from Data.jkx_chart_library import DrawOHLC as JkxDrawOHLC
from Data.ohlc_render import render_ohlc_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

WS, FREQ = 20, 'week'
YEARS = list(range(2000, 2026))
OLD_PANEL = os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_processed_data', 'ashare_ret.feather')
OLD_DIR = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_stocks_dataset', 'I20R5'))
OUT_DIR = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(OLD_DATA_WORK_DIR, 'data', 'ashare_stocks_dataset',
                                                                     'I20R5_rendfix'))
PANEL_COLS = ['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Vol', 'Ret']


def load_old_panel() -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    df = pd.read_feather(OLD_PANEL, columns=PANEL_COLS)
    df['Date'] = pd.to_datetime(df['Date'])
    df['StockID'] = df['StockID'].astype('category')
    df = df.sort_values(['StockID', 'Date'], kind='mergesort').reset_index(drop=True)
    cats = df['StockID'].cat.categories
    key = df['StockID'].cat.codes.to_numpy(np.int64) * 10**6 + _day(df['Date'])
    if not (np.diff(key) > 0).all():
        raise RuntimeError('旧面板存在重复的 (StockID, Date)')
    return df, key, cats


def _day(dates) -> np.ndarray:
    return (pd.DatetimeIndex(dates).values.astype('datetime64[D]').astype(np.int64))


def window_rows(labels: pd.DataFrame, key: np.ndarray, codes: np.ndarray, cats: pd.Index) -> np.ndarray:
    sid = cats.get_indexer(labels['StockID'].astype(str))
    if (sid < 0).any():
        raise RuntimeError(f'{(sid < 0).sum()} 个样本的 StockID 不在旧面板中')
    end = np.searchsorted(key, sid.astype(np.int64) * 10**6 + _day(labels['ending_date']), side='right')
    rows = end[:, None] - WS + np.arange(WS)[None, :]
    if (rows < 0).any() or (codes[rows] != sid[:, None]).any():
        raise RuntimeError('有样本的窗口不足 20 个交易行或跨越股票')
    return rows


def legacy_window_arrays(panel: pd.DataFrame, rows: np.ndarray) -> dict:
    get = lambda c: panel[c].to_numpy(np.float64)[rows]
    O, H, L, C, V, R = (get(c) for c in ('Open', 'High', 'Low', 'Close', 'Vol', 'Ret'))
    growth = np.concatenate([np.ones((len(C), 1)), 1.0 + R[:, 1:]], axis=1)
    Ca = np.cumprod(growth, axis=1)
    ratio = np.where(np.abs(C) > 1e-10, Ca / C, 1.0)
    Oa, Ha, La = O * ratio, H * ratio, L * ratio
    Ha = np.fmax(np.fmax(Ha, Oa), Ca)
    La = np.fmin(np.fmin(La, Oa), Ca)
    MA = pd.DataFrame(Ca.T).rolling(MA_LAGS[0], min_periods=1).mean().to_numpy().T
    return {'Open': Oa, 'High': Ha, 'Low': La, 'Close': Ca, 'Vol': V, 'ma20': MA}


def legacy_draw(w: dict, i: int) -> np.ndarray:
    df = pd.DataFrame({k: v[i] for k, v in w.items()})
    return DrawOHLC(df, window_size=WS, ma_cols=['ma20'], has_volume_bar=VOLUME_BAR,
                    chart_type='bar', bar_style='legacy_candle').draw_to_array()


def official_draw(w: dict, i: int) -> np.ndarray | None:
    df = pd.DataFrame({k: v[i] for k, v in w.items()})
    df['Vol'] = df['Vol'].where(df['Vol'] > 0)
    img = JkxDrawOHLC(df, has_volume_bar=VOLUME_BAR, ma_lags=MA_LAGS).draw_image()
    return None if img is None else np.asarray(img, dtype=np.uint8).reshape(-1)


def process_year(year: int, panel, key, codes, cats, n_check: int, rng, force: bool) -> dict | None:
    fn_img = get_dataset_filename(WS, FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
    fn_lbl = get_dataset_filename(WS, FREQ, VOLUME_BAR, MA_LAGS, year, '_labels.feather')
    old_img, old_lbl = os.path.join(OLD_DIR, fn_img), os.path.join(OLD_DIR, fn_lbl)
    out_img, out_lbl = os.path.join(OUT_DIR, fn_img), os.path.join(OUT_DIR, fn_lbl)
    if not os.path.exists(old_lbl):
        logger.warning(f'{year}: 缺少旧 labels，跳过')
        return None
    if os.path.exists(out_img) and os.path.exists(out_lbl) and not force:
        logger.info(f'{year}: 已存在，跳过')
        return None

    labels = pd.read_feather(old_lbl)
    n = len(labels)
    old = np.memmap(old_img, dtype=np.uint8, mode='r', shape=(n, 64 * 3 * WS))
    w = legacy_window_arrays(panel, window_rows(labels, key, codes, cats))

    check = np.sort(rng.choice(n, size=min(n_check, n), replace=False))
    same = np.array([np.array_equal(legacy_draw(w, i), old[i]) for i in check])
    if not same.all():
        bad = check[~same][:5]
        raise RuntimeError(f'{year}: 旧画法重画与旧 .dat 不一致 {int((~same).sum())}/{len(check)}，'
                           f'例如行 {bad.tolist()}')

    img = render_ohlc_batch(w['Open'], w['High'], w['Low'], w['Close'], w['ma20'], w['Vol'], has_volume=VOLUME_BAR)
    off = np.array([np.array_equal(official_draw(w, i), img[i].reshape(-1)) for i in check])
    if not off.all():
        bad = check[~off][:5]
        raise RuntimeError(f'{year}: 新图像与官方 DrawOHLC 不一致 {int((~off).sum())}/{len(check)}，例如行 {bad.tolist()}')
    mm = np.memmap(out_img, dtype=np.uint8, mode='w+', shape=(n, img.shape[1] * img.shape[2]))
    mm[:] = img.reshape(n, -1)
    mm.flush()
    del mm
    shutil.copy2(old_lbl, out_lbl)

    lo = np.minimum.reduce([w['Low'], w['Open'], w['Close']]).min(axis=1)
    hi = np.maximum.reduce([w['High'], w['Open'], w['Close']]).max(axis=1)
    ma_bound = float(((w['ma20'].max(axis=1) > hi) | (w['ma20'].min(axis=1) < lo)).mean())
    pix_diff = float((img.reshape(n, -1)[check] != old[check]).mean())
    logger.info(f'{year}: {n} 个样本，抽查 {len(check)} 个旧画法、官方画法均完全一致；均线决定纵轴上下界的窗口占比 '
                f'{ma_bound:.4f}；新旧图像像素差异占比 {pix_diff:.4f}')
    return {'year': year, 'n': n, 'n_checked': int(len(check)), 'legacy_identical': True, 'official_identical': True,
            'ma_sets_price_bound_share': ma_bound, 'new_vs_old_pixel_diff_share': pix_diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=YEARS)
    ap.add_argument('--n-check', type=int, default=500)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f'旧面板 {OLD_PANEL}\n旧图像 {OLD_DIR}\n输出 {OUT_DIR}')
    panel, key, cats = load_old_panel()
    codes = panel['StockID'].cat.codes.to_numpy(np.int64)
    rng = np.random.default_rng(20261011)

    rows = []
    for y in args.years:
        r = process_year(y, panel, key, codes, cats, args.n_check, rng, args.force)
        if r is not None:
            rows.append(r)

    check_path = os.path.join(OUT_DIR, 'rerender_check.json')
    prev = json.load(open(check_path, encoding='utf-8')) if os.path.exists(check_path) else {}
    prev.update({str(r['year']): r for r in rows})
    with open(check_path, 'w', encoding='utf-8') as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)
    logger.info(f'写出 {check_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
