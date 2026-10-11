# -*- coding: utf-8 -*-
"""
2_generate_ashare_chart_data.py - 干净版本 I20R5 图像与 140 维数值特征（clean_spec 第 3 节）

样本键完全由 universe_weekly.feather 决定；图像与特征来自同一个窗口（Data/clean_window.py），行序相同。

输出：
  {ASHARE_STOCKS_SAVEPATH}/I20R5_clean/stocks_CHN/dataset_all/20d_week_has_vb_[20]_ma_{year}_images.dat
  {ASHARE_STOCKS_SAVEPATH}/I20R5_clean/stocks_CHN/dataset_all/20d_week_has_vb_[20]_ma_{year}_labels.feather
  {CLEAN_INPUTS_DIR}/features140/X_is.npy, X_oos.npy, meta_is.feather, meta_oos.feather
  {CLEAN_INPUTS_DIR}/features140/generation_check.json

labels / meta 列：StockID, ending_date, ret_val, binary_label, label_raw, MarketCap, split, embargo,
  n_obs_in_win, board, next_date, year
  binary_label：label_raw；但 ret_val 缺失（label_raw = -1）或 embargo = 1 时为 -99（训练与验证中剔除）。
  OOS 预测不按 binary_label 过滤。

用法:
  python ashare_train/2_generate_ashare_chart_data.py [--years 2008 2009] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (  # noqa: E402
    ASHARE_COUNTRY, ASHARE_PANEL_PATH, ASHARE_STOCKS_SAVEPATH, ASHARE_UNIVERSE_PATH, CLEAN_INPUTS_DIR,
)
from Data.clean_window import (  # noqa: E402
    WINDOW, close0_path, features_140, gather_windows, load_grid, ma_out_of_range_share, render_window_images,
    universe_meta,
)
from Data.dgp_config import get_dataset_dir, get_dataset_filename  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('gen_clean')

SPEC = 'I20R5_clean'
BATCH = 40_000
IMG_SIZE = 64 * 60
IS_YEARS = list(range(2000, 2008))
OOS_YEARS = list(range(2008, 2026))


def image_dir() -> str:
    return get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, SPEC))


def feature_dir() -> str:
    return os.path.join(CLEAN_INPUTS_DIR, 'features140')


def process_year(grid, u: pd.DataFrame, year: int, force: bool, features_only: bool = False) -> dict:
    d = image_dir()
    p_img = os.path.join(d, get_dataset_filename(WINDOW, 'week', True, [20], year, '_images.dat'))
    p_lbl = os.path.join(d, get_dataset_filename(WINDOW, 'week', True, [20], year, '_labels.feather'))
    p_x = os.path.join(feature_dir(), 'by_year', f'X_{year}.npy')
    meta = universe_meta(u)
    n = len(meta)
    if not force and all(os.path.exists(p) for p in (p_img, p_lbl, p_x)):
        logger.info(f'{year}: 已存在，跳过')
        return {'year': year, 'n': n, 'skipped': True}

    t_idx = u['t_idx'].to_numpy()
    s_idx = grid.stock_index(meta['StockID'])
    img_mm = None if features_only else np.memmap(p_img, dtype='uint8', mode='w+', shape=(n, IMG_SIZE))
    X = np.empty((n, WINDOW, 7), dtype=np.float32)
    oor_num = 0.0
    for a in range(0, n, BATCH):
        b = min(n, a + BATCH)
        p = close0_path(gather_windows(grid, t_idx[a:b], s_idx[a:b]))
        if img_mm is not None:
            img_mm[a:b] = render_window_images(p).reshape(b - a, IMG_SIZE)
        X[a:b] = features_140(p)
        oor_num += ma_out_of_range_share(p) * p['valid'].sum()
    if img_mm is not None:
        img_mm.flush()
        del img_mm
        meta.to_feather(p_lbl)
    np.save(p_x, X)
    n_valid_days = float(meta['n_obs_in_win'].sum())
    info = {
        'year': year, 'n': n,
        'trainable': int((meta['binary_label'] >= 0).sum()),
        'embargo': int(meta['embargo'].sum()),
        'n_obs_lt_20_share': float((meta['n_obs_in_win'] < 20).mean()),
        'ma_out_of_range_day_share': oor_num / max(n_valid_days, 1.0),
        'x_nonfinite': int((~np.isfinite(X)).sum()),
    }
    logger.info(f'{year}: {info}')
    return info


def concat_features(years: list, tag: str) -> None:
    fd = feature_dir()
    xs = [np.load(os.path.join(fd, 'by_year', f'X_{y}.npy'), mmap_mode='r') for y in years]
    n = sum(len(x) for x in xs)
    out = np.lib.format.open_memmap(os.path.join(fd, f'X_{tag}.npy'), mode='w+', dtype=np.float32,
                                    shape=(n, WINDOW * 7))
    pos = 0
    for x in xs:
        out[pos:pos + len(x)] = np.asarray(x).reshape(len(x), -1)
        pos += len(x)
    out.flush()
    del out
    d = image_dir()
    metas = [pd.read_feather(os.path.join(d, get_dataset_filename(WINDOW, 'week', True, [20], y, '_labels.feather')))
             for y in years]
    pd.concat(metas, ignore_index=True).to_feather(os.path.join(fd, f'meta_{tag}.feather'))
    logger.info(f'{tag}: X {n} × {WINDOW * 7}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--features-only', action='store_true', help='只重写 140 维特征，不动图像和 labels')
    args = ap.parse_args(argv)
    years = args.years or (IS_YEARS + OOS_YEARS)
    os.makedirs(image_dir(), exist_ok=True)
    os.makedirs(os.path.join(feature_dir(), 'by_year'), exist_ok=True)

    t0 = time.time()
    uni = pd.read_feather(ASHARE_UNIVERSE_PATH)
    logger.info('加载稠密网格…')
    grid = load_grid(ASHARE_PANEL_PATH)
    logger.info(f'网格 {grid.C.shape}，用时 {time.time() - t0:.0f}s')
    infos = []
    for y in years:
        u = uni[uni['year'] == y].reset_index(drop=True)
        infos.append(process_year(grid, u, y, args.force or args.features_only, args.features_only))
    del grid

    if set(IS_YEARS + OOS_YEARS) <= set(years) or args.years is None:
        concat_features(IS_YEARS, 'is')
        concat_features(OOS_YEARS, 'oos')
    chk_path = os.path.join(feature_dir(), 'generation_check.json')
    old = json.load(open(chk_path, encoding='utf-8')) if os.path.exists(chk_path) else {}
    old.update({str(i['year']): i for i in infos})
    with open(chk_path, 'w', encoding='utf-8') as f:
        json.dump(old, f, indent=2, ensure_ascii=False)
    logger.info(f'完成，用时 {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
