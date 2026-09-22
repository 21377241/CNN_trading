# -*- coding: utf-8 -*-
"""Recent-5 path-shape features from FEAT_180 (vectorized)."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E2_matched_tabular.e2a_layout import FEAT_180
from experiments.E3_shared_specific.utils import norm_stock_id
from experiments.E4_recent_path.paths import COL_C, COL_H, COL_L, COL_O, COL_VN, OUT_ROOT, PER_DAY, RECENT5, WINDOW

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e4_path')

EPS = 1e-12
PATH_COLS = [
    'CumRet5', 'LateEarly', 'PathVol', 'UpCount5', 'MaxDD5', 'Reversal5', 'Monotonicity',
    'WickMean5', 'GapMean5', 'VolEndOverMean5',
]


def _path_from_closes(c: np.ndarray) -> dict[str, np.ndarray]:
    """c: (n, 20) closes."""
    c5 = c[:, 15:]  # t-4..t
    c14 = c[:, 14]
    cum = c5[:, -1] / (c14 + EPS) - 1.0
    early = c5[:, 1] / (c14 + EPS) - 1.0  # through t-3
    late = c5[:, -1] / (c5[:, 1] + EPS) - 1.0  # t-3→t
    late_early = late - early
    rets = np.diff(np.column_stack([c14, c5]), axis=1) / (np.column_stack([c14, c5[:, :-1]]) + EPS)
    path_vol = np.nanstd(rets, axis=1, ddof=1)
    up_count = (rets > 0).sum(axis=1).astype(float)
    # max drawdown within recent5 path (from running peak of relative level)
    lvl = np.column_stack([np.ones(len(c)), c5 / (c14[:, None] + EPS)])
    peak = np.maximum.accumulate(lvl, axis=1)
    dd = lvl / (peak + EPS) - 1.0
    maxdd = dd.min(axis=1)
    # reversal: last day ret vs first half cum
    rev = rets[:, -1] - rets[:, :2].mean(axis=1)
    # monotonicity: corr(day_index, close_level)
    x = np.arange(5, dtype=float)
    xc = x - x.mean()
    y = c5
    yc = y - y.mean(axis=1, keepdims=True)
    num = (yc * xc).sum(axis=1)
    den = np.sqrt((yc ** 2).sum(axis=1) * (xc ** 2).sum() + EPS)
    mono = num / den
    return {
        'CumRet5': cum,
        'LateEarly': late_early,
        'PathVol': path_vol,
        'UpCount5': up_count,
        'MaxDD5': maxdd,
        'Reversal5': rev,
        'Monotonicity': mono,
    }


def build_pathshape(out_path: str | None = None, chunk: int = 400_000) -> pd.DataFrame:
    X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))
    n = len(meta)
    assert X.shape[0] == n
    parts = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = np.asarray(X[start:end], dtype=np.float64).reshape(-1, WINDOW, PER_DAY)
        o, h, l, c, vn = xb[:, :, COL_O], xb[:, :, COL_H], xb[:, :, COL_L], xb[:, :, COL_C], xb[:, :, COL_VN]
        feat = _path_from_closes(c)
        rng = h[:, RECENT5] - l[:, RECENT5]
        body_hi = np.maximum(o[:, RECENT5], c[:, RECENT5])
        body_lo = np.minimum(o[:, RECENT5], c[:, RECENT5])
        wick = ((h[:, RECENT5] - body_hi) + (body_lo - l[:, RECENT5])) / (rng + EPS)
        feat['WickMean5'] = wick.mean(axis=1)
        gap = o[:, RECENT5] / (c[:, [14, 15, 16, 17, 18]] + EPS) - 1.0
        feat['GapMean5'] = gap.mean(axis=1)
        feat['VolEndOverMean5'] = vn[:, 19] / (vn[:, RECENT5].mean(axis=1) + EPS)
        df = pd.DataFrame(feat)
        df['StockID'] = meta['StockID'].values[start:end]
        df['ending_date'] = meta['ending_date'].values[start:end]
        parts.append(df)
        logger.info(f'  pathshape {end:,}/{n:,}')
    g = pd.concat(parts, ignore_index=True)
    g['StockID'] = norm_stock_id(g['StockID'])
    g['ending_date'] = pd.to_datetime(g['ending_date'])
    out_path = out_path or os.path.join(OUT_ROOT, 'pathshape.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g.to_parquet(out_path, index=False)
    logger.info(f'pathshape -> {out_path}')
    return g


if __name__ == '__main__':
    build_pathshape()
