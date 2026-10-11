# -*- coding: utf-8 -*-
"""Phase-1 geometry features from 180-dim OOS cache + meta (vectorized)."""
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
from experiments.E3_shared_specific.paths import OUT_ROOT, PER_DAY, WINDOW
from experiments.E7_incremental.common import limit_flags_at

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_geom')

EPS = 1e-12


def build_geometry(out_path: str | None = None, chunk: int = 400_000) -> pd.DataFrame:
    X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))
    n = len(meta)
    assert X.shape[0] == n
    logger.info(f'computing geometry n={n:,} vectorized chunks={chunk}')

    parts = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = np.asarray(X[start:end], dtype=np.float64).reshape(-1, WINDOW, PER_DAY)
        o, h, l, c, vn = xb[:, :, 0], xb[:, :, 1], xb[:, :, 2], xb[:, :, 3], xb[:, :, 6]
        T = WINDOW - 1
        feat = {}
        for k in (1, 5, 10, 20):
            j = max(0, T - k)
            feat[f'r_{k}'] = c[:, T] / (c[:, j] + EPS) - 1.0
        rng = h[:, T] - l[:, T]
        feat['body_ratio'] = np.abs(c[:, T] - o[:, T]) / (rng + EPS)
        feat['upper_wick_ratio'] = (h[:, T] - np.maximum(o[:, T], c[:, T])) / (rng + EPS)
        feat['lower_wick_ratio'] = (np.minimum(o[:, T], c[:, T]) - l[:, T]) / (rng + EPS)
        feat['range_rel'] = rng / (c[:, T] + EPS)
        feat['v_end'] = vn[:, T]
        feat['v_end_over_mean'] = vn[:, T] / (vn.mean(axis=1) + EPS)
        rets = np.diff(c, axis=1) / (c[:, :-1] + EPS)
        feat['ret_std'] = np.nanstd(rets, axis=1, ddof=1)
        feat['overnight'] = o[:, T] / (c[:, T - 1] + EPS) - 1.0
        mcap = meta['MarketCap'].values[start:end].astype(float)
        feat['log_mcap'] = np.where(np.isfinite(mcap) & (mcap > 0), np.log(mcap), np.nan)
        df = pd.DataFrame(feat)
        df['StockID'] = meta['StockID'].values[start:end]
        df['ending_date'] = meta['ending_date'].values[start:end]
        if 'Ret' in meta.columns:
            df['Ret'] = meta['Ret'].values[start:end]
        parts.append(df)
        logger.info(f'  geometry {end:,}/{n:,}')

    g = pd.concat(parts, ignore_index=True)
    g['ending_date'] = pd.to_datetime(g['ending_date'])
    lim = limit_flags_at(g)
    g['hit_limit'] = (lim['close_up'].astype(bool) | lim['close_dn'].astype(bool)).astype(float)
    g.loc[(lim['no_limit'].isna() | lim['no_limit'].astype(bool)).to_numpy(), 'hit_limit'] = np.nan
    out_path = out_path or os.path.join(OUT_ROOT, 'geometry.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g.to_parquet(out_path, index=False)
    logger.info(f'geometry -> {out_path}')
    return g


if __name__ == '__main__':
    build_geometry()
