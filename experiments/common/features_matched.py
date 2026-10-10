# -*- coding: utf-8 -*-
"""
Matched numeric features, aligned with the chart image (clean_spec section 3.4).

140 = 20 market days × [O, H, L, C, MA20, V, has_trade], day-major.
  prices (incl. MA20): (p - window low) / (window high - window low), same scaling as the image's price axis;
  V: Vol_adj / window max Vol_adj; missing days: previous valid close, V = 0, has_trade = 0.
Channels 0-5 keep the order of the former 120D features (O, H, L, C, MA20, V); channel 6 is new.

Windows come from trend_code_submit/Data/clean_window.py, the same module that renders the images.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'trend_submit', 'trend_code_submit')
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from Data.clean_window import (  # noqa: E402
    FEATURE_NAMES, PER_DAY, WINDOW, Grid, close0_path, features_140, gather_windows, load_grid,
)

__all__ = ['FEATURE_NAMES', 'PER_DAY', 'WINDOW', 'feature_dim', 'build_feature_matrix', 'load_grid', 'channel']


def feature_dim(window_size: int = WINDOW) -> int:
    return window_size * PER_DAY


def channel(name: str) -> int:
    return FEATURE_NAMES.index(name)


def build_feature_matrix(universe: pd.DataFrame, grid: Grid, batch: int = 50_000) -> np.ndarray:
    """universe 须含 StockID、t_idx；返回 float32 (N, 140)，行序与 universe 相同。"""
    t_idx = universe['t_idx'].to_numpy()
    s_idx = grid.stock_index(universe['StockID'].astype(str))
    X = np.empty((len(universe), feature_dim()), dtype=np.float32)
    for a in range(0, len(universe), batch):
        b = min(len(universe), a + batch)
        X[a:b] = features_140(close0_path(gather_windows(grid, t_idx[a:b], s_idx[a:b]))).reshape(b - a, -1)
    return X
