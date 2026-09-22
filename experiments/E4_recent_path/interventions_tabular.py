# -*- coding: utf-8 -*-
"""MLP/HistGB column-index interventions on FEAT_180."""
from __future__ import annotations

import numpy as np

from experiments.E4_recent_path.paths import (
    COL_C, COL_H, COL_L, COL_O, COL_UP, COL_UV, COL_VN,
    PER_DAY, RECENT5, WINDOW,
)


def _expand(day_cols: list[int], days: list[int] | None = None) -> np.ndarray:
    days = list(range(WINDOW)) if days is None else days
    out = []
    for t in days:
        base = t * PER_DAY
        out.extend(base + c for c in day_cols)
    return np.asarray(out, dtype=int)


def all_cols() -> np.ndarray:
    return _expand(list(range(PER_DAY)))


def drop_day(day: int) -> np.ndarray:
    """Keep all cols except the 9 cols of `day`."""
    keep_days = [t for t in range(WINDOW) if t != day]
    return _expand(list(range(PER_DAY)), keep_days)


def drop_recent5_cols(feat_ids: list[int]) -> np.ndarray:
    """Drop given feature ids on recent5 days only; keep all other days full."""
    drop = set(_expand(feat_ids, RECENT5).tolist())
    return np.asarray([i for i in all_cols().tolist() if i not in drop], dtype=int)


TAB_BLOCKS = {
    'volume': drop_recent5_cols([COL_VN]),
    'wick': drop_recent5_cols([COL_H, COL_L]),
    'direction': drop_recent5_cols([COL_UP, COL_UV]),
    # Return: drop C and delta on recent5 (close path / returns)
    'return': drop_recent5_cols([COL_C, 5]),  # 5 = delta
    # Gap: drop O on recent5 (open relative to prior close)
    'gap': drop_recent5_cols([COL_O]),
    # Intraday body proxies: O,C,uP already partly; drop O,C,uP on recent5
    'intraday': drop_recent5_cols([COL_O, COL_C, COL_UP]),
}


def lag_cols(day: int) -> np.ndarray:
    return drop_day(day)


def block_cols(name: str) -> np.ndarray:
    if name not in TAB_BLOCKS:
        raise ValueError(name)
    return TAB_BLOCKS[name]
