# -*- coding: utf-8 -*-
"""E2A helpers: 180-dim column layouts and chrono paths."""
from __future__ import annotations

import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOW = 20
PER_DAY_180 = 9  # O,H,L,C,MA,delta,Vn,uP,uV

# Per-day indices in 180-dim layout
IDX_OHLC_MA = [0, 1, 2, 3, 4]
IDX_DELTA = 5
IDX_VN = 6
IDX_UP = 7
IDX_UV = 8


def _expand_day_cols(day_cols: list[int], window: int = WINDOW) -> np.ndarray:
    out = []
    for t in range(window):
        base = t * PER_DAY_180
        out.extend(base + c for c in day_cols)
    return np.asarray(out, dtype=int)


COLS_120_FROM_180 = _expand_day_cols(IDX_OHLC_MA + [IDX_VN])  # 120
COLS_60_FROM_180 = _expand_day_cols([IDX_DELTA, IDX_UP, IDX_UV])  # 60
COLS_DELTA20 = _expand_day_cols([IDX_DELTA])  # 20
COLS_UP20 = _expand_day_cols([IDX_UP])
COLS_UV20 = _expand_day_cols([IDX_UV])
# 140 ablations: base 120 layout + one indicator block (7 per day)
COLS_140_DELTA = _expand_day_cols(IDX_OHLC_MA + [IDX_DELTA, IDX_VN])
COLS_140_UP = _expand_day_cols(IDX_OHLC_MA + [IDX_VN, IDX_UP])
COLS_140_UV = _expand_day_cols(IDX_OHLC_MA + [IDX_VN, IDX_UV])

ABLATION_COLS = {
    'delta': COLS_140_DELTA,
    'up': COLS_140_UP,
    'uv': COLS_140_UV,
}

CHRONO_120 = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2_matched_is_chrono')
CHRONO_180 = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2_matched_is_ind_chrono')
FEAT_180 = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2_matched_is_ind', 'features')
CHRONO_RIC = os.getenv(
    'E2_CHRONO_RIC_ROOT',
    os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2_matched_is_ind_chrono_ric'),
)


def ws(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)
