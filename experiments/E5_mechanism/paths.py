# -*- coding: utf-8 -*-
"""E5 paths and constants."""
from __future__ import annotations

import os

from experiments.E2_matched_tabular.e2a_layout import ws

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.getenv(
    'E5_OUT_ROOT', ws('WORK_SPACE', 'paper', 'E5_mechanism_chrono_ric')
)
E4_ROOT = os.getenv(
    'E4_OUT_ROOT', ws('WORK_SPACE', 'paper', 'E4_recent_path_chrono_ric')
)
SCORES_PATH = os.path.join(E4_ROOT, 'baseline_scores.parquet')
PATHSHAPE_PATH = os.path.join(E4_ROOT, 'pathshape.parquet')
ASHARE_RET = ws('WORK_SPACE', 'data', 'ashare_processed_data', 'ashare_ret.feather')
MIN_CS_N = 50
NW_LAG = 4
HORIZONS = list(range(1, 11))
