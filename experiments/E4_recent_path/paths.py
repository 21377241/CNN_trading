# -*- coding: utf-8 -*-
"""E4 paths and constants."""
from __future__ import annotations

import os

from experiments.E2_matched_tabular.e2a_layout import CHRONO_RIC, FEAT_180, ws
from experiments.E3_shared_specific.paths import E1B_RIC_EXP, HISTGB_FROZEN  # noqa: F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.getenv(
    'E4_OUT_ROOT', ws('WORK_SPACE', 'paper', 'E4_recent_path_chrono_ric')
)
PRED_ROOT = os.path.join(CHRONO_RIC, 'preds')
E3_ROOT = os.getenv(
    'E3_OUT_ROOT', ws('WORK_SPACE', 'paper', 'E3_shared_specific_chrono_ric')
)
ANN = 52
MIN_CS_N = 50
NW_LAG = 4
WINDOW = 20
PER_DAY = 9
RECENT5 = list(range(15, 20))
LAG_LABELS = {15: 't-4', 16: 't-3', 17: 't-2', 18: 't-1', 19: 't'}
OOS_YEARS = list(range(2008, 2026))
IMG_H, IMG_W = 64, 60
BATCH = 512
MLP_BAG_KW = dict(
    seeds=list(range(42, 52)),
    es_mode='chrono',
    hidden_layer_sizes=tuple(int(x) for x in os.getenv('MLP_HIDDEN_LAYERS', '128,64,32,16').split(',')),
    max_iter=200,
    n_iter_no_change=15,
    alpha=1e-4,
    lr=1e-3,
    monitor='rank_ic',
)
COL_O, COL_H, COL_L, COL_C = 0, 1, 2, 3
COL_MA, COL_DELTA, COL_VN, COL_UP, COL_UV = 4, 5, 6, 7, 8
