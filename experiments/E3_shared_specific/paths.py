# -*- coding: utf-8 -*-
"""E3 paths and constants."""
from __future__ import annotations

import os

from experiments.E2_matched_tabular.e2a_layout import CHRONO_RIC, FEAT_180, ws

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.getenv(
    'E3_OUT_ROOT', ws('WORK_SPACE', 'paper', 'E3_shared_specific_chrono_ric')
)
PRED_ROOT = os.path.join(CHRONO_RIC, 'preds')
ANN = 52
MIN_CS_N = 50
DISAGREE_HI = 0.8
DISAGREE_LO = 0.2
DISAGREE_MIN_N = 5
NW_LAG = 4
WINDOW = 20
PER_DAY = 9

E1B_RIC_EXP = ws(
    'WORK_SPACE', 'paper', 'E1b_cnn_hp_I20R5_ric',
    'D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C256',
    '20d5p-lr1E-4-dp0.50-maTrue-vbTrue-weekTrained-noDelayedReturn-ric-full-p5',
)

DAILY_FEATHER = ws('WORK_SPACE', 'data', 'ashare_processed_data', 'ashare_cache_step4_precomputed.feather')
DAILY_RET_RAW = ws('WORK_SPACE', 'data', 'ashare_processed_data', 'ashare_daily_ret_raw.feather')

HISTGB_FROZEN = dict(
    learning_rate=0.08,
    max_leaf_nodes=63,
    min_samples_leaf=200,
    l2_regularization=1.0,
    max_iter=300,
    validation_fraction=0.1,
    n_iter_no_change=20,
    early_stopping=True,
    random_state=42,
)
