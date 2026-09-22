# -*- coding: utf-8 -*-
"""E5-1: Future-horizon Rank IC profile for pairwise specific scores."""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.metrics_horizon import (
    daily_hl_spread, daily_rank_ic, summarize_daily_ic, summarize_hl,
)
from experiments.E5_mechanism.paths import HORIZONS, OUT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e51')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RET_COLS = [f'r{h}' for h in HORIZONS] + ['R_1_2', 'R_3_5', 'R_1_5', 'R_1_10', 'Ret_5d', 'Ret_week', 'ret_val']


def run_e51(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    rows = []
    for score in SCORES:
        for ret in RET_COLS:
            if ret not in panel.columns:
                continue
            logger.info(f'IC {score} ~ {ret}')
            daily = daily_rank_ic(panel, score, ret)
            summ = summarize_daily_ic(daily)
            hl = summarize_hl(daily_hl_spread(panel, score, ret))
            rows.append({'score': score, 'ret': ret, **summ, **hl})
            daily.to_csv(
                os.path.join(OUT_ROOT, f'e51_daily_{score}_{ret}.csv'),
                index=False, float_format='%.6f',
            )
    out = pd.DataFrame(rows)
    path = os.path.join(OUT_ROOT, 'e51_horizon_ic.csv')
    out.to_csv(path, index=False, float_format='%.6f')
    logger.info(f'wrote {path}\n{out.to_string(index=False)}')
    return out


if __name__ == '__main__':
    run_e51()
