# -*- coding: utf-8 -*-
"""E5-2: Overnight vs intraday return realization."""
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
from experiments.E5_mechanism.paths import OUT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_e52')

SCORES = ['S_cnn_sp', 'S_mlp_sp']
RET_COLS = ['on1', 'id1', 'ON_1_5', 'ID_1_5', 'r1', 'R_1_5']


def run_e52(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    os.makedirs(OUT_ROOT, exist_ok=True)
    panel = panel if panel is not None else build_forward_panel()
    rows = []
    for score in SCORES:
        for ret in RET_COLS:
            if ret not in panel.columns:
                continue
            logger.info(f'session IC {score} ~ {ret}')
            daily = daily_rank_ic(panel, score, ret)
            summ = summarize_daily_ic(daily)
            hl = summarize_hl(daily_hl_spread(panel, score, ret))
            rows.append({'score': score, 'ret': ret, **summ, **hl})
    out = pd.DataFrame(rows)
    # pairwise ON-ID delta of daily IC
    delta_rows = []
    for score in SCORES:
        for on_col, id_col, tag in [('on1', 'id1', 'next_day'), ('ON_1_5', 'ID_1_5', 'sum_1_5')]:
            d_on = daily_rank_ic(panel, score, on_col).rename(columns={'rank_ic': 'ic_on'})
            d_id = daily_rank_ic(panel, score, id_col).rename(columns={'rank_ic': 'ic_id'})
            m = d_on.merge(d_id[['ending_date', 'ic_id']], on='ending_date', how='inner')
            m['delta_on_minus_id'] = m['ic_on'] - m['ic_id']
            from experiments.E3_shared_specific.utils import newey_west_t
            from experiments.E5_mechanism.paths import NW_LAG
            t, p = newey_west_t(m['delta_on_minus_id'].values, lag=NW_LAG)
            delta_rows.append({
                'score': score, 'window': tag,
                'ic_on_mean': float(m['ic_on'].mean()),
                'ic_id_mean': float(m['ic_id'].mean()),
                'delta_mean': float(m['delta_on_minus_id'].mean()),
                'nw_t': t, 'nw_p': p, 'n_days': int(len(m)),
            })
    delta = pd.DataFrame(delta_rows)
    out_path = os.path.join(OUT_ROOT, 'e52_session_ic.csv')
    delta_path = os.path.join(OUT_ROOT, 'e52_on_minus_id.csv')
    out.to_csv(out_path, index=False, float_format='%.6f')
    delta.to_csv(delta_path, index=False, float_format='%.6f')
    logger.info(f'wrote {out_path}\n{out.to_string(index=False)}')
    logger.info(f'wrote {delta_path}\n{delta.to_string(index=False)}')
    return out


if __name__ == '__main__':
    run_e52()
