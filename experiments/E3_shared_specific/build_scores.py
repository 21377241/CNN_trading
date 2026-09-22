# -*- coding: utf-8 -*-
"""E3-0: Shared + triple residual Specific scores."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.paths import MIN_CS_N, OUT_ROOT
from experiments.E3_shared_specific.utils import cs_zscore, daily_triple_resid

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_scores')


def build_scores(panel_path: str | None = None, out_path: str | None = None) -> pd.DataFrame:
    panel_path = panel_path or os.path.join(OUT_ROOT, 'panel.parquet')
    panel = pd.read_parquet(panel_path)
    panel['ending_date'] = pd.to_datetime(panel['ending_date'])
    pieces = []
    n_skip = 0
    for dt, g in panel.groupby('ending_date', sort=True):
        if len(g) < MIN_CS_N:
            n_skip += 1
            continue
        g = g.copy()
        sc = g['S_cnn'].values.astype(float)
        sm = g['S_mlp'].values.astype(float)
        sh = g['S_hgb'].values.astype(float)
        m = np.isfinite(sc) & np.isfinite(sm) & np.isfinite(sh)
        if m.sum() < MIN_CS_N:
            n_skip += 1
            continue
        g = g.loc[m].copy()
        sc, sm, sh = g['S_cnn'].values.astype(float), g['S_mlp'].values.astype(float), g['S_hgb'].values.astype(float)
        u_c, u_m, u_h = daily_triple_resid(sc, sm, sh)
        zc, zm, zh = cs_zscore(g['S_cnn']), cs_zscore(g['S_mlp']), cs_zscore(g['S_hgb'])
        g['S_sh'] = (zc + zm + zh) / 3.0
        g['S_cnn_sp'] = u_c
        g['S_mlp_sp'] = u_m
        g['S_hgb_sp'] = u_h
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out_path = out_path or os.path.join(OUT_ROOT, 'scores.parquet')
    out.to_parquet(out_path, index=False)
    logger.info(f'scores rows={len(out):,} dates={out["ending_date"].nunique()} skipped_days={n_skip} -> {out_path}')
    return out


if __name__ == '__main__':
    build_scores()
