# -*- coding: utf-8 -*-
"""Build three-model OOS panel."""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.paths import OUT_ROOT, PRED_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_panel')


def build_panel(pred_root: str | None = None, out_path: str | None = None) -> pd.DataFrame:
    root = pred_root or PRED_ROOT
    cnn = pd.read_csv(os.path.join(root, 'pred_cnn_oos.csv'))
    mlp = pd.read_csv(os.path.join(root, 'pred_mlp_oos.csv'))
    hgb = pd.read_csv(os.path.join(root, 'pred_histgb_oos.csv'))
    keys = ['StockID', 'ending_date']
    cnn = cnn.rename(columns={'up_prob': 'S_cnn'})
    mlp = mlp.rename(columns={'up_prob': 'S_mlp'})
    hgb = hgb.rename(columns={'up_prob': 'S_hgb'})

    keep_extra = ['ret_val', 'MarketCap', 'binary_label']
    m = cnn[keys + ['S_cnn'] + [c for c in keep_extra if c in cnn.columns]]
    m = m.merge(mlp[keys + ['S_mlp']], on=keys, how='inner')
    m = m.merge(hgb[keys + ['S_hgb']], on=keys, how='inner')
    for c in keep_extra:
        if c not in m.columns:
            if c in mlp.columns:
                m = m.merge(mlp[keys + [c]], on=keys, how='left')
            elif c in hgb.columns:
                m = m.merge(hgb[keys + [c]], on=keys, how='left')
    m['ending_date'] = pd.to_datetime(m['ending_date'])
    m['year'] = m['ending_date'].dt.year
    out = out_path or os.path.join(OUT_ROOT, 'panel.parquet')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m.to_parquet(out, index=False)
    logger.info(f'panel rows={len(m):,} dates={m["ending_date"].nunique()} -> {out}')
    return m


if __name__ == '__main__':
    build_panel()
