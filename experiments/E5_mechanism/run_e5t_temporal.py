# -*- coding: utf-8 -*-
"""E5T: economic realization of pairwise CNN vs 1D-CNN specific scores.

Builds an isolated E5-compatible panel from E3T, then reuses the established
E5.1--E5.7 implementations without modifying the MLP-4 experiment outputs.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E3T_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E3T_cnn_1d_temporal')
E4T_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E4T_cnn_1d_recent_path')
E4_MLP4_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E4_recent_path_chrono_ric_mlp4')
E5T_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E5T_cnn_1d_mechanism')

# These must precede E5 imports because paths.py reads them at import time.
os.environ['E4_OUT_ROOT'] = E4T_ROOT
os.environ['E5_OUT_ROOT'] = E5T_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def prepare_panel() -> None:
    os.makedirs(E4T_ROOT, exist_ok=True)
    scores = pd.read_parquet(os.path.join(E3T_ROOT, 'scores.parquet')).copy()
    need = ['StockID', 'ending_date', 'cnn', 'temporal', 'S_cnn_sp_1d', 'S_1d_sp_cnn', 'ret_val', 'binary_label']
    missing = set(need).difference(scores.columns)
    if missing:
        raise KeyError(f'E3T score panel missing: {missing}')
    keep = need + (['MarketCap'] if 'MarketCap' in scores.columns else [])
    out = scores[keep].rename(columns={
        'cnn': 'S_cnn',
        'temporal': 'S_mlp',  # compatibility alias; represents the 1D-CNN score only.
        'S_cnn_sp_1d': 'S_cnn_sp',
        'S_1d_sp_cnn': 'S_mlp_sp',
    })
    out.to_parquet(os.path.join(E4T_ROOT, 'baseline_scores.parquet'), index=False)

    # Path variables are representation-invariant and were already built from
    # the identical OOS stock-week panel in E4.  Materialize a private copy.
    src_path = os.path.join(E4_MLP4_ROOT, 'pathshape.parquet')
    dst_path = os.path.join(E4T_ROOT, 'pathshape.parquet')
    path = pd.read_parquet(src_path)
    path.to_parquet(dst_path, index=False)


def main() -> None:
    prepare_panel()
    from experiments.E5_mechanism.build_forward import build_forward_panel
    from experiments.E5_mechanism.run_e51_horizon import run_e51
    from experiments.E5_mechanism.run_e52_session import run_e52
    from experiments.E5_mechanism.run_e53_state import run_e53
    from experiments.E5_mechanism.run_e54_pressure import run_e54
    from experiments.E5_mechanism.run_e55_interaction import run_e55
    from experiments.E5_mechanism.run_e56_cont_rev import run_e56
    from experiments.E5_mechanism.run_e57_institutional import run_e57

    # Force a fresh panel so the score mapping cannot inherit a stale cache.
    os.makedirs(E5T_ROOT, exist_ok=True)
    panel_path = os.path.join(E5T_ROOT, 'forward_labels.parquet')
    if os.path.exists(panel_path):
        os.remove(panel_path)
    panel = build_forward_panel()
    run_e51(panel)
    run_e52(panel)
    run_e53(panel)
    _, pressured = run_e54(panel)
    run_e55(pressured)
    run_e56(pressured)
    run_e57(pressured)


if __name__ == '__main__':
    main()
