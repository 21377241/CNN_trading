# -*- coding: utf-8 -*-
"""E3-4: tabular feature-block ablation with frozen hyperparams."""
from __future__ import annotations

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.common.splits import chrono_train_val_indices
from experiments.E2_matched_tabular.e2a_layout import FEAT_180, PER_DAY_180, WINDOW
from experiments.E2_matched_tabular.mlp_bagging import bag_predict_proba, fit_mlp_bag
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from experiments.E2_matched_tabular.run_train import predict_proba_pos
from experiments.E3_shared_specific.paths import ANN, HISTGB_FROZEN, MIN_CS_N, OUT_ROOT, PRED_ROOT
from Portfolio.portfolio import run_portfolio_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_e34')

MLP_BAG_SEEDS = list(range(42, 52))
MLP_BAG_KW = dict(
    seeds=MLP_BAG_SEEDS,
    es_mode='chrono',
    hidden_layer_sizes=(128, 64),
    max_iter=200,
    n_iter_no_change=15,
    alpha=1e-4,
    lr=1e-3,
    monitor='rank_ic',
)


def _expand(day_cols: list[int]) -> np.ndarray:
    out = []
    for t in range(WINDOW):
        base = t * PER_DAY_180
        out.extend(base + c for c in day_cols)
    return np.asarray(out, dtype=int)


# 180 layout: O H L C MA delta Vn uP uV = 0..8
ABLATIONS = {
    'full180': _expand(list(range(9))),
    'drop_HL': _expand([0, 3, 4, 5, 6, 7, 8]),       # drop H,L
    'drop_volume': _expand([0, 1, 2, 3, 4, 5, 7, 8]),  # drop Vn
    'drop_direction': _expand([0, 1, 2, 3, 4, 5, 6]),  # drop uP uV
    'drop_recent5': _expand(list(range(9)))[: 15 * 9],  # keep first 15 days
}


def _norm_sid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)


def _eval_tab_specific(
    name: str,
    model: str,
    tab_prob: np.ndarray,
    meta_oos: pd.DataFrame,
    partners: pd.DataFrame,
    out_dir: str,
) -> dict:
    """Residualize ablated tabular score on the other two original scores (E3-0)."""
    keys = ['StockID', 'ending_date']
    tab = meta_oos[keys + ['ret_val', 'MarketCap', 'binary_label']].copy()
    tab['StockID'] = _norm_sid(tab['StockID'])
    tab['ending_date'] = pd.to_datetime(tab['ending_date'])
    tab['S_tab'] = tab_prob
    p = partners.copy()
    p['StockID'] = _norm_sid(p['StockID'])
    p['ending_date'] = pd.to_datetime(p['ending_date'])
    m = tab.merge(p[keys + ['S_cnn', 'S_mlp', 'S_hgb']], on=keys, how='inner')

    pieces = []
    for _, g in m.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        g = g.copy()
        st = g['S_tab'].values.astype(float)
        sc = g['S_cnn'].values.astype(float)
        sm = g['S_mlp'].values.astype(float)
        sh = g['S_hgb'].values.astype(float)
        mask = np.isfinite(st) & np.isfinite(sc) & np.isfinite(sm) & np.isfinite(sh)
        g = g.loc[mask]
        if len(g) < MIN_CS_N:
            continue
        if model == 'histgb':
            X = np.column_stack([g['S_cnn'].values, g['S_mlp'].values])
            y = g['S_tab'].values.astype(float)
        else:  # mlp
            X = np.column_stack([g['S_cnn'].values, g['S_hgb'].values])
            y = g['S_tab'].values.astype(float)
        g['S_tab_sp'] = y - LinearRegression().fit(X, y).predict(X)
        pieces.append(g)
    full = pd.concat(pieces, ignore_index=True)
    df = full[keys + ['ret_val', 'MarketCap', 'binary_label']].copy()
    df['up_prob'] = full.groupby('ending_date')['S_tab_sp'].rank(pct=True).values
    ic = summarize_ic(cross_section_ic(df, 'up_prob'))
    mdir = os.path.join(out_dir, name)
    os.makedirs(mdir, exist_ok=True)
    tmp = os.path.join(mdir, 'pred_for_port.csv')
    df.to_csv(tmp, index=False)
    run_portfolio_analysis(result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=ANN)
    hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
    return {'ablation': name, 'model': model, **ic, **hl, 'n': len(df)}


def main():
    out_root = os.path.join(OUT_ROOT, 'e34_ablation')
    os.makedirs(out_root, exist_ok=True)
    X_is = np.load(os.path.join(FEAT_180, 'X_is.npy'), mmap_mode='r')
    X_oos = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta_is = pd.read_feather(os.path.join(FEAT_180, 'meta_is.feather'))
    meta_oos = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))

    cnn = pd.read_csv(os.path.join(PRED_ROOT, 'pred_cnn_oos.csv')).rename(columns={'up_prob': 'S_cnn'})
    mlp = pd.read_csv(os.path.join(PRED_ROOT, 'pred_mlp_oos.csv')).rename(columns={'up_prob': 'S_mlp'})
    hgb = pd.read_csv(os.path.join(PRED_ROOT, 'pred_histgb_oos.csv')).rename(columns={'up_prob': 'S_hgb'})
    keys = ['StockID', 'ending_date']
    partners = cnn[keys + ['S_cnn']].merge(mlp[keys + ['S_mlp']], on=keys).merge(hgb[keys + ['S_hgb']], on=keys)

    y_is = meta_is['binary_label'].values.astype(int)
    valid = np.isin(y_is, [0, 1])
    meta_is_v = meta_is.loc[valid].reset_index(drop=True)
    y_is = y_is[valid]
    tr_idx, va_idx = chrono_train_val_indices(meta_is_v, list(range(2000, 2006)), [2006, 2007])
    meta_va = meta_is_v.iloc[va_idx].reset_index(drop=True)

    rows = []
    for aname, cols in ABLATIONS.items():
        logger.info(f'=== ablation {aname} dim={len(cols)} ===')
        Xv = np.asarray(X_is[valid][:, cols])
        Xo = np.asarray(X_oos[:, cols])
        X_tr, y_tr = Xv[tr_idx], y_is[tr_idx]
        X_va, y_va = Xv[va_idx], y_is[va_idx]

        hgb_m = HistGradientBoostingClassifier(**HISTGB_FROZEN)
        hgb_m.fit(X_tr, y_tr)
        pred_h = predict_proba_pos(hgb_m, Xo)
        rows.append(_eval_tab_specific(f'{aname}_histgb', 'histgb', pred_h, meta_oos, partners, out_root))

        logger.info(f'  MLP chrono bag k={len(MLP_BAG_SEEDS)}…')
        mlp_pipes, mlp_infos = fit_mlp_bag(X_tr, y_tr, X_va, y_va, meta_va, **MLP_BAG_KW)
        pred_m = bag_predict_proba(mlp_pipes, Xo)
        rows.append(_eval_tab_specific(f'{aname}_mlp', 'mlp', pred_m, meta_oos, partners, out_root))
        joblib.dump(
            {'histgb': hgb_m, 'mlp_pipes': mlp_pipes, 'mlp_infos': mlp_infos},
            os.path.join(out_root, f'{aname}_models.joblib'),
        )

    # baselines from chrono_ric preds (align to meta_oos row order)
    mo = meta_oos[keys].copy()
    mo['StockID'] = _norm_sid(mo['StockID'])
    mo['ending_date'] = pd.to_datetime(mo['ending_date'])
    hgb_a = hgb.copy()
    hgb_a['StockID'] = _norm_sid(hgb_a['StockID'])
    hgb_a['ending_date'] = pd.to_datetime(hgb_a['ending_date'])
    mlp_a = mlp.copy()
    mlp_a['StockID'] = _norm_sid(mlp_a['StockID'])
    mlp_a['ending_date'] = pd.to_datetime(mlp_a['ending_date'])
    base_h = mo.merge(hgb_a[keys + ['S_hgb']], on=keys, how='left')['S_hgb'].values
    base_m = mo.merge(mlp_a[keys + ['S_mlp']], on=keys, how='left')['S_mlp'].values
    rows.append(_eval_tab_specific('baseline_histgb', 'histgb', base_h, meta_oos, partners, out_root))
    rows.append(_eval_tab_specific('baseline_mlp', 'mlp', base_m, meta_oos, partners, out_root))

    summary = pd.DataFrame(rows)
    for model in ['histgb', 'mlp']:
        base = summary.loc[summary['ablation'] == f'baseline_{model}', 'ew_hl_sharpe']
        if len(base):
            b = float(base.iloc[0])
            mask = summary['model'] == model
            summary.loc[mask, 'baseline_sharpe'] = b
            summary.loc[mask, 'sharpe_drop_pct'] = (b - summary.loc[mask, 'ew_hl_sharpe']) / abs(b) * 100.0

    out = os.path.join(OUT_ROOT, 'e34_ablation_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    with open(os.path.join(out_root, 'e34_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'ablations': {k: int(len(v)) for k, v in ABLATIONS.items()}, 'histgb': HISTGB_FROZEN}, f, indent=2)
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
