# -*- coding: utf-8 -*-
"""
E2A-A3: Single-indicator 140-dim ablation under chrono (extract cols from 180 cache).

Usage:
  python experiments/E2_matched_tabular/run_e2a_ablation_train.py --ablation all
  python experiments/E2_matched_tabular/run_e2a_ablation_train.py --ablation delta
  python experiments/E2_matched_tabular/run_e2a_ablation_train.py --ablation up,uv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.common.histgb_search import search_histgb
from experiments.common.metrics import cross_section_ic, summarize_ic, yearly_rank_ic_mean
from experiments.common.sgd_logit import search_sgd_logit
from experiments.common.splits import chrono_train_val_indices
from experiments.E2_matched_tabular.e2a_layout import ABLATION_COLS, FEAT_180, ws
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from experiments.E2_matched_tabular.run_train import predict_proba_pos, _write_oos_pred
from Portfolio.portfolio import run_portfolio_analysis, write_port_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2a_ablation')

CFG_TEMPLATE = os.path.join(
    PROJECT_ROOT, 'experiments', 'E2_matched_tabular', 'config_ind_chrono.yaml'
)


def load_cfg():
    with open(CFG_TEMPLATE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one(ablation: str, cfg: dict):
    cols = ABLATION_COLS[ablation]
    out_root = ws('WORK_SPACE', 'paper', f'E2A_ind_{ablation}_chrono')
    model_dir = os.path.join(out_root, 'models')
    pred_dir = os.path.join(out_root, 'preds')
    port_root = os.path.join(out_root, 'portfolio')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    logger.info(f'=== ablation={ablation} dim={len(cols)} out={out_root} ===')
    X_is_full = np.load(os.path.join(FEAT_180, 'X_is.npy'), mmap_mode='r')
    X_oos_full = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta_is = pd.read_feather(os.path.join(FEAT_180, 'meta_is.feather'))
    meta_oos = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))

    y_is = meta_is['binary_label'].values.astype(int)
    valid = np.isin(y_is, [0, 1])
    X_is = np.asarray(X_is_full[valid][:, cols])
    meta_is = meta_is.loc[valid].reset_index(drop=True)
    y_is = y_is[valid]
    X_oos = np.asarray(X_oos_full[:, cols])

    tr_years = cfg.get('chrono_train_years', list(range(2000, 2006)))
    va_years = cfg.get('chrono_val_years', [2006, 2007])
    tr_idx, va_idx = chrono_train_val_indices(meta_is, tr_years, va_years)
    X_tr, y_tr = X_is[tr_idx], y_is[tr_idx]
    X_va, y_va = X_is[va_idx], y_is[va_idx]
    meta_va = meta_is.iloc[va_idx].reset_index(drop=True)
    logger.info(f'train={len(X_tr):,} val={len(X_va):,} oos={len(X_oos):,} dim={X_is.shape[1]}')

    # logit
    logit, rows, best = search_sgd_logit(X_tr, y_tr, X_va, y_va, meta_va, cfg['logit'])
    pd.DataFrame(rows).to_csv(os.path.join(model_dir, 'logit_sgd_search.csv'), index=False)
    joblib.dump(logit, os.path.join(model_dir, 'logit.joblib'))
    with open(os.path.join(model_dir, 'logit_best.json'), 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'search_metric': cfg['logit'].get('search_metric')}, f, indent=2)
    _write_oos_pred('logit', logit, X_oos, meta_oos, pred_dir)

    # histgb
    histgb, rows, best = search_histgb(X_tr, y_tr, X_va, y_va, meta_va, cfg['histgb'])
    pd.DataFrame(rows).to_csv(os.path.join(model_dir, 'histgb_search.csv'), index=False)
    joblib.dump(histgb, os.path.join(model_dir, 'histgb.joblib'))
    with open(os.path.join(model_dir, 'histgb_best.json'), 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'search_metric': cfg['histgb'].get('search_metric')}, f, indent=2)
    _write_oos_pred('histgb', histgb, X_oos, meta_oos, pred_dir)

    # mlp
    mcfg = cfg['mlp']
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=tuple(mcfg['hidden_layer_sizes']),
            alpha=mcfg['alpha'],
            learning_rate_init=mcfg['learning_rate_init'],
            max_iter=mcfg['max_iter'],
            early_stopping=True,
            validation_fraction=mcfg['validation_fraction'],
            n_iter_no_change=mcfg['n_iter_no_change'],
            random_state=cfg['seed'],
        )),
    ])
    mlp.fit(X_tr, y_tr)
    joblib.dump(mlp, os.path.join(model_dir, 'mlp.joblib'))
    _write_oos_pred('mlp', mlp, X_oos, meta_oos, pred_dir)

    # CNN pred copy from meta for panel alignment in optional eval
    cnn = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label', 'up_prob']].copy()
    cnn.to_csv(os.path.join(pred_dir, 'pred_cnn_oos.csv'), index=False)

    # eval tabular models
    rows_sum = []
    for name in ['logit', 'histgb', 'mlp']:
        df = pd.read_csv(os.path.join(pred_dir, f'pred_{name}_oos.csv'))
        df['ending_date'] = pd.to_datetime(df['ending_date'])
        y = df['binary_label'].values
        s = df['up_prob'].values
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        ic_sum = summarize_ic(cross_section_ic(df, 'up_prob'))
        mdir = os.path.join(port_root, name)
        os.makedirs(mdir, exist_ok=True)
        tmp = write_port_input(df, mdir)
        run_portfolio_analysis(
            result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=cfg['ann_factor'],
        )
        hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
        rows_sum.append({'model': name, 'auc': auc, **ic_sum, **hl, 'n': len(df), 'ablation': ablation, 'dim': len(cols)})

    summary = pd.DataFrame(rows_sum)
    summary.to_csv(os.path.join(out_root, 'e2_summary.csv'), index=False, float_format='%.6f')
    with open(os.path.join(model_dir, 'train_info.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'ablation': ablation,
            'dim': len(cols),
            'n_train': int(len(X_tr)),
            'n_val': int(len(X_va)),
            'n_oos': int(len(X_oos)),
            'split_mode': 'chrono',
            'feat_source': FEAT_180,
        }, f, indent=2)
    logger.info(f'\n{summary.to_string(index=False)}')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', default='all', help='comma list: delta,up,uv or all')
    args = parser.parse_args(argv)
    cfg = load_cfg()
    names = ['delta', 'up', 'uv'] if args.ablation.strip().lower() == 'all' else [
        s.strip().lower() for s in args.ablation.split(',') if s.strip()
    ]
    for n in names:
        if n not in ABLATION_COLS:
            raise ValueError(f'unknown ablation {n}')
        train_one(n, cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
