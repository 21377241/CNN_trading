# -*- coding: utf-8 -*-
"""Shared tabular retrain (MLP bag10 / HistGB frozen) under column subsets."""
from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.common.splits import chrono_train_val_indices
from experiments.E2_matched_tabular.e2a_layout import FEAT_180
from experiments.E2_matched_tabular.mlp_bagging import bag_predict_proba, fit_mlp_bag
from experiments.E2_matched_tabular.run_train import predict_proba_pos
from experiments.E3_shared_specific.paths import HISTGB_FROZEN
from experiments.E3_shared_specific.utils import norm_stock_id
from experiments.E4_recent_path.paths import MLP_BAG_KW, OUT_ROOT

logger = logging.getLogger('e4_tab')


def _load_arrays():
    X_is = np.load(os.path.join(FEAT_180, 'X_is.npy'), mmap_mode='r')
    X_oos = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta_is = pd.read_feather(os.path.join(FEAT_180, 'meta_is.feather'))
    meta_oos = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))
    return X_is, X_oos, meta_is, meta_oos


def train_predict_subset(
    cols: np.ndarray,
    tag: str,
    model: str,
    cache_dir: str,
    force: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Train on IS subset cols; return OOS probs aligned to meta_oos rows + meta_oos."""
    os.makedirs(cache_dir, exist_ok=True)
    pred_path = os.path.join(cache_dir, f'pred_{model}_{tag}_oos.csv')
    model_path = os.path.join(cache_dir, f'{model}_{tag}.joblib')
    X_is, X_oos, meta_is, meta_oos = _load_arrays()
    keys = ['StockID', 'ending_date']
    if os.path.exists(pred_path) and not force:
        logger.info(f'resume tabular pred {pred_path}')
        pred = pd.read_csv(pred_path)
        # align to meta_oos order
        mo = meta_oos[keys].copy()
        mo['StockID'] = norm_stock_id(mo['StockID'])
        mo['ending_date'] = pd.to_datetime(mo['ending_date'])
        pred['StockID'] = norm_stock_id(pred['StockID'])
        pred['ending_date'] = pd.to_datetime(pred['ending_date'])
        m = mo.merge(pred[keys + ['up_prob']], on=keys, how='left')
        return m['up_prob'].values.astype(float), meta_oos

    y_is = meta_is['binary_label'].values.astype(int)
    valid = np.isin(y_is, [0, 1])
    meta_is_v = meta_is.loc[valid].reset_index(drop=True)
    y_is = y_is[valid]
    tr_idx, va_idx = chrono_train_val_indices(meta_is_v, list(range(2000, 2006)), [2006, 2007])
    meta_va = meta_is_v.iloc[va_idx].reset_index(drop=True)

    Xv = np.asarray(X_is[valid][:, cols])
    Xo = np.asarray(X_oos[:, cols])
    X_tr, y_tr = Xv[tr_idx], y_is[tr_idx]
    X_va, y_va = Xv[va_idx], y_is[va_idx]
    logger.info(f'train {model}/{tag} dim={len(cols)} n_tr={len(X_tr):,} n_va={len(X_va):,}')

    if model == 'mlp':
        pipes, infos = fit_mlp_bag(X_tr, y_tr, X_va, y_va, meta_va, **MLP_BAG_KW)
        prob = bag_predict_proba(pipes, Xo)
        joblib.dump({'pipes': pipes, 'infos': infos, 'cols': cols}, model_path)
    elif model == 'histgb':
        m = HistGradientBoostingClassifier(**HISTGB_FROZEN)
        m.fit(X_tr, y_tr)
        prob = predict_proba_pos(m, Xo)
        joblib.dump({'model': m, 'cols': cols}, model_path)
    else:
        raise ValueError(model)

    out = meta_oos[keys + ['ret_val', 'MarketCap', 'binary_label']].copy()
    out['StockID'] = norm_stock_id(out['StockID'])
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    out['up_prob'] = prob
    out.to_csv(pred_path, index=False)
    logger.info(f'wrote {pred_path}')
    return prob, meta_oos


def partner_cnn_df() -> pd.DataFrame:
    from experiments.E4_recent_path.paths import PRED_ROOT
    cnn = pd.read_csv(os.path.join(PRED_ROOT, 'pred_cnn_oos.csv')).rename(columns={'up_prob': 'S_cnn'})
    keep = ['StockID', 'ending_date', 'S_cnn'] + [c for c in ['ret_val', 'MarketCap', 'binary_label'] if c in cnn.columns]
    cnn = cnn[keep].copy()
    cnn['StockID'] = norm_stock_id(cnn['StockID'])
    cnn['ending_date'] = pd.to_datetime(cnn['ending_date'])
    return cnn


def partner_mlp_df() -> pd.DataFrame:
    from experiments.E4_recent_path.paths import PRED_ROOT
    mlp = pd.read_csv(os.path.join(PRED_ROOT, 'pred_mlp_oos.csv')).rename(columns={'up_prob': 'S_mlp'})
    keep = ['StockID', 'ending_date', 'S_mlp'] + [c for c in ['ret_val', 'MarketCap', 'binary_label'] if c in mlp.columns]
    mlp = mlp[keep].copy()
    mlp['StockID'] = norm_stock_id(mlp['StockID'])
    mlp['ending_date'] = pd.to_datetime(mlp['ending_date'])
    return mlp
