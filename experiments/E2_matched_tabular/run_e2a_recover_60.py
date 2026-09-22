# -*- coding: utf-8 -*-
"""
E2A-A2: Recover 60-dim indicators from 120-dim base (chrono val).

Binary targets (δ, uP, uV): regress; report per-dim OOS Accuracy (threshold@0.5)
and R² (continuous regression fit). Ridge selects α on val Acc.

Usage:
  python experiments/E2_matched_tabular/run_e2a_recover_60.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.splits import chrono_train_val_indices
from experiments.E2_matched_tabular.e2a_layout import (
    COLS_120_FROM_180, COLS_60_FROM_180, COLS_DELTA20, COLS_UP20, COLS_UV20,
    FEAT_180, ws,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2a_recover')

TARGETS = {
    'all60': COLS_60_FROM_180,
    'delta': COLS_DELTA20,
    'up': COLS_UP20,
    'uv': COLS_UV20,
}

TRAIN_YEARS = list(range(2000, 2006))
VAL_YEARS = [2006, 2007]
THRESHOLD = 0.5


def r2_per_dim(y_true, y_pred) -> np.ndarray:
    """Per-column R^2; handle constant columns."""
    out = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        if ss_tot < 1e-12:
            out[j] = 1.0 if np.allclose(yt, yp) else 0.0
            continue
        ss_res = np.sum((yt - yp) ** 2)
        out[j] = 1.0 - ss_res / ss_tot
    return out


def summarize_r2(r2_dims: np.ndarray) -> dict:
    return {
        'r2_mean': float(np.nanmean(r2_dims)),
        'r2_median': float(np.nanmedian(r2_dims)),
        'r2_min': float(np.nanmin(r2_dims)),
        'r2_max': float(np.nanmax(r2_dims)),
    }


def acc_per_dim(y_true, y_pred, threshold: float = THRESHOLD) -> np.ndarray:
    """Per-column classification accuracy after thresholding continuous predictions."""
    out = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = (y_pred[:, j] >= threshold).astype(np.float64)
        out[j] = float(np.mean(yt == yp))
    return out


def summarize_acc(acc_dims: np.ndarray) -> dict:
    return {
        'acc_mean': float(np.nanmean(acc_dims)),
        'acc_median': float(np.nanmedian(acc_dims)),
        'acc_min': float(np.nanmin(acc_dims)),
        'acc_max': float(np.nanmax(acc_dims)),
    }


def fit_ridge(X_tr, y_tr, X_va, y_va):
    best, best_acc, best_alpha = None, -1.0, None
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', Ridge(alpha=alpha, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        acc = float(np.nanmean(acc_per_dim(y_va, pred)))
        if acc > best_acc:
            best_acc, best, best_alpha = acc, pipe, alpha
    return best, {'alpha': best_alpha, 'val_acc_mean': best_acc}


def fit_mlp(X_tr, y_tr, X_va, y_va):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=6,
            random_state=42,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    acc = float(np.nanmean(acc_per_dim(y_va, pred)))
    return pipe, {'val_acc_mean': acc}


def fit_hgb(X_tr, y_tr, X_va, y_va):
    base = HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=80,
        learning_rate=0.08,
        min_samples_leaf=50,
        random_state=42,
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    n = len(X_tr)
    if n > 80_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=80_000, replace=False)
        X_fit, y_fit = X_tr[idx], y_tr[idx]
    else:
        X_fit, y_fit = X_tr, y_tr
    model.fit(X_fit, y_fit)
    pred = model.predict(X_va)
    acc = float(np.nanmean(acc_per_dim(y_va, pred)))
    return model, {'val_acc_mean': acc, 'n_fit': int(len(X_fit))}


REGRESSORS = {
    'ridge': fit_ridge,
    'mlp': fit_mlp,
    'hgb': fit_hgb,
}


def main():
    out_dir = ws('WORK_SPACE', 'paper', 'E2A_recover_60')
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f'Loading features from {FEAT_180}')
    X_is = np.load(os.path.join(FEAT_180, 'X_is.npy'), mmap_mode='r')
    X_oos = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta_is = pd.read_feather(os.path.join(FEAT_180, 'meta_is.feather'))
    y_bin = meta_is['binary_label'].values
    valid = np.isin(y_bin, [0, 1])
    X_is = np.asarray(X_is[valid])
    meta_is = meta_is.loc[valid].reset_index(drop=True)

    X120_is = X_is[:, COLS_120_FROM_180]
    X120_oos = np.asarray(X_oos[:, COLS_120_FROM_180])

    tr_idx, va_idx = chrono_train_val_indices(meta_is, TRAIN_YEARS, VAL_YEARS)
    logger.info(f'train={len(tr_idx):,} val={len(va_idx):,} oos={len(X_oos):,}')

    rows = []
    for tname, tcols in TARGETS.items():
        Y_is = X_is[:, tcols]
        Y_oos = np.asarray(X_oos[:, tcols])
        X_tr, y_tr = X120_is[tr_idx], Y_is[tr_idx]
        X_va, y_va = X120_is[va_idx], Y_is[va_idx]
        logger.info(f'Target={tname} dim={Y_is.shape[1]}')

        for rname, fit_fn in REGRESSORS.items():
            logger.info(f'  fitting {rname}…')
            model, info = fit_fn(X_tr, y_tr, X_va, y_va)
            pred_oos = model.predict(X120_oos)
            pred_va = model.predict(X_va)
            acc_dims = acc_per_dim(Y_oos, pred_oos)
            r2_dims = r2_per_dim(Y_oos, pred_oos)
            acc_sum = summarize_acc(acc_dims)
            r2_sum = summarize_r2(r2_dims)
            val_r2 = float(np.nanmean(r2_per_dim(y_va, pred_va)))
            row = {
                'target': tname,
                'regressor': rname,
                'n_dims': int(Y_is.shape[1]),
                'threshold': THRESHOLD,
                'val_acc_mean': info.get('val_acc_mean'),
                'val_r2_mean': val_r2,
                'oos_acc_mean': acc_sum['acc_mean'],
                'oos_acc_median': acc_sum['acc_median'],
                'oos_r2_mean': r2_sum['r2_mean'],
                'oos_r2_median': r2_sum['r2_median'],
                'oos_r2_min': r2_sum['r2_min'],
                'oos_r2_max': r2_sum['r2_max'],
                **{k: v for k, v in info.items() if k != 'val_acc_mean'},
            }
            rows.append(row)
            joblib.dump(model, os.path.join(model_dir, f'{tname}_{rname}.joblib'))
            np.save(os.path.join(out_dir, f'acc_dims_{tname}_{rname}.npy'), acc_dims)
            np.save(os.path.join(out_dir, f'r2_dims_{tname}_{rname}.npy'), r2_dims)
            logger.info(
                f'  {rname}: val_acc={row["val_acc_mean"]:.4f} oos_acc={row["oos_acc_mean"]:.4f} '
                f'oos_r2={row["oos_r2_mean"]:.4f}'
            )

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, 'recover_summary.csv')
    summary.to_csv(summary_path, index=False, float_format='%.6f')
    with open(os.path.join(out_dir, 'recover_info.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'feat_root': FEAT_180,
            'train_years': TRAIN_YEARS,
            'val_years': VAL_YEARS,
            'metric': 'accuracy_and_r2',
            'threshold': THRESHOLD,
            'note': 'Binary indicators: regress X120->Y; Acc via threshold@0.5; R2 per-dim SS_res/SS_tot; Ridge selects alpha on val Acc',
        }, f, indent=2)
    logger.info(f'wrote {summary_path}\n{summary.to_string(index=False)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
