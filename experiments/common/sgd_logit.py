# -*- coding: utf-8 -*-
"""SGD logistic pipeline aligned with Bøjstrup-style Component Specification."""
from __future__ import annotations

import itertools
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common.metrics import cross_section_ic, summarize_ic

logger = logging.getLogger('e2_train')


def _log_loss_name():
    try:
        SGDClassifier(loss='log_loss')
        return 'log_loss'
    except ValueError:
        return 'log'


class FiniteFill(BaseEstimator, TransformerMixin):
    """Replace NaN / inf with 0 before fitting (paper spec)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class ClipValues(BaseEstimator, TransformerMixin):
    def __init__(self, lo=-8.0, hi=8.0):
        self.lo = lo
        self.hi = hi

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float64), self.lo, self.hi)


def make_sgd_logit_pipeline(alpha, eta0, max_iter, random_state, clip_lo=-8.0, clip_hi=8.0):
    """Standardize → clip [-8, 8] → SGD logistic (ℓ2, constant lr, averaged)."""
    clf = SGDClassifier(
        loss=_log_loss_name(),
        penalty='l2',
        alpha=float(alpha),
        fit_intercept=True,
        learning_rate='constant',
        eta0=float(eta0),
        average=True,
        max_iter=int(max_iter),
        tol=None,
        shuffle=True,
        early_stopping=False,
        random_state=int(random_state),
    )
    return Pipeline([
        ('fill', FiniteFill()),
        ('scaler', StandardScaler()),
        ('clip', ClipValues(lo=clip_lo, hi=clip_hi)),
        ('clf', clf),
    ])


def _val_rank_ic(meta_va, scores):
    df = meta_va.copy()
    df['up_prob'] = scores
    return summarize_ic(cross_section_ic(df, 'up_prob'))['rank_ic_mean']


def search_sgd_logit(X_tr, y_tr, X_va, y_va, meta_va, cfg_logit):
    """Grid on IS val only. Returns (best_pipeline_fitted_on_train, search_rows, best_row)."""
    grid = cfg_logit['grid']
    metric = str(cfg_logit.get('search_metric', 'val_auc'))
    seed = int(cfg_logit.get('random_state', 47))
    clip_lo = float(cfg_logit.get('clip_low', -8))
    clip_hi = float(cfg_logit.get('clip_high', 8))

    combos = list(itertools.product(grid['alpha'], grid['eta0'], grid['max_iter']))
    logger.info(f'SGD Logit grid size={len(combos)} metric={metric}')

    rows = []
    best = None
    best_pipe = None
    for i, (alpha, eta0, n_pass) in enumerate(combos, 1):
        pipe = make_sgd_logit_pipeline(alpha, eta0, n_pass, seed, clip_lo, clip_hi)
        pipe.fit(X_tr, y_tr)
        classes = list(pipe.named_steps['clf'].classes_)
        col = classes.index(1)
        p_va = pipe.predict_proba(X_va)[:, col]
        auc = float(roc_auc_score(y_va, p_va))
        try:
            ll = float(log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6)))
        except Exception:
            ll = float('nan')
        ric = float(_val_rank_ic(meta_va, p_va))
        row = {
            'alpha': float(alpha),
            'eta0': float(eta0),
            'max_iter': int(n_pass),
            'val_auc': auc,
            'val_logloss': ll,
            'val_rank_ic': ric,
            'val_acc': float(((p_va >= 0.5).astype(int) == y_va).mean()),
            'n_iter_': int(getattr(pipe.named_steps['clf'], 'n_iter_', n_pass)),
        }
        rows.append(row)
        logger.info(
            f'  [{i}/{len(combos)}] alpha={alpha:g} eta0={eta0:g} passes={n_pass}  '
            f'val_auc={auc:.4f} val_rank_ic={ric:.4f} val_logloss={ll:.4f}'
        )
        key = row[metric]
        # AUC/Rank IC: higher better; logloss: lower better
        if metric == 'val_logloss':
            better = best is None or key < best[metric]
        else:
            better = best is None or key > best[metric]
        if better:
            best = row
            best_pipe = pipe

    return best_pipe, rows, best
