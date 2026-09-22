# -*- coding: utf-8 -*-
"""HistGB aligned with Bøjstrup Component Specification + IS-val grid."""
from __future__ import annotations

import itertools
import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.common.sgd_logit import FiniteFill

logger = logging.getLogger('e2_train')


def _loss_name():
    try:
        HistGradientBoostingClassifier(loss='log_loss')
        return 'log_loss'
    except ValueError:
        return 'auto'


def make_histgb(
    learning_rate,
    max_leaf_nodes,
    min_samples_leaf,
    l2_regularization,
    max_iter=300,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
):
    """
    Paper-style HistGB: no standardization / clipping; NaN/inf → 0;
    early stopping via validation_fraction on the fitted training rows.
    Uses max_leaf_nodes (not max_depth).
    """
    clf = HistGradientBoostingClassifier(
        loss=_loss_name(),
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes),
        max_depth=None,
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2_regularization),
        early_stopping=True,
        validation_fraction=float(validation_fraction),
        n_iter_no_change=int(n_iter_no_change),
        random_state=int(random_state),
    )
    return Pipeline([
        ('fill', FiniteFill()),
        ('clf', clf),
    ])


def _val_rank_ic(meta_va, scores):
    df = meta_va.copy()
    df['up_prob'] = scores
    return summarize_ic(cross_section_ic(df, 'up_prob'))['rank_ic_mean']


def search_histgb(X_tr, y_tr, X_va, y_va, meta_va, cfg_histgb):
    """Grid on IS val only. Fit uses paper-style internal validation_fraction."""
    paper = cfg_histgb.get('paper', {})
    grid = cfg_histgb['grid']
    metric = str(cfg_histgb.get('search_metric', 'val_auc'))
    seed = int(cfg_histgb.get('random_state', 42))
    max_iter = int(paper.get('max_iter', 300))
    val_frac = float(paper.get('validation_fraction', 0.1))
    n_iter_no_change = int(paper.get('n_iter_no_change', 20))

    combos = list(itertools.product(
        grid['learning_rate'],
        grid['max_leaf_nodes'],
        grid['min_samples_leaf'],
        grid['l2_regularization'],
    ))
    logger.info(f'HistGB grid size={len(combos)} metric={metric}')

    rows = []
    best = None
    best_pipe = None
    for i, (lr, leaves, min_leaf, l2) in enumerate(combos, 1):
        pipe = make_histgb(
            learning_rate=lr,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            l2_regularization=l2,
            max_iter=max_iter,
            validation_fraction=val_frac,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
        )
        pipe.fit(X_tr, y_tr)
        clf = pipe.named_steps['clf']
        classes = list(clf.classes_)
        col = classes.index(1)
        p_va = pipe.predict_proba(X_va)[:, col]
        auc = float(roc_auc_score(y_va, p_va))
        try:
            ll = float(log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6)))
        except Exception:
            ll = float('nan')
        ric = float(_val_rank_ic(meta_va, p_va))
        row = {
            'learning_rate': float(lr),
            'max_leaf_nodes': int(leaves),
            'min_samples_leaf': int(min_leaf),
            'l2_regularization': float(l2),
            'val_auc': auc,
            'val_logloss': ll,
            'val_rank_ic': ric,
            'val_acc': float(((p_va >= 0.5).astype(int) == y_va).mean()),
            'n_iter_': int(getattr(clf, 'n_iter_', -1)),
        }
        rows.append(row)
        logger.info(
            f'  [{i}/{len(combos)}] lr={lr:g} leaves={leaves} min_leaf={min_leaf} l2={l2:g}  '
            f'n_iter={row["n_iter_"]} val_auc={auc:.4f} val_rank_ic={ric:.4f}'
        )
        key = row[metric]
        if metric == 'val_logloss':
            better = best is None or key < best[metric]
        else:
            better = best is None or key > best[metric]
        if better:
            best = row
            best_pipe = pipe

    return best_pipe, rows, best
