# -*- coding: utf-8 -*-
"""Shared MLP fit + bagging helpers (chrono Rank-IC ES or legacy sklearn ES)."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common.metrics import cross_section_ic, summarize_ic

logger = logging.getLogger('mlp_bagging')


def _rank_ic(meta: pd.DataFrame, scores: np.ndarray) -> float:
    df = meta[['ending_date', 'ret_val']].copy()
    df['s'] = scores
    return float(summarize_ic(cross_section_ic(df, 's'))['rank_ic_mean'])


def _clone_weights(clf: MLPClassifier) -> dict:
    return {
        'coefs_': [c.copy() for c in clf.coefs_],
        'intercepts_': [b.copy() for b in clf.intercepts_],
    }


def _restore_weights(clf: MLPClassifier, state: dict) -> None:
    clf.coefs_ = [c.copy() for c in state['coefs_']]
    clf.intercepts_ = [b.copy() for b in state['intercepts_']]


def predict_proba_pos(model, X) -> np.ndarray:
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    classes = list(clf.classes_)
    if 1 not in classes:
        raise RuntimeError(f'positive class 1 not in classes_={classes}')
    col = classes.index(1)
    return model.predict_proba(X)[:, col]


def fit_mlp_legacy(
    X_tr, y_tr, seed: int,
    hidden_layer_sizes=(128, 64),
    max_iter: int = 100,
    n_iter_no_change: int = 8,
    validation_fraction: float = 0.15,
    alpha: float = 1e-4,
    lr: float = 1e-3,
) -> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def fit_mlp_chrono_es(
    X_tr, y_tr, X_va, y_va, meta_va, seed: int,
    hidden_layer_sizes=(128, 64),
    max_iter: int = 200,
    n_iter_no_change: int = 15,
    alpha: float = 1e-4,
    lr: float = 1e-3,
    monitor: str = 'rank_ic',
) -> tuple[Pipeline, dict]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        alpha=alpha,
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        random_state=seed,
        shuffle=True,
    )
    if monitor not in ('rank_ic', 'auc', 'loss'):
        raise ValueError(f'unknown monitor {monitor}')
    best_score, best_state, best_epoch, patience = -np.inf, None, 0, 0
    best_row = {}
    history = []
    for epoch in range(1, max_iter + 1):
        clf.fit(X_tr_s, y_tr)
        col = list(clf.classes_).index(1)
        p_va = clf.predict_proba(X_va_s)[:, col]
        row = {'epoch': epoch, 'val_loss': float(log_loss(y_va, np.clip(p_va, 1e-7, 1 - 1e-7), labels=[0, 1])),
               'val_rank_ic': _rank_ic(meta_va, p_va)}
        if monitor == 'auc':
            row['val_auc'] = float(roc_auc_score(y_va, p_va))
        score = {'rank_ic': row['val_rank_ic'], 'loss': -row['val_loss']}.get(monitor, row.get('val_auc'))
        row['val_score'] = score
        history.append(row)
        if score > best_score + 1e-6:
            best_score, best_state, best_epoch, patience = score, _clone_weights(clf), epoch, 0
            best_row = row
        else:
            patience += 1
            if patience >= n_iter_no_change:
                logger.info(f'    seed={seed} stop@{epoch} best_epoch={best_epoch} best={best_score:.4f}')
                break
    else:
        logger.info(f'    seed={seed} max_iter best_epoch={best_epoch} best={best_score:.4f}')
    if best_state is not None:
        _restore_weights(clf, best_state)
    return Pipeline([('scaler', scaler), ('clf', clf)]), {
        'best_epoch': best_epoch,
        'best_val_score': float(best_score),
        'best_val_loss': float(best_row.get('val_loss', np.nan)),
        'best_val_rank_ic': float(best_row.get('val_rank_ic', np.nan)),
        'stopped_epoch': int(history[-1]['epoch']) if history else 0,
        'n_parameters': int(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_)),
        'monitor': monitor,
        'history': history,
    }


def fit_mlp_bag(
    X_tr, y_tr, X_va, y_va, meta_va,
    *,
    seeds: list[int],
    es_mode: str = 'chrono',
    hidden_layer_sizes=(128, 64),
    max_iter: int = 200,
    n_iter_no_change: int = 15,
    validation_fraction: float = 0.15,
    alpha: float = 1e-4,
    lr: float = 1e-3,
    monitor: str = 'rank_ic',
) -> tuple[list[Pipeline], list[dict[str, Any]]]:
    """Fit K MLPs; return list of pipelines and per-seed info."""
    pipes: list[Pipeline] = []
    infos: list[dict[str, Any]] = []
    es_mode = str(es_mode).lower()
    for seed in seeds:
        if es_mode == 'chrono':
            pipe, info = fit_mlp_chrono_es(
                X_tr, y_tr, X_va, y_va, meta_va, seed=seed,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                n_iter_no_change=n_iter_no_change,
                alpha=alpha,
                lr=lr,
                monitor=monitor,
            )
            info = {**info, 'seed': seed, 'es_mode': 'chrono'}
        else:
            pipe = fit_mlp_legacy(
                X_tr, y_tr, seed=seed,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                n_iter_no_change=n_iter_no_change,
                validation_fraction=validation_fraction,
                alpha=alpha,
                lr=lr,
            )
            info = {'seed': seed, 'es_mode': 'legacy', 'n_iter': int(getattr(pipe.named_steps['clf'], 'n_iter_', -1))}
        pipes.append(pipe)
        infos.append(info)
    return pipes, infos


def bag_predict_proba(pipes: list[Pipeline], X) -> np.ndarray:
    if not pipes:
        raise ValueError('empty pipe list')
    probs = np.stack([predict_proba_pos(p, X) for p in pipes], axis=0)
    return probs.mean(axis=0)
