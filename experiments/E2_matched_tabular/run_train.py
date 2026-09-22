# -*- coding: utf-8 -*-
"""Train Logit / HistGB / MLP on matched features; write OOS predictions."""
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.histgb_search import make_histgb, search_histgb
from experiments.common.sgd_logit import make_sgd_logit_pipeline, search_sgd_logit
from experiments.common.splits import chrono_train_val_indices, train_val_indices
from experiments.E2_matched_tabular.mlp_bagging import bag_predict_proba, fit_mlp_bag

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2_train')


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _estimator(model):
    return model.named_steps['clf'] if hasattr(model, 'named_steps') else model


def predict_proba_pos(model, X):
    """P(y=1), not predict_proba[:,1] (that column is whichever class is index 1)."""
    clf = _estimator(model)
    classes = list(clf.classes_)
    if 1 not in classes:
        raise RuntimeError(f'positive class 1 not in classes_={classes}')
    col = classes.index(1)
    return model.predict_proba(X)[:, col]


def _write_oos_pred(name, model, X_oos, meta_oos, pred_dir, prob=None):
    if prob is None:
        prob = predict_proba_pos(model, X_oos)
    out = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    if 'up_prob' in meta_oos.columns:
        out['cnn_up_prob'] = meta_oos['up_prob'].values
    out['up_prob'] = prob
    path = os.path.join(pred_dir, f'pred_{name}_oos.csv')
    out.to_csv(path, index=False)
    logger.info(f'wrote {path} rows={len(out):,}')


def train_logit_sgd(cfg, X_tr, y_tr, X_va, y_va, meta_va, model_dir):
    lcfg = cfg['logit']
    paper = lcfg.get('paper', {})
    logit, rows, best = search_sgd_logit(X_tr, y_tr, X_va, y_va, meta_va, lcfg)
    search_df = pd.DataFrame(rows)
    search_path = os.path.join(model_dir, 'logit_sgd_search.csv')
    search_df.to_csv(search_path, index=False, float_format='%.6g')
    logger.info(f'wrote {search_path}')
    logger.info(
        f"  selected by {lcfg.get('search_metric', 'val_auc')}: "
        f"alpha={best['alpha']:g} eta0={best['eta0']:g} passes={best['max_iter']}  "
        f"val_auc={best['val_auc']:.4f} val_rank_ic={best['val_rank_ic']:.4f}"
    )

    # also fit paper-default spec on the same train split (for comparison)
    paper_pipe = make_sgd_logit_pipeline(
        paper.get('alpha', 1e-4),
        paper.get('eta0', 0.01),
        paper.get('max_iter', 3),
        lcfg.get('random_state', 47),
        float(lcfg.get('clip_low', -8)),
        float(lcfg.get('clip_high', 8)),
    )
    paper_pipe.fit(X_tr, y_tr)
    joblib.dump(logit, os.path.join(model_dir, 'logit.joblib'))
    joblib.dump(paper_pipe, os.path.join(model_dir, 'logit_paper.joblib'))
    with open(os.path.join(model_dir, 'logit_best.json'), 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'search_metric': lcfg.get('search_metric', 'val_auc')}, f, indent=2)
    return {'logit': logit, 'logit_paper': paper_pipe}


def train_histgb(cfg, X_tr, y_tr, X_va, y_va, meta_va, model_dir):
    hcfg = cfg['histgb']
    paper = hcfg.get('paper', {})
    histgb, rows, best = search_histgb(X_tr, y_tr, X_va, y_va, meta_va, hcfg)
    search_df = pd.DataFrame(rows)
    search_path = os.path.join(model_dir, 'histgb_search.csv')
    search_df.to_csv(search_path, index=False, float_format='%.6g')
    logger.info(f'wrote {search_path}')
    logger.info(
        f"  selected by {hcfg.get('search_metric', 'val_auc')}: "
        f"lr={best['learning_rate']:g} leaves={best['max_leaf_nodes']} "
        f"min_leaf={best['min_samples_leaf']} l2={best['l2_regularization']:g}  "
        f"n_iter={best['n_iter_']} val_auc={best['val_auc']:.4f} val_rank_ic={best['val_rank_ic']:.4f}"
    )

    paper_pipe = make_histgb(
        learning_rate=paper.get('learning_rate', 0.05),
        max_leaf_nodes=paper.get('max_leaf_nodes', 31),
        min_samples_leaf=paper.get('min_samples_leaf', 100),
        l2_regularization=paper.get('l2_regularization', 0.0),
        max_iter=paper.get('max_iter', 300),
        validation_fraction=paper.get('validation_fraction', 0.1),
        n_iter_no_change=paper.get('n_iter_no_change', 20),
        random_state=hcfg.get('random_state', 42),
    )
    paper_pipe.fit(X_tr, y_tr)
    joblib.dump(histgb, os.path.join(model_dir, 'histgb.joblib'))
    joblib.dump(paper_pipe, os.path.join(model_dir, 'histgb_paper.joblib'))
    with open(os.path.join(model_dir, 'histgb_best.json'), 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'search_metric': hcfg.get('search_metric', 'val_auc')}, f, indent=2)
    return {'histgb': histgb, 'histgb_paper': paper_pipe}


def train_mlp(cfg, X_tr, y_tr, X_va, y_va, meta_va, model_dir):
    """Train MLP; if mlp.bagging, fit n_seeds models and return (pipes, meta)."""
    mcfg = cfg['mlp']
    bagging = bool(mcfg.get('bagging', False))
    n_seeds = int(mcfg.get('n_seeds', 1))
    es_mode = str(mcfg.get('es_mode', 'legacy')).lower()
    base_seed = int(cfg['seed'])
    seeds = list(range(base_seed, base_seed + (n_seeds if bagging else 1)))

    if bagging or es_mode == 'chrono':
        pipes, infos = fit_mlp_bag(
            X_tr, y_tr, X_va, y_va, meta_va,
            seeds=seeds,
            es_mode=es_mode if bagging or es_mode == 'chrono' else 'legacy',
            hidden_layer_sizes=tuple(mcfg['hidden_layer_sizes']),
            max_iter=int(mcfg.get('max_iter', 200 if es_mode == 'chrono' else 100)),
            n_iter_no_change=int(mcfg.get('n_iter_no_change', 15 if es_mode == 'chrono' else 8)),
            validation_fraction=float(mcfg.get('validation_fraction', 0.15)),
            alpha=float(mcfg['alpha']),
            lr=float(mcfg['learning_rate_init']),
            monitor=str(mcfg.get('monitor', 'rank_ic')),
        )
        bag_dir = os.path.join(model_dir, 'mlp_bag')
        os.makedirs(bag_dir, exist_ok=True)
        for pipe, info in zip(pipes, infos):
            joblib.dump(pipe, os.path.join(bag_dir, f'mlp_seed{info["seed"]}.joblib'))
        meta = {
            'bagging': bagging,
            'n_seeds': len(seeds),
            'seeds': seeds,
            'es_mode': es_mode,
            'infos': [{k: v for k, v in info.items() if k != 'history'} for info in infos],
        }
        with open(os.path.join(bag_dir, 'mlp_bag_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        # keep first seed as mlp.joblib for backward compat
        joblib.dump(pipes[0], os.path.join(model_dir, 'mlp.joblib'))
        logger.info(f'  MLP bagging={bagging} es_mode={es_mode} seeds={seeds}')
        return {'pipes': pipes, 'meta': meta, 'bagging': True}

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
    logger.info(f'  val acc={mlp.score(X_va, y_va):.4f}')
    return mlp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default='', help='comma list: logit,histgb,mlp (empty = all)')
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        help='path to config yaml',
    )
    args = parser.parse_args(argv)
    only = {s.strip().lower() for s in args.only.split(',') if s.strip()}
    do_all = not only

    cfg = load_config(args.config)
    split_mode = str(cfg.get('split_mode', 'random')).lower()
    logger.info(
        f'config={args.config}  include_indicators={cfg.get("include_indicators", False)}  '
        f'split_mode={split_mode}'
    )
    out_root = os.path.join(PROJECT_ROOT, cfg['paths']['out_root'].replace('/', os.sep))
    feat_rel = cfg['paths'].get('features_root') or cfg['paths']['out_root']
    feat_dir = os.path.join(PROJECT_ROOT, str(feat_rel).replace('/', os.sep), 'features')
    if not os.path.isdir(feat_dir):
        # allow features_root to already point at .../features
        alt = os.path.join(PROJECT_ROOT, str(feat_rel).replace('/', os.sep))
        if os.path.isfile(os.path.join(alt, 'X_is.npy')):
            feat_dir = alt
    model_dir = os.path.join(out_root, 'models')
    pred_dir = os.path.join(out_root, 'preds')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    logger.info(f'features_dir={feat_dir}')
    logger.info(f'out_root={out_root}')

    X_is = np.load(os.path.join(feat_dir, 'X_is.npy'))
    meta_is = pd.read_feather(os.path.join(feat_dir, 'meta_is.feather'))
    X_oos = np.load(os.path.join(feat_dir, 'X_oos.npy'))
    meta_oos = pd.read_feather(os.path.join(feat_dir, 'meta_oos.feather'))
    y_is = meta_is['binary_label'].values.astype(int)
    valid = np.isin(y_is, [0, 1])
    n_drop = int((~valid).sum())
    if n_drop:
        logger.info(f'drop invalid IS labels (not in {{0,1}}): {n_drop:,}')
        X_is, meta_is, y_is = X_is[valid], meta_is.loc[valid].reset_index(drop=True), y_is[valid]

    if split_mode == 'chrono':
        tr_years = cfg.get('chrono_train_years', list(range(2000, 2006)))
        va_years = cfg.get('chrono_val_years', [2006, 2007])
        tr_idx, va_idx = chrono_train_val_indices(meta_is, tr_years, va_years)
        logger.info(f'chrono train_years={tr_years} val_years={va_years}')
    else:
        tr_idx, va_idx = train_val_indices(len(X_is), cfg['train_val_split'], cfg['seed'])

    X_tr, y_tr = X_is[tr_idx], y_is[tr_idx]
    X_va, y_va = X_is[va_idx], y_is[va_idx]
    meta_va = meta_is.iloc[va_idx].reset_index(drop=True)
    logger.info(f'train={len(X_tr):,} val={len(X_va):,} oos={len(X_oos):,} dim={X_is.shape[1]}')

    models = {}
    if do_all or 'logit' in only:
        logger.info('Training SGD Logit (IS-val grid)…')
        models.update(train_logit_sgd(cfg, X_tr, y_tr, X_va, y_va, meta_va, model_dir))
        logger.info(f'  classes={list(_estimator(models["logit"]).classes_)}')

    if do_all or 'histgb' in only:
        logger.info('Training HistGB (IS-val grid around paper spec)…')
        models.update(train_histgb(cfg, X_tr, y_tr, X_va, y_va, meta_va, model_dir))

    mlp_bag_result = None
    if do_all or 'mlp' in only:
        logger.info('Training MLP…')
        mlp_out = train_mlp(cfg, X_tr, y_tr, X_va, y_va, meta_va, model_dir)
        if isinstance(mlp_out, dict) and mlp_out.get('bagging'):
            mlp_bag_result = mlp_out
            models['mlp'] = mlp_out['pipes'][0]
        else:
            models['mlp'] = mlp_out

    for name, model in models.items():
        if name == 'mlp' and mlp_bag_result is not None:
            prob = bag_predict_proba(mlp_bag_result['pipes'], X_oos)
            _write_oos_pred(name, model, X_oos, meta_oos, pred_dir, prob=prob)
            with open(os.path.join(model_dir, 'mlp_bag', 'mlp_bag_meta.json'), 'r', encoding='utf-8') as f:
                bag_meta = json.load(f)
            bag_meta['oos_pred'] = 'mean_of_seed_probs'
            with open(os.path.join(model_dir, 'mlp_bag', 'mlp_bag_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(bag_meta, f, indent=2)
        else:
            _write_oos_pred(name, model, X_oos, meta_oos, pred_dir)

    # Do NOT overwrite E1b-ric CNN preds: only write if missing (meta up_prob is not the paper CNN).
    cnn_path = os.path.join(pred_dir, 'pred_cnn_oos.csv')
    if not os.path.exists(cnn_path):
        cnn = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label', 'up_prob']].copy()
        cnn.to_csv(cnn_path, index=False)
        logger.info(f'wrote placeholder {cnn_path} (no existing CNN pred)')
    else:
        logger.info(f'kept existing CNN pred: {cnn_path}')

    info_path = os.path.join(model_dir, 'train_info.json')
    info = {}
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    info.update({
        'n_train': int(len(X_tr)),
        'n_val': int(len(X_va)),
        'n_oos': int(len(X_oos)),
        'dim': int(X_is.shape[1]),
        'seed': cfg['seed'],
        'logit_estimator': 'SGDClassifier',
        'split_mode': split_mode,
        'features_dir': feat_dir,
        'out_root': out_root,
    })
    if split_mode == 'chrono':
        info['chrono_train_years'] = list(cfg.get('chrono_train_years', list(range(2000, 2006))))
        info['chrono_val_years'] = list(cfg.get('chrono_val_years', [2006, 2007]))
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    logger.info('[OK] training done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
