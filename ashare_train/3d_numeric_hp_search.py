# -*- coding: utf-8 -*-
"""
3d_numeric_hp_search.py - 数值模型（MLP-140、1D-CNN-140）的两阶段搜参（clean_spec 第 9 节“数值模型的搜参”）

  python ashare_train/3d_numeric_hp_search.py --model mlp --phase all
  python ashare_train/3d_numeric_hp_search.py --model oned --phase all
  python ashare_train/3d_numeric_hp_search.py --model oned --predict     # 选定组合的 OOS 预测

第一阶段：全网格，seed 42。第二阶段：第一阶段验证 Rank IC 前 3 名，加上锚点（原固定设定），seed 42-46
（seed 42 沿用第一阶段的模型）。选定：5 个成员验证 Rank IC 均值最高者，均值之差 < 1e-4 时取参数少者。
早停按验证 Rank IC；试验得分为 Rank IC 最优 checkpoint 的验证 Rank IC。只输出验证集指标。

输出：WORK_SPACE/paper/hp_numeric_140/{mlp,oned}/
  trials/<名称>/models/seed<s>/   模型与 info.json
  phase1.csv、phase2.csv、selection.json、preds/pred_<tag>_oos.parquet
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.metrics import cross_section_ic, summarize_ic  # noqa: E402
from experiments.common.splits import chrono_train_val_indices  # noqa: E402

FEATURE_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'clean_inputs', 'features140')
OUT_BASE = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'hp_numeric_140')
ONED_BASE_CONFIG = os.path.join(PROJECT_ROOT, 'experiments', 'E2C_temporal', 'config_1dcnn_140.yaml')
TRAIN_YEARS, VAL_YEARS = list(range(2000, 2006)), [2006, 2007]
SEEDS = [42, 43, 44, 45, 46]
PHASE1_SEED = 42
TOP_K = 3
TIE_TOL = 1e-4

GRIDS = {
    'mlp': {
        'hidden': [(128, 64), (128, 64, 32), (128, 64, 32, 16), (128, 64, 32, 16, 8)],
        'lr': [3e-4, 1e-3, 3e-3],
        'alpha': [1e-5, 1e-4, 1e-3],
    },
    'oned': {
        'channels': [(16, 32), (32, 64), (64, 128), (16, 32, 64), (32, 64, 128), (64, 128, 256)],
        'dropout': [0.1, 0.25, 0.5],
        'lr': [3e-4, 1e-3],
    },
}
ANCHORS = {
    'mlp': {'hidden': (128, 64, 32, 16), 'lr': 1e-3, 'alpha': 1e-4},
    'oned': {'channels': (32, 64), 'dropout': 0.25, 'lr': 1e-3},
}
MLP_FIXED = {'max_iter': 200, 'n_iter_no_change': 15}
MONITOR = 'rank_ic'
PRED_TAG = {'mlp': 'mlp140', 'oned': '1dcnn140'}

logger = logging.getLogger('numeric_hp')


def trial_name(model: str, hp: dict) -> str:
    if model == 'mlp':
        return f"h{'-'.join(map(str, hp['hidden']))}_lr{hp['lr']:g}_a{hp['alpha']:g}"
    return f"c{'-'.join(map(str, hp['channels']))}_dp{hp['dropout']:g}_lr{hp['lr']:g}"


def grid(model: str) -> list[dict]:
    g = GRIDS[model]
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def load_data():
    X_is = np.load(os.path.join(FEATURE_ROOT, 'X_is.npy'))
    meta_is = pd.read_feather(os.path.join(FEATURE_ROOT, 'meta_is.feather'))
    meta_is['ending_date'] = pd.to_datetime(meta_is['ending_date'])
    y = meta_is['binary_label'].to_numpy(dtype=int)
    valid = np.isin(y, [0, 1])
    X_is, meta_is, y = X_is[valid], meta_is.loc[valid].reset_index(drop=True), y[valid]
    tr_idx, va_idx = chrono_train_val_indices(meta_is, TRAIN_YEARS, VAL_YEARS)
    return X_is, y, meta_is, tr_idx, va_idx


def oned_cfg(hp: dict) -> dict:
    with open(ONED_BASE_CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg['model']['hidden_channels'] = list(hp['channels'])
    cfg['model']['dropout'] = float(hp['dropout'])
    cfg['train']['learning_rate'] = float(hp['lr'])
    cfg['train']['monitor'] = MONITOR
    return cfg


def run_seed(model: str, hp: dict, seed: int, data, out_dir: str) -> dict:
    seed_dir = os.path.join(out_dir, 'models', f'seed{seed}')
    info_path = os.path.join(seed_dir, 'info.json')
    if os.path.exists(info_path):
        with open(info_path, encoding='utf-8') as f:
            info = json.load(f)
        if info.get('monitor') != MONITOR:
            raise RuntimeError(f'{info_path} 按 monitor={info.get("monitor")} 训练，当前要求 {MONITOR}')
        return info
    os.makedirs(seed_dir, exist_ok=True)
    X_is, y, meta_is, tr_idx, va_idx = data
    t0 = time.time()
    if model == 'mlp':
        from experiments.E2_matched_tabular.mlp_bagging import fit_mlp_chrono_es
        pipe, info = fit_mlp_chrono_es(
            X_is[tr_idx], y[tr_idx], X_is[va_idx], y[va_idx], meta_is.iloc[va_idx].reset_index(drop=True),
            seed=seed, hidden_layer_sizes=hp['hidden'], alpha=hp['alpha'], lr=hp['lr'], monitor=MONITOR, **MLP_FIXED)
        joblib.dump(pipe, os.path.join(seed_dir, 'model.joblib'))
        pd.DataFrame(info.pop('history')).to_csv(os.path.join(seed_dir, 'history.csv'), index=False)
        info['seed'] = seed
    else:
        from experiments.E2C_temporal.train_1dcnn import fit_seed
        info = fit_seed(oned_cfg(hp), seed, X_is, y, meta_is, tr_idx, va_idx, out_dir)
    info['seconds'] = round(time.time() - t0, 1)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return info


def predict(model: str, hp: dict, seed: int, out_dir: str, X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    if model == 'mlp':
        pipe = joblib.load(os.path.join(out_dir, 'models', f'seed{seed}', 'model.joblib'))
        clf = pipe.named_steps['clf']
        return pipe.predict_proba(X)[:, list(clf.classes_).index(1)]
    from experiments.E2C_temporal.train_1dcnn import predict_oos_seed
    return predict_oos_seed(oned_cfg(hp), seed, X, meta, out_dir)


def trial_row(model: str, hp: dict, info: dict) -> dict:
    return {**{k: str(v) if isinstance(v, tuple) else v for k, v in hp.items()},
            'name': trial_name(model, hp), 'seed': info['seed'], 'val_rank_ic': info['best_val_rank_ic'],
            'val_loss': info['best_val_loss'], 'best_epoch': info['best_epoch'],
            'stopped_epoch': info.get('stopped_epoch'), 'n_parameters': info['n_parameters'],
            'seconds': info.get('seconds')}


def phase1(model: str, data, root: str) -> pd.DataFrame:
    rows = []
    hps = grid(model)
    for i, hp in enumerate(hps, 1):
        info = run_seed(model, hp, PHASE1_SEED, data, os.path.join(root, 'trials', trial_name(model, hp)))
        rows.append(trial_row(model, hp, info))
        logger.info(f'phase1 {i}/{len(hps)} {rows[-1]["name"]}: val_rank_ic={info["best_val_rank_ic"]:.5f} '
                    f'val_loss={info["best_val_loss"]:.5f} best_epoch={info["best_epoch"]}')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, 'phase1.csv'), index=False, float_format='%.6f')
    return df


def phase2(model: str, data, root: str) -> dict:
    p1 = pd.read_csv(os.path.join(root, 'phase1.csv'))
    names = list(p1.sort_values('val_rank_ic', ascending=False)['name'].head(TOP_K))
    anchor = trial_name(model, ANCHORS[model])
    if anchor not in names:
        names.append(anchor)
    hp_of = {trial_name(model, hp): hp for hp in grid(model)}
    X_is, y, meta_is, tr_idx, va_idx = data
    meta_va = meta_is.iloc[va_idx].reset_index(drop=True)
    rows = []
    for name in names:
        hp, out_dir = hp_of[name], os.path.join(root, 'trials', name)
        infos = [run_seed(model, hp, s, data, out_dir) for s in SEEDS]
        val_pred = np.mean([predict(model, hp, s, out_dir, X_is[va_idx], meta_va) for s in SEEDS], axis=0)
        ens = summarize_ic(cross_section_ic(meta_va[['ending_date', 'ret_val']].assign(up_prob=val_pred), 'up_prob'))
        rics = [i['best_val_rank_ic'] for i in infos]
        rows.append({'name': name, **{k: str(v) if isinstance(v, tuple) else v for k, v in hp.items()},
                     'is_anchor': name == anchor, 'in_top_k': name in names[:TOP_K],
                     'member_val_rank_ic': json.dumps([round(r, 6) for r in rics]),
                     'mean_member_val_rank_ic': float(np.mean(rics)), 'ensemble_val_rank_ic': float(ens['rank_ic_mean']),
                     'n_parameters': int(infos[0]['n_parameters'])})
        logger.info(f'phase2 {name}: members={rics} mean={np.mean(rics):.5f} ensemble={ens["rank_ic_mean"]:.5f}')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, 'phase2.csv'), index=False, float_format='%.6f')
    best = df['mean_member_val_rank_ic'].max()
    tied = df[df['mean_member_val_rank_ic'] >= best - TIE_TOL]
    pick = tied.sort_values(['n_parameters', 'mean_member_val_rank_ic'], ascending=[True, False]).iloc[0]
    sel = {'model': model, 'name': pick['name'], 'hp': {k: list(v) if isinstance(v, tuple) else v
                                                        for k, v in hp_of[pick['name']].items()},
           'seeds': SEEDS, 'trial_dir': os.path.relpath(os.path.join(root, 'trials', pick['name']), PROJECT_ROOT),
           'mean_member_val_rank_ic': float(pick['mean_member_val_rank_ic']),
           'ensemble_val_rank_ic': float(pick['ensemble_val_rank_ic']), 'n_parameters': int(pick['n_parameters']),
           'n_tied_within_tol': int(len(tied)), 'rule': 'max mean member val Rank IC; ties (< 1e-4) -> fewer parameters'}
    with open(os.path.join(root, 'selection.json'), 'w', encoding='utf-8') as f:
        json.dump(sel, f, indent=2)
    logger.info(f'selected {sel["name"]}: mean member val Rank IC {sel["mean_member_val_rank_ic"]:.5f}')
    return sel


def predict_selected(model: str, root: str) -> str:
    with open(os.path.join(root, 'selection.json'), encoding='utf-8') as f:
        sel = json.load(f)
    hp = {k: tuple(v) if isinstance(v, list) else v for k, v in sel['hp'].items()}
    out_dir = os.path.join(PROJECT_ROOT, sel['trial_dir'])
    X_oos = np.load(os.path.join(FEATURE_ROOT, 'X_oos.npy'))
    meta_oos = pd.read_feather(os.path.join(FEATURE_ROOT, 'meta_oos.feather'))
    out = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    out['StockID'] = out['StockID'].astype(str)
    preds = {s: predict(model, hp, s, out_dir, X_oos, meta_oos) for s in sel['seeds']}
    out['up_prob'] = np.mean(list(preds.values()), axis=0)
    for s, p in preds.items():
        out[f'up_prob_seed{s}'] = p
    os.makedirs(os.path.join(root, 'preds'), exist_ok=True)
    path = os.path.join(root, 'preds', f'pred_{PRED_TAG[model]}_oos.parquet')
    out.to_parquet(path, index=False)
    logger.info(f'wrote {path} rows={len(out):,}')
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['mlp', 'oned'], required=True)
    ap.add_argument('--phase', choices=['1', '2', 'all', 'none'], default='all')
    ap.add_argument('--predict', action='store_true', help='OOS predictions for the selected configuration')
    args = ap.parse_args(argv)
    root = os.path.join(OUT_BASE, args.model)
    os.makedirs(root, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(root, 'search.log'),
                                                                               encoding='utf-8')])
    if args.phase != 'none':
        data = load_data()
        logger.info(f'{args.model}: train {len(data[3]):,} val {len(data[4]):,}; grid {len(grid(args.model))}')
        if args.phase in ('1', 'all'):
            phase1(args.model, data, root)
        if args.phase in ('2', 'all'):
            phase2(args.model, data, root)
    if args.predict:
        predict_selected(args.model, root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
