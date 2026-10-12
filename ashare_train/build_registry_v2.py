# -*- coding: utf-8 -*-
"""
build_registry_v2.py — 登记三个规范模型的 OOS 预测（clean_spec 第 9 节“C3 训练前补充”）。

  chartcnn_clean：第三阶段完整训练的各组中，5 个成员 loss 最优 checkpoint 的验证 Rank IC 均值最高的一组；
                  由 checkpoint 重新预测（stitch_cnn_oos.py --regenerate）并与逐年文件比对
  onedcnn_140、mlp_140：hp_numeric_140 两阶段搜参选定的组合（3d_numeric_hp_search.py），由各 seed 的权重重新预测并比对

验收：重新预测最大绝对差 ≤ 1e-4；键与 universe OOS 键完全相同；up_prob 有限且在 [0, 1]；成员数 5。
不计算任何 OOS 指标。

用法:
  python ashare_train/build_registry_v2.py [--models chartcnn_clean onedcnn_140 mlp_140]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
for p in (CODE_PKG, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from Misc.config import ASHARE_UNIVERSE_PATH, CLEAN_INPUTS_DIR  # noqa: E402

PAPER = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper')
REGISTRY = os.path.join(PAPER, 'registry_v2.json')
PRED_DIR = os.path.join(PAPER, 'registry_v2', 'preds')
CHART_ROOT = os.path.join(PAPER, 'E1c_cnn_hp_I20R5_clean')
HP_NUMERIC_ROOT = os.path.join(PAPER, 'hp_numeric_140')
SEEDS = [42, 43, 44, 45, 46]
KEYS = ['StockID', 'ending_date']
TOL = 1e-4


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, '/')


def oos_keys() -> pd.DataFrame:
    u = pd.read_feather(ASHARE_UNIVERSE_PATH, columns=KEYS + ['split'])
    u = u[u['split'] == 'oos'][KEYS].reset_index(drop=True)
    u['StockID'] = u['StockID'].astype(str)
    u['ending_date'] = pd.to_datetime(u['ending_date'])
    return u


def accept(df: pd.DataFrame, member_cols: list[str], keys: pd.DataFrame) -> dict:
    if len(member_cols) != len(SEEDS):
        raise RuntimeError(f'成员数 {len(member_cols)} != {len(SEEDS)}')
    k = df[KEYS].copy()
    k['StockID'] = k['StockID'].astype(str)
    k['ending_date'] = pd.to_datetime(k['ending_date'])
    if k.duplicated().any():
        raise RuntimeError(f'重复键 {int(k.duplicated().sum())} 行')
    m = k.merge(keys, on=KEYS, how='outer', indicator=True)
    only_pred, only_univ = int((m['_merge'] == 'left_only').sum()), int((m['_merge'] == 'right_only').sum())
    if only_pred or only_univ or len(k) != len(keys):
        raise RuntimeError(f'键与 universe OOS 不一致：预测 {len(k)}，universe {len(keys)}，'
                           f'只在预测 {only_pred}，只在 universe {only_univ}')
    up = df['up_prob'].to_numpy(dtype=float)
    if not np.isfinite(up).all() or up.min() < 0 or up.max() > 1:
        raise RuntimeError('up_prob 含非有限值或超出 [0, 1]')
    mean_gap = float(np.abs(df[member_cols].to_numpy(dtype=float).mean(axis=1) - up).max())
    if mean_gap > 1e-6:
        raise RuntimeError(f'up_prob 不等于成员均值（最大差 {mean_gap:.2e}）')
    return {'rows': int(len(df)), 'keys_equal_universe_oos': True, 'up_prob_min': float(up.min()),
            'up_prob_max': float(up.max()), 'up_prob_equals_member_mean': True}


def freeze(df: pd.DataFrame, name: str) -> tuple[str, str]:
    os.makedirs(PRED_DIR, exist_ok=True)
    path = os.path.join(PRED_DIR, f'pred_{name}_oos.parquet')
    out = df.copy()
    out['StockID'] = out['StockID'].astype(str)
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    out.to_parquet(path, index=False)
    return path, sha1_of(path)


def chart_selection() -> dict:
    import torch
    p3 = pd.read_csv(os.path.join(CHART_ROOT, 'phase3_full_train.csv'))
    rows = []
    for _, r in p3.iterrows():
        cks = sorted(glob.glob(os.path.join(r['exp_dir'], 'checkpoint*.pth.tar')))
        if len(cks) != len(SEEDS):
            raise RuntimeError(f'{r["exp_dir"]}: checkpoint 数 {len(cks)} != {len(SEEDS)}')
        rics = [float(torch.load(c, map_location='cpu', weights_only=False)['val_rank_ic']) for c in cks]
        rows.append({'lr': float(r['lr']), 'dropout': float(r['dropout']), 'patience': int(r['patience']),
                     'label': r['label'], 'exp_dir': r['exp_dir'], 'val_rank_ic_members': rics,
                     'val_rank_ic_mean': float(np.mean(rics))})
    best = max(rows, key=lambda x: x['val_rank_ic_mean'])
    return {'candidates': rows, 'selected': best}


def build_chart(keys: pd.DataFrame) -> dict:
    sel = chart_selection()
    best = sel['selected']
    tmp = os.path.join(CHART_ROOT, 'stitched_oos.parquet')
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'ashare_train', 'stitch_cnn_oos.py'),
                    '--exp-dir', best['exp_dir'], '--dataset-subdir', 'I20R5_clean', '--training', 'clean',
                    '--out', rel(tmp), '--regenerate'], check=True)
    info = json.load(open(tmp.replace('.parquet', '.json'), encoding='utf-8'))
    bad = [c for c in info['checks'] if c['same_keys'] is not True or not c['max_abs_diff'] <= TOL]
    if bad or info['n_ensembles'] != len(SEEDS):
        raise RuntimeError(f'chart-CNN 重新预测核对未通过：{bad}')
    df = pd.read_parquet(tmp)
    members = [f'up_prob_m{i}' for i in range(len(SEEDS))]
    df = df[KEYS + ['up_prob'] + members]
    acc = accept(df, members, keys)
    path, sha1 = freeze(df, 'chartcnn_clean')
    return {'path': rel(path), 'sha1': sha1, **acc, 'members': members, 'seeds': SEEDS,
            'source': rel(best['exp_dir']), 'hp': {k: best[k] for k in ('lr', 'dropout', 'patience')},
            'selection': {'rule': 'phase-3 configuration with the highest mean validation Rank IC '
                                  '(loss-best checkpoint) over its 5 members',
                          'candidates': [{**{k: c[k] for k in ('lr', 'dropout', 'patience', 'label',
                                                                'val_rank_ic_members', 'val_rank_ic_mean')},
                                          'exp_dir': rel(c['exp_dir'])} for c in sel['candidates']]},
            'regenerated_max_abs_diff': max(c['max_abs_diff'] for c in info['checks'])}


def numeric_selection(model: str) -> tuple[dict, str]:
    root = os.path.join(HP_NUMERIC_ROOT, model)
    sel = json.load(open(os.path.join(root, 'selection.json'), encoding='utf-8'))
    if sel['seeds'] != SEEDS or sel['model'] != model:
        raise RuntimeError(f'{model} 选定记录不符：{sel}')
    return sel, root


def build_numeric(keys: pd.DataFrame, model: str, name: str, tag: str) -> dict:
    """由选定组合各 seed 的权重重新预测 OOS，与 3d_numeric_hp_search.py --predict 写出的文件逐行比对。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('hp_search', os.path.join(PROJECT_ROOT, 'ashare_train',
                                                                            '3d_numeric_hp_search.py'))
    hp_search = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp_search)
    sel, root = numeric_selection(model)
    hp = {k: tuple(v) if isinstance(v, list) else v for k, v in sel['hp'].items()}
    trial_dir = os.path.join(PROJECT_ROOT, sel['trial_dir'])
    src = os.path.join(root, 'preds', f'pred_{tag}_oos.parquet')
    df = pd.read_parquet(src)
    feat = os.path.join(CLEAN_INPUTS_DIR, 'features140')
    X_oos = np.load(os.path.join(feat, 'X_oos.npy'))
    meta = pd.read_feather(os.path.join(feat, 'meta_oos.feather'))
    if not (meta['StockID'].astype(str).to_numpy() == df['StockID'].astype(str).to_numpy()).all():
        raise RuntimeError(f'{name} 预测与特征 meta 行序不同')
    members, diffs = [f'up_prob_seed{s}' for s in SEEDS], []
    for s, col in zip(SEEDS, members):
        p = hp_search.predict(model, hp, s, trial_dir, X_oos, meta)
        diffs.append(float(np.abs(p - df[col].to_numpy()).max()))
        print(f'{name} seed {s}: max|Δ|={diffs[-1]:.2e}', flush=True)
    if max(diffs) > TOL:
        raise RuntimeError(f'{name} 重新预测核对未通过：{diffs}')
    df = df[KEYS + ['up_prob'] + members]
    acc = accept(df, members, keys)
    path, sha1 = freeze(df, name)
    return {'path': rel(path), 'sha1': sha1, **acc, 'members': members, 'seeds': SEEDS, 'source': rel(src),
            'hp': sel['hp'], 'n_parameters': sel['n_parameters'], 'source_models': sel['trial_dir'].replace(os.sep, '/'),
            'selection': {'rule': sel['rule'], 'mean_member_val_rank_ic': sel['mean_member_val_rank_ic'],
                          'ensemble_val_rank_ic': sel['ensemble_val_rank_ic'],
                          'phase2': rel(os.path.join(root, 'phase2.csv'))},
            'regenerated_max_abs_diff': max(diffs)}


def build_oned(keys: pd.DataFrame) -> dict:
    return build_numeric(keys, 'oned', 'onedcnn_140', '1dcnn140')


def build_mlp(keys: pd.DataFrame) -> dict:
    return build_numeric(keys, 'mlp', 'mlp_140', 'mlp140')


BUILDERS = {'chartcnn_clean': build_chart, 'onedcnn_140': build_oned, 'mlp_140': build_mlp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(BUILDERS), choices=list(BUILDERS))
    args = ap.parse_args()
    reg = json.load(open(REGISTRY, encoding='utf-8')) if os.path.exists(REGISTRY) else {'version': 2, 'predictions': {}}
    keys = oos_keys()
    for name in args.models:
        print(f'== {name}', flush=True)
        entry = BUILDERS[name](keys)
        entry['registered_at'] = datetime.now().isoformat(timespec='seconds')
        reg['predictions'][name] = entry
        print(f'{name}: rows={entry["rows"]:,} sha1={entry["sha1"][:16]}', flush=True)
    reg['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True,
                                       text=True).stdout.strip()
    tmp = REGISTRY + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    shutil.move(tmp, REGISTRY)
    print(f'wrote {REGISTRY}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
