# -*- coding: utf-8 -*-
"""
3b_cnn_hp_search.py — E1b A-share CNN hyperparameter search (chrono val, 3 phases).

用法:
  python ashare_train/3b_cnn_hp_search.py --phase 1
  python ashare_train/3b_cnn_hp_search.py --phase 2
  python ashare_train/3b_cnn_hp_search.py --phase 3
  python ashare_train/3b_cnn_hp_search.py --phase all
  python ashare_train/3b_cnn_hp_search.py --phase all --force
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Windows: avoid OpenMP dual-init crash (MKL + PyTorch)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_STOCKS_SAVEPATH, ASHARE_COUNTRY, MA_LAGS, VOLUME_BAR,
)
from Data.dgp_config import get_dataset_dir, get_dataset_filename
from Experiments.cnn_experiment import run_cnn_trial, train_us_model, _build_exp_dir
from Model.cnn_model import get_full_model_name

DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT, 'experiments', 'E1_image_baseline', 'config_cnn_hp_clean.yaml'
)


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _metric_col(cfg: dict) -> str:
    m = str(cfg.get('search_metric', 'val_rank_ic')).lower()
    if m in ('val_mcc', 'mcc'):
        return 'best_val_mcc'
    return 'best_val_rank_ic'


def _channels(cfg) -> tuple[int, ...] | None:
    ch = cfg.get('channels')
    return tuple(int(x) for x in ch) if ch else None


def _trial_suffix(patience: int, stage: str = 'hp', metric_tag: str = 'ric',
                  channels: tuple[int, ...] | None = None) -> str:
    base = f'{metric_tag}-{stage}-p{patience}'
    if channels:
        base += '-ch' + '-'.join(str(c) for c in channels)
    return base


def _trial_row(trial_id: str, phase: str, lr: float, dropout: float, patience: int,
               train_patience: int, result: dict, stage: str,
               channels: tuple[int, ...] | None = None) -> dict:
    return {
        'trial_id': trial_id,
        'phase': phase,
        'stage': stage,
        'lr': lr,
        'dropout': dropout,
        'patience': patience,
        'train_patience': train_patience,
        'channels': '-'.join(map(str, channels)) if channels else '',
        'best_val_mcc': result.get('best_val_mcc'),
        'best_val_rank_ic': result.get('best_val_rank_ic'),
        'best_val_loss': result.get('best_val_loss'),
        'best_epoch': result.get('best_epoch'),
        'wall_time_sec': result.get('wall_time_sec'),
        'exp_dir': result.get('exp_dir'),
        'dir_suffix': result.get('dir_suffix', ''),
        'training': result.get('training', ''),
        'train_up_share': result.get('train_up_share'),
        'val_up_share': result.get('val_up_share'),
    }


def _run_trial(cfg, lr, dropout, patience, train_patience, n_ensembles,
               dataset_dir, out_root, stage, trial_id, phase, force):
    ws = cfg['window_size']
    pw = cfg['predict_window']
    metric = str(cfg.get('search_metric', 'val_rank_ic'))
    tag = 'mcc' if metric in ('val_mcc', 'mcc') else 'ric'
    channels = _channels(cfg)
    result = run_cnn_trial(
        ws=ws,
        pw=pw,
        freq=cfg['freq'],
        lr=lr,
        dropout=dropout,
        is_years=cfg['train_years'],
        val_years=cfg['val_years'],
        country=cfg.get('country', ASHARE_COUNTRY),
        base_exp_dir=out_root,
        dataset_dir=dataset_dir,
        ma_lags=MA_LAGS,
        has_vb=VOLUME_BAR,
        n_ensembles=n_ensembles,
        batch_size=cfg.get('batch_size', 128),
        max_epochs=cfg.get('max_epochs', 50),
        early_stop_patience=train_patience,
        # cached checkpoints are keyed by the patience actually used for early stopping
        dir_suffix=_trial_suffix(train_patience, stage='hp', metric_tag=tag, channels=channels),
        force_retrain=force,
        search_metric=metric,
        training=cfg['training'],
        monitor=cfg.get('monitor', 'loss'),
        channels=channels,
    )
    return _trial_row(trial_id, phase, lr, dropout, patience, train_patience, result, stage,
                      channels=channels)


def _best_lr(rows: list, metric_col: str) -> float:
    df = pd.DataFrame(rows)
    return float(df.loc[df[metric_col].idxmax(), 'lr'])


def _best_dropout(rows: list, lr: float, metric_col: str) -> float:
    df = pd.DataFrame(rows)
    sub = df[df['lr'] == lr]
    return float(sub.loc[sub[metric_col].idxmax(), 'dropout'])


def run_phase1_factorial(cfg, dataset_dir, out_root, force: bool) -> pd.DataFrame:
    """与数值模型对称：全网格 × 各组自身 patience，seed/ensemble 数由 phase1.n_ensembles 决定。"""
    grid = cfg['grid']
    p1 = cfg['phases']['phase1']
    metric_col = _metric_col(cfg)
    rows = []
    n = 0
    for lr in grid['lr']:
        for dp in grid['dropout']:
            for pat in grid['patience']:
                n += 1
                rows.append(_run_trial(
                    cfg, float(lr), float(dp), int(pat), int(pat), p1['n_ensembles'],
                    dataset_dir, out_root, 'factorial', f'p1_{n:02d}', '1', force,
                ))
                logging.info(
                    f'phase1 factorial {n}/{len(grid["lr"])*len(grid["dropout"])*len(grid["patience"])} '
                    f'lr={lr:g} dp={dp} p={pat} {metric_col}={rows[-1].get(metric_col)}'
                )
    df = pd.DataFrame(rows)
    path = os.path.join(out_root, 'search_phase1.csv')
    df.to_csv(path, index=False, float_format='%.6g')
    logging.info(
        f'wrote {path}  best_{metric_col}={df[metric_col].max():.4f}  '
        f'(search_metric={cfg.get("search_metric")} mode=factorial)'
    )
    return df


def run_phase1(cfg, dataset_dir, out_root, force: bool) -> pd.DataFrame:
    p1 = cfg['phases']['phase1']
    mode = str(p1.get('mode', 'sequential')).lower()
    if mode in ('factorial', 'symmetric', 'full_grid'):
        return run_phase1_factorial(cfg, dataset_dir, out_root, force)

    grid = cfg['grid']
    anchor = cfg['anchor']
    coarse_patience = int(p1.get('patience_override', 1))
    metric_col = _metric_col(cfg)
    rows = []
    n = 0

    # 1) sweep lr
    dp_mid = anchor['dropout']
    for lr in grid['lr']:
        n += 1
        rows.append(_run_trial(
            cfg, lr, dp_mid, anchor['patience'], coarse_patience, p1['n_ensembles'],
            dataset_dir, out_root, 'lr_sweep', f'p1_{n:02d}', '1', force,
        ))
    best_lr = _best_lr(rows, metric_col)

    # 2) sweep dropout
    for dp in grid['dropout']:
        n += 1
        rows.append(_run_trial(
            cfg, best_lr, dp, anchor['patience'], coarse_patience, p1['n_ensembles'],
            dataset_dir, out_root, 'dp_sweep', f'p1_{n:02d}', '1', force,
        ))
    best_dp = _best_dropout(rows, best_lr, metric_col)

    # 3) sweep patience (grid values, train with that patience)
    for pat in grid['patience']:
        n += 1
        rows.append(_run_trial(
            cfg, best_lr, best_dp, pat, pat, p1['n_ensembles'],
            dataset_dir, out_root, 'pat_sweep', f'p1_{n:02d}', '1', force,
        ))

    # 4) anchor if missing
    if not any(r['lr'] == anchor['lr'] and r['dropout'] == anchor['dropout']
               and r['patience'] == anchor['patience'] for r in rows):
        n += 1
        rows.append(_run_trial(
            cfg, anchor['lr'], anchor['dropout'], anchor['patience'],
            anchor['patience'], p1['n_ensembles'],
            dataset_dir, out_root, 'anchor', f'p1_{n:02d}', '1', force,
        ))

    df = pd.DataFrame(rows)
    path = os.path.join(out_root, 'search_phase1.csv')
    df.to_csv(path, index=False, float_format='%.6g')
    logging.info(
        f'wrote {path}  best_{metric_col}={df[metric_col].max():.4f}  '
        f'(search_metric={cfg.get("search_metric")})'
    )
    return df


def _top_trials(df: pd.DataFrame, k: int, anchor: dict, metric_col: str) -> pd.DataFrame:
    top = (df.sort_values(metric_col, ascending=False)
             .drop_duplicates(subset=['lr', 'dropout', 'patience'])
             .head(k).copy())
    has_anchor = (
        (top['lr'] == anchor['lr']) &
        (top['dropout'] == anchor['dropout']) &
        (top['patience'] == anchor['patience'])
    ).any()
    if not has_anchor:
        anchor_rows = df[
            (df['lr'] == anchor['lr']) &
            (df['dropout'] == anchor['dropout']) &
            (df['patience'] == anchor['patience'])
        ]
        if len(anchor_rows):
            top = pd.concat([top, anchor_rows.head(1)], ignore_index=True)
    return top.drop_duplicates(subset=['lr', 'dropout', 'patience'])


def run_phase2(cfg, dataset_dir, out_root, force: bool) -> pd.DataFrame:
    p1_path = os.path.join(out_root, 'search_phase1.csv')
    if not os.path.exists(p1_path):
        raise FileNotFoundError(f'缺少 {p1_path}，请先跑 phase 1')
    p1 = pd.read_csv(p1_path)
    p2cfg = cfg['phases']['phase2']
    anchor = cfg['anchor']
    metric_col = _metric_col(cfg)
    top = _top_trials(p1, int(p2cfg['top_k']), anchor, metric_col)
    rows = []
    for i, row in top.iterrows():
        lr, dp, pat = float(row['lr']), float(row['dropout']), int(row['patience'])
        rows.append(_run_trial(
            cfg, lr, dp, pat, pat, p2cfg['n_ensembles'],
            dataset_dir, out_root, 'refine', f'p2_{i:02d}', '2', force,
        ))
    df = pd.DataFrame(rows)
    path = os.path.join(out_root, 'search_phase2.csv')
    df.to_csv(path, index=False, float_format='%.6g')
    logging.info(f'wrote {path}  best_{metric_col}={df[metric_col].max():.4f}')
    return df


def _phase3_candidates(p2: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p3 = cfg['phases']['phase3']
    anchor = cfg['anchor']
    metric_col = _metric_col(cfg)
    gap = float(p3.get('val_metric_gap_for_top2', p3.get('val_mcc_gap_for_top2', 0.002)))
    sorted_df = p2.sort_values(metric_col, ascending=False).reset_index(drop=True)
    picks = [sorted_df.iloc[0]]
    if len(sorted_df) > 1:
        if float(sorted_df.iloc[0][metric_col]) - float(sorted_df.iloc[1][metric_col]) < gap:
            picks.append(sorted_df.iloc[1])
    pick_df = pd.DataFrame(picks)
    if p3.get('always_full_train_anchor_if_not_in_top', True):
        in_top = (
            (pick_df['lr'] == anchor['lr']) &
            (pick_df['dropout'] == anchor['dropout']) &
            (pick_df['patience'] == anchor['patience'])
        ).any()
        if not in_top:
            anchor_in_p2 = p2[
                (p2['lr'] == anchor['lr']) &
                (p2['dropout'] == anchor['dropout']) &
                (p2['patience'] == anchor['patience'])
            ]
            if len(anchor_in_p2):
                pick_df = pd.concat([pick_df, anchor_in_p2.head(1)], ignore_index=True)
            else:
                pick_df = pd.concat([pick_df, pd.DataFrame([{
                    'lr': anchor['lr'], 'dropout': anchor['dropout'],
                    'patience': anchor['patience'],
                }])], ignore_index=True)
    return pick_df.drop_duplicates(subset=['lr', 'dropout', 'patience'])


def run_phase3(cfg, dataset_dir, out_root, force: bool) -> pd.DataFrame:
    p2_path = os.path.join(out_root, 'search_phase2.csv')
    if not os.path.exists(p2_path):
        raise FileNotFoundError(f'缺少 {p2_path}，请先跑 phase 2')
    p2 = pd.read_csv(p2_path)
    p3cfg = cfg['phases']['phase3']
    candidates = _phase3_candidates(p2, cfg)
    rows = []
    p3cfg = cfg['phases']['phase3']
    ws, pw, freq = cfg['window_size'], cfg['predict_window'], cfg['freq']
    model_name = get_full_model_name(ws)

    channels = _channels(cfg)
    for i, row in candidates.iterrows():
        lr = float(row['lr'])
        dp = float(row['dropout'])
        pat = int(row['patience'])
        suffix = _trial_suffix(pat, stage='full', metric_tag='ric', channels=channels)
        logging.info(f'Phase3 full train lr={lr:g} dp={dp} patience={pat} suffix={suffix} ch={channels}')
        train_us_model(
            ws_list=[cfg['window_size']],
            pw_list=[cfg['predict_window']],
            freq=cfg['freq'],
            ma_lags=MA_LAGS,
            has_vb=VOLUME_BAR,
            lr=lr,
            dropout=dp,
            n_ensembles=int(p3cfg['n_ensembles']),
            batch_size=cfg.get('batch_size', 128),
            max_epochs=cfg.get('max_epochs', 50),
            early_stop_patience=pat,
            is_years=cfg['train_years'],
            oos_years=cfg['oos_years'],
            val_years=cfg['val_years'],
            country=cfg.get('country', ASHARE_COUNTRY),
            base_exp_dir=out_root,
            dataset_dir=dataset_dir,
            dir_suffix=suffix,
            skip_oos=False,
            force_retrain=force,
            training=cfg['training'],
            monitor=cfg.get('monitor', 'loss'),
            channels=channels,
        )
        exp_leaf = _build_exp_dir(
            model_name, ws, pw, lr, dp, freq,
            len(MA_LAGS) > 0, VOLUME_BAR,
            base_exp_dir=out_root,
            dir_suffix=suffix,
        )
        rows.append({
            'lr': lr, 'dropout': dp, 'patience': pat,
            'dir_suffix': suffix,
            'channels': '-'.join(map(str, channels)) if channels else '',
            'exp_dir': exp_leaf,
            'label': 'anchor' if (
                lr == cfg['anchor']['lr'] and dp == cfg['anchor']['dropout']
                and pat == cfg['anchor']['patience']
            ) else f'candidate_{i}',
        })

    df = pd.DataFrame(rows)
    path = os.path.join(out_root, 'phase3_full_train.csv')
    df.to_csv(path, index=False)
    logging.info(f'wrote {path}')
    return df


def _apply_retrain_year(cfg: dict, retrain_year: int) -> tuple[dict, str]:
    """主分析 EW：按 R 改切分，写动态 embargo overlay，输出到 ew_main/R{R}/chart。"""
    from ashare_train.ew_panel import prepare_chart_overlay
    from ashare_train.ew_splits import assert_fair_split, ew_split

    split = ew_split(retrain_year)
    assert_fair_split(split)
    cfg = dict(cfg)
    cfg['train_years'] = list(split['train_years'])
    cfg['val_years'] = list(split['val_years'])
    cfg['oos_years'] = list(split['pred_years'])
    cfg['out_root'] = f'WORK_SPACE/paper/ew_main/R{retrain_year}/chart'
    # phase1 必须 factorial（对称）
    cfg.setdefault('phases', {}).setdefault('phase1', {})['mode'] = 'factorial'
    out_root = os.path.join(PROJECT_ROOT, cfg['out_root'].replace('/', os.sep))
    os.makedirs(out_root, exist_ok=True)
    overlay = prepare_chart_overlay(
        {'train_years': cfg['train_years'], 'val_years': cfg['val_years'],
         'pred_years': cfg['oos_years']},
        os.path.join(out_root, 'dataset_overlay'),
    )
    split_path = os.path.join(out_root, 'split.json')
    with open(split_path, 'w', encoding='utf-8') as f:
        import json
        json.dump({**split, 'overlay': str(overlay)}, f, indent=2)
    return cfg, str(overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='1')
    parser.add_argument('--force', action='store_true', help='忽略已有 checkpoint 重训')
    parser.add_argument('--retrain-year', type=int, default=None,
                        help='主分析 EW 重训年 R；改切分并写动态 embargo overlay')
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_dir_override = None
    if args.retrain_year is not None:
        cfg, dataset_dir_override = _apply_retrain_year(cfg, args.retrain_year)

    out_root = os.path.join(PROJECT_ROOT, cfg['out_root'].replace('/', os.sep))
    os.makedirs(out_root, exist_ok=True)
    # copy config snapshot
    try:
        import shutil
        cfg_dst = os.path.join(out_root, 'config_cnn_hp.yaml')
        with open(cfg_dst, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        if os.path.isfile(args.config) and args.retrain_year is None and not os.path.isfile(
                os.path.join(out_root, 'config_cnn_hp_src.yaml')):
            shutil.copy2(args.config, os.path.join(out_root, 'config_cnn_hp_src.yaml'))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(out_root, 'cnn_hp_search.log'),
                encoding='utf-8',
            ),
        ],
    )

    if dataset_dir_override:
        dataset_dir = dataset_dir_override
    else:
        base_save = os.path.join(ASHARE_STOCKS_SAVEPATH, cfg['dataset_subdir'])
        dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=base_save)
    ws, pw, freq = cfg['window_size'], cfg['predict_window'], cfg['freq']

    def available(years):
        out = []
        for y in years:
            fn_i = get_dataset_filename(ws, freq, VOLUME_BAR, MA_LAGS, y, '_images.dat')
            fn_l = get_dataset_filename(ws, freq, VOLUME_BAR, MA_LAGS, y, '_labels.feather')
            if os.path.exists(os.path.join(dataset_dir, fn_i)) and \
               os.path.exists(os.path.join(dataset_dir, fn_l)):
                out.append(y)
        return out

    for k in ('train_years', 'val_years', 'oos_years'):
        missing = sorted(set(cfg[k]) - set(available(cfg[k])))
        if missing:
            logging.error(f'{k} 缺少图像或标签文件：{missing}（{dataset_dir}）')
            return 1

    logging.info(f'out_root={out_root}  dataset={dataset_dir}')
    logging.info(
        f'train={cfg["train_years"]}  val={cfg["val_years"]}  '
        f'oos={len(cfg["oos_years"])} yrs  search_metric={cfg.get("search_metric")} '
        f'retrain_year={args.retrain_year}'
    )
    if args.phase in ('1', 'all'):
        run_phase1(cfg, dataset_dir, out_root, args.force)
    if args.phase in ('2', 'all'):
        run_phase2(cfg, dataset_dir, out_root, args.force)
    if args.phase in ('3', 'all'):
        run_phase3(cfg, dataset_dir, out_root, args.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
