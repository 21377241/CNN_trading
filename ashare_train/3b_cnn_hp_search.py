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
import subprocess
import sys

# Windows: avoid OpenMP dual-init crash (MKL + PyTorch)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_STOCKS_SAVEPATH, ASHARE_COUNTRY, MA_LAGS, VOLUME_BAR, ASHARE_LOG_DIR,
)
from Data.dgp_config import get_dataset_dir, get_dataset_filename
from Experiments.cnn_experiment import run_cnn_trial, train_us_model, _build_exp_dir
from Model.cnn_model import get_full_model_name

DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT, 'experiments', 'E1_image_baseline', 'config_cnn_hp.yaml'
)


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _metric_col(cfg: dict) -> str:
    m = str(cfg.get('search_metric', 'val_rank_ic')).lower()
    if m in ('val_mcc', 'mcc'):
        return 'best_val_mcc'
    return 'best_val_rank_ic'


def _trial_suffix(patience: int, stage: str = 'hp', metric_tag: str = 'ric') -> str:
    return f'{metric_tag}-{stage}-p{patience}'


def _trial_row(trial_id: str, phase: str, lr: float, dropout: float, patience: int,
               result: dict, stage: str) -> dict:
    return {
        'trial_id': trial_id,
        'phase': phase,
        'stage': stage,
        'lr': lr,
        'dropout': dropout,
        'patience': patience,
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
        dir_suffix=_trial_suffix(patience, stage='hp', metric_tag=tag),
        force_retrain=force,
        search_metric=metric,
        training=cfg.get('training', 'legacy'),
    )
    return _trial_row(trial_id, phase, lr, dropout, patience, result, stage)


def _best_lr(rows: list, metric_col: str) -> float:
    df = pd.DataFrame(rows)
    return float(df.loc[df[metric_col].idxmax(), 'lr'])


def _best_dropout(rows: list, lr: float, metric_col: str) -> float:
    df = pd.DataFrame(rows)
    sub = df[df['lr'] == lr]
    return float(sub.loc[sub[metric_col].idxmax(), 'dropout'])


def run_phase1(cfg, dataset_dir, out_root, force: bool) -> pd.DataFrame:
    grid = cfg['grid']
    anchor = cfg['anchor']
    p1 = cfg['phases']['phase1']
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
    top = df.sort_values(metric_col, ascending=False).head(k).copy()
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

    for i, row in candidates.iterrows():
        lr = float(row['lr'])
        dp = float(row['dropout'])
        pat = int(row['patience'])
        suffix = _trial_suffix(pat, stage='full', metric_tag='ric')
        logging.info(f'Phase3 full train lr={lr:g} dp={dp} patience={pat} suffix={suffix}')
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
            training=cfg.get('training', 'legacy'),
        )
        exp_leaf = _build_exp_dir(
            model_name, ws, pw, lr, dp, freq,
            len(MA_LAGS) > 0, VOLUME_BAR,
            base_exp_dir=out_root,
            dir_suffix=suffix,
        )
        if cfg.get('run_portfolio_analysis', True):
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, 'ashare_train', '4_analyze_ashare_portfolio.py'),
                    '--spec', cfg.get('spec', 'I20R5'),
                    '--exp-dir', exp_leaf,
                    '--n-ensembles', str(p3cfg['n_ensembles']),
                ],
                check=False,
            )
        rows.append({
            'lr': lr, 'dropout': dp, 'patience': pat,
            'dir_suffix': suffix,
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
    _summarize_phase3_sharpe(out_root, cfg, df)
    return df


def _summarize_phase3_sharpe(out_root: str, cfg: dict, phase3_df: pd.DataFrame):
    summaries = []
    for _, row in phase3_df.iterrows():
        summary_path = os.path.join(row.get('exp_dir', ''), 'portfolio', 'portfolio_summary.txt')
        sharpe = None
        if summary_path and os.path.isfile(summary_path):
            with open(summary_path, 'r', encoding='utf-8', errors='replace') as f:
                section = None
                for line in f:
                    if '等权' in line or 'Equal-Weight' in line:
                        section = 'ew'
                    if section == 'ew' and line.strip().startswith('H-L'):
                        parts = line.split()
                        if len(parts) >= 5:
                            sharpe = float(parts[4])
                            break
        summaries.append({
            'lr': row['lr'], 'dropout': row['dropout'], 'patience': row['patience'],
            'label': row['label'], 'ew_hl_sharpe': sharpe,
            'summary_path': summary_path or '',
        })
    out = pd.DataFrame(summaries)
    out_path = os.path.join(out_root, 'phase3_portfolio_summary.csv')
    out.to_csv(out_path, index=False)
    logging.info(f'wrote {out_path}')
    baseline = float(cfg.get('baseline_ew_hl_sharpe', 2.65))
    best = pd.to_numeric(out['ew_hl_sharpe'], errors='coerce').max()
    if pd.notna(best):
        logging.info(f'Phase3 best EW H-L Sharpe={best:.4f} vs baseline={baseline:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='1')
    parser.add_argument('--force', action='store_true', help='忽略已有 checkpoint 重训')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = os.path.join(PROJECT_ROOT, cfg['out_root'].replace('/', os.sep))
    os.makedirs(out_root, exist_ok=True)
    # copy config snapshot
    try:
        import shutil
        cfg_dst = os.path.join(out_root, 'config_cnn_hp.yaml')
        if os.path.isfile(args.config) and not os.path.isfile(cfg_dst):
            shutil.copy2(args.config, cfg_dst)
    except OSError:
        pass
    os.makedirs(ASHARE_LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(ASHARE_LOG_DIR, 'cnn_hp_search.log'),
                encoding='utf-8',
            ),
        ],
    )

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

    cfg['train_years'] = available(cfg['train_years'])
    cfg['val_years'] = available(cfg['val_years'])
    cfg['oos_years'] = available(cfg['oos_years'])
    if not cfg['train_years'] or not cfg['val_years']:
        logging.error('train/val 年份数据不足')
        return 1

    logging.info(f'out_root={out_root}  dataset={dataset_dir}')
    logging.info(
        f'train={cfg["train_years"]}  val={cfg["val_years"]}  '
        f'oos={len(cfg["oos_years"])} yrs  search_metric={cfg.get("search_metric")}'
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
