# -*- coding: utf-8 -*-
"""Run E2G proportional-width chart-CNN controls without retraining the baseline CNN."""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_PKG = PROJECT_ROOT / 'trend_submit' / 'trend_code_submit'
for path in (PROJECT_ROOT, CODE_PKG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Data.dgp_config import get_dataset_dir
from Experiments.cnn_experiment import _build_exp_dir, _generate_oos_predictions, _get_device, run_cnn_trial
from Misc.config import ASHARE_COUNTRY, ASHARE_STOCKS_SAVEPATH, MA_LAGS, VOLUME_BAR
from Model.cnn_model import CNNModel, get_full_model_name


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def best(rows: list[dict]) -> dict:
    valid = [r for r in rows if pd.notna(r.get('best_val_rank_ic'))]
    if not valid:
        raise RuntimeError('no valid validation Rank IC result')
    return max(valid, key=lambda r: r['best_val_rank_ic'])


def check_parameter_count(arch_name: str, arch: dict) -> dict:
    model = CNNModel(img_h=64, img_w=60, dropout=0.5, n_classes=2, window_size=20,
                     channels=tuple(arch['channels']))
    breakdown = model.parameter_breakdown()
    if model.channels != tuple(arch['channels']) or breakdown['total'] != int(arch['expected_parameters']):
        raise RuntimeError(
            f'{arch_name}: built channels={model.channels}, params={breakdown["total"]}; '
            f'expected channels={tuple(arch["channels"])}, params={arch["expected_parameters"]}'
        )
    return breakdown


def run_trial(cfg: dict, arch_name: str, arch: dict, root: Path, lr: float, dropout: float,
              patience: int, n_seeds: int, force: bool) -> dict:
    splits = cfg['splits']
    search = cfg['hyperparameter_search']
    dataset_dir = get_dataset_dir(
        ASHARE_COUNTRY,
        base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5'),
    )
    tag = f'e2g-{arch_name}-ch{"-".join(map(str, arch["channels"]))}-p{patience}'
    result = run_cnn_trial(
        ws=20, pw=5, freq='week', lr=float(lr), dropout=float(dropout),
        is_years=splits['train_years'], val_years=splits['validation_years'],
        country=ASHARE_COUNTRY, base_exp_dir=str(root), dataset_dir=dataset_dir,
        ma_lags=MA_LAGS, has_vb=VOLUME_BAR, n_ensembles=n_seeds,
        batch_size=int(search['batch_size']), max_epochs=int(search['max_epochs']),
        early_stop_patience=int(patience), dir_suffix=tag, force_retrain=force,
        ens_seed_base=int(search['final_seeds'][0]), search_metric='val_rank_ic',
        channels=tuple(arch['channels']), save_history=bool(cfg['diagnostics']['save_epoch_history']),
    )
    return {
        'architecture': arch_name, 'channels': '-'.join(map(str, arch['channels'])),
        'lr': lr, 'dropout': dropout, 'patience': patience, 'n_seeds': n_seeds,
        **result,
    }


def search_architecture(cfg: dict, arch_name: str, arch: dict, root: Path, force: bool) -> pd.DataFrame:
    search = cfg['hyperparameter_search']
    anchor = (1e-5, 0.50, 2)
    rows: list[dict] = []
    # Same sequential validation budget used by the existing chart-CNN localization.
    for lr in search['learning_rate']:
        rows.append(run_trial(cfg, arch_name, arch, root, lr, anchor[1], 1, 1, force))
    selected_lr = best(rows)['lr']
    for dropout in search['dropout']:
        rows.append(run_trial(cfg, arch_name, arch, root, selected_lr, dropout, 1, 1, force))
    selected_dropout = best([r for r in rows if r['lr'] == selected_lr])['dropout']
    for patience in search['patience']:
        rows.append(run_trial(cfg, arch_name, arch, root, selected_lr, selected_dropout, patience, 1, force))
    if not any((r['lr'], r['dropout'], r['patience']) == anchor for r in rows):
        rows.append(run_trial(cfg, arch_name, arch, root, *anchor, 1, force))

    phase1 = pd.DataFrame(rows)
    phase1.to_csv(root / f'{arch_name}_search_phase1.csv', index=False)
    # Patience=1 is only a screening device for lr/dropout; final candidates must use the registered patience grid.
    allowed = phase1[phase1['patience'].isin([int(p) for p in search['patience']])]
    top = allowed.sort_values('best_val_rank_ic', ascending=False).head(3)
    candidates = {(float(r.lr), float(r.dropout), int(r.patience)) for r in top.itertuples()}
    candidates.add(anchor)
    refine = [run_trial(cfg, arch_name, arch, root, *candidate, 1, force) for candidate in sorted(candidates)]
    phase2 = pd.DataFrame(refine)
    phase2.to_csv(root / f'{arch_name}_search_phase2.csv', index=False)
    return phase2


def full_train_and_predict(cfg: dict, arch_name: str, arch: dict, root: Path, phase2: pd.DataFrame, force: bool) -> pd.DataFrame:
    selected = best(phase2.to_dict('records'))
    final = run_trial(
        cfg, arch_name, arch, root, selected['lr'], selected['dropout'], int(selected['patience']),
        len(cfg['hyperparameter_search']['final_seeds']), force,
    )
    model_name = get_full_model_name(20)
    exp_dir = _build_exp_dir(
        model_name, 20, 5, final['lr'], final['dropout'], 'week', True, VOLUME_BAR,
        base_exp_dir=str(root),
        dir_suffix=f'e2g-{arch_name}-ch{"-".join(map(str, arch["channels"]))}-p{int(final["patience"])}',
    )
    dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5'))
    _generate_oos_predictions(
        exp_dir=exp_dir, ws=20, pw=5, img_h=64, img_w=60, freq='week', has_vb=VOLUME_BAR,
        ma_lags=MA_LAGS, n_ensembles=len(cfg['hyperparameter_search']['final_seeds']),
        oos_years=cfg['splits']['oos_years'], batch_size=int(cfg['hyperparameter_search']['batch_size']),
        device=_get_device(), country=ASHARE_COUNTRY, dropout=float(final['dropout']), model_name=model_name,
        dataset_dir=dataset_dir, channels=tuple(arch['channels']),
    )
    breakdown = check_parameter_count(arch_name, arch)
    row = {**final, 'exp_dir': exp_dir, **{f'params_{k}': v for k, v in breakdown.items()}}
    return pd.DataFrame([row])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config_chart_capacity.yaml')))
    parser.add_argument('--architecture', choices=['chartcnn_s', 'chartcnn_m', 'all'], default='all')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    base = PROJECT_ROOT / 'WORK_SPACE/paper/E2G_chart_capacity'
    base.mkdir(parents=True, exist_ok=True)
    (base / 'config_snapshot.yaml').write_text(Path(args.config).read_text(encoding='utf-8'), encoding='utf-8')
    selected = cfg['architectures'] if args.architecture == 'all' else {args.architecture: cfg['architectures'][args.architecture]}
    for name, arch in selected.items():
        check_parameter_count(name, arch)
    for name, arch in selected.items():
        root = base / name
        root.mkdir(exist_ok=True)
        phase2 = search_architecture(cfg, name, arch, root, args.force)
        final = full_train_and_predict(cfg, name, arch, root, phase2, args.force)
        final.to_csv(root / f'{name}_final_training_summary.csv', index=False)
    summaries = [pd.read_csv(p) for p in sorted(base.glob('*/*_final_training_summary.csv'))]
    pd.concat(summaries, ignore_index=True).to_csv(base / 'e2g_final_training_summary.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
