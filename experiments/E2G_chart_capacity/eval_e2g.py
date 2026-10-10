# -*- coding: utf-8 -*-
"""Evaluate E2G width-scaled chart-CNNs against the full chart-CNN, 1D-CNN-120 and MLP-4-120."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.E2C_temporal.eval_e2c import nw_t_mean, score_summary
from experiments.common.metrics import cross_section_ic

ANN_FACTOR = 52
KEYS = ['StockID', 'ending_date']
BASE = PROJECT_ROOT / 'WORK_SPACE/paper/E2G_chart_capacity'
BENCHMARKS = {
    'chartcnn_full': ('WORK_SPACE/paper/E2_matched_is_ind_chrono_ric_mlp4/preds/pred_cnn_oos.csv', 708866),
    '1dcnn_120': ('WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv', 7073),
    'mlp4_120': ('WORK_SPACE/paper/E2F_mlp_depth/preds/pred_mlp4_oos.csv', 26369),
}


def load_e2g_scores(exp_dir: Path) -> pd.DataFrame:
    files = sorted((exp_dir / 'ensem_res').glob('ensem5_res_*.csv'))
    if len(files) != 18:
        raise RuntimeError(f'{exp_dir}: expected 18 OOS years, found {len(files)}')
    df = pd.concat([pd.read_csv(f, usecols=KEYS + ['up_prob']) for f in files], ignore_index=True)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    return df


def training_diagnostics(exp_dir: Path) -> dict:
    rows = []
    for f in sorted(exp_dir.glob('history_seed*.csv')):
        h = pd.read_csv(f)
        best = h.loc[h['val_rank_ic'].idxmax()]
        rows.append({
            'best_epoch': int(best['epoch']),
            'stop_epoch': int(h['epoch'].max()),
            'val_rank_ic': float(best['val_rank_ic']),
            'train_loss': float(best['train_loss']),
            'val_loss': float(best['val_loss']),
        })
    d = pd.DataFrame(rows)
    return {
        'n_seed_histories': len(d),
        'seed_val_rank_ic_mean': d['val_rank_ic'].mean(),
        'train_loss_at_best': d['train_loss'].mean(),
        'val_loss_at_best': d['val_loss'].mean(),
        'train_val_gap': (d['val_loss'] - d['train_loss']).mean(),
        'best_epoch_mean': d['best_epoch'].mean(),
        'stop_epoch_mean': d['stop_epoch'].mean(),
    }


def mean_rank_corr(df: pd.DataFrame, a: str, b: str) -> float:
    vals = [spearmanr(g[a], g[b])[0] for _, g in df.groupby('ending_date') if len(g) >= 10]
    return float(np.nanmean(vals))


def main() -> int:
    summary = pd.read_csv(BASE / 'e2g_final_training_summary.csv')
    base_path, _ = BENCHMARKS['chartcnn_full']
    df = pd.read_csv(PROJECT_ROOT / base_path, usecols=KEYS + ['ret_val', 'MarketCap', 'binary_label', 'up_prob'])
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df = df.rename(columns={'up_prob': 'chartcnn_full'})
    for name, (path, _) in BENCHMARKS.items():
        if name == 'chartcnn_full':
            continue
        b = pd.read_csv(PROJECT_ROOT / path, usecols=KEYS + ['up_prob'])
        b['ending_date'] = pd.to_datetime(b['ending_date'])
        df = df.merge(b.rename(columns={'up_prob': name}), on=KEYS, how='inner')

    params = {name: p for name, (_, p) in BENCHMARKS.items()}
    diag_rows = []
    for r in summary.itertuples():
        exp_dir = Path(r.exp_dir)
        s = load_e2g_scores(exp_dir)
        df = df.merge(s.rename(columns={'up_prob': r.architecture}), on=KEYS, how='inner')
        params[r.architecture] = int(r.params_total)
        diag_rows.append({
            'model': r.architecture, 'channels': r.channels, 'lr': r.lr, 'dropout': r.dropout,
            'patience': int(r.patience), 'params_conv': int(r.params_conv),
            'params_bn': int(r.params_batch_norm), 'params_head': int(r.params_head),
            'params_total': int(r.params_total), 'selected_val_rank_ic': r.best_val_rank_ic,
            **training_diagnostics(exp_dir),
        })
    df = df.dropna(subset=['ret_val']).reset_index(drop=True)
    models = list(summary['architecture']) + [m for m in BENCHMARKS]

    standalone = []
    for m in models:
        ic = cross_section_ic(df, m)
        standalone.append({
            'model': m, 'params': params[m],
            **score_summary(df.rename(columns={m: 'score'}), 'score', ANN_FACTOR),
            'rank_ic_nw_t': nw_t_mean(ic['rank_ic'].values),
        })
    standalone = pd.DataFrame(standalone)

    corr = pd.DataFrame(
        [{'model': m, **{f'rho_{o}': mean_rank_corr(df, m, o) for o in BENCHMARKS}} for m in summary['architecture']]
    )
    diag = pd.DataFrame(diag_rows)

    res = BASE / 'results'
    res.mkdir(exist_ok=True)
    standalone.to_csv(res / 'e2g_standalone.csv', index=False)
    corr.to_csv(res / 'e2g_rank_corr.csv', index=False)
    diag.to_csv(res / 'e2g_training_diagnostics.csv', index=False)
    md = [
        '# E2G chart-CNN capacity control (common OOS sample)',
        '',
        f'Common sample: {len(df):,} stock-weeks, {df["ending_date"].nunique()} weeks, '
        f'{df["ending_date"].min().date()} to {df["ending_date"].max().date()}.',
        '',
        '## Standalone', '', standalone.to_markdown(index=False, floatfmt='.4f'), '',
        '## Mean daily Spearman correlation with benchmarks', '', corr.to_markdown(index=False, floatfmt='.3f'), '',
        '## Training diagnostics', '', diag.round({c: 4 for c in diag.columns if c != 'lr'}).to_markdown(index=False, floatfmt='g'),
    ]
    (res / 'e2g_report.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
