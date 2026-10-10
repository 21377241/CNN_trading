"""E2F: fixed-protocol MLP depth robustness on the 120D chrono panel.

Clean-data MLP-4-140 (clean_spec 5.3):
  python experiments/E2F_mlp_depth/run_depth.py --features-root WORK_SPACE/paper/clean_inputs/features140 \
      --out-root WORK_SPACE/paper/E2F_mlp_depth_140 --depths MLP-4 --tag 140 --no-portfolio
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.splits import chrono_train_val_indices
from experiments.E2_matched_tabular.mlp_bagging import fit_mlp_bag, bag_predict_proba
from experiments.common.metrics import cross_section_ic, summarize_ic
from trend_submit.trend_code_submit.Portfolio.portfolio import run_portfolio_analysis


FEATURE_ROOT = PROJECT_ROOT / 'WORK_SPACE/paper/E2_matched_is/features'
OUT_ROOT = PROJECT_ROOT / 'WORK_SPACE/paper/E2F_mlp_depth'
SEEDS = [42, 43, 44, 45, 46]
DEPTHS = {
    'MLP-2': (128, 64),
    'MLP-3': (128, 64, 32),
    'MLP-4': (128, 64, 32, 16),
    'MLP-5': (128, 64, 32, 16, 8),
}


def seed_metrics(out: pd.DataFrame) -> dict:
    ic = summarize_ic(cross_section_ic(out, 'up_prob'))
    ew, vw = [], []
    for _, g in out.groupby('ending_date'):
        g = g.dropna(subset=['up_prob', 'ret_val', 'MarketCap']).copy()
        if len(g) < 20:
            continue
        g['q'] = pd.qcut(g['up_prob'].rank(method='first'), 10, labels=False)
        lo, hi = g[g['q'] == 0], g[g['q'] == 9]
        ew.append(float(hi['ret_val'].mean() - lo['ret_val'].mean()))
        def wmean(x):
            w = np.maximum(x.index.to_series().map(g['MarketCap']).to_numpy(dtype=float), 0.0)
            return float(np.average(x.to_numpy(dtype=float), weights=w)) if w.sum() > 0 else float(x.mean())
        vw.append(wmean(hi['ret_val']) - wmean(lo['ret_val']))
    def sharpe(x):
        x = np.asarray(x, dtype=float)
        return float(np.sqrt(52.0) * x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan
    return {'rank_ic_mean': ic['rank_ic_mean'], 'n_dates': ic['n_dates'],
            'ew_hl_sharpe': sharpe(ew), 'vw_hl_sharpe': sharpe(vw)}


def main() -> int:
    global FEATURE_ROOT, OUT_ROOT, DEPTHS
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-root', default=None)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--depths', nargs='*', default=None, help=f'subset of {list(DEPTHS)}')
    ap.add_argument('--tag', default='', help='suffix for model names and prediction files, e.g. 140')
    ap.add_argument('--no-portfolio', action='store_true',
                    help='skip run_portfolio_analysis; EW/VW Sharpe from seed_metrics on labelled rows')
    args = ap.parse_args()
    if args.features_root:
        FEATURE_ROOT = PROJECT_ROOT / args.features_root
    if args.out_root:
        OUT_ROOT = PROJECT_ROOT / args.out_root
    if args.depths:
        DEPTHS = {k: DEPTHS[k] for k in args.depths}
    if args.tag:
        DEPTHS = {f'{k}-{args.tag}': v for k, v in DEPTHS.items()}
    pred_ext = 'parquet' if args.no_portfolio else 'csv'

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    X_is = np.load(FEATURE_ROOT / 'X_is.npy')
    X_oos = np.load(FEATURE_ROOT / 'X_oos.npy')
    meta_is = pd.read_feather(FEATURE_ROOT / 'meta_is.feather')
    meta_oos = pd.read_feather(FEATURE_ROOT / 'meta_oos.feather')
    y = meta_is['binary_label'].to_numpy(dtype=int)
    valid = np.isin(y, [0, 1])
    X_is, meta_is, y = X_is[valid], meta_is.loc[valid].reset_index(drop=True), y[valid]
    tr_idx, va_idx = chrono_train_val_indices(meta_is, list(range(2000, 2006)), [2006, 2007])
    X_tr, y_tr = X_is[tr_idx], y[tr_idx]
    X_va, y_va = X_is[va_idx], y[va_idx]
    meta_va = meta_is.iloc[va_idx].reset_index(drop=True)

    rows = []
    for name, hidden in DEPTHS.items():
        model_dir = OUT_ROOT / 'models' / name
        pred_dir = OUT_ROOT / 'preds'
        port_dir = OUT_ROOT / 'portfolio' / name
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        port_dir.mkdir(parents=True, exist_ok=True)
        pipes, infos = fit_mlp_bag(
            X_tr, y_tr, X_va, y_va, meta_va,
            seeds=SEEDS, es_mode='chrono', hidden_layer_sizes=hidden,
            max_iter=200, n_iter_no_change=15, alpha=1e-4,
            lr=1e-3, monitor='rank_ic',
        )
        val_seed_scores = [pipe.predict_proba(X_va)[:, 1] for pipe in pipes]
        val_ensemble = meta_va[['ending_date', 'ret_val']].copy()
        val_ensemble['up_prob'] = np.mean(np.stack(val_seed_scores, axis=0), axis=0)
        val_ensemble_ic = summarize_ic(cross_section_ic(val_ensemble, 'up_prob'))['rank_ic_mean']
        with open(model_dir / 'fit_info.json', 'w', encoding='utf-8') as f:
            json.dump({'hidden_layer_sizes': hidden, 'seeds': SEEDS,
                       'infos': infos, 'input_dim': int(X_is.shape[1])}, f, indent=2)
        for seed, pipe in zip(SEEDS, pipes):
            joblib.dump(pipe, model_dir / f'seed{seed}.joblib')
        # Each fitted Pipeline owns its StandardScaler; pass raw features once.
        seed_scores = [pipe.predict_proba(X_oos)[:, 1] for pipe in pipes]
        score = np.mean(np.stack(seed_scores, axis=0), axis=0)
        out = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
        out['up_prob'] = score
        pred_path = pred_dir / f'pred_{name.lower().replace("-", "")}_oos.{pred_ext}'
        ew = vw = np.nan
        if args.no_portfolio:
            full = out.assign(StockID=out['StockID'].astype(str))
            for seed, s in zip(SEEDS, seed_scores):
                full[f'up_prob_seed{seed}'] = s
            full.to_parquet(pred_path, index=False)
            out = out[np.isfinite(out['ret_val'])].copy()
            sm = seed_metrics(out)
            ew, vw = sm['ew_hl_sharpe'], sm['vw_hl_sharpe']
            ic = summarize_ic(cross_section_ic(out, 'up_prob'))
            text = ''
        else:
            out.to_csv(pred_path, index=False)
            run_portfolio_analysis(result_files=[str(pred_path)], output_dir=str(port_dir),
                                   n_ensembles=5, n_decile=10, ann_factor=52)
            ic = summarize_ic(cross_section_ic(out, 'up_prob'))
            summary = port_dir / 'portfolio_summary.txt'
            text = summary.read_text(encoding='utf-8', errors='ignore') if summary.exists() else ''
        section = None
        for line in text.splitlines():
            if 'Equal-Weight' in line or '等权' in line:
                section = 'ew'
            elif 'Value-Weight' in line or '市值加权' in line:
                section = 'vw'
            if section and line.strip().startswith('H-L'):
                parts = line.split()
                if len(parts) >= 5:
                    val = float(parts[4])
                    if section == 'ew': ew = val
                    else: vw = val
        rows.append({'model': name, 'hidden_layer_sizes': str(hidden),
                     **ic, 'ew_hl_sharpe': ew, 'vw_hl_sharpe': vw,
                     'val_ensemble_rank_ic': val_ensemble_ic,
                     'n_seeds': len(SEEDS), 'n_oos': len(out)})
        seed_rows = []
        for seed, seed_score in zip(SEEDS, seed_scores):
            seed_out = out.drop(columns=['up_prob']).copy()
            seed_out['up_prob'] = seed_score[out.index.to_numpy()]
            sm = seed_metrics(seed_out)
            seed_rows.append({'model': name, 'seed': seed, **sm})
        pd.DataFrame(seed_rows).to_csv(
            OUT_ROOT / f'{name.lower().replace("-", "")}_seed_metrics.csv',
            index=False, float_format='%.6f')
    pd.DataFrame(rows).to_csv(OUT_ROOT / 'e2f_depth_summary.csv', index=False, float_format='%.6f')
    seed_df = pd.concat([
        pd.read_csv(OUT_ROOT / f'{name.lower().replace("-", "")}_seed_metrics.csv')
        for name in DEPTHS
    ], ignore_index=True)
    seed_summary = seed_df.groupby('model', as_index=False).agg(
        rank_ic_mean=('rank_ic_mean', 'mean'), rank_ic_std=('rank_ic_mean', 'std'),
        ew_hl_sharpe_mean=('ew_hl_sharpe', 'mean'), ew_hl_sharpe_std=('ew_hl_sharpe', 'std'),
        vw_hl_sharpe_mean=('vw_hl_sharpe', 'mean'), vw_hl_sharpe_std=('vw_hl_sharpe', 'std'))
    seed_summary.to_csv(OUT_ROOT / 'e2f_depth_seed_summary.csv', index=False, float_format='%.6f')
    print(pd.DataFrame(rows).to_string(index=False))
    print(seed_summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
