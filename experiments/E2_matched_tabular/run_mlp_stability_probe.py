# -*- coding: utf-8 -*-
"""
MLP stability probe (extended): Rank IC + H-L Sharpe + seed corr + bagging curve.

Protocols:
  A) legacy_es: sklearn random validation_fraction early stop
  B) chrono_es: fixed chrono val Rank-IC early stop (warm_start)
  C) bagging curve k=1..K on chrono probs; also legacy bag at K

Usage:
  python experiments/E2_matched_tabular/run_mlp_stability_probe.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.common.splits import chrono_train_val_indices
from experiments.E2_matched_tabular.e2a_layout import FEAT_180, ws
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from experiments.E2_matched_tabular.run_train import predict_proba_pos
from Portfolio.portfolio import run_portfolio_analysis

warnings.filterwarnings('ignore', category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('mlp_stab')

OUT_DIR = ws('WORK_SPACE', 'paper', 'E2_mlp_stability_probe')
ANN = 52


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


def fit_mlp_legacy(
    X_tr, y_tr, seed: int, max_iter: int = 100, n_iter_no_change: int = 8,
    validation_fraction: float = 0.15, alpha: float = 1e-4, lr: float = 1e-3,
) -> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=alpha, learning_rate_init=lr,
            max_iter=max_iter, early_stopping=True, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, random_state=seed,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def fit_mlp_chrono_es(
    X_tr, y_tr, X_va, y_va, meta_va, seed: int,
    max_iter: int = 200, n_iter_no_change: int = 15,
    alpha: float = 1e-4, lr: float = 1e-3, monitor: str = 'rank_ic',
) -> tuple[Pipeline, dict]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=alpha, learning_rate_init=lr,
        max_iter=1, warm_start=True, early_stopping=False, random_state=seed, shuffle=True,
    )
    best_score, best_state, best_epoch, patience = -np.inf, None, 0, 0
    history = []
    for epoch in range(1, max_iter + 1):
        clf.fit(X_tr_s, y_tr)
        col = list(clf.classes_).index(1)
        p_va = clf.predict_proba(X_va_s)[:, col]
        score = float(roc_auc_score(y_va, p_va)) if monitor == 'auc' else _rank_ic(meta_va, p_va)
        history.append({'epoch': epoch, 'val_score': score})
        if score > best_score + 1e-6:
            best_score, best_state, best_epoch, patience = score, _clone_weights(clf), epoch, 0
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
        'best_epoch': best_epoch, 'best_val_score': float(best_score),
        'stopped_epoch': int(history[-1]['epoch']) if history else 0,
        'monitor': monitor, 'history': history,
    }


def eval_ic(prob: np.ndarray, meta_oos: pd.DataFrame) -> dict:
    df = meta_oos[['ending_date', 'ret_val']].copy()
    df['up_prob'] = prob
    return summarize_ic(cross_section_ic(df, 'up_prob'))


def eval_portfolio(prob: np.ndarray, meta_oos: pd.DataFrame, out_dir: str, tag: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    pred = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    pred['up_prob'] = prob
    csv_path = os.path.join(out_dir, f'pred_{tag}.csv')
    pred.to_csv(csv_path, index=False)
    run_portfolio_analysis(
        result_files=[csv_path], output_dir=out_dir, n_ensembles=5, n_decile=10, ann_factor=ANN,
    )
    return parse_hl_sharpe(os.path.join(out_dir, 'portfolio_summary.txt'))


def pairwise_mean_spearman(prob_mat: np.ndarray) -> float:
    """prob_mat: (n_seeds, n_samples). Mean off-diagonal Spearman."""
    K = prob_mat.shape[0]
    if K < 2:
        return float('nan')
    vals = []
    for i in range(K):
        for j in range(i + 1, K):
            # subsample for speed if huge
            n = prob_mat.shape[1]
            if n > 300_000:
                rng = np.random.default_rng(0)
                idx = rng.choice(n, size=300_000, replace=False)
                a, b = prob_mat[i, idx], prob_mat[j, idx]
            else:
                a, b = prob_mat[i], prob_mat[j]
            r, _ = spearmanr(a, b)
            if np.isfinite(r):
                vals.append(float(r))
    return float(np.mean(vals)) if vals else float('nan')


def bagging_curve(prob_mat: np.ndarray, meta_oos: pd.DataFrame, port_root: str, prefix: str) -> pd.DataFrame:
    """Incremental mean of first k seeds; IC + Sharpe."""
    rows = []
    for k in range(1, prob_mat.shape[0] + 1):
        bag = prob_mat[:k].mean(axis=0)
        ic = eval_ic(bag, meta_oos)
        hl = eval_portfolio(bag, meta_oos, os.path.join(port_root, f'{prefix}_bag_k{k}'), f'{prefix}_bag_k{k}')
        rows.append({'protocol': prefix, 'k': k, **ic, **hl})
        logger.info(
            f'  bag {prefix} k={k}: rank_ic={ic["rank_ic_mean"]:.4f} '
            f'ew_sharpe={hl.get("ew_hl_sharpe")}'
        )
    return pd.DataFrame(rows)


def protocol_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for proto, g in detail.groupby('protocol'):
        if proto.endswith('_bag') or proto.startswith('bag_'):
            continue
        row = {
            'protocol': proto,
            'n_runs': int(len(g)),
            'rank_ic_mean': float(g['rank_ic_mean'].mean()),
            'rank_ic_std': float(g['rank_ic_mean'].std(ddof=1)) if len(g) > 1 else 0.0,
            'rank_ic_se': float(g['rank_ic_mean'].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0,
            'rank_ic_cv': float(g['rank_ic_mean'].std(ddof=1) / abs(g['rank_ic_mean'].mean())) if len(g) > 1 else 0.0,
            'rank_ic_min': float(g['rank_ic_mean'].min()),
            'rank_ic_max': float(g['rank_ic_mean'].max()),
            'ew_sharpe_mean': float(g['ew_hl_sharpe'].mean()) if 'ew_hl_sharpe' in g else np.nan,
            'ew_sharpe_std': float(g['ew_hl_sharpe'].std(ddof=1)) if 'ew_hl_sharpe' in g and len(g) > 1 else 0.0,
            'ew_sharpe_min': float(g['ew_hl_sharpe'].min()) if 'ew_hl_sharpe' in g else np.nan,
            'ew_sharpe_max': float(g['ew_hl_sharpe'].max()) if 'ew_hl_sharpe' in g else np.nan,
            'vw_sharpe_mean': float(g['vw_hl_sharpe'].mean()) if 'vw_hl_sharpe' in g else np.nan,
            'vw_sharpe_std': float(g['vw_hl_sharpe'].std(ddof=1)) if 'vw_hl_sharpe' in g and len(g) > 1 else 0.0,
            'best_epoch_mean': float(g['best_epoch'].mean()),
            'seed_pred_spearman': float(g['seed_pred_spearman'].iloc[0]) if 'seed_pred_spearman' in g else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--max-iter-legacy', type=int, default=100)
    ap.add_argument('--max-iter-chrono', type=int, default=200)
    ap.add_argument('--patience-legacy', type=int, default=8)
    ap.add_argument('--patience-chrono', type=int, default=15)
    ap.add_argument('--monitor', choices=['rank_ic', 'auc'], default='rank_ic')
    ap.add_argument('--skip-portfolio', action='store_true')
    ap.add_argument('--protocols', default='legacy,chrono', help='comma list: legacy,chrono')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    protocols = {s.strip().lower() for s in args.protocols.split(',') if s.strip()}
    do_legacy = 'legacy' in protocols
    do_chrono = 'chrono' in protocols
    if not (do_legacy or do_chrono):
        raise SystemExit('need at least one of --protocols legacy,chrono')

    os.makedirs(args.out_dir, exist_ok=True)
    port_root = os.path.join(args.out_dir, 'portfolio')
    os.makedirs(port_root, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))
    logger.info(f'seeds={seeds} n={len(seeds)} protocols={sorted(protocols)} out={args.out_dir}')

    X_is = np.load(os.path.join(FEAT_180, 'X_is.npy'), mmap_mode='r')
    X_oos = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    meta_is = pd.read_feather(os.path.join(FEAT_180, 'meta_is.feather'))
    meta_oos = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))
    y_is = meta_is['binary_label'].values.astype(int)
    valid = np.isin(y_is, [0, 1])
    meta_is_v = meta_is.loc[valid].reset_index(drop=True)
    y_is = y_is[valid]
    X_is_v = np.asarray(X_is[valid])
    tr_idx, va_idx = chrono_train_val_indices(meta_is_v, list(range(2000, 2006)), [2006, 2007])
    X_tr, y_tr = X_is_v[tr_idx], y_is[tr_idx]
    X_va, y_va = X_is_v[va_idx], y_is[va_idx]
    meta_va = meta_is_v.iloc[va_idx].reset_index(drop=True)
    X_oos_arr = np.asarray(X_oos)
    logger.info(f'n_tr={len(X_tr):,} n_va={len(X_va):,} n_oos={len(X_oos_arr):,}')

    rows = []
    legacy_probs, chrono_probs = [], []

    if do_legacy:
        for seed in seeds:
            t0 = time.time()
            npy = os.path.join(args.out_dir, f'prob_legacy_seed{seed}.npy')
            logger.info(f'=== legacy seed={seed} ===')
            if os.path.exists(npy):
                prob = np.load(npy)
                logger.info(f'  resume {npy}')
                pipe = None
                n_iter = -1
            else:
                pipe = fit_mlp_legacy(
                    X_tr, y_tr, seed=seed, max_iter=args.max_iter_legacy,
                    n_iter_no_change=args.patience_legacy,
                )
                prob = predict_proba_pos(pipe, X_oos_arr)
                np.save(npy, prob)
                n_iter = int(getattr(pipe.named_steps['clf'], 'n_iter_', -1))
            legacy_probs.append(prob)
            ev = eval_ic(prob, meta_oos)
            hl = {} if args.skip_portfolio else eval_portfolio(
                prob, meta_oos, os.path.join(port_root, f'legacy_seed{seed}'), f'legacy_seed{seed}',
            )
            val_ric = (
                _rank_ic(meta_va, predict_proba_pos(pipe, X_va)) if pipe is not None
                else float('nan')
            )
            rows.append({
                'protocol': 'legacy_es', 'seed': seed,
                'n_iter': n_iter, 'best_epoch': n_iter,
                'val_rank_ic': val_ric,
                'wall_sec': time.time() - t0, **ev, **hl,
            })
            logger.info(f'  legacy seed={seed}: ric={ev["rank_ic_mean"]:.4f} sharpe={hl.get("ew_hl_sharpe")} ({time.time()-t0:.0f}s)')

    if do_chrono:
        for seed in seeds:
            t0 = time.time()
            npy = os.path.join(args.out_dir, f'prob_chrono_seed{seed}.npy')
            hist_path = os.path.join(args.out_dir, f'chrono_hist_seed{seed}.json')
            logger.info(f'=== chrono_es seed={seed} ===')
            if os.path.exists(npy):
                prob = np.load(npy)
                info = {'best_epoch': -1, 'best_val_score': float('nan'), 'stopped_epoch': -1}
                if os.path.exists(hist_path):
                    with open(hist_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                logger.info(f'  resume {npy}')
            else:
                pipe, info = fit_mlp_chrono_es(
                    X_tr, y_tr, X_va, y_va, meta_va, seed=seed,
                    max_iter=args.max_iter_chrono, n_iter_no_change=args.patience_chrono,
                    monitor=args.monitor,
                )
                prob = predict_proba_pos(pipe, X_oos_arr)
                np.save(npy, prob)
                with open(hist_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f)
            chrono_probs.append(prob)
            ev = eval_ic(prob, meta_oos)
            hl = {} if args.skip_portfolio else eval_portfolio(
                prob, meta_oos, os.path.join(port_root, f'chrono_seed{seed}'), f'chrono_seed{seed}',
            )
            rows.append({
                'protocol': 'chrono_es', 'seed': seed,
                'n_iter': info.get('stopped_epoch', -1), 'best_epoch': info.get('best_epoch', -1),
                'val_rank_ic': info.get('best_val_score', float('nan')),
                'wall_sec': time.time() - t0, **ev, **hl,
            })
            logger.info(
                f'  chrono seed={seed}: ric={ev["rank_ic_mean"]:.4f} sharpe={hl.get("ew_hl_sharpe")} '
                f'best_ep={info.get("best_epoch")} ({time.time()-t0:.0f}s)'
            )

    leg_sp = chr_sp = float('nan')
    legacy_mat = np.stack(legacy_probs, axis=0) if legacy_probs else None
    chrono_mat = np.stack(chrono_probs, axis=0) if chrono_probs else None
    if legacy_mat is not None:
        leg_sp = pairwise_mean_spearman(legacy_mat)
    if chrono_mat is not None:
        chr_sp = pairwise_mean_spearman(chrono_mat)
    for r in rows:
        if r['protocol'] == 'legacy_es':
            r['seed_pred_spearman'] = leg_sp
        elif r['protocol'] == 'chrono_es':
            r['seed_pred_spearman'] = chr_sp

    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(args.out_dir, 'mlp_stability_detail.csv'), index=False, float_format='%.6f')

    bag_parts = []
    mats = []
    if legacy_mat is not None:
        mats.append(('legacy', legacy_mat))
    if chrono_mat is not None:
        mats.append(('chrono', chrono_mat))

    if args.skip_portfolio:
        bag_rows = []
        for prefix, mat in mats:
            for k in range(1, mat.shape[0] + 1):
                ic = eval_ic(mat[:k].mean(axis=0), meta_oos)
                bag_rows.append({'protocol': prefix, 'k': k, **ic})
        bag_df = pd.DataFrame(bag_rows)
    else:
        for prefix, mat in mats:
            bag_parts.append(bagging_curve(mat, meta_oos, port_root, prefix))
        bag_df = pd.concat(bag_parts, ignore_index=True) if bag_parts else pd.DataFrame()
    bag_df.to_csv(os.path.join(args.out_dir, 'mlp_bagging_curve.csv'), index=False, float_format='%.6f')

    # enough-seeds: 5, 10, K and deltas
    enough_rows = []
    for prefix, mat in mats:
        K = mat.shape[0]
        checkpoints = sorted({k for k in (5, 10, K) if k <= K})
        by_k = {}
        for k in checkpoints:
            bag = mat[:k].mean(axis=0)
            ic = eval_ic(bag, meta_oos)
            sub = bag_df[(bag_df['protocol'] == prefix) & (bag_df['k'] == k)]
            row = {
                'protocol': prefix, 'k': k,
                'rank_ic': ic['rank_ic_mean'],
                'ew_hl_sharpe': float(sub['ew_hl_sharpe'].iloc[0]) if len(sub) and 'ew_hl_sharpe' in sub.columns else np.nan,
            }
            enough_rows.append(row)
            by_k[k] = row
        for a, b in ((5, 10), (10, 20), (5, K), (10, K)):
            if a in by_k and b in by_k and a < b:
                enough_rows.append({
                    'protocol': prefix, 'k': f'{a}_to_{b}_delta',
                    'rank_ic': by_k[b]['rank_ic'] - by_k[a]['rank_ic'],
                    'ew_hl_sharpe': (
                        (by_k[b]['ew_hl_sharpe'] - by_k[a]['ew_hl_sharpe'])
                        if np.isfinite(by_k[b]['ew_hl_sharpe']) and np.isfinite(by_k[a]['ew_hl_sharpe']) else np.nan
                    ),
                })
    enough_df = pd.DataFrame(enough_rows)
    enough_df.to_csv(os.path.join(args.out_dir, 'mlp_enough_seeds_check.csv'), index=False, float_format='%.6f')

    summary = protocol_summary(detail)
    for prefix, mat in mats:
        K = mat.shape[0]
        sub = bag_df[(bag_df['protocol'] == prefix) & (bag_df['k'] == K)]
        if len(sub):
            summary = pd.concat([summary, pd.DataFrame([{
                'protocol': f'{prefix}_bag_k{K}',
                'n_runs': 1,
                'rank_ic_mean': float(sub['rank_ic_mean'].iloc[0]),
                'rank_ic_std': 0.0, 'rank_ic_se': 0.0, 'rank_ic_cv': 0.0,
                'rank_ic_min': float(sub['rank_ic_mean'].iloc[0]),
                'rank_ic_max': float(sub['rank_ic_mean'].iloc[0]),
                'ew_sharpe_mean': float(sub['ew_hl_sharpe'].iloc[0]) if 'ew_hl_sharpe' in sub else np.nan,
                'ew_sharpe_std': 0.0,
                'ew_sharpe_min': float(sub['ew_hl_sharpe'].iloc[0]) if 'ew_hl_sharpe' in sub else np.nan,
                'ew_sharpe_max': float(sub['ew_hl_sharpe'].iloc[0]) if 'ew_hl_sharpe' in sub else np.nan,
                'vw_sharpe_mean': float(sub['vw_hl_sharpe'].iloc[0]) if 'vw_hl_sharpe' in sub else np.nan,
                'vw_sharpe_std': 0.0,
                'best_epoch_mean': np.nan,
                'seed_pred_spearman': np.nan,
            }])], ignore_index=True)

    summary.to_csv(os.path.join(args.out_dir, 'mlp_stability_summary.csv'), index=False, float_format='%.6f')

    verdict = []
    thr = 'If |delta_ic|<0.001 and |delta_sharpe|<0.05, treat as practically converged.'
    for prefix in ([p for p, _ in mats]):
        for a, b in ((5, 10), (10, 20), (10, args.seeds)):
            if b > args.seeds or a >= b:
                continue
            ra = enough_df[(enough_df['protocol'] == prefix) & (enough_df['k'] == a)]
            rb = enough_df[(enough_df['protocol'] == prefix) & (enough_df['k'] == b)]
            if len(ra) and len(rb):
                d_ic = float(rb['rank_ic'].iloc[0] - ra['rank_ic'].iloc[0])
                d_sh = float(rb['ew_hl_sharpe'].iloc[0] - ra['ew_hl_sharpe'].iloc[0])
                ok = abs(d_ic) < 0.001 and abs(d_sh) < 0.05
                verdict.append(
                    f'{prefix} bag k={a} vs k={b}: delta_rank_ic={d_ic:+.4f}, delta_ew_sharpe={d_sh:+.4f}. '
                    f'converged={ok}. {thr}'
                )
    leg = summary[summary['protocol'] == 'legacy_es']
    chr_ = summary[summary['protocol'] == 'chrono_es']
    if len(leg):
        verdict.append(
            f"single-seed std rank_ic legacy={leg['rank_ic_std'].iloc[0]:.4f}; "
            f"ew_sharpe_std={leg['ew_sharpe_std'].iloc[0]:.4f}; seed_spearman={leg_sp:.3f}."
        )
    if len(chr_):
        verdict.append(
            f"single-seed std rank_ic chrono={chr_['rank_ic_std'].iloc[0]:.4f}; "
            f"ew_sharpe_std={chr_['ew_sharpe_std'].iloc[0]:.4f}; seed_spearman={chr_sp:.3f}."
        )

    with open(os.path.join(args.out_dir, 'mlp_stability_verdict.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(verdict) + '\n')

    meta = {
        'seeds': seeds, 'n_seeds': len(seeds), 'protocols': sorted(protocols),
        'max_iter_legacy': args.max_iter_legacy, 'max_iter_chrono': args.max_iter_chrono,
        'patience_legacy': args.patience_legacy, 'patience_chrono': args.patience_chrono,
        'monitor': args.monitor, 'legacy_seed_spearman': leg_sp, 'chrono_seed_spearman': chr_sp,
        'verdict': verdict,
    }
    with open(os.path.join(args.out_dir, 'mlp_stability_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f'summary:\n{summary.to_string(index=False)}')
    logger.info(f'bagging curve:\n{bag_df.to_string(index=False)}')
    logger.info(f'enough-seeds check:\n{enough_df.to_string(index=False)}')
    for v in verdict:
        logger.info(f'VERDICT: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
