# -*- coding: utf-8 -*-
"""Evaluate E2C temporal benchmark: standalone, spanning, residuals, ensemble."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.metrics import cross_section_ic, summarize_ic


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ann_stats(r: pd.Series, ann_factor: int) -> dict:
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {'ann': np.nan, 'vol': np.nan, 'sharpe': np.nan, 'n_dates': 0}
    mean = r.mean()
    ann = (1.0 + mean) ** ann_factor - 1.0
    vol = r.std(ddof=1) * np.sqrt(ann_factor)
    # Sharpe uses arithmetic mean return; compounded annual return is reported separately.
    sharpe = float(mean / r.std(ddof=1) * np.sqrt(ann_factor)) if r.std(ddof=1) > 1e-12 else np.nan
    return {'ann': float(ann), 'vol': float(vol), 'sharpe': sharpe, 'n_dates': int(len(r))}


def portfolio_hl(df: pd.DataFrame, score_col: str, ann_factor: int, n_decile: int = 10) -> dict:
    rows = []
    for dt, g in df.dropna(subset=[score_col, 'ret_val']).groupby('ending_date'):
        if len(g) < n_decile:
            continue
        g = g.copy()
        try:
            g['decile'] = pd.qcut(g[score_col].rank(method='first'), n_decile, labels=False) + 1
        except Exception:
            continue
        lo = g[g['decile'] == 1]
        hi = g[g['decile'] == n_decile]
        if len(lo) == 0 or len(hi) == 0:
            continue
        ew = hi['ret_val'].mean() - lo['ret_val'].mean()
        vw = np.average(hi['ret_val'], weights=np.maximum(hi['MarketCap'], 0)) - np.average(
            lo['ret_val'], weights=np.maximum(lo['MarketCap'], 0)
        )
        rows.append({'ending_date': dt, 'ew_hl': ew, 'vw_hl': vw})
    ret = pd.DataFrame(rows)
    ew = ann_stats(ret['ew_hl'], ann_factor) if not ret.empty else ann_stats([], ann_factor)
    vw = ann_stats(ret['vw_hl'], ann_factor) if not ret.empty else ann_stats([], ann_factor)
    return {
        'ew_hl_ann': ew['ann'],
        'ew_hl_sharpe': ew['sharpe'],
        'vw_hl_ann': vw['ann'],
        'vw_hl_sharpe': vw['sharpe'],
        'n_dates': ew['n_dates'],
    }


def score_summary(df: pd.DataFrame, score_col: str, ann_factor: int) -> dict:
    y = df['binary_label'].values
    s = df[score_col].values
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
    ic = summarize_ic(cross_section_ic(df, score_col))
    port = portfolio_hl(df, score_col, ann_factor)
    return {'auc': float(auc), **ic, **port, 'n': int(len(df))}


def z_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby('ending_date')[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12))


def mean_daily_corr(df: pd.DataFrame, a: str, b: str) -> dict:
    rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < 10:
            continue
        m = np.isfinite(g[a]) & np.isfinite(g[b])
        if m.sum() < 10:
            continue
        rows.append({'ending_date': dt, 'pearson': pearsonr(g.loc[m, a], g.loc[m, b])[0], 'spearman': spearmanr(g.loc[m, a], g.loc[m, b])[0]})
    r = pd.DataFrame(rows)
    return {'pearson_mean': float(r['pearson'].mean()), 'spearman_mean': float(r['spearman'].mean()), 'n_dates': int(len(r))}


def top_overlap(df: pd.DataFrame, a: str, b: str, q: float = 0.10) -> dict:
    vals = []
    jac = []
    for dt, g in df.groupby('ending_date'):
        n = len(g)
        k = max(1, int(np.floor(n * q)))
        ia = set(g.nlargest(k, a).index)
        ib = set(g.nlargest(k, b).index)
        vals.append(len(ia & ib) / len(ia))
        jac.append(len(ia & ib) / len(ia | ib))
    return {'top10_overlap': float(np.mean(vals)), 'top10_jaccard': float(np.mean(jac))}


def nw_t_mean(x: np.ndarray, lag: int = 4) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    u = x - x.mean()
    gamma0 = np.dot(u, u) / n
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        gamma = np.dot(u[l:], u[:-l]) / n
        var += 2 * (1 - l / (lag + 1)) * gamma
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 1e-12 else np.nan


def fm_joint(df: pd.DataFrame, ret_col='ret_val') -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby('ending_date'):
        if len(g) < 20:
            continue
        y = g[ret_col].values.astype(float)
        X = np.column_stack([np.ones(len(g)), g['z_cnn'].values, g['z_temporal'].values])
        m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if m.sum() < 20:
            continue
        beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        rows.append({'ending_date': dt, 'beta_cnn': beta[1], 'beta_temporal': beta[2]})
    out = pd.DataFrame(rows)
    return pd.DataFrame([
        {'term': 'CNN controlling temporal', 'beta_bar': out['beta_cnn'].mean(), 'nw_t': nw_t_mean(out['beta_cnn'].values)},
        {'term': 'Temporal controlling CNN', 'beta_bar': out['beta_temporal'].mean(), 'nw_t': nw_t_mean(out['beta_temporal'].values)},
    ])


def residualize_by_date(df: pd.DataFrame, ycol: str, xcol: str, outcol: str) -> pd.DataFrame:
    vals = np.full(len(df), np.nan, dtype=float)
    for _, idx in df.groupby('ending_date').groups.items():
        g = df.loc[idx]
        y = g[ycol].values.astype(float)
        x = g[xcol].values.astype(float)
        X = np.column_stack([np.ones(len(g)), x])
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 10:
            continue
        beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        vals[df.index.get_indexer(idx)] = y - X @ beta
    out = df.copy()
    out[outcol] = vals
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config_1dcnn_120.yaml'))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out_root = os.path.join(PROJECT_ROOT, cfg['paths']['out_root'].replace('/', os.sep))
    pred_temporal = os.path.join(out_root, 'preds', 'pred_1dcnn120_oos.csv')
    pred_cnn = os.path.join(PROJECT_ROOT, cfg['paths']['cnn_pred'].replace('/', os.sep))
    ann_factor = int(cfg.get('ann_factor', 52))
    res_dir = os.path.join(out_root, 'results')
    os.makedirs(res_dir, exist_ok=True)

    t = pd.read_csv(pred_temporal)
    c = pd.read_csv(pred_cnn)
    for d in (t, c):
        d['ending_date'] = pd.to_datetime(d['ending_date'])
    keys = ['StockID', 'ending_date']
    df = c[keys + ['up_prob', 'ret_val', 'MarketCap', 'binary_label']].rename(columns={'up_prob': 'cnn'})
    df = df.merge(t[keys + ['up_prob']].rename(columns={'up_prob': 'temporal'}), on=keys, how='inner')
    df = df.dropna(subset=['cnn', 'temporal', 'ret_val'])
    df['z_cnn'] = z_by_date(df, 'cnn')
    df['z_temporal'] = z_by_date(df, 'temporal')
    df['ens'] = 0.5 * (df['z_cnn'] + df['z_temporal'])

    standalone = pd.DataFrame([
        {'model': 'cnn', **score_summary(df.rename(columns={'cnn': 'score'}), 'score', ann_factor)},
        {'model': '1dcnn120', **score_summary(df.rename(columns={'temporal': 'score'}), 'score', ann_factor)},
        {'model': 'cnn_1dcnn120_ens', **score_summary(df.rename(columns={'ens': 'score'}), 'score', ann_factor)},
    ])
    corr = mean_daily_corr(df, 'cnn', 'temporal')
    overlap = top_overlap(df, 'cnn', 'temporal')
    fm = fm_joint(df)
    df_res = residualize_by_date(df, 'cnn', 'temporal', 'cnn_resid')
    df_res = residualize_by_date(df_res, 'temporal', 'cnn', 'temporal_resid')
    residual = pd.DataFrame([
        {'score': 'cnn_resid_on_temporal', **score_summary(df_res.rename(columns={'cnn_resid': 'score'}), 'score', ann_factor)},
        {'score': 'temporal_resid_on_cnn', **score_summary(df_res.rename(columns={'temporal_resid': 'score'}), 'score', ann_factor)},
    ])

    standalone.to_csv(os.path.join(res_dir, 'standalone_summary.csv'), index=False)
    pd.DataFrame([{**corr, **overlap}]).to_csv(os.path.join(res_dir, 'cnn_temporal_similarity.csv'), index=False)
    fm.to_csv(os.path.join(res_dir, 'fm_joint.csv'), index=False)
    residual.to_csv(os.path.join(res_dir, 'residual_summary.csv'), index=False)
    df.to_csv(os.path.join(res_dir, 'common_scores.csv'), index=False)

    key = {
        'rank_ic_temporal': float(standalone.loc[standalone['model'] == '1dcnn120', 'rank_ic_mean'].iloc[0]),
        'ew_sharpe_temporal': float(standalone.loc[standalone['model'] == '1dcnn120', 'ew_hl_sharpe'].iloc[0]),
        'corr_cnn_temporal': corr['pearson_mean'],
        'top10_overlap': overlap['top10_overlap'],
        'cnn_residual_sharpe': float(residual.loc[residual['score'] == 'cnn_resid_on_temporal', 'ew_hl_sharpe'].iloc[0]),
        'ensemble_sharpe': float(standalone.loc[standalone['model'] == 'cnn_1dcnn120_ens', 'ew_hl_sharpe'].iloc[0]),
    }
    with open(os.path.join(res_dir, 'key_six.json'), 'w', encoding='utf-8') as f:
        json.dump(key, f, indent=2)

    md = [
        '# E2C Temporal Numerical Benchmark: 1D-CNN-120',
        '',
        '## Key six numbers',
        '',
        '| Metric | Value |',
        '|---|---:|',
    ]
    for k, v in key.items():
        md.append(f'| {k} | {v:.6f} |')
    md += [
        '',
        '## Standalone / ensemble',
        '',
        standalone.to_markdown(index=False),
        '',
        '## CNN vs 1D-CNN similarity',
        '',
        pd.DataFrame([{**corr, **overlap}]).to_markdown(index=False),
        '',
        '## Joint Fama--MacBeth',
        '',
        fm.to_markdown(index=False),
        '',
        '## Residualized scores',
        '',
        residual.to_markdown(index=False),
    ]
    with open(os.path.join(res_dir, 'e2c_1dcnn120_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md[:14]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
