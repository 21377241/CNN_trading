# -*- coding: utf-8 -*-
"""E4T: recent-path mechanisms for pairwise CNN vs 1D-CNN specific scores.

The experiment keeps the frozen E2C five-seed 1D-CNN and the previously
generated CNN chart-intervention predictions.  It does not retrain either
model.  Every intervention is followed by a cross-sectional residualization
against the untouched partner score.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E2C_temporal.train_1dcnn import Conv1DTemporal
from experiments.E3_shared_specific.utils import newey_west_t, norm_stock_id
from experiments.E4_recent_path.build_pathshape import PATH_COLS, build_pathshape
from experiments.E4_recent_path.eval_specific import attach_deltas, eval_specific_df
from experiments.E4_recent_path.interventions_ohlcv import CNN_BLOCK_FNS, flatten_day
from experiments.E4_recent_path.paths import LAG_LABELS, NW_LAG

OUT_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E4T_cnn_1d_recent_path')
E2C_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2C_temporal_1dcnn_120')
E3T_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E3T_cnn_1d_temporal')
E4_CACHE = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E4_recent_path_chrono_ric_mlp4')
FEATURE_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2_matched_is', 'features')
SEEDS = (42, 43, 44, 45, 46)
RECENT5 = range(15, 20)
BLOCKS = tuple(CNN_BLOCK_FNS)
KEYS = ['StockID', 'ending_date']


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['StockID'] = norm_stock_id(out['StockID'])
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    return out


def _load_scores() -> pd.DataFrame:
    scores = _keys(pd.read_parquet(os.path.join(E3T_ROOT, 'scores.parquet')))
    required = {'cnn', 'temporal', 'ret_val', 'MarketCap', 'binary_label'}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f'E3T scores missing columns: {missing}')
    return scores


def _residualize(score: np.ndarray, partner: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Daily linear residual of `score` on the frozen partner score."""
    out = np.full(len(score), np.nan, dtype=np.float64)
    for _, idx in pd.Series(np.arange(len(score))).groupby(dates, sort=False):
        ii = idx.to_numpy()
        x, y = partner[ii], score[ii]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 50:
            continue
        fit = LinearRegression().fit(x[valid].reshape(-1, 1), y[valid])
        out[ii[valid]] = y[valid] - fit.predict(x[valid].reshape(-1, 1))
    return out


def _specific_frame(scores: pd.DataFrame, perturbed: np.ndarray, side: str) -> pd.DataFrame:
    partner = scores['temporal'].to_numpy(float) if side == 'cnn' else scores['cnn'].to_numpy(float)
    sp = _residualize(perturbed, partner, scores['ending_date'].to_numpy())
    out = scores[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    out['S_sp'] = sp
    return out.dropna(subset=['S_sp'])


def _recompute_ma(x: np.ndarray) -> None:
    # Matched 120D channel 4 is the chart-aligned rolling MA20 of adjusted close.
    x[:, :, 4] = np.cumsum(x[:, :, 3], axis=1) / np.arange(1, 21, dtype=np.float32)[None, :]


def _intervene_1d(x: np.ndarray, name: str) -> np.ndarray:
    """Apply an OHLCV intervention to 20x6 matched input, then rebuild MA20."""
    z = np.asarray(x, dtype=np.float32).copy()
    for i in range(len(z)):
        o, h, l, c, v = z[i, :, 0], z[i, :, 1], z[i, :, 2], z[i, :, 3], z[i, :, 5]
        if name.startswith('flatten_day_'):
            oo, hh, ll, cc, vv = flatten_day(o, h, l, c, v, int(name.rsplit('_', 1)[1]))
        else:
            oo, hh, ll, cc, vv = CNN_BLOCK_FNS[name](o, h, l, c, v)
        z[i, :, 0], z[i, :, 1], z[i, :, 2], z[i, :, 3], z[i, :, 5] = oo, hh, ll, cc, vv
    _recompute_ma(z)
    return z


def _load_1d_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for seed in SEEDS:
        ckpt = torch.load(os.path.join(E2C_ROOT, 'models', f'seed{seed}', 'best.pt'), map_location=device)
        model = Conv1DTemporal(in_channels=6, hidden_channels=(32, 64), kernel_size=3, dropout=.25).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        models.append((model, np.asarray(ckpt['mean'], dtype=np.float32), np.asarray(ckpt['std'], dtype=np.float32)))
    return device, models


def _predict_1d(name: str | None, batch_size: int = 16384) -> np.ndarray:
    """Frozen five-seed ensemble prediction, optionally after an intervention."""
    x = np.load(os.path.join(FEATURE_ROOT, 'X_oos.npy'), mmap_mode='r')
    device, models = _load_1d_models()
    ans = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        z = np.asarray(x[start:end], dtype=np.float32).reshape(-1, 20, 6)
        if name is not None:
            z = _intervene_1d(z, name)
        probs = []
        with torch.no_grad():
            for model, mean, std in models:
                inp = torch.from_numpy(((z - mean[None, None, :]) / std[None, None, :]).transpose(0, 2, 1)).to(device)
                probs.append(torch.sigmoid(model(inp)).cpu().numpy())
        ans[start:end] = np.mean(np.vstack(probs), axis=0)
        if end % (batch_size * 16) == 0 or end == len(x):
            print(f'1D forward {name or "baseline"}: {end:,}/{len(x):,}', flush=True)
    return ans


def _load_cnn_cache(name: str) -> pd.DataFrame:
    sub = 'e41_lag' if name.startswith('flatten_day_') else 'e43_blocks'
    path = os.path.join(E4_CACHE, sub, 'cnn', 'preds', f'pred_cnn_{name}_oos.csv')
    return _keys(pd.read_csv(path)).rename(columns={'up_prob': 'cnn_perturbed'})


def _baseline_rows(scores: pd.DataFrame, out_dir: str, side: str) -> list[dict]:
    raw = scores['cnn'].to_numpy(float) if side == 'cnn' else scores['temporal'].to_numpy(float)
    return [eval_specific_df(_specific_frame(scores, raw, side), 'baseline', out_dir)]


def _run_interventions(kind: str) -> None:
    scores = _load_scores()
    # The E2C score panel and 120D cache must be in identical record order.
    meta = _keys(pd.read_feather(os.path.join(FEATURE_ROOT, 'meta_oos.feather')))
    meta['_feature_pos'] = np.arange(len(meta), dtype=np.int64)
    panel_pos = scores[KEYS].merge(meta[KEYS + ['_feature_pos']], on=KEYS, how='left', validate='one_to_one')['_feature_pos']
    if len(meta) != len(scores) or panel_pos.isna().any():
        raise ValueError('E2C scores and 120D feature metadata do not define the same one-to-one panel.')
    panel_pos = panel_pos.to_numpy(dtype=np.int64)
    names = [f'flatten_day_{d}' for d in RECENT5] if kind == 'lag' else list(BLOCKS)
    label_col = 'lag' if kind == 'lag' else 'block'
    label = (lambda n: LAG_LABELS[int(n.rsplit('_', 1)[1])]) if kind == 'lag' else (lambda n: n)

    for side in ('cnn', '1dcnn'):
        out_dir = os.path.join(OUT_ROOT, f'e4t_{kind}', side)
        os.makedirs(out_dir, exist_ok=True)
        rows = _baseline_rows(scores, out_dir, side)
        for name in names:
            if side == 'cnn':
                cached = _load_cnn_cache(name)
                m = scores[KEYS].merge(cached[KEYS + ['cnn_perturbed']], on=KEYS, how='left', validate='one_to_one')
                if m['cnn_perturbed'].isna().any():
                    raise ValueError(f'CNN cache does not cover E2C panel: {name}')
                perturbed = m['cnn_perturbed'].to_numpy(float)
                tag = name
            else:
                perturbed = _predict_1d(name)[panel_pos]
                tag = f'block_{name}' if kind == 'blocks' else name
            row = eval_specific_df(_specific_frame(scores, perturbed, 'cnn' if side == 'cnn' else '1dcnn'), tag, out_dir)
            row[label_col] = label(name)
            row['intervention'] = name
            rows.append(row)
        summary = attach_deltas(pd.DataFrame(rows))
        if kind == 'blocks':
            drops = summary.loc[summary['tag'] != 'baseline', 'delta_ew_sharpe'].to_numpy(float)
            summary['drop_norm'] = np.nan
            denom = np.abs(drops).sum()
            if denom > 1e-12:
                summary.loc[summary['tag'] != 'baseline', 'drop_norm'] = drops / denom
        summary.to_csv(os.path.join(out_dir, f'e4t_{side}_{kind}_summary.csv'), index=False, float_format='%.6f')
        print(summary.to_string(index=False), flush=True)


def _cs_z(x: pd.Series) -> np.ndarray:
    a = x.to_numpy(float)
    s = a.std(ddof=0)
    return (a - a.mean()) / s if np.isfinite(s) and s > 1e-12 else np.zeros(len(a))


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    if len(y) < 50 or not np.isfinite(y).all() or not np.isfinite(X).all():
        return np.nan
    return float(LinearRegression().fit(X, y).score(X, y))


def run_pathshape() -> None:
    scores = _load_scores()
    path_file = os.path.join(E4_CACHE, 'pathshape.parquet')
    path = _keys(pd.read_parquet(path_file)) if os.path.exists(path_file) else build_pathshape(path_file)
    df = scores.merge(path[KEYS + PATH_COLS], on=KEYS, how='inner', validate='one_to_one')
    ycols = {'cnn_specific_given_1d': 'S_cnn_sp_1d', '1d_specific_given_cnn': 'S_1d_sp_cnn'}
    shape = ['LateEarly', 'PathVol', 'UpCount5', 'MaxDD5', 'Reversal5', 'Monotonicity']
    rows = []
    for date, g in df.groupby('ending_date', sort=True):
        for name, ycol in ycols.items():
            y = _cs_z(g[ycol])
            x_cum = _cs_z(g['CumRet5']).reshape(-1, 1)
            x_full = np.column_stack([x_cum] + [_cs_z(g[c]) for c in shape])
            r2_c = _r2(y, x_cum)
            r2_f = _r2(y, x_full)
            rows.append({'ending_date': date, 'score': name, 'kind': 'all_shape', 'r2_cum': r2_c, 'r2_aug': r2_f, 'delta_r2': r2_f - r2_c})
            for col in shape:
                r2_s = _r2(y, np.column_stack([x_cum, _cs_z(g[col])]))
                rows.append({'ending_date': date, 'score': name, 'kind': col, 'r2_cum': r2_c, 'r2_aug': r2_s, 'delta_r2': r2_s - r2_c})
    daily = pd.DataFrame(rows)
    summary = []
    for (score, kind), g in daily.groupby(['score', 'kind'], sort=True):
        t, p = newey_west_t(g['delta_r2'].to_numpy(float), lag=NW_LAG)
        summary.append({'score': score, 'kind': kind, 'r2_cum_mean': g['r2_cum'].mean(), 'r2_aug_mean': g['r2_aug'].mean(), 'delta_r2_mean': g['delta_r2'].mean(), 'nw_t_delta_r2': t, 'nw_p_delta_r2': p, 'n_dates': len(g)})
    out_dir = os.path.join(OUT_ROOT, 'e4t_pathshape')
    os.makedirs(out_dir, exist_ok=True)
    daily.to_csv(os.path.join(out_dir, 'e4t_daily_delta_r2.csv'), index=False, float_format='%.8f')
    pd.DataFrame(summary).sort_values(['score', 'kind']).to_csv(os.path.join(out_dir, 'e4t_delta_r2_summary.csv'), index=False, float_format='%.8f')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pathshape', 'lag', 'blocks', 'all'], default='all')
    args = parser.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    if args.stage in ('pathshape', 'all'):
        run_pathshape()
    if args.stage in ('lag', 'all'):
        _run_interventions('lag')
    if args.stage in ('blocks', 'all'):
        _run_interventions('blocks')


if __name__ == '__main__':
    main()
