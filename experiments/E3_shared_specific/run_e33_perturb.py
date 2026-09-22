# -*- coding: utf-8 -*-
"""E3-3: CNN image perturbations (online DrawOHLC + E1b-ric ensemble).

Perturb window OHLC/V from 180-dim matched features → redraw 60×64 →
5-model forward → residualize vs original S_M,S_H → IC/Sharpe.
Year-level resume under e33_perturb/.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Data.chart_library import DrawOHLC
from Model.cnn_model import MODEL_NAME_I20, init_model
from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.E2_matched_tabular.e2a_layout import FEAT_180
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from experiments.E3_shared_specific.paths import ANN, E1B_RIC_EXP, MIN_CS_N, OUT_ROOT, WINDOW
from Portfolio.portfolio import run_portfolio_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_e33')

IMG_H, IMG_W = 64, 60
PER_DAY = 9
BATCH = 512
PERTURBS = ('mask_last5', 'flatten_wick', 'shuffle_days', 'zero_volume')
OOS_YEARS = list(range(2008, 2026))


def _norm_sid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)


def _load_models(device: torch.device):
    models = []
    for i in range(5):
        path = os.path.join(E1B_RIC_EXP, f'checkpoint{i}.pth.tar')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        m = init_model(MODEL_NAME_I20, img_h=IMG_H, img_w=IMG_W, dropout=0.0, n_classes=2, window_size=20)
        m.load_state_dict(ckpt['model_state_dict'])
        m.to(device).eval()
        models.append(m)
    logger.info(f'loaded {len(models)} checkpoints from {E1B_RIC_EXP}')
    return models


def _perturb_ohlcv(o, h, l, c, v, name: str, rng: np.random.Generator):
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    if name == 'mask_last5':
        mo, mh, ml, mc, mv = o.mean(), h.mean(), l.mean(), c.mean(), v.mean()
        o[-5:], h[-5:], l[-5:], c[-5:], v[-5:] = mo, mh, ml, mc, mv
    elif name == 'flatten_wick':
        hi = np.maximum(o, c)
        lo = np.minimum(o, c)
        h[:], l[:] = hi, lo
    elif name == 'shuffle_days':
        idx = rng.permutation(len(o))
        o, h, l, c, v = o[idx], h[idx], l[idx], c[idx], v[idx]
    elif name == 'zero_volume':
        v[:] = 0.0
    else:
        raise ValueError(name)
    # keep H/L consistency
    h = np.maximum(np.maximum(h, o), c)
    l = np.minimum(np.minimum(l, o), c)
    return o, h, l, c, v


def _draw_one(o, h, l, c, v) -> np.ndarray:
    ma = pd.Series(c).rolling(20, min_periods=1).mean().values
    df = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Vol': v, 'ma20': ma})
    arr = DrawOHLC(
        ohlcv_df=df, window_size=WINDOW, ma_cols=['ma20'],
        has_volume_bar=True, chart_type='bar',
        img_width=IMG_W, img_height=IMG_H,
    ).draw_to_array()
    return arr.astype(np.float32).reshape(1, IMG_H, IMG_W) / 255.0


@torch.no_grad()
def _infer_batch(models, imgs: np.ndarray, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(imgs).to(device)
    probs = []
    for m in models:
        probs.append(m.predict_proba(t)[:, 1].cpu().numpy())
    return np.mean(probs, axis=0)


def _predict_year(X_year: np.ndarray, names: list[str], dates: list, perturb: str,
                  models, device) -> pd.DataFrame:
    n = X_year.shape[0]
    xb = X_year.reshape(n, WINDOW, PER_DAY)
    up = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        imgs = np.empty((end - start, 1, IMG_H, IMG_W), dtype=np.float32)
        for i, j in enumerate(range(start, end)):
            row = xb[j]
            o, h, l, c, v = row[:, 0], row[:, 1], row[:, 2], row[:, 3], row[:, 6]
            seed = int(hashlib.md5(f'{names[j]}|{dates[j]}|{perturb}'.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            o, h, l, c, v = _perturb_ohlcv(o, h, l, c, v, perturb, rng)
            imgs[i] = _draw_one(o, h, l, c, v)
        up[start:end] = _infer_batch(models, imgs, device)
        if (start // BATCH) % 20 == 0:
            logger.info(f'    {perturb} {end:,}/{n:,} elapsed={time.time()-t0:.0f}s')
    return pd.DataFrame({
        'StockID': names,
        'ending_date': dates,
        'up_prob': up,
    })


def _cnn_specific_from_mask(pred: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    keys = ['StockID', 'ending_date']
    pred = pred.copy()
    pred['StockID'] = _norm_sid(pred['StockID'])
    pred['ending_date'] = pd.to_datetime(pred['ending_date'])
    p = panel[keys + ['S_mlp', 'S_hgb', 'ret_val', 'MarketCap', 'binary_label']].copy()
    p['StockID'] = _norm_sid(p['StockID'])
    p['ending_date'] = pd.to_datetime(p['ending_date'])
    m = pred.merge(p, on=keys, how='inner')
    pieces = []
    for _, g in m.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        g = g.copy()
        sc = g['up_prob'].values.astype(float)
        sm = g['S_mlp'].values.astype(float)
        sh = g['S_hgb'].values.astype(float)
        mask = np.isfinite(sc) & np.isfinite(sm) & np.isfinite(sh)
        g = g.loc[mask]
        if len(g) < MIN_CS_N:
            continue
        X = np.column_stack([g['S_mlp'].values, g['S_hgb'].values])
        y = g['up_prob'].values.astype(float)
        g['S_cnn_sp'] = y - LinearRegression().fit(X, y).predict(X)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _eval_specific(name: str, sp_df: pd.DataFrame, out_root: str) -> dict:
    df = sp_df[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    df['up_prob'] = sp_df.groupby('ending_date')['S_cnn_sp'].rank(pct=True).values
    ic = summarize_ic(cross_section_ic(df, 'up_prob'))
    mdir = os.path.join(out_root, name)
    os.makedirs(mdir, exist_ok=True)
    tmp = os.path.join(mdir, 'pred_for_port.csv')
    df.to_csv(tmp, index=False)
    run_portfolio_analysis(result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=ANN)
    hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
    return {'perturb': name, **ic, **hl, 'n': len(df)}


def main():
    out_root = os.path.join(OUT_ROOT, 'e33_perturb')
    os.makedirs(out_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}')
    models = _load_models(device)

    meta = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))
    meta['StockID'] = _norm_sid(meta['StockID'])
    meta['ending_date'] = pd.to_datetime(meta['ending_date'])
    meta['year'] = meta['ending_date'].dt.year
    X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    assert len(meta) == X.shape[0]

    panel = pd.read_parquet(os.path.join(OUT_ROOT, 'panel.parquet'))
    panel['StockID'] = _norm_sid(panel['StockID'])
    panel['ending_date'] = pd.to_datetime(panel['ending_date'])

    # optional baseline unperturbed (sanity) — skip; use scores baseline later
    for perturb in PERTURBS:
        logger.info(f'======== perturb={perturb} ========')
        year_parts = []
        for year in OOS_YEARS:
            ypath = os.path.join(out_root, f'pred_cnn_{perturb}_oos_{year}.csv')
            if os.path.exists(ypath):
                logger.info(f'  resume skip {year}')
                year_parts.append(pd.read_csv(ypath))
                continue
            idx = np.where(meta['year'].values == year)[0]
            if len(idx) == 0:
                continue
            logger.info(f'  year={year} n={len(idx):,}')
            Xy = np.asarray(X[idx], dtype=np.float64)
            names = meta['StockID'].values[idx].tolist()
            dates = meta['ending_date'].values[idx]
            pred_y = _predict_year(Xy, names, dates, perturb, models, device)
            pred_y.to_csv(ypath, index=False)
            year_parts.append(pred_y)
            logger.info(f'  wrote {ypath}')
        full = pd.concat(year_parts, ignore_index=True)
        full_path = os.path.join(out_root, f'pred_cnn_{perturb}_oos.csv')
        full.to_csv(full_path, index=False)

    # evaluate specific Sharpe
    rows = []
    scores = pd.read_parquet(os.path.join(OUT_ROOT, 'scores.parquet'))
    # baseline CNN-spec from scores
    base = scores[['StockID', 'ending_date', 'S_cnn_sp', 'ret_val', 'MarketCap', 'binary_label']].copy()
    base = base.rename(columns={'S_cnn_sp': 'S_cnn_sp'})
    rows.append(_eval_specific('baseline', base, out_root))

    for perturb in PERTURBS:
        pred = pd.read_csv(os.path.join(out_root, f'pred_cnn_{perturb}_oos.csv'))
        sp = _cnn_specific_from_mask(pred, panel)
        rows.append(_eval_specific(perturb, sp, out_root))

    summary = pd.DataFrame(rows)
    b = float(summary.loc[summary['perturb'] == 'baseline', 'ew_hl_sharpe'].iloc[0])
    summary['baseline_sharpe'] = b
    summary['sharpe_drop_pct'] = (b - summary['ew_hl_sharpe']) / abs(b) * 100.0
    summary.loc[summary['perturb'] == 'baseline', 'sharpe_drop_pct'] = 0.0
    out = os.path.join(OUT_ROOT, 'e33_perturb_summary.csv')
    summary.to_csv(out, index=False, float_format='%.6f')
    logger.info(f'wrote {out}\n{summary.to_string(index=False)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
