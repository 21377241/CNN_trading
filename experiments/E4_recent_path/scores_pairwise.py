# -*- coding: utf-8 -*-
"""Pairwise CNN↔MLP specific scores (no HistGB in residualization)."""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from experiments.E3_shared_specific.utils import norm_stock_id
from experiments.E4_recent_path.paths import MIN_CS_N, OUT_ROOT, PRED_ROOT

logger = logging.getLogger('e4_scores')


def _load_raw_preds() -> pd.DataFrame:
    cnn = pd.read_csv(os.path.join(PRED_ROOT, 'pred_cnn_oos.csv')).rename(columns={'up_prob': 'S_cnn'})
    mlp = pd.read_csv(os.path.join(PRED_ROOT, 'pred_mlp_oos.csv')).rename(columns={'up_prob': 'S_mlp'})
    keys = ['StockID', 'ending_date']
    keep = ['ret_val', 'MarketCap', 'binary_label']
    m = cnn[keys + ['S_cnn'] + [c for c in keep if c in cnn.columns]]
    m = m.merge(mlp[keys + ['S_mlp']], on=keys, how='inner')
    for c in keep:
        if c not in m.columns and c in mlp.columns:
            m = m.merge(mlp[keys + [c]], on=keys, how='left')
    m['StockID'] = norm_stock_id(m['StockID'])
    m['ending_date'] = pd.to_datetime(m['ending_date'])
    return m


def pairwise_resid_day(sc: np.ndarray, sm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S_c ~ S_m and S_m ~ S_c residuals."""
    sc = np.asarray(sc, dtype=float)
    sm = np.asarray(sm, dtype=float)
    Xc = sm.reshape(-1, 1)
    Xm = sc.reshape(-1, 1)
    u_c = sc - LinearRegression().fit(Xc, sc).predict(Xc)
    u_m = sm - LinearRegression().fit(Xm, sm).predict(Xm)
    return u_c, u_m


def build_baseline_scores(out_path: str | None = None) -> pd.DataFrame:
    panel = _load_raw_preds()
    pieces = []
    n_skip = 0
    for _, g in panel.groupby('ending_date', sort=True):
        if len(g) < MIN_CS_N:
            n_skip += 1
            continue
        g = g.copy()
        sc, sm = g['S_cnn'].values.astype(float), g['S_mlp'].values.astype(float)
        m = np.isfinite(sc) & np.isfinite(sm)
        if m.sum() < MIN_CS_N:
            n_skip += 1
            continue
        g = g.loc[m].copy()
        u_c, u_m = pairwise_resid_day(g['S_cnn'].values, g['S_mlp'].values)
        g['S_cnn_sp'] = u_c
        g['S_mlp_sp'] = u_m
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out_path = out_path or os.path.join(OUT_ROOT, 'baseline_scores.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(f'baseline pairwise scores rows={len(out):,} dates={out["ending_date"].nunique()} skip={n_skip} -> {out_path}')
    return out


def _ensure_labels(m: pd.DataFrame, *sources: pd.DataFrame) -> pd.DataFrame:
    """Fill ret_val / MarketCap / binary_label from any available source."""
    for col in ('ret_val', 'MarketCap', 'binary_label'):
        if col in m.columns and m[col].notna().any():
            continue
        for src in sources:
            if src is None or col not in src.columns:
                continue
            s = src[['StockID', 'ending_date', col]].copy()
            s['StockID'] = norm_stock_id(s['StockID'])
            s['ending_date'] = pd.to_datetime(s['ending_date'])
            m = m.drop(columns=[col], errors='ignore').merge(s, on=['StockID', 'ending_date'], how='left')
            break
    return m


def residualize_cnn_on_mlp(pred_cnn: pd.DataFrame, partner_mlp: pd.DataFrame) -> pd.DataFrame:
    """pred_cnn has up_prob; partner_mlp has S_mlp (+ optional labels)."""
    keys = ['StockID', 'ending_date']
    a = pred_cnn.copy()
    a['StockID'] = norm_stock_id(a['StockID'])
    a['ending_date'] = pd.to_datetime(a['ending_date'])
    if 'up_prob' in a.columns and 'S_cnn' not in a.columns:
        a = a.rename(columns={'up_prob': 'S_cnn'})
    b = partner_mlp.copy()
    b['StockID'] = norm_stock_id(b['StockID'])
    b['ending_date'] = pd.to_datetime(b['ending_date'])
    keep_b = keys + ['S_mlp'] + [c for c in ['ret_val', 'MarketCap', 'binary_label'] if c in b.columns]
    m = a.merge(b[keep_b], on=keys, how='inner', suffixes=('', '_p'))
    m = _ensure_labels(m, a, b)
    pieces = []
    for _, g in m.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        g = g.copy()
        sc, sm = g['S_cnn'].values.astype(float), g['S_mlp'].values.astype(float)
        mask = np.isfinite(sc) & np.isfinite(sm)
        g = g.loc[mask]
        if len(g) < MIN_CS_N:
            continue
        u_c, _ = pairwise_resid_day(g['S_cnn'].values, g['S_mlp'].values)
        g['S_sp'] = u_c
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def residualize_mlp_on_cnn(pred_mlp: pd.DataFrame, partner_cnn: pd.DataFrame) -> pd.DataFrame:
    keys = ['StockID', 'ending_date']
    a = pred_mlp.copy()
    a['StockID'] = norm_stock_id(a['StockID'])
    a['ending_date'] = pd.to_datetime(a['ending_date'])
    if 'up_prob' in a.columns and 'S_mlp' not in a.columns:
        a = a.rename(columns={'up_prob': 'S_mlp'})
    b = partner_cnn.copy()
    b['StockID'] = norm_stock_id(b['StockID'])
    b['ending_date'] = pd.to_datetime(b['ending_date'])
    keep_b = keys + ['S_cnn'] + [c for c in ['ret_val', 'MarketCap', 'binary_label'] if c in b.columns]
    m = a.merge(b[keep_b], on=keys, how='inner', suffixes=('', '_p'))
    m = _ensure_labels(m, a, b)
    pieces = []
    for _, g in m.groupby('ending_date'):
        if len(g) < MIN_CS_N:
            continue
        g = g.copy()
        sc, sm = g['S_cnn'].values.astype(float), g['S_mlp'].values.astype(float)
        mask = np.isfinite(sc) & np.isfinite(sm)
        g = g.loc[mask]
        if len(g) < MIN_CS_N:
            continue
        _, u_m = pairwise_resid_day(g['S_cnn'].values, g['S_mlp'].values)
        g['S_sp'] = u_m
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    build_baseline_scores()
