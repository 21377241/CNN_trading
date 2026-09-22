# -*- coding: utf-8 -*-
"""Shared CNN forward after OHLCV intervention (year resume)."""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import torch

from Data.chart_library import DrawOHLC
from Model.cnn_model import MODEL_NAME_I20, init_model
from experiments.E2_matched_tabular.e2a_layout import FEAT_180
from experiments.E3_shared_specific.paths import E1B_RIC_EXP
from experiments.E3_shared_specific.utils import norm_stock_id
from experiments.E4_recent_path.interventions_ohlcv import apply_named, extract_ohlcv_from_row
from experiments.E4_recent_path.paths import BATCH, IMG_H, IMG_W, OOS_YEARS, PER_DAY, WINDOW

logger = logging.getLogger('e4_cnn')


def load_cnn_models(device: torch.device):
    models = []
    for i in range(5):
        path = os.path.join(E1B_RIC_EXP, f'checkpoint{i}.pth.tar')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        m = init_model(MODEL_NAME_I20, img_h=IMG_H, img_w=IMG_W, dropout=0.0, n_classes=2, window_size=20)
        m.load_state_dict(ckpt['model_state_dict'])
        m.to(device).eval()
        models.append(m)
    return models


def draw_one(o, h, l, c, v) -> np.ndarray:
    ma = pd.Series(c).rolling(20, min_periods=1).mean().values
    df = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Vol': v, 'ma20': ma})
    arr = DrawOHLC(
        ohlcv_df=df, window_size=WINDOW, ma_cols=['ma20'],
        has_volume_bar=True, chart_type='bar',
        img_width=IMG_W, img_height=IMG_H,
    ).draw_to_array()
    return arr.astype(np.float32).reshape(1, IMG_H, IMG_W) / 255.0


@torch.no_grad()
def infer_batch(models, imgs: np.ndarray, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(imgs).to(device)
    probs = [m.predict_proba(t)[:, 1].cpu().numpy() for m in models]
    return np.mean(probs, axis=0)


def predict_intervention(
    name: str,
    out_dir: str,
    models,
    device: torch.device,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_feather(os.path.join(FEAT_180, 'meta_oos.feather'))
    meta['StockID'] = norm_stock_id(meta['StockID'])
    meta['ending_date'] = pd.to_datetime(meta['ending_date'])
    meta['year'] = meta['ending_date'].dt.year
    X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    parts = []
    for year in OOS_YEARS:
        ypath = os.path.join(out_dir, f'pred_cnn_{name}_oos_{year}.csv')
        if os.path.exists(ypath):
            logger.info(f'  resume {name} {year}')
            parts.append(pd.read_csv(ypath))
            continue
        idx = np.where(meta['year'].values == year)[0]
        if len(idx) == 0:
            continue
        logger.info(f'  {name} year={year} n={len(idx):,}')
        Xy = np.asarray(X[idx], dtype=np.float64).reshape(-1, WINDOW, PER_DAY)
        n = len(idx)
        up = np.empty(n, dtype=np.float32)
        t0 = time.time()
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            imgs = np.empty((end - start, 1, IMG_H, IMG_W), dtype=np.float32)
            for i, j in enumerate(range(start, end)):
                o, h, l, c, v = extract_ohlcv_from_row(Xy[j])
                o, h, l, c, v = apply_named(o, h, l, c, v, name)
                imgs[i] = draw_one(o, h, l, c, v)
            up[start:end] = infer_batch(models, imgs, device)
            if (start // BATCH) % 40 == 0:
                logger.info(f'    {name} {end:,}/{n:,} ({time.time()-t0:.0f}s)')
        pred_y = pd.DataFrame({
            'StockID': meta['StockID'].values[idx],
            'ending_date': meta['ending_date'].values[idx],
            'up_prob': up,
        })
        pred_y.to_csv(ypath, index=False)
        parts.append(pred_y)
    full = pd.concat(parts, ignore_index=True)
    full.to_csv(os.path.join(out_dir, f'pred_cnn_{name}_oos.csv'), index=False)
    return full
