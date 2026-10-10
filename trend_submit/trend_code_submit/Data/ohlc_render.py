# -*- coding: utf-8 -*-
"""
Data/ohlc_render.py - Jiang, Kelly and Xiu (2023) OHLC 图像的批量渲染（numpy）

每天 3 列：左列开盘价刻度、中列高低价竖线、右列收盘价刻度；均线点与成交量柱画在中列。
价格区纵轴按有交易日的 OHLC 最高/最低价定标，价格到像素四舍五入；超出该范围的均线点不画，
均线只在相邻两天都可画时连线。缺失日（OHLC 任一为 NaN）三列留空。
"""
from __future__ import annotations

import numpy as np

from Data.dgp_config import IMAGE_HEIGHT, VOLUME_CHART_GAP, VOLUME_CHART_HEIGHT_RATIO, CHART_COLOR


def layout(window: int, has_volume: bool = True) -> dict:
    h = IMAGE_HEIGHT.get(window, 64)
    if has_volume:
        vol_h = max(4, int(h * VOLUME_CHART_HEIGHT_RATIO))
        price_h = h - vol_h - VOLUME_CHART_GAP
        vol_top, vol_bottom = price_h + VOLUME_CHART_GAP, h - 1
    else:
        price_h, vol_top, vol_bottom = h, None, None
    return {'height': h, 'width': 3 * window, 'price_h': price_h, 'vol_top': vol_top, 'vol_bottom': vol_bottom}


def _round(x):
    return np.floor(x + 0.5)


def price_bounds(O, H, L, C) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & np.isfinite(C)
    stack = np.stack([O, H, L, C])
    lo = np.where(valid, stack, np.inf).min(axis=(0, 2))
    hi = np.where(valid, stack, -np.inf).max(axis=(0, 2))
    return lo, hi, valid


def render_ohlc_batch(O, H, L, C, MA=None, V=None, has_volume: bool = True,
                      color: int = CHART_COLOR) -> np.ndarray:
    """O/H/L/C/MA/V: (N, T)；返回 uint8 (N, height, 3T)。"""
    O, H, L, C = (np.asarray(a, dtype=np.float64) for a in (O, H, L, C))
    N, T = C.shape
    lay = layout(T, has_volume and V is not None)
    ph = lay['price_h']
    img = np.zeros((N, lay['height'], lay['width']), dtype=np.uint8)
    if N == 0:
        return img

    lo, hi, valid = price_bounds(O, H, L, C)
    has_any = valid.any(axis=1)
    lo = np.where(has_any, lo, 0.0)
    hi = np.where(has_any, hi, 1.0)
    rng = hi - lo
    rng = np.where(rng < 1e-10, 1e-10, rng)

    def to_y(p):
        return _round((1.0 - (p - lo[:, None]) / rng[:, None]) * (ph - 1))

    yO, yH, yL, yC = (np.clip(np.nan_to_num(to_y(a), nan=0), 0, ph - 1).astype(np.int64) for a in (O, H, L, C))
    rows = np.arange(ph)
    for t in range(T):
        idx = np.flatnonzero(valid[:, t])
        if len(idx) == 0:
            continue
        img[idx, yO[idx, t], 3 * t] = color
        img[idx, yC[idx, t], 3 * t + 2] = color
        bar = (rows[None, :] >= yH[idx, t, None]) & (rows[None, :] <= yL[idx, t, None])
        img[idx, :ph, 3 * t + 1] = np.where(bar, color, img[idx, :ph, 3 * t + 1])

    if MA is not None:
        MA = np.asarray(MA, dtype=np.float64)
        ok = valid & np.isfinite(MA) & (MA >= lo[:, None]) & (MA <= hi[:, None])
        yM = np.clip(np.nan_to_num(to_y(MA), nan=0), 0, ph - 1).astype(np.int64)
        for t in range(T):
            idx = np.flatnonzero(ok[:, t])
            img[idx, yM[idx, t], 3 * t + 1] = color
        for t in range(T - 1):
            idx = np.flatnonzero(ok[:, t] & ok[:, t + 1])
            if len(idx) == 0:
                continue
            y0 = yM[idx, t].astype(np.float64)
            dy = yM[idx, t + 1] - y0
            steps = np.maximum(3.0, np.abs(dy))
            for k in range(int(steps.max()) + 1):
                kk = np.minimum(k, steps)
                x = 3 * t + 1 + _round(kk * 3.0 / steps).astype(np.int64)
                y = _round(y0 + kk * dy / steps).astype(np.int64)
                img[idx, y, x] = color

    if lay['vol_top'] is not None:
        V = np.where(valid, np.asarray(V, dtype=np.float64), np.nan)
        vmax = np.nanmax(np.where(np.isfinite(V), V, -np.inf), axis=1)
        span = lay['vol_bottom'] - lay['vol_top']
        vrows = np.arange(lay['vol_top'], lay['vol_bottom'] + 1)
        for t in range(T):
            idx = np.flatnonzero(np.isfinite(V[:, t]) & (vmax > 1e-10))
            if len(idx) == 0:
                continue
            bar_h = np.maximum(1, (V[idx, t] / vmax[idx] * span).astype(np.int64))
            bar = vrows[None, :] >= (lay['vol_bottom'] - bar_h)[:, None]
            img[idx, lay['vol_top']:, 3 * t + 1] = np.where(bar, color, 0).astype(np.uint8)
    return img
