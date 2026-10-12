# -*- coding: utf-8 -*-
"""
Data/ohlc_render.py - Jiang, Kelly and Xiu (2023) OHLC 图像的批量渲染

与原作者的 DrawOHLC（Data/jkx_chart_library.py）逐像素相同（tests/test_chart_library.py 检查）：
- 纵轴：O/H/L/C 与均线一起取最小、最大值定标；y = np.around((p − min) × (价格区行数 − 1) / (max − min))，
  在“底部为 0”的坐标中取整（.5 取偶），再上下翻转；
- 64 行 = 价格区 51 行 + 间隔 1 行 + 成交量区 12 行；
- 每天 3 列：开盘刻度画在左列和中列，高低价竖线在中列，收盘刻度在右列；高或低缺失的日子不画；
- 均线：相邻两天都有值时用 PIL 连线，另画最后一天的点；
- 成交量：柱高 np.around(V × 12 / Vmax) 像素，至少 1 像素；缺失或非正的成交量不画。
价格区间为零（最大 = 最小）的窗口原作者不生成图像，这里报错，由 universe 事先剔除。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from Data.dgp_config import IMAGE_HEIGHT, VOLUME_CHART_GAP, CHART_COLOR


def layout(window: int, has_volume: bool = True) -> dict:
    h = IMAGE_HEIGHT.get(window, 64)
    if has_volume:
        vol_h = int(h / 5)
        price_h = h - vol_h - VOLUME_CHART_GAP
        vol_top, vol_bottom = price_h + VOLUME_CHART_GAP, h - 1
    else:
        price_h, vol_top, vol_bottom = h, None, None
    return {'height': h, 'width': 3 * window, 'price_h': price_h, 'vol_top': vol_top, 'vol_bottom': vol_bottom}


def price_bounds(O, H, L, C, MA=None) -> tuple[np.ndarray, np.ndarray]:
    """每个窗口 O/H/L/C/MA 中有限值的最小、最大值；没有有限值时为 NaN。"""
    arrs = [np.asarray(a, dtype=np.float64) for a in (O, H, L, C)]
    if MA is not None:
        arrs.append(np.asarray(MA, dtype=np.float64))
    stack = np.stack(arrs)
    ok = np.isfinite(stack)
    lo = np.where(ok, stack, np.inf).min(axis=(0, 2))
    hi = np.where(ok, stack, -np.inf).max(axis=(0, 2))
    none = ~ok.any(axis=(0, 2))
    return np.where(none, np.nan, lo), np.where(none, np.nan, hi)


def _draw_ma(ma: np.ndarray, y: np.ndarray, ph: int, width: int, color: int) -> np.ndarray:
    """PIL 画均线，返回 (N, ph, width) 的布尔数组（已上下翻转）。每个窗口占画布中一段高 ph 的条带。"""
    N, T = ma.shape
    ok = np.isfinite(ma)
    canvas = Image.new('L', (width, ph * N), 0)
    draw = ImageDraw.Draw(canvas)
    xs = 3 * np.arange(T) + 1
    seg = ok[:, :-1] & ok[:, 1:]
    for n, t in zip(*np.nonzero(seg)):
        off = ph * int(n)
        draw.line((int(xs[t]), off + int(y[n, t]), int(xs[t + 1]), off + int(y[n, t + 1])), width=1, fill=color)
    px = canvas.load()
    for n in np.flatnonzero(ok[:, -1]):
        px[int(xs[-1]), ph * int(n) + int(y[n, -1])] = color
    arr = np.asarray(canvas, dtype=np.uint8).reshape(N, ph, width)
    return arr[:, ::-1, :] > 0


def render_ohlc_batch(O, H, L, C, MA=None, V=None, has_volume: bool = True,
                      color: int = CHART_COLOR) -> np.ndarray:
    """O/H/L/C/MA/V: (N, T)；返回 uint8 (N, height, 3T)。"""
    O, H, L, C = (np.asarray(a, dtype=np.float64) for a in (O, H, L, C))
    MA = None if MA is None else np.asarray(MA, dtype=np.float64)
    N, T = C.shape
    lay = layout(T, has_volume and V is not None)
    ph = lay['price_h']
    img = np.zeros((N, lay['height'], lay['width']), dtype=np.uint8)
    if N == 0:
        return img

    lo, hi = price_bounds(O, H, L, C, MA)
    bad = ~np.isfinite(lo) | ~(hi > lo)
    if bad.any():
        raise ValueError(f'{int(bad.sum())} 个窗口没有价格或价格区间为零（原作者不生成图像）')
    ppu = (ph - 1.0) / (hi - lo)

    def to_row(p):
        y = np.around((p - lo[:, None]) * ppu[:, None])
        return (ph - 1) - np.nan_to_num(y, nan=0).astype(np.int64), y

    rH, yH = to_row(H)
    rL, yL = to_row(L)
    rO, yO = to_row(O)
    rC, yC = to_row(C)
    bar_ok = np.isfinite(yH) & np.isfinite(yL)
    rows = np.arange(ph)
    for t in range(T):
        idx = np.flatnonzero(bar_ok[:, t])
        if len(idx) == 0:
            continue
        bar = (rows[None, :] >= rH[idx, t, None]) & (rows[None, :] <= rL[idx, t, None])
        img[idx, :ph, 3 * t + 1] = np.where(bar, color, img[idx, :ph, 3 * t + 1])
        io = idx[np.isfinite(yO[idx, t])]
        img[io, rO[io, t], 3 * t] = color
        img[io, rO[io, t], 3 * t + 1] = color
        ic = idx[np.isfinite(yC[idx, t])]
        img[ic, rC[ic, t], 3 * t + 2] = color

    if MA is not None:
        yM = np.nan_to_num(np.around((MA - lo[:, None]) * ppu[:, None]), nan=0).astype(np.int64)
        lit = _draw_ma(MA, yM, ph, lay['width'], color)
        img[:, :ph, :][lit] = color

    if lay['vol_top'] is not None:
        V = np.asarray(V, dtype=np.float64)
        V = np.where(np.isfinite(V) & (V > 0), V, np.nan)
        vmax = np.nanmax(np.where(np.isfinite(V), V, -np.inf), axis=1)
        vol_h = lay['vol_bottom'] - lay['vol_top'] + 1
        vrows = np.arange(lay['vol_top'], lay['vol_bottom'] + 1)
        for t in range(T):
            idx = np.flatnonzero(np.isfinite(V[:, t]) & (vmax > 0))
            if len(idx) == 0:
                continue
            ppv = 1.0 * vol_h / vmax[idx]
            bar_h = np.maximum(1, np.around(V[idx, t] * ppv).astype(np.int64))
            bar = vrows[None, :] > (lay['vol_bottom'] - bar_h)[:, None]
            img[idx, lay['vol_top']:, 3 * t + 1] = np.where(bar, color, 0).astype(np.uint8)
    return img
