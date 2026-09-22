# -*- coding: utf-8 -*-
"""NumPy bar-chart renderer matching DrawOHLC (bar + ma20 + volume)."""
from __future__ import annotations

import numpy as np

# From Data.dgp_config (E1b charts)
BACKGROUND_COLOR = 0
CHART_COLOR = 255
VOLUME_CHART_GAP = 2
VOLUME_CHART_HEIGHT_RATIO = 0.2


def fast_draw_bar(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray,
    img_w: int = 60, img_h: int = 64,
) -> np.ndarray:
    """Return float32 image (1, H, W) in [0,1], matching DrawOHLC.draw_to_array()/255."""
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    T = len(c)
    vol_h = max(4, int(img_h * VOLUME_CHART_HEIGHT_RATIO))
    gap = VOLUME_CHART_GAP
    price_h = img_h - vol_h - gap
    img = np.full((img_h, img_w), BACKGROUND_COLOR, dtype=np.uint8)

    prices = np.concatenate([o, h, l, c])
    valid = prices[np.isfinite(prices)]
    if len(valid) == 0:
        return (img.astype(np.float32) / 255.0).reshape(1, img_h, img_w)
    pmin = float(valid.min())
    pmax = float(valid.max())
    pr = pmax - pmin
    if pr < 1e-10:
        pr = 1e-10

    def price_to_y(p: float) -> int:
        norm = (p - pmin) / pr
        return int((1.0 - norm) * (price_h - 1))

    bar_pixel_w = img_w / max(T, 1)
    body_w = max(1, int(bar_pixel_w * 0.6))

    for t in range(T):
        if not (np.isfinite(o[t]) and np.isfinite(h[t]) and np.isfinite(l[t]) and np.isfinite(c[t])):
            continue
        cx = int((t + 0.5) * bar_pixel_w)
        x0 = max(0, cx - body_w // 2)
        x1 = min(img_w - 1, cx + body_w // 2)
        y_high = price_to_y(h[t])
        y_low = price_to_y(l[t])
        y_open = price_to_y(o[t])
        y_close = price_to_y(c[t])
        y0l, y1l = min(y_high, y_low), max(y_high, y_low)
        img[y0l:y1l + 1, cx] = CHART_COLOR
        y_top = min(y_open, y_close)
        y_bot = max(y_open, y_close)
        if y_top == y_bot:
            y_bot = y_top + 1
        img[y_top:y_bot + 1, x0:x1 + 1] = CHART_COLOR

    ma = np.cumsum(c) / np.arange(1, T + 1)
    pts = []
    for t in range(T):
        if np.isfinite(ma[t]):
            cx = int((t + 0.5) * bar_pixel_w)
            pts.append((cx, price_to_y(float(ma[t]))))
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, n + 1).astype(int)
        ys = np.linspace(y0, y1, n + 1).astype(int)
        img[ys, xs] = CHART_COLOR

    vv = v[np.isfinite(v)]
    if len(vv):
        vmax = float(vv.max())
        if vmax >= 1e-10:
            vol_top = price_h + gap
            vol_bottom = img_h - 1
            for t in range(T):
                if not np.isfinite(v[t]):
                    continue
                bar_hh = max(1, int((v[t] / vmax) * (vol_bottom - vol_top)))
                cx = int((t + 0.5) * bar_pixel_w)
                x0 = max(0, cx - body_w // 2)
                x1 = min(img_w - 1, cx + body_w // 2)
                y_top = vol_bottom - bar_hh
                img[y_top:vol_bottom + 1, x0:x1 + 1] = CHART_COLOR

    return (img.astype(np.float32) / 255.0).reshape(1, img_h, img_w)
