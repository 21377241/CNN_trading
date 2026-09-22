# -*- coding: utf-8 -*-
"""CNN-side OHLCV interventions (numeric → redraw)."""
from __future__ import annotations

import numpy as np

from experiments.E4_recent_path.paths import COL_C, COL_H, COL_L, COL_O, COL_VN, RECENT5

EPS = 1e-12


def _fix_hl(o, h, l, c):
    h = np.maximum(np.maximum(h, o), c)
    l = np.minimum(np.minimum(l, o), c)
    return h, l


def flatten_day(o, h, l, c, v, day: int):
    """Eliminate price change on `day` by pinning to previous close."""
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    assert 0 <= day < len(o)
    prev = c[day - 1] if day > 0 else c[day]
    o[day] = h[day] = l[day] = c[day] = prev
    lo = max(0, day - 2)
    hi = min(len(v), day + 3)
    v[day] = float(np.mean(v[lo:hi])) if hi > lo else float(v[day])
    h, l = _fix_hl(o, h, l, c)
    return o, h, l, c, v


def block_volume_recent5(o, h, l, c, v):
    """Replace recent5 volume with mean of earlier days."""
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    base = float(np.mean(v[:15])) if len(v) >= 15 else float(np.mean(v))
    v[15:] = base
    return o, h, l, c, v


def block_wick_recent5(o, h, l, c, v):
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    for k in RECENT5:
        h[k] = max(o[k], c[k])
        l[k] = min(o[k], c[k])
    return o, h, l, c, v


def block_direction_recent5(o, h, l, c, v):
    """Neutralize body sign but keep range: set C=O, keep H/L."""
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    for k in RECENT5:
        mid = 0.5 * (o[k] + c[k])
        o[k] = mid
        c[k] = mid
        h, l = _fix_hl(o, h, l, c)
    return o, h, l, c, v


def block_return_recent5(o, h, l, c, v):
    """Zero cumret over recent5: pin path at C_14, keep each day's H-L range."""
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    anchor = c[14]
    for k in RECENT5:
        rng = max(h[k] - l[k], EPS)
        half = 0.5 * rng
        o[k] = c[k] = anchor
        h[k] = anchor + half
        l[k] = anchor - half
    h, l = _fix_hl(o, h, l, c)
    return o, h, l, c, v


def block_gap_recent5(o, h, l, c, v):
    """Remove overnight gaps in recent5: set O_k = C_{k-1}, shift H/L/C by same delta."""
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    for k in RECENT5:
        target_o = c[k - 1]
        delta = target_o - o[k]
        o[k] = target_o
        h[k] += delta
        l[k] += delta
        c[k] += delta
    h, l = _fix_hl(o, h, l, c)
    return o, h, l, c, v


def block_intraday_recent5(o, h, l, c, v):
    """Compress intraday body: set O=C=mid, keep H/L range."""
    o, h, l, c, v = o.copy(), h.copy(), l.copy(), c.copy(), v.copy()
    for k in RECENT5:
        mid = 0.5 * (h[k] + l[k])
        o[k] = mid
        c[k] = mid
    h, l = _fix_hl(o, h, l, c)
    return o, h, l, c, v


CNN_BLOCK_FNS = {
    'volume': block_volume_recent5,
    'wick': block_wick_recent5,
    'direction': block_direction_recent5,
    'return': block_return_recent5,
    'gap': block_gap_recent5,
    'intraday': block_intraday_recent5,
}


def apply_named(o, h, l, c, v, name: str):
    if name.startswith('flatten_day_'):
        day = int(name.split('_')[-1])
        return flatten_day(o, h, l, c, v, day)
    if name in CNN_BLOCK_FNS:
        return CNN_BLOCK_FNS[name](o, h, l, c, v)
    raise ValueError(name)


def extract_ohlcv_from_row(row_2d: np.ndarray):
    """row_2d: (20, 9)."""
    return row_2d[:, COL_O], row_2d[:, COL_H], row_2d[:, COL_L], row_2d[:, COL_C], row_2d[:, COL_VN]
