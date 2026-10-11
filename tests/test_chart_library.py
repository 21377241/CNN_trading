# -*- coding: utf-8 -*-
"""渲染器与窗口特征的检查。运行：python tests/test_chart_library.py（也兼容 pytest）。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'trend_submit', 'trend_code_submit')):
    if p not in sys.path:
        sys.path.insert(0, p)

from Data.chart_library import DrawOHLC  # noqa: E402
from Data.clean_window import close0_path, features_140  # noqa: E402
from Data.ohlc_render import layout, render_ohlc_batch  # noqa: E402
from experiments.E4_recent_path.fast_draw import fast_draw_bar  # noqa: E402

LAY = layout(20)
PH, VT, VB = LAY['price_h'], LAY['vol_top'], LAY['vol_bottom']


def _window(seed=0, T=20):
    rng = np.random.default_rng(seed)
    c = np.cumprod(1 + rng.normal(0, 0.02, T))
    o = c * (1 + rng.normal(0, 0.01, T))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, T)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, T)))
    v = rng.uniform(1, 10, T)
    ma = pd.Series(c).rolling(5, min_periods=1).mean().to_numpy()
    return o, h, l, c, v, ma


def _y(p, lo, hi):
    return int(np.floor((1 - (p - lo) / (hi - lo)) * (PH - 1) + 0.5))


def test_layout_and_tick_positions():
    assert (LAY['height'], LAY['width'], PH, VT, VB) == (64, 60, 50, 52, 63)
    o, h, l, c, v, _ = _window()
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], None, v[None])[0]
    lo, hi = l.min(), h.max()
    for t in range(20):
        assert img[_y(o[t], lo, hi), 3 * t] == 255
        assert img[_y(c[t], lo, hi), 3 * t + 2] == 255
        assert img[:PH, 3 * t].sum() == 255 and img[:PH, 3 * t + 2].sum() == 255
        col = img[:PH, 3 * t + 1]
        assert np.array_equal(np.flatnonzero(col), np.arange(_y(h[t], lo, hi), _y(l[t], lo, hi) + 1))
    assert img[PH:VT].sum() == 0


def test_open_close_visible_vs_legacy():
    o, h, l, c, v, ma = _window(1)
    df = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Vol': v, 'ma': ma})
    sw = df.assign(Open=c, Close=o)
    new_a = DrawOHLC(df, 20, ['ma'], bar_style='ohlc').draw_to_array()
    new_b = DrawOHLC(sw, 20, ['ma'], bar_style='ohlc').draw_to_array()
    old_a = DrawOHLC(df, 20, ['ma'], bar_style='legacy_candle').draw_to_array()
    old_b = DrawOHLC(sw, 20, ['ma'], bar_style='legacy_candle').draw_to_array()
    assert not np.array_equal(new_a, new_b), '交换开盘价与收盘价后新图像应改变'
    assert np.array_equal(old_a, old_b), '旧画法看不到开盘价与收盘价（复现旧缺陷）'
    old_img = DrawOHLC(df, 20, [], bar_style='legacy_candle').draw_to_array().reshape(64, 60)
    assert old_img[:, 0::3].sum() == 0 and old_img[:, 2::3].sum() == 0


def test_rounding_extremes():
    o, h, l, c, v, _ = _window(2)
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], None, v[None])[0]
    th, tl = int(h.argmax()), int(l.argmin())
    assert img[0, 3 * th + 1] == 255 and img[PH - 1, 3 * tl + 1] == 255
    lo, hi = 1.0, 2.0
    p = lo + (hi - lo) * (1 - 10.5 / (PH - 1))
    assert _y(p, lo, hi) == 11


def test_ma_and_missing_days():
    o, h, l, c = (np.full(20, 1.9) for _ in range(4))
    h[0], l[0] = 2.0, 1.0
    v = np.ones(20)
    lo, hi = 1.0, 2.0
    ma = np.full(20, 1.5)
    ma[5] = 2.5
    for a in (o, h, l, c, v):
        a[10] = np.nan
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], ma[None], v[None])[0]
    assert img[:, 30:33].sum() == 0, '缺失日三列留空'
    ym = _y(ma[0], lo, hi)
    assert ym != _y(1.9, lo, hi)
    assert img[ym, 3 * 2 + 1] == 255
    assert img[ym, 3 * 2 + 2] == 255 and img[ym, 3 * 3] == 255, '相邻有效日之间连线'
    assert img[ym, 3 * 4 + 2] == 0 and img[ym, 3 * 6] == 0, '超出范围的均线点不画、不连线'
    assert img[ym, 3 * 9 + 2] == 0 and img[ym, 3 * 11] == 0, '缺失日两侧不连线'


def test_volume_and_entry_points_agree():
    o, h, l, c, v, ma = _window(4)
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], ma[None], v[None])[0]
    tm = int(v.argmax())
    assert np.array_equal(np.flatnonzero(img[VT:, 3 * tm + 1]), np.arange(VB - VT + 1))
    assert img[VT:, 0::3].sum() == 0 and img[VT:, 2::3].sum() == 0
    for t in range(20):
        bh = max(1, int(v[t] / v.max() * (VB - VT)))
        assert img[VT:, 3 * t + 1].sum() == 255 * (bh + 1)
    df = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Vol': v, 'ma': ma})
    a = DrawOHLC(df, 20, ['ma'], bar_style='ohlc').draw_to_array().reshape(64, 60)
    b = (fast_draw_bar(o, h, l, c, v, ma)[0] * 255).round().astype(np.uint8)
    assert np.array_equal(a, img) and np.array_equal(b, img)


def test_features_140_match_image_coordinates():
    o, h, l, c, v, ma = _window(5)
    w = {'O': o[None] * 7, 'H': h[None] * 7, 'L': l[None] * 7, 'C': c[None] * 7, 'MA': ma[None] * 7, 'V': v[None]}
    for k in ('O', 'H', 'L', 'C', 'V'):
        w[k] = w[k].copy()
        w[k][0, [0, 10]] = np.nan
    p = close0_path(w)
    x = features_140(p)[0]
    assert x.shape == (20, 7)
    assert np.allclose(x[[0, 10], 6], 0) and np.allclose(x[[0, 10], 5], 0)
    assert np.allclose(x[0, :5], x[1, 3]) and np.allclose(x[10, :5], x[9, 3])
    ok = p['valid'][0]
    assert np.isclose(x[ok, 1].max(), 1) and np.isclose(x[ok, 2].min(), 0)
    assert np.isclose(x[ok, 5].max(), 1)
    pix = np.floor((1 - x[ok, 3]) * (PH - 1) + 0.5).astype(int)
    rendered = render_ohlc_batch(p['O'], p['H'], p['L'], p['C'], None, p['V'])[0]
    days = np.flatnonzero(ok)
    assert all(rendered[pix[i], 3 * d + 2] == 255 for i, d in enumerate(days))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'{len(fns)} passed')
