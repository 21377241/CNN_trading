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
from Data.clean_window import close0_path, features_140, ma_sets_bound_share  # noqa: E402
from Data.jkx_chart_library import DrawOHLC as JkxDrawOHLC  # noqa: E402
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


def _raw_windows(n, seed, missing_share=0.1, tick=True):
    """原始价格窗口（含 19 天均线回溯）；价格在 0.01 网格上以制造 .5 像素并列；部分日子停牌。"""
    rng = np.random.default_rng(seed)
    T = 39
    c = 10 * np.cumprod(1 + rng.normal(0, rng.uniform(0.005, 0.04, (n, 1)), (n, T)), axis=1)
    o = c * (1 + rng.normal(0, 0.01, (n, T)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, (n, T))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, (n, T))))
    if tick:
        o, h, l, c = (np.round(a, 2) for a in (o, h, l, c))
    v = rng.lognormal(10, 1, (n, T))
    v[rng.random((n, T)) < 0.02] *= 1e-4
    miss = rng.random((n, T)) < missing_share
    miss[:, -1] = False
    for a in (o, h, l, c, v):
        a[miss] = np.nan
    ma = pd.DataFrame(c.T).apply(lambda s: s.dropna().rolling(20).mean().reindex(s.index).ffill()).T.to_numpy()
    sl = slice(T - 20, T)
    return {'O': o[:, sl], 'H': h[:, sl], 'L': l[:, sl], 'C': c[:, sl], 'MA': ma[:, sl], 'V': v[:, sl]}


def _official(p, n):
    df = pd.DataFrame({'Open': p['O'][n], 'High': p['H'][n], 'Low': p['L'][n], 'Close': p['C'][n],
                       'ma20': p['MA'][n], 'Vol': np.where(p['V'][n] > 0, p['V'][n], np.nan)})
    img = JkxDrawOHLC(df, has_volume_bar=True, ma_lags=[20]).draw_image()
    return None if img is None else np.asarray(img, dtype=np.uint8)


def test_layout():
    assert (LAY['height'], LAY['width'], PH, VT, VB) == (64, 60, 51, 52, 63)


def test_matches_official_drawohlc():
    n_cmp = 0
    for seed, miss, tick in ((0, 0.0, True), (1, 0.1, True), (2, 0.3, True), (3, 0.1, False)):
        p = close0_path(_raw_windows(300, seed, miss, tick))
        mine = render_ohlc_batch(p['O'], p['H'], p['L'], p['C'], p['MA'], p['V'])
        for n in range(len(mine)):
            ref = _official(p, n)
            assert ref is not None
            assert np.array_equal(mine[n], ref), f'seed {seed} 窗口 {n} 与官方 DrawOHLC 不同'
            n_cmp += 1
    assert n_cmp == 1200


def test_ma_sets_vertical_scale():
    o, h, l, c, v, _ = _window(6)
    ma = np.full(20, h.max() * 1.2)
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], ma[None], v[None])[0]
    assert img[0, 1::3].all(), '均线高于全部价格时占据价格区顶行'
    assert not (img[0, 1::3] & img[1, 1::3]).any(), '顶行没有高低价竖线'
    p = close0_path({'O': o[None], 'H': h[None], 'L': l[None], 'C': c[None], 'MA': ma[None], 'V': v[None]})
    assert ma_sets_bound_share(p) == 1.0


def test_missing_days_blank_and_ma_not_connected():
    p = close0_path(_raw_windows(1, 7, 0.0))
    for k in ('O', 'H', 'L', 'C', 'MA', 'V'):
        p[k][0, 10] = np.nan
    p['valid'][0, 10] = False
    img = render_ohlc_batch(p['O'], p['H'], p['L'], p['C'], p['MA'], p['V'])[0]
    assert img[:, 30:33].sum() == 0, '缺失日三列留空，两侧均线不连'
    assert np.array_equal(img, _official(p, 0))


def test_flat_window_raises():
    one = np.ones((1, 20))
    try:
        render_ohlc_batch(one, one, one, one, one, one)
    except ValueError:
        return
    raise AssertionError('价格区间为零应报错')


def test_volume_bar_heights():
    o, h, l, c, v, ma = _window(4)
    v[3] = v.max() * 1e-4
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], ma[None], v[None])[0]
    for t in range(20):
        bh = max(1, int(np.around(v[t] * 12 / v.max())))
        assert (img[VT:, 3 * t + 1] > 0).sum() == bh
        assert img[VB, 3 * t + 1] == 255
    assert img[VT:, 0::3].sum() == 0 and img[VT:, 2::3].sum() == 0
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


def test_entry_points_agree():
    o, h, l, c, v, ma = _window(5)
    img = render_ohlc_batch(o[None], h[None], l[None], c[None], ma[None], v[None])[0]
    df = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Vol': v, 'ma': ma})
    a = DrawOHLC(df, 20, ['ma'], bar_style='ohlc').draw_to_array().reshape(64, 60)
    b = (fast_draw_bar(o, h, l, c, v, ma)[0] * 255).round().astype(np.uint8)
    assert np.array_equal(a, img) and np.array_equal(b, img)


def test_features_140_match_image_coordinates():
    p = close0_path(_raw_windows(200, 8, 0.1, tick=False))
    x = features_140(p)
    img = render_ohlc_batch(p['O'], p['H'], p['L'], p['C'], p['MA'], p['V'])
    assert np.isfinite(x).all() and x[..., :5].min() >= 0 and x[..., :5].max() <= 1 + 1e-6
    ok = p['valid']
    assert np.allclose(x[..., :5].max(axis=(1, 2)), 1, atol=1e-6) and np.allclose(x[..., :5].min(axis=(1, 2)), 0, atol=1e-6)
    rows = (PH - 1) - np.around(x[..., 3].astype(np.float64) * (PH - 1)).astype(int)
    n, t = np.nonzero(ok)
    assert (img[n, rows[n, t], 3 * t + 2] == 255).all(), '收盘刻度位置与特征一致'
    assert np.allclose(x[~ok][:, 5:], 0), '缺失日成交量与交易标记为 0'


def test_features_140_flat_window_raises():
    w = {k: np.full((1, 20), 8.0) for k in ('O', 'H', 'L', 'C', 'MA')}
    w['V'] = np.full((1, 20), 100.0)
    try:
        features_140(close0_path(w))
    except ValueError:
        return
    raise AssertionError('价格区间为零应报错')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'{len(fns)} passed')
