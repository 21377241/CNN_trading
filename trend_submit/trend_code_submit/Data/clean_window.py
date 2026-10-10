# -*- coding: utf-8 -*-
"""
Data/clean_window.py - 干净版本的市场日窗口（clean_spec 第 3.2-3.4 节）

所有模型（chart-CNN 图像、1D-CNN-140、MLP-4-140）以及事后分析都从这里取窗口，保证样本和输入一致。

  grid = load_grid()                         日面板 → 稠密数组 (市场日 × 股票)
  w = gather_windows(grid, t_idx, s_idx)     信号日 t 往前 20 个市场日
  p = close0_path(w)                         Close₀ = 1 的价格路径（缺失日为 NaN）
  img = render_window_images(p)              uint8 (N, 64, 60)
  x = features_140(p)                        float32 (N, 20, 7)，每天 [O, H, L, C, MA20, V, has_trade]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Data.ohlc_render import price_bounds, render_ohlc_batch

WINDOW = 20
FEATURE_NAMES = ('O', 'H', 'L', 'C', 'MA20', 'V', 'has_trade')
PER_DAY = len(FEATURE_NAMES)


@dataclass
class Grid:
    dates: pd.DatetimeIndex
    stocks: pd.Index
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    MA: np.ndarray
    V: np.ndarray

    def stock_index(self, stock_ids) -> np.ndarray:
        idx = self.stocks.get_indexer(pd.Index(stock_ids).astype(str))
        if (idx < 0).any():
            raise KeyError(f'{int((idx < 0).sum())} 个 StockID 不在网格中')
        return idx

    def date_index(self, dates) -> np.ndarray:
        idx = self.dates.get_indexer(pd.DatetimeIndex(dates))
        if (idx < 0).any():
            raise KeyError(f'{int((idx < 0).sum())} 个日期不在市场日历中')
        return idx


def load_grid(panel_path: str) -> Grid:
    """MA 按股票沿时间前向填充：缺失日的 MA20 = 最近一个交易日的 MA20（即最近 20 个有效收盘价的均值）。"""
    p = pd.read_feather(panel_path, columns=['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'MA20', 'Vol'])
    dates = pd.DatetimeIndex(np.sort(p['Date'].unique()))
    stocks = pd.Index(p['StockID'].cat.categories.astype(str))
    di = dates.get_indexer(p['Date'])
    si = p['StockID'].cat.codes.to_numpy()
    shape = (len(dates), len(stocks))

    def dense(col, dtype):
        a = np.full(shape, np.nan, dtype=dtype)
        a[di, si] = p[col].to_numpy(dtype=dtype)
        return a

    O, H, L, C = (dense(c, np.float64) for c in ('Open', 'High', 'Low', 'Close'))
    MA = pd.DataFrame(dense('MA20', np.float64)).ffill().to_numpy()
    V = dense('Vol', np.float64)
    return Grid(dates, stocks, O, H, L, C, MA, V)


def gather_windows(grid: Grid, t_idx, s_idx, window: int = WINDOW) -> dict:
    t_idx = np.asarray(t_idx, dtype=np.int64)
    s_idx = np.asarray(s_idx, dtype=np.int64)
    if (t_idx < window - 1).any():
        raise ValueError('信号日之前不足一个窗口的市场日')
    rows = t_idx[:, None] + np.arange(-window + 1, 1)[None, :]
    cols = s_idx[:, None]
    return {k: getattr(grid, k)[rows, cols] for k in ('O', 'H', 'L', 'C', 'MA', 'V')}


def close0_path(w: dict) -> dict:
    """窗口内第一个有效收盘价归一为 1；O/H/L/MA 同比例缩放；修正 H/L 包住 O/C。缺失日 O/H/L/C/V 为 NaN。"""
    valid = np.isfinite(w['C'])
    if not valid.any(axis=1).all():
        raise ValueError('存在窗口内没有任何交易日的样本')
    first = valid.argmax(axis=1)
    c0 = w['C'][np.arange(len(first)), first][:, None]
    O, H, L, C, MA = (w[k] / c0 for k in ('O', 'H', 'L', 'C', 'MA'))
    H = np.where(valid, np.fmax(H, np.fmax(O, C)), np.nan)
    L = np.where(valid, np.fmin(L, np.fmin(O, C)), np.nan)
    O = np.where(valid, O, np.nan)
    C = np.where(valid, C, np.nan)
    V = np.where(valid, w['V'], np.nan)
    return {'O': O, 'H': H, 'L': L, 'C': C, 'MA': MA, 'V': V, 'valid': valid}


def render_window_images(p: dict) -> np.ndarray:
    return render_ohlc_batch(p['O'], p['H'], p['L'], p['C'], p['MA'], p['V'], has_volume=True)


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    """沿时间轴前向填充；开头的 NaN 用第一个有效值。"""
    n, T = a.shape
    ok = np.isfinite(a)
    idx = np.where(ok, np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = a[np.arange(n)[:, None], idx]
    first = ok.argmax(axis=1)
    lead = np.arange(T)[None, :] < first[:, None]
    return np.where(lead, a[np.arange(n), first][:, None], out)


def features_140(p: dict) -> np.ndarray:
    """返回 float32 (N, 20, 7)。展平为 140 维时按天优先（reshape(N, -1)）。"""
    lo, hi, valid = price_bounds(p['O'], p['H'], p['L'], p['C'])
    rng = hi - lo
    rng = np.where(rng < 1e-10, 1e-10, rng)

    def norm(a):
        return (a - lo[:, None]) / rng[:, None]

    C = norm(p['C'])
    C_fill = _ffill_rows(C)
    O, H, L = (np.where(valid, norm(p[k]), C_fill) for k in ('O', 'H', 'L'))
    MA = norm(p['MA'])
    MA = np.where(np.isfinite(MA), MA, C_fill)
    V = np.where(valid, p['V'], 0.0)
    vmax = V.max(axis=1, keepdims=True)
    V = np.where(vmax > 1e-10, V / np.where(vmax > 1e-10, vmax, 1.0), 0.0)
    feats = np.stack([O, H, L, C_fill, MA, V, valid.astype(np.float64)], axis=2)
    return feats.astype(np.float32)


def ma_out_of_range_share(p: dict) -> float:
    lo, hi, valid = price_bounds(p['O'], p['H'], p['L'], p['C'])
    ma = p['MA']
    out = valid & np.isfinite(ma) & ((ma < lo[:, None]) | (ma > hi[:, None]))
    return float(out.sum() / max(valid.sum(), 1))
