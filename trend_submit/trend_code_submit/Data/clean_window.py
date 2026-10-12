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
META_COLS = ['StockID', 'ending_date', 'ret_val', 'binary_label', 'label_raw', 'MarketCap', 'split', 'embargo',
             'n_obs_in_win', 'board', 'is_bj', 'next_date', 'year']


def universe_meta(u: pd.DataFrame) -> pd.DataFrame:
    """universe 行 → 图像 labels / 特征 meta。binary_label 在无标签或 embargo 行为 -99。"""
    m = u.rename(columns={'ret_label': 'ret_val', 'label': 'label_raw'}).copy()
    m['StockID'] = m['StockID'].astype(str)
    bl = m['label_raw'].astype(np.int64)
    m['binary_label'] = np.where((bl < 0) | (m['embargo'] == 1), -99, bl).astype(np.int64)
    m['split'] = m['split'].astype(str)
    m['board'] = m['board'].astype(str)
    return m[META_COLS].reset_index(drop=True)


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
    MA = dense('MA20', np.float64)
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
    """窗口内第一个有效收盘价归一为 1；O/H/L/MA 同比例缩放；修正 H/L 包住 O/C。缺失日 O/H/L/C/MA/V 为 NaN。

    缺失日的均线为 NaN：Jiang 官方代码按股票自己的交易行取窗口，停牌日没有这一行，也就没有均线值。"""
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
    MA = np.where(valid, MA, np.nan)
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
    """返回 float32 (N, 20, 7)。展平为 140 维时按天优先（reshape(N, -1)）。

    价格（含 MA20）与图像纵轴同一定标：O/H/L/C 与均线一起取最小、最大值，缩放到 [0, 1]。
    无交易日的 O/H/L/C/MA 都取占位收盘价、V 为 0、有交易标记为 0：图像在这些日子是空白列。
    均线不可用的交易日（有效收盘价不足 20 个）MA 也取占位收盘价。"""
    valid = p['valid']
    lo, hi = price_bounds(p['O'], p['H'], p['L'], p['C'], p['MA'])
    if not (hi > lo).all():
        raise ValueError(f'{int((~(hi > lo)).sum())} 个窗口价格区间为零，应已由 universe 剔除')
    rng = hi - lo

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


def ma_sets_bound_share(p: dict) -> float:
    """均线高于全部 OHLC 或低于全部 OHLC（即均线决定纵轴上界或下界）的窗口占比。"""
    lo, hi = price_bounds(p['O'], p['H'], p['L'], p['C'])
    ma = p['MA']
    with np.errstate(invalid='ignore'):
        above = np.nanmax(np.where(np.isfinite(ma), ma, -np.inf), axis=1) > hi
        below = np.nanmin(np.where(np.isfinite(ma), ma, np.inf), axis=1) < lo
    return float((above | below).mean())
