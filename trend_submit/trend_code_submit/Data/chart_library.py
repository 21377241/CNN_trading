# -*- coding: utf-8 -*-
"""
Data/chart_library.py - 图表绘制库

DrawOHLC: 将归一化后的 OHLCV + 均线数据绘制成灰度 PIL Image。
支持 bar（K 线柱状）/ pixel / centered_pixel 三种图表类型。
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pandas as pd

from Data.dgp_config import (
    BAR_WIDTH, IMAGE_WIDTH, IMAGE_HEIGHT,
    VOLUME_CHART_GAP, VOLUME_CHART_HEIGHT_RATIO,
    BACKGROUND_COLOR, CHART_COLOR
)


class DrawOHLC:
    """
    将归一化后的 OHLCV 数据绘制为灰度图像。

    Args:
        ohlcv_df: DataFrame，含列 ['Open','High','Low','Close'] 及可选 ['Vol']，
                  行索引为时间序列（T 行 = window_size）
        window_size: K 线数量（决定图像宽度）
        ma_cols: 均线列名列表，如 ['ma20']；对应 ohlcv_df 的列
        has_volume_bar: 是否在图像底部绘制成交量柱
        chart_type: 'bar' | 'pixel' | 'centered_pixel'
        img_width: 覆盖默认图像宽度（像素）
        img_height: 覆盖默认图像高度（像素）
    """

    def __init__(self,
                 ohlcv_df: pd.DataFrame,
                 window_size: int = 20,
                 ma_cols: list = None,
                 has_volume_bar: bool = True,
                 chart_type: str = 'bar',
                 img_width: int = None,
                 img_height: int = None):
        self.df = ohlcv_df.reset_index(drop=True)
        self.T = len(self.df)
        self.window_size = window_size
        self.ma_cols = ma_cols or []
        self.has_volume_bar = has_volume_bar and ('Vol' in self.df.columns)
        self.chart_type = chart_type

        self.img_width = img_width or IMAGE_WIDTH.get(window_size, window_size * BAR_WIDTH)
        self.img_height = img_height or IMAGE_HEIGHT.get(window_size, 64)

        # 计算价格区与成交量区的像素分配
        if self.has_volume_bar:
            vol_h = max(4, int(self.img_height * VOLUME_CHART_HEIGHT_RATIO))
            gap = VOLUME_CHART_GAP
            self.price_h = self.img_height - vol_h - gap
            self.vol_h = vol_h
            self.vol_gap = gap
        else:
            self.price_h = self.img_height
            self.vol_h = 0
            self.vol_gap = 0

    # ─────────────────────────────────────────────
    # 公开绘制方法
    # ─────────────────────────────────────────────
    def draw(self) -> Image.Image:
        """绘制并返回 PIL Image (mode='L', 尺寸 img_width × img_height)"""
        img = Image.new('L', (self.img_width, self.img_height), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)

        ohlc = self._get_ohlc_arrays()
        if ohlc is None:
            return img

        open_arr, high_arr, low_arr, close_arr = ohlc

        # 价格轴映射（归一化到像素）
        all_prices = np.concatenate([open_arr, high_arr, low_arr, close_arr])
        valid = all_prices[~np.isnan(all_prices)]
        if len(valid) == 0:
            return img

        price_min = np.min(valid)
        price_max = np.max(valid)
        price_range = price_max - price_min
        if price_range < 1e-10:
            price_range = 1e-10

        def price_to_y(p):
            """价格→像素 y 坐标（上方=高价，下方=低价）"""
            norm = (p - price_min) / price_range
            # y=0 在顶部，y=price_h-1 在底部
            return int((1.0 - norm) * (self.price_h - 1))

        if self.chart_type == 'bar':
            self._draw_bar_chart(draw, price_to_y, open_arr, high_arr, low_arr, close_arr)
        elif self.chart_type == 'pixel':
            self._draw_pixel_chart(draw, price_to_y, close_arr)
        elif self.chart_type == 'centered_pixel':
            self._draw_centered_pixel_chart(draw, price_to_y, open_arr, close_arr)

        # 均线
        for col in self.ma_cols:
            if col in self.df.columns:
                ma_arr = self.df[col].values.astype(float)
                self._draw_ma_line(draw, price_to_y, ma_arr)

        # 成交量
        if self.has_volume_bar:
            self._draw_volume_bars(draw)

        return img

    def draw_to_array(self) -> np.ndarray:
        """绘制并返回 uint8 一维 numpy 数组（按行展开）"""
        img = self.draw()
        return np.array(img, dtype=np.uint8).flatten()

    # ─────────────────────────────────────────────
    # 内部绘制方法
    # ─────────────────────────────────────────────
    def _get_ohlc_arrays(self):
        """提取 OHLC 数组，做基本校验"""
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in self.df.columns:
                return None

        o = self.df['Open'].values.astype(float)
        h = self.df['High'].values.astype(float)
        l = self.df['Low'].values.astype(float)
        c = self.df['Close'].values.astype(float)
        return o, h, l, c

    def _bar_x(self, t: int):
        """计算第 t 根 K 线的像素 x 中心位置"""
        bar_pixel_w = self.img_width / max(self.T, 1)
        return int((t + 0.5) * bar_pixel_w)

    def _bar_x_range(self, t: int):
        """返回第 t 根 K 线主体的 (x_left, x_right)"""
        bar_pixel_w = self.img_width / max(self.T, 1)
        body_w = max(1, int(bar_pixel_w * 0.6))
        cx = self._bar_x(t)
        x0 = max(0, cx - body_w // 2)
        x1 = min(self.img_width - 1, cx + body_w // 2)
        return x0, x1

    def _draw_bar_chart(self, draw, price_to_y, o, h, l, c):
        """绘制 K 线柱（OHLC）"""
        for t in range(self.T):
            if any(np.isnan(v) for v in [o[t], h[t], l[t], c[t]]):
                continue
            cx = self._bar_x(t)
            x0, x1 = self._bar_x_range(t)

            y_high = price_to_y(h[t])
            y_low = price_to_y(l[t])
            y_open = price_to_y(o[t])
            y_close = price_to_y(c[t])

            # High-Low 竖线
            draw.line([(cx, y_high), (cx, y_low)], fill=CHART_COLOR, width=1)

            # 主体矩形（Open-Close）
            y_top = min(y_open, y_close)
            y_bot = max(y_open, y_close)
            if y_top == y_bot:
                y_bot = y_top + 1
            draw.rectangle([x0, y_top, x1, y_bot], fill=CHART_COLOR)

    def _draw_pixel_chart(self, draw, price_to_y, c):
        """像素图：每根 bar 只画收盘价一个像素点"""
        for t in range(self.T):
            if np.isnan(c[t]):
                continue
            cx = self._bar_x(t)
            y = price_to_y(c[t])
            draw.point([(cx, y)], fill=CHART_COLOR)

    def _draw_centered_pixel_chart(self, draw, price_to_y, o, c):
        """居中像素图：画 open-close 垂线"""
        for t in range(self.T):
            if any(np.isnan(v) for v in [o[t], c[t]]):
                continue
            cx = self._bar_x(t)
            y0 = price_to_y(o[t])
            y1 = price_to_y(c[t])
            draw.line([(cx, y0), (cx, y1)], fill=CHART_COLOR, width=1)

    def _draw_ma_line(self, draw, price_to_y, ma_arr):
        """绘制均线折线"""
        points = []
        for t in range(self.T):
            if not np.isnan(ma_arr[t]):
                cx = self._bar_x(t)
                y = price_to_y(ma_arr[t])
                points.append((cx, y))
        if len(points) >= 2:
            draw.line(points, fill=CHART_COLOR, width=1)

    def _draw_volume_bars(self, draw):
        """在图像底部绘制成交量柱"""
        vol_arr = self.df['Vol'].values.astype(float)
        valid_vol = vol_arr[~np.isnan(vol_arr)]
        if len(valid_vol) == 0:
            return

        vol_max = np.max(valid_vol)
        if vol_max < 1e-10:
            return

        # 成交量区 y 坐标范围（图像下方）
        vol_top = self.price_h + self.vol_gap
        vol_bottom = self.img_height - 1

        for t in range(self.T):
            if np.isnan(vol_arr[t]):
                continue
            ratio = vol_arr[t] / vol_max
            bar_h = max(1, int(ratio * (vol_bottom - vol_top)))
            x0, x1 = self._bar_x_range(t)
            y_top = vol_bottom - bar_h
            y_bot = vol_bottom
            draw.rectangle([x0, y_top, x1, y_bot], fill=CHART_COLOR)

