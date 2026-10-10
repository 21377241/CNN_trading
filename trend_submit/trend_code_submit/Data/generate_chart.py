# -*- coding: utf-8 -*-
"""
Data/generate_chart.py - 图表数据生成

GenerateStockData:
  每年、每只股票、每个周期末日，生成：
    - 归一化 K 线 + 量 + 均线的灰度图像
    - 对应的标签与元数据

输出（按年）：
  {DATASET_DIR}/{window_size}d_{freq}_has_vb_[20]_ma_{year}_images.dat
  {DATASET_DIR}/{window_size}d_{freq}_has_vb_[20]_ma_{year}_labels.feather
  {DATASET_DIR}/{window_size}d_{freq}_has_vb_[20]_ma_{year}.txt
"""
import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Data.dgp_config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, FREQ_DICT,
    get_dataset_filename, get_dataset_dir
)
from Data.chart_library import DrawOHLC
import Data.equity_data as equity_data

logger = logging.getLogger(__name__)


class GenerateStockData:
    """
    为指定年份生成图表图像数据集。

    Args:
        country: 国家标识，用于目录命名
        year: 目标年份（仅取该年的周期末日）
        window_size: 历史窗口（天数），如 20
        freq: 预测频率，如 'month'
        chart_freq: 图表频率（通常为 1，即每个周期末都生成）
        ma_lags: 均线周期列表，如 [20]
        volume_bar: 是否含成交量柱
        need_adjust_price: 是否做价格归一化
        chart_type: 'bar' | 'pixel' | 'centered_pixel'
        ret_col: 用作标签的收益列名，如 'Ret_month'
    """

    def __init__(self,
                 country: str = 'USA',
                 year: int = 2024,
                 window_size: int = 20,
                 freq: str = 'month',
                 chart_freq: int = 1,
                 ma_lags: list = None,
                 volume_bar: bool = True,
                 need_adjust_price: bool = True,
                 chart_type: str = 'bar',
                 ret_col: str = None,
                 dataset_dir: str = None):

        self.country = country
        self.year = year
        self.window_size = window_size
        self.freq = freq
        self.chart_freq = chart_freq
        self.ma_lags = ma_lags if ma_lags is not None else [20]
        self.volume_bar = volume_bar
        self.need_adjust_price = need_adjust_price
        self.chart_type = chart_type

        # 根据 FREQ_DICT 确定 ret_col
        if ret_col is None:
            self.ret_col = f'Ret_{freq}'
        else:
            self.ret_col = ret_col

        # 图像尺寸
        self.img_w = IMAGE_WIDTH.get(window_size, window_size * 3)
        self.img_h = IMAGE_HEIGHT.get(window_size, 64)
        self.img_size = self.img_w * self.img_h

        # 输出目录（可由外部传入以支持不同数据集路径）
        self.dataset_dir = dataset_dir if dataset_dir is not None else get_dataset_dir(country)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # 加载数据
        self._data: pd.DataFrame = None
        self._period_ends: pd.DatetimeIndex = None

    # ─────────────────────────────────────────────
    # 数据加载
    # ─────────────────────────────────────────────
    def _load_data(self):
        """懒加载：按需读取当年 + 前一年的面板数据"""
        if self._data is None:
            self._data = equity_data.get_processed_US_data_by_year(
                self.year, lookback_years=1
            )
            logger.info(f"已加载 {self.year} 年数据，行数: {len(self._data)}")

    def _load_period_ends(self):
        """懒加载：计算当年的周期末日期"""
        if self._period_ends is None:
            all_ends = equity_data.get_period_end_dates(self.freq)
            self._period_ends = all_ends[all_ends.year == self.year]
            logger.info(f"{self.year} 年 {self.freq} 期末日数量: {len(self._period_ends)}")

    # ─────────────────────────────────────────────
    # 价格归一化
    # ─────────────────────────────────────────────
    @staticmethod
    def adjust_price(window_df: pd.DataFrame) -> pd.DataFrame:
        """
        归一化价格：窗口第一天 Close=1，后续按 Ret 递推，
        Open/High/Low 按当日 Close 与原始 Close 的比例缩放。

        Args:
            window_df: 含 Open, High, Low, Close, Ret 列的 DataFrame（T 行）

        Returns:
            归一化后的 DataFrame（原地修改副本）
        """
        df = window_df.copy()
        T = len(df)
        if T == 0:
            return df

        close_orig = df['Close'].values.astype(float)
        open_orig  = df['Open'].values.astype(float)
        high_orig  = df['High'].values.astype(float)
        low_orig   = df['Low'].values.astype(float)
        ret_arr    = df['Ret'].values.astype(float)

        # 第一天 Close 设为 1
        close_adj = np.zeros(T, dtype=float)
        close_adj[0] = 1.0

        for t in range(1, T):
            close_adj[t] = close_adj[t - 1] * (1.0 + ret_arr[t])

        # 按比例缩放 O/H/L
        ratio = np.where(np.abs(close_orig) > 1e-10,
                         close_adj / close_orig,
                         1.0)

        df['Close'] = close_adj
        df['Open']  = open_orig  * ratio
        df['High']  = high_orig  * ratio
        df['Low']   = low_orig   * ratio

        # 保证 High >= Close/Open >= Low
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low']  = df[['Low',  'Open', 'Close']].min(axis=1)

        return df

    # ─────────────────────────────────────────────
    # 特征生成（添加均线）
    # ─────────────────────────────────────────────
    def _add_ma_features(self, window_df: pd.DataFrame) -> pd.DataFrame:
        """在 window_df 上计算并附加均线列"""
        df = window_df.copy()
        for lag in self.ma_lags:
            col = f'ma{lag}'
            df[col] = df['Close'].rolling(lag, min_periods=1).mean()
        return df

    # ─────────────────────────────────────────────
    # 主要生成逻辑
    # ─────────────────────────────────────────────
    def save_annual_data(self):
        """
        生成并保存当年的图像数据集。

        输出：
          - *_images.dat   (N, H*W) uint8 memmap
          - *_labels.feather
          - *.txt 日志
        """
        self._load_data()
        self._load_period_ends()

        if len(self._period_ends) == 0:
            logger.warning(f"{self.year} 年无有效周期末日期，跳过。")
            return

        # 文件名（不含目录）
        has_ma = len(self.ma_lags) > 0
        fn_images  = get_dataset_filename(self.window_size, self.freq, self.volume_bar,
                                          self.ma_lags, self.year, '_images.dat')
        fn_labels  = get_dataset_filename(self.window_size, self.freq, self.volume_bar,
                                          self.ma_lags, self.year, '_labels.feather')
        fn_log     = get_dataset_filename(self.window_size, self.freq, self.volume_bar,
                                          self.ma_lags, self.year, '.txt')

        path_images = os.path.join(self.dataset_dir, fn_images)
        path_labels = os.path.join(self.dataset_dir, fn_labels)
        path_log    = os.path.join(self.dataset_dir, fn_log)

        stock_ids = self._data['StockID'].unique()
        logger.info(f"{self.year} 年共 {len(stock_ids)} 只股票，"
                    f"{len(self._period_ends)} 个期末日")

        all_images = []     # 每条样本的图像 uint8 数组
        all_labels = []     # 元数据字典列表

        # 统计
        cnt_total = 0
        cnt_missing_window = 0
        cnt_missing_ret = 0
        cnt_missing_ewma = 0

        for sid in stock_ids:
            sdf = self._data[self._data['StockID'] == sid].sort_values('Date').reset_index(drop=True)

            for end_dt in self._period_ends:
                # 取 end_dt 当日或之前最近的交易日
                available = sdf[sdf['Date'] <= end_dt]
                if len(available) < self.window_size:
                    cnt_missing_window += 1
                    continue

                window = available.tail(self.window_size).copy()

                # 检查收益列
                if self.ret_col not in window.columns:
                    cnt_missing_ret += 1
                    continue

                # 取期末日的标签（最后一行）
                last_row = window.iloc[-1]
                ret_label = last_row.get(self.ret_col, np.nan)

                # 检查 EWMA_vol
                ewma_val = last_row.get('EWMA_vol', np.nan)
                if pd.isna(ewma_val) or ewma_val == 0:
                    cnt_missing_ewma += 1
                    continue

                # 价格归一化
                if self.need_adjust_price:
                    window = self.adjust_price(window)

                # 添加均线
                ma_col_names = [f'ma{lag}' for lag in self.ma_lags]
                window = self._add_ma_features(window)

                # 绘图
                try:
                    drawer = DrawOHLC(
                        ohlcv_df=window,
                        window_size=self.window_size,
                        ma_cols=ma_col_names if has_ma else [],
                        has_volume_bar=self.volume_bar,
                        chart_type=self.chart_type,
                        img_width=self.img_w,
                        img_height=self.img_h
                    )
                    img_arr = drawer.draw_to_array()
                except Exception as e:
                    logger.debug(f"绘图失败 {sid} {end_dt}: {e}")
                    continue

                # 二值标签：Ret > 0 → 1，否则 0
                if pd.isna(ret_label):
                    binary_label = -99
                else:
                    binary_label = 1 if ret_label > 0 else 0

                all_images.append(img_arr)
                all_labels.append({
                    'StockID':       sid,
                    'ending_date':   end_dt,
                    'ret_val':       float(last_row.get(self.ret_col, np.nan)),
                    'Ret':           float(last_row.get('Ret', np.nan)),
                    'Ret_5d':        float(last_row.get('Ret_5d', np.nan)),
                    'Ret_20d':       float(last_row.get('Ret_20d', np.nan)),
                    'Ret_month':     float(last_row.get('Ret_month', np.nan)),
                    'MarketCap':     float(last_row.get('MarketCap', np.nan)),
                    'EWMA_vol':      float(ewma_val),
                    'binary_label':  binary_label,
                })
                cnt_total += 1

        N = len(all_images)
        logger.info(f"{self.year} 年生成样本数: {N}，"
                    f"缺窗口: {cnt_missing_window}，"
                    f"缺标签: {cnt_missing_ret}，"
                    f"缺EWMA: {cnt_missing_ewma}")

        if N == 0:
            logger.warning(f"{self.year} 年无有效样本，不写出文件。")
            return

        # ── 保存图像（memmap）──
        img_memmap = np.memmap(path_images, dtype='uint8',
                               mode='w+', shape=(N, self.img_size))
        for i, arr in enumerate(all_images):
            img_memmap[i] = arr
        del img_memmap   # flush

        # ── 保存标签 ──
        labels_df = pd.DataFrame(all_labels)
        labels_df['ending_date'] = pd.to_datetime(labels_df['ending_date'])
        labels_df.reset_index(drop=True).to_feather(path_labels)

        # ── 写日志 ──
        with open(path_log, 'w', encoding='utf-8') as f:
            f.write(f"Year:          {self.year}\n")
            f.write(f"Window:        {self.window_size}\n")
            f.write(f"Freq:          {self.freq}\n")
            f.write(f"Has volume:    {self.volume_bar}\n")
            f.write(f"MA lags:       {self.ma_lags}\n")
            f.write(f"Total samples: {N}\n")
            f.write(f"Missing window:{cnt_missing_window}\n")
            f.write(f"Missing ret:   {cnt_missing_ret}\n")
            f.write(f"Missing EWMA:  {cnt_missing_ewma}\n")

        logger.info(f"已保存: {path_images}")
        logger.info(f"已保存: {path_labels}")

    # ─────────────────────────────────────────────
    # 1D 时间序列数据（可选）
    # ─────────────────────────────────────────────
    def save_annual_ts_data(self):
        """
        生成并保存 1D 时间序列数据。

        特征维度：[Close, Open, High, Low, Vol, ma20] → 6 维
        形状：(N, 6, window_size)
        输出：*_ts_data.npz
        """
        self._load_data()
        self._load_period_ends()

        if len(self._period_ends) == 0:
            return

        fn_ts = get_dataset_filename(self.window_size, self.freq, self.volume_bar,
                                     self.ma_lags, self.year, '_ts_data.npz')
        path_ts = os.path.join(self.dataset_dir, fn_ts)

        stock_ids = self._data['StockID'].unique()
        ts_list = []
        label_list = []

        for sid in stock_ids:
            sdf = self._data[self._data['StockID'] == sid].sort_values('Date').reset_index(drop=True)
            for end_dt in self._period_ends:
                available = sdf[sdf['Date'] <= end_dt]
                if len(available) < self.window_size:
                    continue
                window = available.tail(self.window_size).copy()
                window = self._add_ma_features(window)

                if self.need_adjust_price:
                    window = self.adjust_price(window)

                last_row = window.iloc[-1]
                ret_label = last_row.get(self.ret_col, np.nan)
                if pd.isna(ret_label):
                    continue

                # 6 维特征
                feats = []
                for col in ['Close', 'Open', 'High', 'Low', 'Vol'] + [f'ma{l}' for l in self.ma_lags]:
                    if col in window.columns:
                        feats.append(window[col].fillna(0).values)
                    else:
                        feats.append(np.zeros(self.window_size))

                # 取前 6 维
                feat_arr = np.array(feats[:6], dtype=np.float32)  # (6, T)
                if feat_arr.shape[0] < 6:
                    pad = np.zeros((6 - feat_arr.shape[0], self.window_size), dtype=np.float32)
                    feat_arr = np.vstack([feat_arr, pad])

                ts_list.append(feat_arr)
                label_list.append(1 if ret_label > 0 else 0)

        if ts_list:
            ts_arr = np.stack(ts_list, axis=0)          # (N, 6, T)
            label_arr = np.array(label_list, dtype=np.int64)
            np.savez(path_ts, ts=ts_arr, labels=label_arr)
            logger.info(f"1D TS 数据已保存: {path_ts}")

