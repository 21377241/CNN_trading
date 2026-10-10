# -*- coding: utf-8 -*-
"""
Data/chart_dataset.py - PyTorch 数据集

EquityDataset:  2D CNN 训练/推理用数据集（图像）
TS1DDataset:    1D CNN 训练/推理用数据集（时间序列）
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Data.dgp_config import (
    IMAGE_WIDTH, IMAGE_HEIGHT,
    get_dataset_filename, get_dataset_dir
)
from Misc.config import COUNTRY


class EquityDataset(Dataset):
    """
    2D CNN 图像数据集。

    从 memmap .dat 文件读取图像，从 .feather 文件读取标签。

    Args:
        years: 年份列表，如 [2023, 2024]
        window_size: 窗口大小，如 20
        predict_window: 预测窗口（用于确定标签列名），如 20
        freq: 频率字符串，如 'month'
        has_volume_bar: 是否含成交量柱
        ma_lags: 均线周期列表
        country: 国家标识
        image_scale: 图像归一化方式 'norm255'(除255) | 'standardize' | 'none'
        label_col: 标签列名，默认 'binary_label'
    """

    def __init__(self,
                 years: list,
                 window_size: int = 20,
                 predict_window: int = 20,
                 freq: str = 'month',
                 has_volume_bar: bool = True,
                 ma_lags: list = None,
                 country: str = COUNTRY,
                 image_scale: str = 'norm255',
                 label_col: str = 'binary_label',
                 dataset_dir: str = None):
        """
        Args:
            dataset_dir: 图表数据集目录（可选）。
                         默认由 get_dataset_dir(country) 推断（SPX路径）。
                         CSI800 请显式传入 get_dataset_dir(CSI800_COUNTRY,
                         base_savepath=CSI800_STOCKS_SAVEPATH)。
        """
        self.years = years
        self.window_size = window_size
        self.predict_window = predict_window
        self.freq = freq
        self.has_volume_bar = has_volume_bar
        self.ma_lags = ma_lags if ma_lags is not None else [20]
        self.country = country
        self.image_scale = image_scale
        self.label_col = label_col

        self.img_w = IMAGE_WIDTH.get(window_size, window_size * 3)
        self.img_h = IMAGE_HEIGHT.get(window_size, 64)
        self.img_size = self.img_w * self.img_h

        self.dataset_dir = dataset_dir if dataset_dir is not None else get_dataset_dir(country)

        # 加载所有年份的数据
        self._images: np.ndarray = None   # (N, H, W) float32
        self._labels: np.ndarray = None   # (N,) int64
        self._meta: pd.DataFrame = None   # 元数据
        self._load_all()

    def _load_all(self):
        """加载并合并所有年份的数据"""
        img_list = []
        meta_list = []

        for year in self.years:
            fn_images = get_dataset_filename(self.window_size, self.freq,
                                             self.has_volume_bar, self.ma_lags,
                                             year, '_images.dat')
            fn_labels = get_dataset_filename(self.window_size, self.freq,
                                             self.has_volume_bar, self.ma_lags,
                                             year, '_labels.feather')

            p_img = os.path.join(self.dataset_dir, fn_images)
            p_lbl = os.path.join(self.dataset_dir, fn_labels)

            if not os.path.exists(p_img) or not os.path.exists(p_lbl):
                print(f"[EquityDataset] 警告: 找不到 {year} 年数据文件，跳过。")
                continue

            meta = pd.read_feather(p_lbl)
            N = len(meta)
            if N == 0:
                continue

            # 读取 memmap
            mmap = np.memmap(p_img, dtype='uint8', mode='r', shape=(N, self.img_size))
            imgs = np.array(mmap, dtype=np.float32)  # copy to RAM
            del mmap

            img_list.append(imgs)
            meta_list.append(meta)

        if not img_list:
            raise RuntimeError(f"未找到任何有效数据，年份: {self.years}")

        imgs_all = np.concatenate(img_list, axis=0)  # (N, H*W)
        meta_all = pd.concat(meta_list, ignore_index=True)

        # 过滤无效样本：label=-99 或 EWMA_vol=0
        valid_mask = (meta_all[self.label_col] != -99)
        if 'EWMA_vol' in meta_all.columns:
            valid_mask &= (meta_all['EWMA_vol'] > 0)

        imgs_all = imgs_all[valid_mask.values]
        meta_all = meta_all[valid_mask].reset_index(drop=True)

        # 图像归一化
        if self.image_scale == 'norm255':
            imgs_all /= 255.0
        elif self.image_scale == 'standardize':
            mean = imgs_all.mean()
            std = imgs_all.std() + 1e-8
            imgs_all = (imgs_all - mean) / std

        # reshape 为 (N, 1, H, W)
        self._images = imgs_all.reshape(-1, 1, self.img_h, self.img_w)
        self._labels = meta_all[self.label_col].values.astype(np.int64)
        self._meta = meta_all

        print(f"[EquityDataset] 年份 {self.years}: 共 {len(self._labels)} 条有效样本，"
              f"正例 {(self._labels == 1).sum()}，负例 {(self._labels == 0).sum()}")

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        img = torch.tensor(self._images[idx], dtype=torch.float32)
        lbl = torch.tensor(self._labels[idx], dtype=torch.long)
        return img, lbl

    @property
    def meta(self) -> pd.DataFrame:
        """返回元数据 DataFrame"""
        return self._meta


class TS1DDataset(Dataset):
    """
    1D CNN 时间序列数据集。

    从 .npz 文件读取时间序列特征与标签。

    Args:
        years: 年份列表
        window_size: 时间序列长度
        freq: 频率
        has_volume_bar: 是否包含成交量（影响文件名）
        ma_lags: 均线周期
        country: 国家标识
        ts_scale: 特征归一化方式 'none' | 'standardize'
    """

    def __init__(self,
                 years: list,
                 window_size: int = 20,
                 freq: str = 'month',
                 has_volume_bar: bool = True,
                 ma_lags: list = None,
                 country: str = COUNTRY,
                 ts_scale: str = 'none'):

        self.years = years
        self.window_size = window_size
        self.freq = freq
        self.has_volume_bar = has_volume_bar
        self.ma_lags = ma_lags if ma_lags is not None else [20]
        self.country = country
        self.ts_scale = ts_scale

        self.dataset_dir = get_dataset_dir(country)

        self._ts: np.ndarray = None      # (N, 6, T)
        self._labels: np.ndarray = None  # (N,)
        self._load_all()

    def _load_all(self):
        ts_list = []
        lbl_list = []

        for year in self.years:
            fn_ts = get_dataset_filename(self.window_size, self.freq,
                                         self.has_volume_bar, self.ma_lags,
                                         year, '_ts_data.npz')
            p_ts = os.path.join(self.dataset_dir, fn_ts)
            if not os.path.exists(p_ts):
                print(f"[TS1DDataset] 警告: 找不到 {year} 年 TS 数据文件，跳过。")
                continue
            data = np.load(p_ts)
            ts_list.append(data['ts'])
            lbl_list.append(data['labels'])

        if not ts_list:
            raise RuntimeError(f"未找到任何 TS 数据，年份: {self.years}")

        ts_all  = np.concatenate(ts_list, axis=0).astype(np.float32)
        lbl_all = np.concatenate(lbl_list, axis=0).astype(np.int64)

        if self.ts_scale == 'standardize':
            mean = ts_all.mean(axis=(0, 2), keepdims=True)
            std  = ts_all.std(axis=(0, 2), keepdims=True) + 1e-8
            ts_all = (ts_all - mean) / std

        self._ts = ts_all
        self._labels = lbl_all

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        ts  = torch.tensor(self._ts[idx], dtype=torch.float32)
        lbl = torch.tensor(self._labels[idx], dtype=torch.long)
        return ts, lbl

