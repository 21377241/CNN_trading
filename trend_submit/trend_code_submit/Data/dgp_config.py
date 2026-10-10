# -*- coding: utf-8 -*-
"""
Data/dgp_config.py - 图表生成配置

定义图像尺寸、K 线样式、颜色等参数。
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Misc.config import (
    WORK_DIR, PROCESSED_DATA_DIR, STOCKS_SAVEPATH, CACHE_DIR, COUNTRY
)

# ──────────────────────────────────────────────
# 目录派生
# ──────────────────────────────────────────────
DATA_DIR = os.path.join(WORK_DIR, 'data')

# ──────────────────────────────────────────────
# K 线像素参数
# ──────────────────────────────────────────────
BAR_WIDTH = 3          # 每根 K 线宽度（像素），含主体 + 空白

# 图像宽度 = window_size * BAR_WIDTH
# 图像高度（含成交量区域）按窗口大小设置：
#   key = window_size (天数)
IMAGE_WIDTH = {
    5:  15,    # 5日  → 15px
    20: 60,    # 20日 → 60px
    60: 180,   # 60日 → 180px
}
IMAGE_HEIGHT = {
    5:  32,    # 5日  → 32px
    20: 64,    # 20日 → 64px
    60: 128,   # 60日 → 128px
}

# 成交量区域参数
VOLUME_CHART_GAP = 2          # 价格区与成交量区之间的间隔（像素）
VOLUME_CHART_HEIGHT_RATIO = 0.2   # 成交量区占总高度的比例

# ──────────────────────────────────────────────
# 颜色（灰度图）
# ──────────────────────────────────────────────
BACKGROUND_COLOR = 0    # 背景：黑色
CHART_COLOR = 255       # K 线、均线：白色

# ──────────────────────────────────────────────
# 频率映射
# ──────────────────────────────────────────────
FREQ_DICT = {
    5:  'week',
    20: 'month',
    60: 'quarter',
    65: 'quarter',
}

# ──────────────────────────────────────────────
# 数据文件名格式（与步骤 2 命名保持一致）
# ──────────────────────────────────────────────
# 示例: 20d_month_has_vb_[20]_ma_2024_images.dat
def get_dataset_filename(window_size: int, freq: str, has_vb: bool,
                         ma_lags: list, year: int, suffix: str) -> str:
    """
    生成数据集文件名（不含目录前缀）。

    Args:
        window_size: 窗口大小（天数），如 20
        freq: 频率字符串，如 'month'
        has_vb: 是否含成交量柱
        ma_lags: 均线周期列表，如 [20]
        year: 年份
        suffix: 文件后缀，如 '_images.dat', '_labels.feather', '.txt'

    Returns:
        文件名字符串
    """
    vb_str = 'has_vb' if has_vb else 'no_vb'
    ma_str = str(ma_lags).replace(' ', '')  # [20]
    return f"{window_size}d_{freq}_{vb_str}_{ma_str}_ma_{year}{suffix}"


def get_dataset_dir(country: str = COUNTRY, base_savepath: str = None) -> str:
    """
    返回数据集保存目录。

    Args:
        country: 国家标识，如 'USA' 或 'CHN'
        base_savepath: 根保存路径，默认使用 config.STOCKS_SAVEPATH（SPX路径）。
                       CSI800 请传入 config.CSI800_STOCKS_SAVEPATH。
    """
    savepath = base_savepath if base_savepath is not None else STOCKS_SAVEPATH
    return os.path.join(savepath, f'stocks_{country}', 'dataset_all')

