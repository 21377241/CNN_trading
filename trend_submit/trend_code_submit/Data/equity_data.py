# -*- coding: utf-8 -*-
"""
Data/equity_data.py - 数据加载层

提供以下接口：
  - processed_US_data()         → 全部已处理的面板 DataFrame
  - get_processed_US_data_by_year(year) → 某年及之前若干年的数据（供图表生成用）
  - get_period_end_dates(period) → 周期末日期列表

本模块设计为可 Monkey Patch：
  步骤 2 入口脚本会将这些函数替换为从 us_ret_spx.feather 读取的版本。
"""
import os
import sys
import numpy as np
import pandas as pd

# 确保 Misc 可导入
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Misc.config import PROCESSED_DATA_DIR, CACHE_DIR
from Misc.utilities import get_period_end_dates_from_dates

# ─────────────────────────────────────────────
# 全局缓存
# ─────────────────────────────────────────────
_cached_df: pd.DataFrame = None
_FEATHER_PATH = os.path.join(PROCESSED_DATA_DIR, 'us_ret_spx.feather')


def _load_feather() -> pd.DataFrame:
    """加载并缓存 feather 文件"""
    global _cached_df
    if _cached_df is None:
        if not os.path.exists(_FEATHER_PATH):
            raise FileNotFoundError(
                f"找不到预处理数据文件: {_FEATHER_PATH}\n"
                "请先运行 1_preprocess_spx_data.py 生成该文件。"
            )
        _cached_df = pd.read_feather(_FEATHER_PATH)
        # 确保 Date 为 datetime
        _cached_df['Date'] = pd.to_datetime(_cached_df['Date'])
    return _cached_df


# ─────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────
def processed_US_data() -> pd.DataFrame:
    """
    返回全部已处理的面板 DataFrame，索引为 (Date, StockID)。

    列包含：Date, StockID, Open, High, Low, Close, Vol,
            Ret, MarketCap, EWMA_vol, log_ret, cum_log_ret,
            Ret_5d, Ret_20d, Ret_60d, ... ,
            Ret_week, Ret_month, Ret_quarter, Ret_year,
            Shares
    """
    df = _load_feather().copy()
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['Date', 'StockID'])
    return df


def get_processed_US_data_by_year(year: int, lookback_years: int = 1) -> pd.DataFrame:
    """
    返回 [year - lookback_years, year] 范围内的数据，重置索引。

    Args:
        year: 目标年份
        lookback_years: 额外向前取几年数据（用于 MA 等 lookback）

    Returns:
        DataFrame，含 Date, StockID 等列，按 (StockID, Date) 排序
    """
    df = _load_feather()
    start_year = year - lookback_years
    mask = (df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= year)
    return df[mask].copy().sort_values(['StockID', 'Date']).reset_index(drop=True)


def get_period_end_dates(period: str) -> pd.DatetimeIndex:
    """
    从已处理数据的交易日历推算周期末日期。

    优先查找 CACHE_DIR/spy_{period}_ret.csv 中的 date 列；
    若无，则从 feather 的 Date 自算。

    Args:
        period: 'week' | 'month' | 'quarter' | 'year'

    Returns:
        DatetimeIndex
    """
    spy_file = os.path.join(CACHE_DIR, f'spy_{period}_ret.csv')
    if os.path.exists(spy_file):
        try:
            spy = pd.read_csv(spy_file)
            date_col = 'date' if 'date' in spy.columns else spy.columns[0]
            return pd.DatetimeIndex(pd.to_datetime(spy[date_col]).sort_values().unique())
        except Exception:
            pass

    # 备用：从 feather 的交易日历推算
    df = _load_feather()
    return get_period_end_dates_from_dates(df['Date'], period)


def get_period_end_dates_for_year(year: int, period: str) -> pd.DatetimeIndex:
    """
    返回指定年份的周期末日期序列。

    Args:
        year: 目标年份
        period: 'week' | 'month' | 'quarter' | 'year'

    Returns:
        DatetimeIndex（仅含 year 年的日期）
    """
    all_ends = get_period_end_dates(period)
    return all_ends[all_ends.year == year]


def invalidate_cache():
    """清除缓存，强制下次重新读取 feather"""
    global _cached_df
    _cached_df = None

