# -*- coding: utf-8 -*-
"""
2_generate_csi800_chart_data.py - CSI800 图表数据生成

从 csi800_ret.feather（步骤1输出）生成K线图像数据集：
  - 按年遍历 CSI800_IS_YEARS + CSI800_OOS_YEARS
  - Monkey Patch equity_data 的数据源与期末日函数
  - 每年生成：*_images.dat + *_labels.feather + *.txt

输出目录：
  WORK_SPACE/data/csi800_stocks_dataset/stocks_CHN/dataset_all/
"""
import sys
import os
import logging
import pandas as pd

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CSI800 专属配置
# ──────────────────────────────────────────────
from Misc.config import (
    CSI800_PROCESSED_DATA_DIR, CSI800_STOCKS_SAVEPATH,
    CSI800_IS_YEARS, CSI800_OOS_YEARS, CSI800_COUNTRY,
    BENCHMARK_MODEL_WINDOW, FREQ, MA_LAGS, VOLUME_BAR
)
from Misc.utilities import get_period_end_dates_from_dates

FEATHER_PATH = os.path.join(CSI800_PROCESSED_DATA_DIR, 'csi800_ret.feather')
ALL_YEARS    = CSI800_IS_YEARS + CSI800_OOS_YEARS

# ──────────────────────────────────────────────
# Monkey Patch：替换 equity_data 的数据源与期末日函数
# ──────────────────────────────────────────────
import Data.equity_data as equity_data


def _patched_processed_US_data():
    """从 csi800_ret.feather 加载全量数据，设置 (Date, StockID) 为索引"""
    if not hasattr(_patched_processed_US_data, '_cache'):
        df = pd.read_feather(FEATHER_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        _patched_processed_US_data._cache = df.set_index(['Date', 'StockID'])
    return _patched_processed_US_data._cache


def _patched_get_processed_US_data_by_year(year: int, lookback_years: int = 1):
    """从 csi800_ret.feather 取 year 及前 lookback_years 年数据"""
    if not hasattr(_patched_get_processed_US_data_by_year, '_cache'):
        df = pd.read_feather(FEATHER_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        _patched_get_processed_US_data_by_year._cache = df
    df = _patched_get_processed_US_data_by_year._cache
    start_year = year - lookback_years
    mask = (df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= year)
    return df[mask].copy().sort_values(['StockID', 'Date']).reset_index(drop=True)


def _patched_get_period_end_dates(period: str):
    """从 csi800_ret.feather 的交易日历推算周期末日期"""
    if not hasattr(_patched_get_period_end_dates, '_cache'):
        df = pd.read_feather(FEATHER_PATH)
        _patched_get_period_end_dates._cache = pd.to_datetime(df['Date']).unique()
    dates = _patched_get_period_end_dates._cache
    return get_period_end_dates_from_dates(pd.Series(dates), period)


# 执行 Monkey Patch
equity_data.processed_US_data = _patched_processed_US_data
equity_data.get_processed_US_data_by_year = _patched_get_processed_US_data_by_year
equity_data.get_period_end_dates = _patched_get_period_end_dates
logger.info("Monkey Patch 完成：equity_data 数据源已替换为 CSI800 feather")

# ──────────────────────────────────────────────
from Data.generate_chart import GenerateStockData
from Data.dgp_config import get_dataset_dir, get_dataset_filename


def main():
    logger.info("=" * 60)
    logger.info("  步骤 2（CSI800）：图表数据生成")
    logger.info("=" * 60)
    logger.info(f"数据源   : {FEATHER_PATH}")
    logger.info(f"处理年份 : {ALL_YEARS}")
    logger.info(f"窗口大小 : {BENCHMARK_MODEL_WINDOW}，频率: {FREQ}，均线: {MA_LAGS}")
    logger.info(f"输出目录 : {CSI800_STOCKS_SAVEPATH}")

    if not os.path.exists(FEATHER_PATH):
        logger.error(f"找不到预处理数据文件: {FEATHER_PATH}")
        logger.error("请先运行: python 1_preprocess_csi800_data.py")
        return 1

    dataset_dir = get_dataset_dir(CSI800_COUNTRY, base_savepath=CSI800_STOCKS_SAVEPATH)
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info(f"图表数据集目录: {dataset_dir}")

    for year in ALL_YEARS:
        logger.info(f"\n{'─'*40}")
        logger.info(f"处理年份: {year}")

        # 检查是否已生成
        fn_images = get_dataset_filename(
            BENCHMARK_MODEL_WINDOW, FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
        p_images = os.path.join(dataset_dir, fn_images)
        if os.path.exists(p_images):
            logger.info(f"  已存在，跳过: {fn_images}")
            continue

        try:
            gen = GenerateStockData(
                country=CSI800_COUNTRY,
                year=year,
                window_size=BENCHMARK_MODEL_WINDOW,
                freq=FREQ,
                chart_freq=1,
                ma_lags=MA_LAGS,
                volume_bar=VOLUME_BAR,
                need_adjust_price=True,
                chart_type='bar',
                ret_col=f'Ret_{FREQ}',
                dataset_dir=dataset_dir
            )
            gen.save_annual_data()
        except Exception as e:
            logger.error(f"  年份 {year} 生成失败: {e}", exc_info=True)
            continue

    # 验证输出
    logger.info(f"\n{'='*60}")
    logger.info("验证生成结果：")
    ok_count = 0
    for year in ALL_YEARS:
        fn_img = get_dataset_filename(
            BENCHMARK_MODEL_WINDOW, FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
        fn_lbl = get_dataset_filename(
            BENCHMARK_MODEL_WINDOW, FREQ, VOLUME_BAR, MA_LAGS, year, '_labels.feather')
        p_img = os.path.join(dataset_dir, fn_img)
        p_lbl = os.path.join(dataset_dir, fn_lbl)

        if os.path.exists(p_img) and os.path.exists(p_lbl):
            lbl = pd.read_feather(p_lbl)
            logger.info(f"  {year}: {len(lbl)} 条样本  [OK]")
            ok_count += 1
        else:
            logger.warning(f"  {year}: 文件缺失  [XX]")

    if ok_count == len(ALL_YEARS):
        logger.info(f"\n[OK] 步骤 2（CSI800）完成！所有 {len(ALL_YEARS)} 个年份已生成。")
        logger.info("  下一步请运行: python 3_train_csi800_model.py")
    else:
        logger.warning(f"\n[!] 步骤 2（CSI800）部分完成：{ok_count}/{len(ALL_YEARS)} 个年份成功。")

    return 0


if __name__ == '__main__':
    sys.exit(main())
