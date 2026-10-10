# -*- coding: utf-8 -*-
"""
2w_generate_spx_week5d_chart_data.py - SPX 周频图表数据生成（5日预测）

从 us_ret_spx.feather（步骤 1 输出）生成 K 线图像数据集：
  - 输入窗口：20 个交易日（与月频模型相同）
  - 预测频率：week（每周末生成一个样本）
  - 标签列：Ret_week（未来 5 个交易日收益）
  - 每年生成：*_images.dat + *_labels.feather + *.txt

输出目录：
  WORK_SPACE/data/stocks_dataset/stocks_USA/dataset_all/
  文件名格式：20d_week_has_vb_[20]_ma_{year}_images.dat
"""
import sys
import os
import logging

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
# 配置（从全局 config 读取通用参数，本脚本覆盖频率相关参数）
# ──────────────────────────────────────────────
from Misc.config import (
    PROCESSED_DATA_DIR, IS_YEARS, OOS_YEARS,
    BENCHMARK_MODEL_WINDOW, MA_LAGS, VOLUME_BAR, COUNTRY
)
from Misc.utilities import get_period_end_dates_from_dates
import pandas as pd

# ── 周频专属参数 ──
WEEK_FREQ = 'week'           # 预测频率：周
WEEK_PREDICT_WINDOW = 5      # 预测天数：5个交易日
WINDOW_SIZE = BENCHMARK_MODEL_WINDOW  # 输入窗口：20天（与月频相同）

FEATHER_PATH = os.path.join(PROCESSED_DATA_DIR, 'us_ret_spx.feather')
ALL_YEARS = IS_YEARS + OOS_YEARS

# ──────────────────────────────────────────────
# Monkey Patch：替换 equity_data 的数据源与期末日函数
# ──────────────────────────────────────────────
import Data.equity_data as equity_data


def _patched_processed_US_data():
    if not hasattr(_patched_processed_US_data, '_cache'):
        df = pd.read_feather(FEATHER_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        _patched_processed_US_data._cache = df.set_index(['Date', 'StockID'])
    return _patched_processed_US_data._cache


def _patched_get_processed_US_data_by_year(year: int, lookback_years: int = 1):
    if not hasattr(_patched_get_processed_US_data_by_year, '_cache'):
        df = pd.read_feather(FEATHER_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        _patched_get_processed_US_data_by_year._cache = df
    df = _patched_get_processed_US_data_by_year._cache
    start_year = year - lookback_years
    mask = (df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= year)
    return df[mask].copy().sort_values(['StockID', 'Date']).reset_index(drop=True)


def _patched_get_period_end_dates(period: str):
    if not hasattr(_patched_get_period_end_dates, '_cache'):
        df = pd.read_feather(FEATHER_PATH)
        _patched_get_period_end_dates._cache = pd.to_datetime(df['Date']).unique()
    dates = _patched_get_period_end_dates._cache
    return get_period_end_dates_from_dates(pd.Series(dates), period)


equity_data.processed_US_data = _patched_processed_US_data
equity_data.get_processed_US_data_by_year = _patched_get_processed_US_data_by_year
equity_data.get_period_end_dates = _patched_get_period_end_dates
logger.info("Monkey Patch 完成：equity_data 数据源已替换为 SPX feather（周频）")

# ──────────────────────────────────────────────
from Data.generate_chart import GenerateStockData
from Data.dgp_config import get_dataset_dir, get_dataset_filename


def main():
    logger.info("=" * 60)
    logger.info("  步骤 2w：SPX 周频图表数据生成（20d输入 / 5d预测）")
    logger.info("=" * 60)
    logger.info(f"数据源    : {FEATHER_PATH}")
    logger.info(f"处理年份  : {ALL_YEARS}")
    logger.info(f"输入窗口  : {WINDOW_SIZE} 天")
    logger.info(f"预测频率  : {WEEK_FREQ}（{WEEK_PREDICT_WINDOW} 个交易日）")
    logger.info(f"标签列    : Ret_{WEEK_FREQ}")
    logger.info(f"均线      : {MA_LAGS}，成交量柱：{VOLUME_BAR}")

    if not os.path.exists(FEATHER_PATH):
        logger.error(f"找不到预处理数据文件: {FEATHER_PATH}")
        logger.error("请先运行 1_preprocess_spx_data.py")
        return 1

    dataset_dir = get_dataset_dir(COUNTRY)
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info(f"输出目录  : {dataset_dir}")

    for year in ALL_YEARS:
        logger.info(f"\n{'─'*40}")
        logger.info(f"处理年份: {year}")

        fn_images = get_dataset_filename(
            WINDOW_SIZE, WEEK_FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
        p_images = os.path.join(dataset_dir, fn_images)
        if os.path.exists(p_images):
            logger.info(f"  已存在，跳过: {fn_images}")
            continue

        try:
            gen = GenerateStockData(
                country=COUNTRY,
                year=year,
                window_size=WINDOW_SIZE,
                freq=WEEK_FREQ,
                chart_freq=1,
                ma_lags=MA_LAGS,
                volume_bar=VOLUME_BAR,
                need_adjust_price=True,
                chart_type='bar',
                ret_col=f'Ret_{WEEK_FREQ}'
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
            WINDOW_SIZE, WEEK_FREQ, VOLUME_BAR, MA_LAGS, year, '_images.dat')
        fn_lbl = get_dataset_filename(
            WINDOW_SIZE, WEEK_FREQ, VOLUME_BAR, MA_LAGS, year, '_labels.feather')
        p_img = os.path.join(dataset_dir, fn_img)
        p_lbl = os.path.join(dataset_dir, fn_lbl)

        if os.path.exists(p_img) and os.path.exists(p_lbl):
            lbl = pd.read_feather(p_lbl)
            logger.info(f"  {year}: {len(lbl)} 条样本  [OK]")
            ok_count += 1
        else:
            logger.warning(f"  {year}: 文件缺失  [XX]")

    if ok_count == len(ALL_YEARS):
        logger.info(f"\n[OK] 步骤 2w 完成！所有 {len(ALL_YEARS)} 个年份已生成。")
        logger.info("  下一步请运行: python 3w_train_spx_week5d_model.py")
    else:
        logger.warning(f"\n[!] 步骤 2w 部分完成：{ok_count}/{len(ALL_YEARS)} 个年份成功。")

    return 0


if __name__ == '__main__':
    sys.exit(main())
