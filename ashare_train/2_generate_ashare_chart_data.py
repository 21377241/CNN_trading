# -*- coding: utf-8 -*-
"""
2_generate_ashare_chart_data.py - A股全市场图表生成

用法:
  python ashare_train/2_generate_ashare_chart_data.py --spec I20R20
  python ashare_train/2_generate_ashare_chart_data.py --spec I5R5
"""
import sys
import os
import argparse
import logging
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_PROCESSED_DATA_DIR, ASHARE_STOCKS_SAVEPATH,
    ASHARE_IS_YEARS, ASHARE_OOS_YEARS, ASHARE_COUNTRY,
    MA_LAGS, VOLUME_BAR
)
from Misc.utilities import get_period_end_dates_from_dates

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

FEATHER_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_ret.feather')
ALL_YEARS = ASHARE_IS_YEARS + ASHARE_OOS_YEARS

SPECS = {
    'I20R20': dict(window_size=20, freq='month', predict_window=20, ma_lags=MA_LAGS),
    'I5R5':   dict(window_size=5,  freq='week',  predict_window=5,  ma_lags=MA_LAGS),
    'I20R5':  dict(window_size=20, freq='week',  predict_window=5,  ma_lags=MA_LAGS),
}

import Data.equity_data as equity_data


def _patch_equity_data():
    def _processed():
        if not hasattr(_processed, '_cache'):
            df = pd.read_feather(FEATHER_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            _processed._cache = df.set_index(['Date', 'StockID'])
        return _processed._cache

    def _by_year(year: int, lookback_years: int = 1):
        if not hasattr(_by_year, '_cache'):
            df = pd.read_feather(FEATHER_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            _by_year._cache = df
        df = _by_year._cache
        start_year = year - lookback_years
        mask = (df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= year)
        return df[mask].copy().sort_values(['StockID', 'Date']).reset_index(drop=True)

    def _period_ends(period: str):
        if not hasattr(_period_ends, '_cache'):
            df = pd.read_feather(FEATHER_PATH)
            _period_ends._cache = pd.to_datetime(df['Date']).unique()
        return get_period_end_dates_from_dates(pd.Series(_period_ends._cache), period)

    equity_data.processed_US_data = _processed
    equity_data.get_processed_US_data_by_year = _by_year
    equity_data.get_period_end_dates = _period_ends
    logger.info("Monkey Patch 完成：数据源 → ashare_ret.feather")


_patch_equity_data()

from Data.generate_chart import GenerateStockData
from Data.dgp_config import get_dataset_dir, get_dataset_filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', choices=list(SPECS.keys()) + ['all'], default='I20R20')
    args = parser.parse_args()
    spec = SPECS[args.spec]
    ws, freq = spec['window_size'], spec['freq']
    ma_lags = spec['ma_lags']

    logger.info("=" * 60)
    logger.info(f"  图表生成 [{args.spec}]  ws={ws} freq={freq}")
    logger.info("=" * 60)

    if not os.path.exists(FEATHER_PATH):
        logger.error(f"请先运行预处理: {FEATHER_PATH}")
        return 1

    # 分子设定子目录，避免 I5/I20 文件名冲突（freq 不同通常不冲突，但仍隔离）
    base_save = os.path.join(ASHARE_STOCKS_SAVEPATH, args.spec)
    dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=base_save)
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info(f"输出: {dataset_dir}")

    for year in ALL_YEARS:
        fn = get_dataset_filename(ws, freq, VOLUME_BAR, ma_lags, year, '_images.dat')
        p = os.path.join(dataset_dir, fn)
        if os.path.exists(p):
            logger.info(f"  {year}: 已存在，跳过")
            continue
        logger.info(f"\n── 处理 {year} ──")
        try:
            gen = GenerateStockData(
                country=ASHARE_COUNTRY,
                year=year,
                window_size=ws,
                freq=freq,
                chart_freq=1,
                ma_lags=ma_lags,
                volume_bar=VOLUME_BAR,
                need_adjust_price=True,
                chart_type='bar',
                ret_col=f'Ret_{freq}',
                dataset_dir=dataset_dir,
            )
            gen.save_annual_data()
        except Exception as e:
            logger.error(f"  {year} 失败: {e}", exc_info=True)

    ok = 0
    for year in ALL_YEARS:
        fn_img = get_dataset_filename(ws, freq, VOLUME_BAR, ma_lags, year, '_images.dat')
        fn_lbl = get_dataset_filename(ws, freq, VOLUME_BAR, ma_lags, year, '_labels.feather')
        if os.path.exists(os.path.join(dataset_dir, fn_img)) and \
           os.path.exists(os.path.join(dataset_dir, fn_lbl)):
            ok += 1
            logger.info(f"  {year}: OK")
        else:
            logger.warning(f"  {year}: 缺失")
    logger.info(f"[OK] {args.spec}: {ok}/{len(ALL_YEARS)} 年完成")
    return 0 if ok == len(ALL_YEARS) else 1


if __name__ == '__main__':
    sys.exit(main())
