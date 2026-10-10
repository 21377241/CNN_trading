# -*- coding: utf-8 -*-
"""
Misc/utilities.py - 通用工具函数
"""
import os
import logging
import numpy as np
import pandas as pd


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """创建 Logger，可选同时输出到文件"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def read_csv_with_encoding(filepath: str, **kwargs) -> pd.DataFrame:
    """
    尝试多种编码读取 CSV，返回 DataFrame
    优先顺序：gbk -> gb18030 -> utf-8 -> latin1
    """
    encodings = ['gbk', 'gb18030', 'utf-8', 'utf-8-sig', 'latin1']

    # 先尝试 chardet 检测
    try:
        import chardet
        with open(filepath, 'rb') as f:
            raw = f.read(min(100000, os.path.getsize(filepath)))
        detected = chardet.detect(raw)
        if detected['encoding']:
            encodings = [detected['encoding']] + encodings
    except ImportError:
        pass

    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            return df
        except (UnicodeDecodeError, Exception):
            continue

    raise ValueError(f"无法读取文件 {filepath}，尝试了编码: {encodings}")


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 Matthews Correlation Coefficient (MCC)
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def annualize_return(monthly_returns: np.ndarray) -> float:
    """将月度收益序列年化"""
    mean_monthly = np.mean(monthly_returns)
    return (1 + mean_monthly) ** 12 - 1


def annualize_sharpe(monthly_returns: np.ndarray, rf: float = 0.0) -> float:
    """计算年化 Sharpe 比率（月度数据）"""
    excess = monthly_returns - rf / 12
    if len(excess) < 2 or np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(12))


def make_dir(path: str):
    """安全创建目录"""
    os.makedirs(path, exist_ok=True)
    return path


def get_period_end_dates_from_dates(dates: pd.Series, period: str) -> pd.DatetimeIndex:
    """
    从交易日日期序列中推算各周期末最后一个交易日

    Args:
        dates: 所有交易日的日期序列（无重复）
        period: 'week' | 'month' | 'quarter' | 'year'

    Returns:
        周期末日期的 DatetimeIndex（排序）
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values()
    df = pd.DataFrame({'Date': dates})

    if period == 'week':
        df['key'] = df['Date'].dt.isocalendar().year * 100 + df['Date'].dt.isocalendar().week
    elif period == 'month':
        df['key'] = df['Date'].dt.year * 100 + df['Date'].dt.month
    elif period == 'quarter':
        df['key'] = df['Date'].dt.year * 10 + df['Date'].dt.quarter
    elif period == 'year':
        df['key'] = df['Date'].dt.year
    else:
        raise ValueError(f"不支持的 period: {period}")

    end_dates = df.groupby('key')['Date'].max()
    return pd.DatetimeIndex(sorted(end_dates.values))

