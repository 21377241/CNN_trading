# -*- coding: utf-8 -*-
"""
1_preprocess_ashare_data.py - A股全市场复权数据预处理

从 F:/BaiduNetdiskDownload/股票4.14/每只股票一个文件/复权/*.csv
读取每股日线，过滤 ST，计算 Ret / MarketCap / EWMA / 未来收益，
写出 WORK_SPACE/data/ashare_processed_data/ashare_ret.feather

样本：约 5000 只股票，2000–2026
"""
import sys
import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_RAW_ROOT, ASHARE_PROCESSED_DATA_DIR, ASHARE_CSV_ENCODING,
    ASHARE_RET_CLIP
)
from Misc.utilities import get_period_end_dates_from_dates

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

FEATHER_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_ret.feather')
CACHE_STEP1 = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_cache_step1_merged.feather')
CACHE_STEP4 = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_cache_step4_precomputed.feather')


def _find_col(columns, *keywords):
    for c in columns:
        for k in keywords:
            if k in str(c):
                return c
    return None


def resolve_adjusted_dir(raw_root: str = None) -> Path:
    """定位「复权」目录（排除「不复权」）"""
    root = Path(raw_root or ASHARE_RAW_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"原始数据根目录不存在: {root}")

    # 可能是 股票4.14/每只股票一个文件/复权 或直接 复权
    candidates = []
    for p in root.rglob('*'):
        if not p.is_dir():
            continue
        name = p.name
        if '复权' in name and '不复权' not in name:
            # 必须含 csv
            if any(p.glob('*.csv')):
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"未找到「复权」目录于 {root}。请确认数据结构。"
        )
    # 选 csv 最多的
    best = max(candidates, key=lambda d: len(list(d.glob('*.csv'))))
    logger.info(f"使用复权目录: {best}  (csv≈{len(list(best.glob('*.csv')))})")
    return best


def _save_feather(df: pd.DataFrame, path: str, label: str):
    tmp = df.copy()
    for col in tmp.columns:
        if tmp[col].dtype == object:
            tmp[col] = tmp[col].astype(str)
    tmp = tmp.reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp.to_feather(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"  [OK] [{label}] → {os.path.basename(path)}  ({size_mb:.1f} MB)")


def _load_feather(path: str, label: str) -> pd.DataFrame:
    logger.info(f"  [加载] [{label}]: {os.path.basename(path)}")
    df = pd.read_feather(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"    形状 {df.shape}，"
                f"{df['Date'].min().date()} ~ {df['Date'].max().date()}")
    return df


def _is_st_value(v) -> bool:
    s = str(v).strip()
    if s in ('', '-', 'nan', 'None', '否', '0', 'False'):
        return False
    if '否' in s:
        return False
    if '是' in s or 'ST' in s.upper() or s == '1':
        return True
    return False


def read_and_merge(adj_dir: Path = None) -> pd.DataFrame:
    """逐股读取复权 CSV，标准化列，过滤 ST 行，合并面板。"""
    adj_dir = adj_dir or resolve_adjusted_dir()
    files = sorted([
        f for f in adj_dir.glob('*.csv')
        if not f.stem.endswith('(1)')
    ])
    if not files:
        raise FileNotFoundError(f"复权目录无 CSV: {adj_dir}")

    logger.info(f"待读取股票文件: {len(files)}")
    chunks = []
    n_skip_st = 0
    n_fail = 0

    for fp in tqdm(files, desc='读取复权CSV'):
        try:
            df = pd.read_csv(fp, encoding=ASHARE_CSV_ENCODING)
        except Exception:
            try:
                df = pd.read_csv(fp, encoding='gb18030')
            except Exception:
                n_fail += 1
                continue

        if df is None or len(df) == 0:
            n_fail += 1
            continue

        cols = list(df.columns)
        c_date = _find_col(cols, '日期', 'date', 'Date')
        c_code = _find_col(cols, '代码', 'code', 'Code')
        c_open = _find_col(cols, '开盘')
        c_high = _find_col(cols, '最高')
        c_low = _find_col(cols, '最低')
        c_close = _find_col(cols, '收盘')
        # 成交量列优先匹配「成交量」避免误匹配成交额
        c_vol = None
        for c in cols:
            if '成交量' in str(c):
                c_vol = c
                break
        c_mcap = _find_col(cols, '流通市值') or _find_col(cols, '总市值')
        c_st = _find_col(cols, '是否ST', 'ST')
        c_name = _find_col(cols, '名称', 'name')

        if not all([c_date, c_open, c_high, c_low, c_close]):
            n_fail += 1
            continue

        out = pd.DataFrame({
            'Date': pd.to_datetime(df[c_date], errors='coerce'),
            'Open': pd.to_numeric(df[c_open], errors='coerce'),
            'High': pd.to_numeric(df[c_high], errors='coerce'),
            'Low': pd.to_numeric(df[c_low], errors='coerce'),
            'Close': pd.to_numeric(df[c_close], errors='coerce'),
        })
        # StockID：优先文件名（6位代码）；代码列为空则回退文件名
        sid = fp.stem
        if c_code is not None and len(df[c_code].dropna()) > 0:
            code_val = str(df[c_code].dropna().iloc[0]).zfill(6)
            if code_val.replace('.0', '').isdigit() or len(code_val) >= 6:
                sid = code_val.split('.')[0].zfill(6)
        out['StockID'] = sid

        if c_vol is not None:
            out['Vol'] = pd.to_numeric(df[c_vol], errors='coerce').fillna(0)
        else:
            out['Vol'] = 0.0

        if c_mcap is not None:
            out['MarketCap'] = pd.to_numeric(df[c_mcap], errors='coerce').fillna(0)
        else:
            out['MarketCap'] = out['Vol'] * out['Close']

        if c_name is not None:
            out['sec_name'] = df[c_name].astype(str)
        else:
            out['sec_name'] = sid

        if c_st is not None:
            st_mask = df[c_st].map(_is_st_value)
            n_skip_st += int(st_mask.sum())
            out = out.loc[~st_mask.values].copy()

        out = out.dropna(subset=['Date', 'Close'])
        out = out[(out['Close'] > 0) & (out['Open'] > 0)]
        if len(out) < 30:
            continue
        chunks.append(out)

    logger.info(f"成功 {len(chunks)} 只，失败 {n_fail}，剔除 ST 行 {n_skip_st}")
    if not chunks:
        raise RuntimeError("未读到任何有效股票数据")

    # 分批 concat 降低峰值内存
    batch = 200
    parts = []
    for i in range(0, len(chunks), batch):
        parts.append(pd.concat(chunks[i:i + batch], ignore_index=True))
        chunks[i:i + batch] = [None] * min(batch, len(chunks) - i)
    df = pd.concat(parts, ignore_index=True)
    del parts, chunks

    df = df.sort_values(['StockID', 'Date']).reset_index(drop=True)
    logger.info(f"合并后: {df.shape}，"
                f"{df['Date'].min().date()} ~ {df['Date'].max().date()}")
    return df


def compute_ret(df: pd.DataFrame) -> pd.DataFrame:
    df['Ret'] = df.groupby('StockID')['Close'].pct_change()
    df['Ret'] = df['Ret'].fillna(0.0).clip(-ASHARE_RET_CLIP, ASHARE_RET_CLIP)
    return df


def finalize_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """确保 MarketCap 为正；缺失时用 Vol×Close"""
    if 'MarketCap' not in df.columns:
        df['MarketCap'] = df['Vol'] * df['Close']
    miss = df['MarketCap'].isna() | (df['MarketCap'] <= 0)
    if miss.any():
        df.loc[miss, 'MarketCap'] = (
            df.loc[miss, 'Vol'] * df.loc[miss, 'Close']
        ).fillna(0)
    df['MarketCap'] = df['MarketCap'].fillna(0.0)
    return df


def compute_ewma_vol(df: pd.DataFrame) -> pd.DataFrame:
    df['EWMA_vol'] = (
        df.groupby('StockID')['Ret']
        .transform(lambda x: (x ** 2).ewm(alpha=0.05, adjust=False).mean().shift(1))
    )
    df['EWMA_vol'] = df['EWMA_vol'].fillna(0.0)
    return df


def compute_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['log_ret'] = np.log1p(df['Ret'])
    df['cum_log_ret'] = df.groupby('StockID')['log_ret'].cumsum()

    for h in tqdm([5, 20, 60, 65, 180, 250, 260], desc='固定 horizon'):
        col = f'Ret_{h}d'
        df[col] = (
            df.groupby('StockID')['cum_log_ret']
            .transform(lambda x: np.exp(x.shift(-h) - x) - 1)
        )

    all_dates = df['Date'].unique()
    date_to_idx = {}
    for date_val, grp in df.groupby('Date'):
        date_to_idx[date_val] = grp.index

    for period in ['week', 'month', 'quarter', 'year']:
        col = f'Ret_{period}'
        df[col] = np.nan
        ends = sorted(set(get_period_end_dates_from_dates(
            pd.Series(all_dates), period)))
        logger.info(f"  计算 {period}: {len(ends)} 个周期")
        for i, end_dt in enumerate(tqdm(ends, desc=period, leave=False)):
            if i + 1 >= len(ends):
                break
            end_ts = pd.Timestamp(end_dt)
            next_ts = pd.Timestamp(ends[i + 1])
            idx_end = date_to_idx.get(end_ts)
            idx_next = date_to_idx.get(next_ts)
            if idx_end is None or idx_next is None:
                continue
            cum_end = df.loc[idx_end].set_index('StockID')['cum_log_ret']
            cum_next = df.loc[idx_next].set_index('StockID')['cum_log_ret']
            common = cum_end.index.intersection(cum_next.index)
            if len(common) == 0:
                continue
            period_ret = np.exp(cum_next[common] - cum_end[common]) - 1
            df.loc[idx_end, col] = df.loc[idx_end, 'StockID'].map(period_ret)

    df = df.drop(columns=['log_ret', 'cum_log_ret'])
    return df


def validate_output(df: pd.DataFrame):
    logger.info("=" * 60)
    logger.info(f"行数={len(df):,}  股票={df['StockID'].nunique()}  "
                f"日期={df['Date'].min().date()}~{df['Date'].max().date()}")
    logger.info(f"列={list(df.columns)}")
    logger.info(f"Ret: mean={df['Ret'].mean():.5f} std={df['Ret'].std():.5f} "
                f"min={df['Ret'].min():.3f} max={df['Ret'].max():.3f}")
    pos = (df['MarketCap'] > 0).mean()
    logger.info(f"MarketCap>0 比例: {pos:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-from', type=int, default=0,
                        help='0=自动; 1=从头读CSV; 4=从step1重算Ret; 5=只重算未来收益')
    parser.add_argument('--raw-root', type=str, default=None)
    args = parser.parse_args()
    force = args.force_from

    logger.info("=" * 60)
    logger.info("  步骤 1：A股全市场复权数据预处理")
    logger.info("=" * 60)

    if force == 0 and os.path.exists(FEATHER_PATH):
        logger.info(f"最终文件已存在: {FEATHER_PATH}")
        logger.info("重跑: --force-from 1 / 4 / 5")
        return 0

    need_read = force == 1 or not os.path.exists(CACHE_STEP1)
    need_feat = force in (1, 4) or not os.path.exists(CACHE_STEP4)

    if need_read:
        adj = resolve_adjusted_dir(args.raw_root)
        df = read_and_merge(adj)
        _save_feather(df, CACHE_STEP1, 'step1_merged')
    else:
        df = _load_feather(CACHE_STEP1, 'step1')

    if need_feat:
        logger.info("[2-4] Ret / MarketCap / EWMA ...")
        df = compute_ret(df)
        df = finalize_market_cap(df)
        df = compute_ewma_vol(df)
        _save_feather(df, CACHE_STEP4, 'step4_precomputed')
    else:
        df = _load_feather(CACHE_STEP4, 'step4')

    logger.info("[5/5] 计算未来收益...")
    df = compute_future_returns(df)
    validate_output(df)
    _save_feather(df, FEATHER_PATH, 'ashare_ret')
    logger.info("[OK] 预处理完成。下一步: python ashare_train/2_generate_ashare_chart_data.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
