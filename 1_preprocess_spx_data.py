# -*- coding: utf-8 -*-
"""
1_preprocess_spx_data.py - 数据预处理

从 SPX_volume_price/ 读取原始美股日线 CSV 文件，
目录结构：SPX_volume_price/<YYYY>/<YYYYMMDD>.csv
每个 CSV 包含当日全部股票数据（横截面）

执行以下处理后保存为 us_ret_spx.feather：

  1. 读取与合并（按年份目录逐文件读取）
  2. 列名标准化
  3. 日收益率 Ret
  4. 市值代理 MarketCap = Vol × Close（按日 median 缩放）
  5. EWMA 波动率
  6. 未来收益（Ret_5d, Ret_20d, Ret_60d, Ret_65d, Ret_180d, Ret_250d, Ret_260d,
               Ret_week, Ret_month, Ret_quarter, Ret_year）
  7. 写出 feather

─── 中间文件缓存策略 ────────────────────────────────────────────
  cache_step1_merged.feather    : 步骤1完成后（CSV读取合并）
  cache_step4_precomputed.feather : 步骤4完成后（Ret/MarketCap/EWMA_vol，未计算未来收益）

  运行逻辑：
    - 若 cache_step4_precomputed.feather 存在 → 直接跳到步骤5
    - 若 cache_step1_merged.feather 存在      → 从步骤2开始
    - 否则从步骤1（读取CSV）开始

  强制重跑：
    python 1_preprocess_spx_data.py --force          # 从头全部重算
    python 1_preprocess_spx_data.py --force-from 1   # 从步骤1重算（默认）
    python 1_preprocess_spx_data.py --force-from 2   # 从步骤2重算（跳过CSV读取）
    python 1_preprocess_spx_data.py --force-from 5   # 只重算未来收益
"""
import sys
import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    SPX_DIR, PROCESSED_DATA_DIR, CACHE_DIR, SPX_CSV_ENCODING
)
from Misc.utilities import get_period_end_dates_from_dates

# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── 路径定义 ──
FEATHER_PATH        = os.path.join(PROCESSED_DATA_DIR, 'us_ret_spx.feather')
CACHE_STEP1_PATH    = os.path.join(PROCESSED_DATA_DIR, 'cache_step1_merged.feather')
CACHE_STEP4_PATH    = os.path.join(PROCESSED_DATA_DIR, 'cache_step4_precomputed.feather')


# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────
def _save_feather(df: pd.DataFrame, path: str, label: str):
    """安全保存 feather，自动处理 object 列"""
    tmp = df.copy()
    for col in tmp.columns:
        if tmp[col].dtype == 'object':
            tmp[col] = tmp[col].astype(str)
    tmp = tmp.reset_index(drop=True)
    tmp.to_feather(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"  ✓ [{label}] 已缓存 → {os.path.basename(path)}  ({size_mb:.1f} MB)")


def _load_feather(path: str, label: str) -> pd.DataFrame:
    """加载 feather 并恢复日期列类型"""
    logger.info(f"  ↩ 加载缓存 [{label}]: {os.path.basename(path)}")
    df = pd.read_feather(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"    形状: {df.shape}，"
                f"日期: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    return df


# ────────────────────────────────────────────────
# 步骤 1：读取与合并
# ────────────────────────────────────────────────
def read_and_merge() -> pd.DataFrame:
    """
    读取 SPX_volume_price/<YYYY>/<YYYYMMDD>.csv 所有文件并合并为面板 DataFrame。
    每个 CSV 为当日横截面，列：date, wind_code, sec_name, OPEN, CLOSE, HIGH, LOW, VOLUME
    """
    # 找所有年份目录
    year_dirs = sorted([
        d for d in glob.glob(os.path.join(SPX_DIR, '*'))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ])

    if not year_dirs:
        raise FileNotFoundError(
            f"未找到任何年份子目录于: {SPX_DIR}\n"
            "请确认 SPX_volume_price/ 目录存在且包含数据（如 2011/, 2012/, ...）。"
        )

    logger.info(f"发现年份目录: {[os.path.basename(d) for d in year_dirs]}")

    dfs = []
    total_files = 0

    for year_dir in year_dirs:
        year = int(os.path.basename(year_dir))
        csv_files = sorted(glob.glob(os.path.join(year_dir, '*.csv')))
        if not csv_files:
            logger.warning(f"年份 {year} 目录下无 CSV 文件，跳过。")
            continue

        logger.info(f"  年份 {year}: 发现 {len(csv_files)} 个 CSV 文件")

        year_dfs = []
        for fp in tqdm(csv_files, desc=f"读取 {year}", leave=False):
            try:
                df = pd.read_csv(fp, encoding=SPX_CSV_ENCODING)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(fp, encoding='utf-8')
                except Exception as e:
                    logger.warning(f"读取失败 {os.path.basename(fp)}: {e}，跳过。")
                    continue
            except Exception as e:
                logger.warning(f"读取失败 {os.path.basename(fp)}: {e}，跳过。")
                continue

            year_dfs.append(df)
            total_files += 1

        if year_dfs:
            year_df = pd.concat(year_dfs, ignore_index=True)
            dfs.append(year_df)

    if not dfs:
        raise RuntimeError("所有文件读取失败。")

    logger.info(f"共读取 {total_files} 个 CSV 文件，开始合并...")
    panel = pd.concat(dfs, ignore_index=True)

    # ── 列名标准化 ──
    col_map = {
        'date':      'Date',
        'wind_code': 'StockID',
        'sec_name':  'sec_name',
        'OPEN':      'Open',
        'CLOSE':     'Close',
        'HIGH':      'High',
        'LOW':       'Low',
        'VOLUME':    'Vol',
    }
    panel = panel.rename(columns=col_map)

    # 确保日期列为 datetime
    panel['Date'] = pd.to_datetime(panel['Date'])

    # 去重（同一天同一股票可能因多文件重复）
    panel = panel.drop_duplicates(subset=['Date', 'StockID'])

    # 排序
    panel = panel.sort_values(['StockID', 'Date']).reset_index(drop=True)

    # 过滤无效价格
    panel = panel[panel['Close'] > 0].copy()

    logger.info(f"合并后总行数: {len(panel)}，"
                f"股票数: {panel['StockID'].nunique()}，"
                f"日期范围: {panel['Date'].min().date()} ~ {panel['Date'].max().date()}")
    return panel


# ────────────────────────────────────────────────
# 步骤 2：日收益率
# ────────────────────────────────────────────────
def compute_ret(df: pd.DataFrame) -> pd.DataFrame:
    """计算日收益率，首日置 0"""
    df['Ret'] = df.groupby('StockID')['Close'].pct_change()
    df['Ret'] = df['Ret'].fillna(0.0)
    # 处理极端值（如停牌、退市等）
    df['Ret'] = df['Ret'].clip(-0.5, 0.5)
    return df


# ────────────────────────────────────────────────
# 步骤 3：市值代理
# ────────────────────────────────────────────────
def compute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    MarketCap = Vol * Close（成交额代理）
    按整体中位数缩放至 ~1e9 量级
    """
    df['MarketCap'] = df['Vol'] * df['Close']
    median_val = df['MarketCap'].median()
    if median_val > 0:
        scale = 1e9 / median_val
        df['MarketCap'] = df['MarketCap'] * scale
    df['Shares'] = np.where(df['Close'] > 0,
                             df['MarketCap'] / df['Close'], 0)
    return df


# ────────────────────────────────────────────────
# 步骤 4：EWMA 波动率
# ────────────────────────────────────────────────
def compute_ewma_vol(df: pd.DataFrame) -> pd.DataFrame:
    """EWMA 日波动率（前一日），alpha=0.05"""
    df['EWMA_vol'] = (df.groupby('StockID')['Ret']
                       .transform(lambda x: (x ** 2)
                                  .ewm(alpha=0.05, adjust=False)
                                  .mean()
                                  .shift(1)))
    df['EWMA_vol'] = df['EWMA_vol'].fillna(0.0)
    return df


# ────────────────────────────────────────────────
# 步骤 5：未来收益（性能优化版）
# ────────────────────────────────────────────────
def compute_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算固定 horizon 与周期末未来收益。

    【性能优化说明】
    原版周期收益的内层循环：
        for stock_id in common:
            stock_mask = mask_end & (df['StockID'] == stock_id)
            df.loc[stock_mask, col] = period_ret[stock_id]
    复杂度 O(N_periods × N_stocks × len(df))，week 约 500×3000×5M = 极慢。

    改为向量化赋值：
        idx_end = df.index[mask_end]
        df.loc[idx_end, col] = df.loc[idx_end, 'StockID'].map(period_ret)
    复杂度降为 O(N_periods × len(df))，速度提升约 3000×。
    """
    # 对数收益 & 累积对数收益
    df['log_ret'] = np.log1p(df['Ret'])
    df['cum_log_ret'] = df.groupby('StockID')['log_ret'].cumsum()

    # ── 固定 horizon 未来收益 ──
    horizons = [5, 20, 60, 65, 180, 250, 260]
    for h in tqdm(horizons, desc='计算固定 horizon 收益'):
        col = f'Ret_{h}d'
        df[col] = (df.groupby('StockID')['cum_log_ret']
                    .transform(lambda x: np.exp(x.shift(-h) - x) - 1))

    # ── 周期末收益（向量化版本）──
    all_dates = df['Date'].unique()

    # 预先建立 日期→行索引 的字典，大幅加速 mask 查找
    # 比每次 df['Date'] == ts 的全量扫描快约 100x
    logger.info("  构建日期→索引映射...")
    date_to_idx: dict = {}
    for _date_val, _grp in df.groupby('Date'):
        date_to_idx[_date_val] = _grp.index

    for period in ['week', 'month', 'quarter', 'year']:
        col = f'Ret_{period}'
        df[col] = np.nan

        end_dates_set = set(get_period_end_dates_from_dates(
            pd.Series(all_dates), period))
        sorted_ends = sorted(end_dates_set)

        logger.info(f"  计算 {period} 收益：共 {len(sorted_ends)} 个周期...")

        for i, end_dt in enumerate(tqdm(sorted_ends,
                                        desc=f'计算 {period} 收益',
                                        leave=False)):
            if i + 1 >= len(sorted_ends):
                break
            next_end = sorted_ends[i + 1]

            end_ts   = pd.Timestamp(end_dt)
            next_ts  = pd.Timestamp(next_end)

            # 取该日期的行索引
            idx_end  = date_to_idx.get(end_ts)
            idx_next = date_to_idx.get(next_ts)
            if idx_end is None or idx_next is None:
                continue

            # 按 StockID 建立 cum_log_ret 映射
            cum_end  = df.loc[idx_end].set_index('StockID')['cum_log_ret']
            cum_next = df.loc[idx_next].set_index('StockID')['cum_log_ret']

            common = cum_end.index.intersection(cum_next.index)
            if len(common) == 0:
                continue

            period_ret = np.exp(cum_next[common] - cum_end[common]) - 1

            # ✅ 向量化赋值（原始内层循环已删除，速度提升 ~3000x）
            df.loc[idx_end, col] = df.loc[idx_end, 'StockID'].map(period_ret)

    # 清理中间列
    df = df.drop(columns=['log_ret', 'cum_log_ret'])
    return df


# ────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='SPX 数据预处理')
    parser.add_argument(
        '--force', action='store_true',
        help='强制从头重算所有步骤（忽略所有缓存）'
    )
    parser.add_argument(
        '--force-from', type=int, default=None, choices=[1, 2, 3, 4, 5],
        metavar='STEP',
        help=(
            '从指定步骤强制重算：\n'
            '  1 = 重新读取 CSV（最慢）\n'
            '  2 = 从 step1 缓存重算 Ret/MarketCap/EWMA\n'
            '  5 = 从 step4 缓存只重算未来收益（最快）'
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析强制重算起始步骤
    force_from = 1 if args.force else (args.force_from or 999)

    logger.info("=" * 60)
    logger.info("  步骤 1：数据预处理（SPX_volume_price）")
    logger.info("=" * 60)
    logger.info(f"数据来源 : {SPX_DIR}")
    logger.info(f"输出路径 : {FEATHER_PATH}")
    logger.info(f"缓存目录 : {PROCESSED_DATA_DIR}")
    if args.force:
        logger.info("⚠ --force 模式：忽略所有缓存，从头重算")
    elif args.force_from:
        logger.info(f"⚠ --force-from {args.force_from} 模式：从步骤 {args.force_from} 重算")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # ────────────────────────────────────────
    # 决定从哪个步骤开始
    # ────────────────────────────────────────
    cache4_ok = (force_from > 4) and os.path.exists(CACHE_STEP4_PATH)
    cache1_ok = (force_from > 1) and os.path.exists(CACHE_STEP1_PATH)

    if cache4_ok:
        # ── 直接从缓存4开始（只重算未来收益）──
        logger.info("\n✅ 检测到步骤4缓存，直接跳过步骤1-4")
        df = _load_feather(CACHE_STEP4_PATH, 'step4_precomputed')

    elif cache1_ok:
        # ── 从缓存1开始（步骤2-4需重算）──
        logger.info("\n✅ 检测到步骤1缓存，跳过 CSV 读取")
        df = _load_feather(CACHE_STEP1_PATH, 'step1_merged')

        logger.info("\n[2/5] 计算日收益率...")
        df = compute_ret(df)

        logger.info("[3/5] 计算市值代理...")
        df = compute_market_cap(df)

        logger.info("[4/5] 计算 EWMA 波动率...")
        df = compute_ewma_vol(df)

        logger.info("[缓存] 保存步骤4中间结果...")
        _save_feather(df, CACHE_STEP4_PATH, 'step4_precomputed')

    else:
        # ── 从步骤1开始（读取所有 CSV）──
        logger.info("\n[1/5] 读取与合并 CSV 文件（最耗时，首次运行约数分钟）...")
        df = read_and_merge()

        logger.info("[缓存] 保存步骤1中间结果...")
        _save_feather(df, CACHE_STEP1_PATH, 'step1_merged')

        logger.info("[2/5] 计算日收益率...")
        df = compute_ret(df)

        logger.info("[3/5] 计算市值代理...")
        df = compute_market_cap(df)

        logger.info("[4/5] 计算 EWMA 波动率...")
        df = compute_ewma_vol(df)

        logger.info("[缓存] 保存步骤4中间结果...")
        _save_feather(df, CACHE_STEP4_PATH, 'step4_precomputed')

    # ────────────────────────────────────────
    # 步骤 5：未来收益（始终运行或按缓存决定）
    # ────────────────────────────────────────
    logger.info("\n[5/5] 计算未来收益（已向量化，速度大幅提升）...")
    df = compute_future_returns(df)

    # ── 打印预览 ──
    logger.info(f"\n最终数据形状: {df.shape}")
    logger.info(f"列名: {list(df.columns)}")
    logger.info(f"\n数据预览（前 3 行）:\n{df.head(3).to_string()}")
    logger.info(f"\n股票数: {df['StockID'].nunique()}")
    logger.info(f"日期范围: {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    # ── 保存最终结果 ──
    logger.info(f"\n正在保存到 {FEATHER_PATH}...")
    _save_feather(df, FEATHER_PATH, 'us_ret_spx（最终）')
    logger.info("\n[OK] 步骤 1 完成！下一步请运行: python 2_generate_spx_chart_data.py")


if __name__ == '__main__':
    main()
