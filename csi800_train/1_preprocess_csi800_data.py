# -*- coding: utf-8 -*-
"""
1_preprocess_csi800_data.py - CSI800 数据预处理

从 CSI800_volume_price/ 读取中证800日线 CSV 文件，
目录结构：CSI800_volume_price/<YYYY>/<YYYYMMDD>.csv
每个 CSV 包含当日全部800只成分股数据（横截面）

与 SPX 版本的关键差异：
  1. CSV 中无 date 列，日期从文件名 YYYYMMDD 提取
  2. 列名为小写：open/high/low/close（SPX 为大写 OPEN/CLOSE/HIGH/LOW）
  3. 额外含 i_weight（指数权重）和 industry（行业）列
  4. A 股涨跌停 ±10%，收益率 clip 改为 ±0.11
  5. MarketCap 优先使用 i_weight，也支持 Vol × Close

执行以下处理后保存为 csi800_ret.feather：

  1. 读取与合并（按年份目录逐文件读取，注入 date）
  2. 列名标准化
  3. 日收益率 Ret（clip ±11%）
  4. 市值代理 MarketCap（优先 i_weight，否则 Vol × Close）
  5. EWMA 波动率
  6. 未来收益（Ret_5d, Ret_20d, Ret_60d, Ret_65d, Ret_180d, Ret_250d, Ret_260d,
               Ret_week, Ret_month, Ret_quarter, Ret_year）
  7. 写出 feather

─── 中间文件缓存策略 ────────────────────────────────────────────
  csi800_cache_step1_merged.feather     : 步骤1完成后（CSV读取合并）
  csi800_cache_step4_precomputed.feather: 步骤4完成后（Ret/MarketCap/EWMA，未来收益待算）

  运行逻辑：
    - 若 step4 缓存存在 → 直接跳到步骤5
    - 若 step1 缓存存在 → 从步骤2开始
    - 否则从步骤1（读取CSV）开始

  强制重跑：
    python 1_preprocess_csi800_data.py --force          # 从头全部重算
    python 1_preprocess_csi800_data.py --force-from 1   # 从步骤1重算（默认）
    python 1_preprocess_csi800_data.py --force-from 2   # 从步骤2重算（跳过CSV读取）
    python 1_preprocess_csi800_data.py --force-from 5   # 只重算未来收益
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    CSI800_DIR, CSI800_PROCESSED_DATA_DIR, CACHE_DIR,
    CSI800_CSV_ENCODING, CSI800_RET_CLIP
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
FEATHER_PATH     = os.path.join(CSI800_PROCESSED_DATA_DIR, 'csi800_ret.feather')
CACHE_STEP1_PATH = os.path.join(CSI800_PROCESSED_DATA_DIR, 'csi800_cache_step1_merged.feather')
CACHE_STEP4_PATH = os.path.join(CSI800_PROCESSED_DATA_DIR, 'csi800_cache_step4_precomputed.feather')


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
    logger.info(f"  [OK] [{label}] 已缓存 → {os.path.basename(path)}  ({size_mb:.1f} MB)")


def _load_feather(path: str, label: str) -> pd.DataFrame:
    """加载 feather 并恢复日期列类型"""
    logger.info(f"  [加载缓存] [{label}]: {os.path.basename(path)}")
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
    读取 CSI800_volume_price/<YYYY>/<YYYYMMDD>.csv 所有文件并合并为面板 DataFrame。

    与 SPX 的关键差异：
      - CSV 中无 date 列，日期从文件名（如 20200102.csv）中提取
      - 列名为小写：wind_code, sec_name, i_weight, industry, open, high, low, close, VOLUME
    """
    year_dirs = sorted([
        d for d in glob.glob(os.path.join(CSI800_DIR, '*'))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ])

    if not year_dirs:
        raise FileNotFoundError(
            f"未找到任何年份子目录于: {CSI800_DIR}\n"
            "请确认 CSI800_volume_price/ 目录存在且包含数据（如 2007/, 2008/, ...）。"
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
            # ── 关键差异1：从文件名提取日期 ──
            fname = os.path.basename(fp)  # 如 20200102.csv
            date_str = fname.replace('.csv', '')
            try:
                trade_date = pd.to_datetime(date_str, format='%Y%m%d')
            except ValueError:
                logger.warning(f"无法从文件名解析日期: {fname}，跳过。")
                continue

            try:
                df = pd.read_csv(fp, encoding=CSI800_CSV_ENCODING)
            except UnicodeDecodeError:
                for enc in ['gb18030', 'utf-8', 'latin1']:
                    try:
                        df = pd.read_csv(fp, encoding=enc)
                        break
                    except Exception:
                        continue
                else:
                    logger.warning(f"读取失败（编码）{fname}，跳过。")
                    continue
            except Exception as e:
                logger.warning(f"读取失败 {fname}: {e}，跳过。")
                continue

            # 注入日期列
            df['date'] = trade_date

            year_dfs.append(df)
            total_files += 1

        if year_dfs:
            year_df = pd.concat(year_dfs, ignore_index=True)
            dfs.append(year_df)

    if not dfs:
        raise RuntimeError("所有文件读取失败。")

    logger.info(f"共读取 {total_files} 个 CSV 文件，开始合并...")
    panel = pd.concat(dfs, ignore_index=True)

    # ── 关键差异2：列名映射（CSI800 列名为小写）──
    col_map = {
        'date':       'Date',
        'wind_code':  'StockID',
        'sec_name':   'sec_name',
        'open':       'Open',
        'close':      'Close',
        'high':       'High',
        'low':        'Low',
        'VOLUME':     'Vol',
        'i_weight':   'IndexWeight',   # 指数权重（可用于市值加权）
        'industry':   'Industry',      # 行业分类
    }
    panel = panel.rename(columns=col_map)

    # 确保日期列为 datetime
    panel['Date'] = pd.to_datetime(panel['Date'])

    # 去重
    panel = panel.drop_duplicates(subset=['Date', 'StockID'])

    # 排序
    panel = panel.sort_values(['StockID', 'Date']).reset_index(drop=True)

    # 过滤无效价格（含退市/停牌零价）
    panel = panel[panel['Close'] > 0].copy()

    logger.info(f"合并后总行数: {len(panel)}，"
                f"股票数: {panel['StockID'].nunique()}，"
                f"日期范围: {panel['Date'].min().date()} ~ {panel['Date'].max().date()}")
    return panel


# ────────────────────────────────────────────────
# 步骤 2：日收益率
# ────────────────────────────────────────────────
def compute_ret(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算日收益率。
    关键差异：A 股涨跌停 ±10%，clip 范围收紧至 ±CSI800_RET_CLIP（默认 0.11）
    """
    df['Ret'] = df.groupby('StockID')['Close'].pct_change()
    df['Ret'] = df['Ret'].fillna(0.0)
    df['Ret'] = df['Ret'].clip(-CSI800_RET_CLIP, CSI800_RET_CLIP)
    return df


# ────────────────────────────────────────────────
# 步骤 3：市值代理
# ────────────────────────────────────────────────
def compute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    MarketCap 计算策略：
      - 若含 IndexWeight 列且非全零 → 直接使用指数权重（更准确）
      - 否则回退到 Vol × Close（与 SPX 保持一致）

    按整体中位数缩放至 ~1e9 量级，便于模型权重计算。
    """
    if 'IndexWeight' in df.columns and df['IndexWeight'].fillna(0).gt(0).any():
        logger.info("  使用 IndexWeight（指数权重）作为 MarketCap 代理")
        df['MarketCap'] = df['IndexWeight'].fillna(0)
        # 缩放至 ~1e2 量级（原始权重为百分比 0~2 左右）
        median_val = df['MarketCap'][df['MarketCap'] > 0].median()
        if median_val > 0:
            df['MarketCap'] = df['MarketCap'] / median_val * 1e9
    else:
        logger.info("  IndexWeight 不可用，使用 Vol × Close 作为 MarketCap 代理")
        df['MarketCap'] = df['Vol'] * df['Close']
        median_val = df['MarketCap'].median()
        if median_val > 0:
            df['MarketCap'] = df['MarketCap'] / median_val * 1e9

    df['MarketCap'] = df['MarketCap'].fillna(0.0)
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
# 步骤 5：未来收益（向量化版本，与 SPX 完全一致）
# ────────────────────────────────────────────────
def compute_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算固定 horizon 与周期末未来收益。
    向量化实现，速度比循环版快约 3000x。
    """
    df['log_ret'] = np.log1p(df['Ret'])
    df['cum_log_ret'] = df.groupby('StockID')['log_ret'].cumsum()

    # ── 固定 horizon 未来收益 ──
    horizons = [5, 20, 60, 65, 180, 250, 260]
    for h in tqdm(horizons, desc='计算固定 horizon 收益'):
        col = f'Ret_{h}d'
        df[col] = (df.groupby('StockID')['cum_log_ret']
                    .transform(lambda x: np.exp(x.shift(-h) - x) - 1))

    # ── 周期末收益（向量化）──
    all_dates = df['Date'].unique()

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

            idx_end  = date_to_idx.get(end_ts)
            idx_next = date_to_idx.get(next_ts)
            if idx_end is None or idx_next is None:
                continue

            cum_end  = df.loc[idx_end].set_index('StockID')['cum_log_ret']
            cum_next = df.loc[idx_next].set_index('StockID')['cum_log_ret']

            common = cum_end.index.intersection(cum_next.index)
            if len(common) == 0:
                continue

            period_ret = np.exp(cum_next[common] - cum_end[common]) - 1
            df.loc[idx_end, col] = df.loc[idx_end, 'StockID'].map(period_ret)

    df = df.drop(columns=['log_ret', 'cum_log_ret'])
    return df


# ────────────────────────────────────────────────
# 数据验证
# ────────────────────────────────────────────────
def validate_output(df: pd.DataFrame):
    """输出基本统计信息，便于发现异常"""
    logger.info("=" * 60)
    logger.info("数据验证")
    logger.info("=" * 60)
    logger.info(f"总行数       : {len(df):,}")
    logger.info(f"股票数       : {df['StockID'].nunique()}")
    logger.info(f"日期范围     : {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    logger.info(f"列名         : {list(df.columns)}")

    # 收益率分布检查
    ret_stats = df['Ret'].describe()
    logger.info(f"\n日收益率统计:")
    logger.info(f"  均值={ret_stats['mean']:.5f}  std={ret_stats['std']:.5f}")
    logger.info(f"  min={ret_stats['min']:.4f}  max={ret_stats['max']:.4f}")

    # 检查是否有超出涨跌停的值
    over_limit = (df['Ret'].abs() > CSI800_RET_CLIP + 1e-6).sum()
    if over_limit > 0:
        logger.warning(f"  ! {over_limit} 行收益率超出 ±{CSI800_RET_CLIP} clip 范围（已裁剪）")
    else:
        logger.info(f"  收益率均在 ±{CSI800_RET_CLIP} 范围内")

    # 月度收益率检查
    if 'Ret_month' in df.columns:
        valid_month = df['Ret_month'].dropna()
        logger.info(f"\n月度收益率: {len(valid_month):,} 条有效值")
        logger.info(f"  均值={valid_month.mean():.4f}  std={valid_month.std():.4f}")

    # MarketCap 检查
    if 'MarketCap' in df.columns:
        mc_pos = (df['MarketCap'] > 0).mean()
        logger.info(f"\nMarketCap 正值比例: {mc_pos:.2%}")

    # 行业分布（若有）
    if 'Industry' in df.columns:
        n_industry = df['Industry'].nunique()
        logger.info(f"\n行业数: {n_industry}")
        top5 = df['Industry'].value_counts().head(5)
        logger.info(f"  Top5行业（按记录数）:\n{top5.to_string()}")

    logger.info("=" * 60)


# ────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='CSI800 数据预处理')
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
            '  5 = 从 step4 缓存只重算未来收益'
        )
    )
    parser.add_argument(
        '--years', type=str, default=None,
        metavar='START-END',
        help='只处理指定年份范围，如 --years 2015-2025'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    force_from = 1 if args.force else (args.force_from or 999)

    # 解析年份过滤
    year_filter = None
    if args.years:
        try:
            start_yr, end_yr = map(int, args.years.split('-'))
            year_filter = list(range(start_yr, end_yr + 1))
            logger.info(f"年份过滤: {year_filter[0]} ~ {year_filter[-1]}")
        except Exception:
            logger.warning(f"无法解析 --years 参数: {args.years}，将处理全部年份")

    logger.info("=" * 60)
    logger.info("  步骤 1：CSI800 数据预处理")
    logger.info("=" * 60)
    logger.info(f"数据来源  : {CSI800_DIR}")
    logger.info(f"输出路径  : {FEATHER_PATH}")
    logger.info(f"缓存目录  : {CSI800_PROCESSED_DATA_DIR}")
    logger.info(f"收益率clip: ±{CSI800_RET_CLIP}")
    if args.force:
        logger.info("--force 模式：忽略所有缓存，从头重算")
    elif args.force_from:
        logger.info(f"--force-from {args.force_from} 模式：从步骤 {args.force_from} 重算")

    os.makedirs(CSI800_PROCESSED_DATA_DIR, exist_ok=True)

    # ────────────────────────────────────────
    # 决定从哪个步骤开始
    # ────────────────────────────────────────
    cache4_ok = (force_from > 4) and os.path.exists(CACHE_STEP4_PATH)
    cache1_ok = (force_from > 1) and os.path.exists(CACHE_STEP1_PATH)

    if cache4_ok:
        logger.info("\n[缓存命中] 检测到步骤4缓存，直接跳过步骤1-4")
        df = _load_feather(CACHE_STEP4_PATH, 'csi800_step4_precomputed')

    elif cache1_ok:
        logger.info("\n[缓存命中] 检测到步骤1缓存，跳过 CSV 读取")
        df = _load_feather(CACHE_STEP1_PATH, 'csi800_step1_merged')

        # 应用年份过滤
        if year_filter:
            df = df[df['Date'].dt.year.isin(year_filter)].copy()
            logger.info(f"  年份过滤后行数: {len(df)}")

        logger.info("\n[2/5] 计算日收益率...")
        df = compute_ret(df)

        logger.info("[3/5] 计算市值代理...")
        df = compute_market_cap(df)

        logger.info("[4/5] 计算 EWMA 波动率...")
        df = compute_ewma_vol(df)

        logger.info("[缓存] 保存步骤4中间结果...")
        _save_feather(df, CACHE_STEP4_PATH, 'csi800_step4_precomputed')

    else:
        logger.info("\n[1/5] 读取与合并 CSV 文件（首次运行约数分钟）...")
        df = read_and_merge()

        # 应用年份过滤
        if year_filter:
            df = df[df['Date'].dt.year.isin(year_filter)].copy()
            logger.info(f"  年份过滤后行数: {len(df)}")

        logger.info("[缓存] 保存步骤1中间结果...")
        _save_feather(df, CACHE_STEP1_PATH, 'csi800_step1_merged')

        logger.info("[2/5] 计算日收益率...")
        df = compute_ret(df)

        logger.info("[3/5] 计算市值代理...")
        df = compute_market_cap(df)

        logger.info("[4/5] 计算 EWMA 波动率...")
        df = compute_ewma_vol(df)

        logger.info("[缓存] 保存步骤4中间结果...")
        _save_feather(df, CACHE_STEP4_PATH, 'csi800_step4_precomputed')

    # ────────────────────────────────────────
    # 步骤 5：未来收益
    # ────────────────────────────────────────
    logger.info("\n[5/5] 计算未来收益（向量化）...")
    df = compute_future_returns(df)

    # ── 验证输出 ──
    validate_output(df)

    # ── 保存最终结果 ──
    logger.info(f"\n正在保存到 {FEATHER_PATH}...")
    _save_feather(df, FEATHER_PATH, 'csi800_ret（最终）')

    logger.info("\n[OK] CSI800 预处理完成！")
    logger.info("     下一步请运行: python 2_generate_csi800_chart_data.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
