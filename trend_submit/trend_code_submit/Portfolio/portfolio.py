# -*- coding: utf-8 -*-
"""
Portfolio/portfolio.py - 投资组合分析

主要函数：
  run_portfolio_analysis(result_csv_list, output_dir)
    - 读取 ensem5_res_*.csv
    - 按 up_prob 十分位分组
    - 计算等权 / 市值加权 / H-L 月度收益、年化收益、Sharpe
    - 输出报告文本、CSV、4 张图
"""
import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Misc.utilities import annualize_return, annualize_sharpe

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 核心分析函数
# ─────────────────────────────────────────────
def _load_results(result_files: list) -> pd.DataFrame:
    """读取并合并多个 ensem*_res_*.csv"""
    dfs = []
    for fp in result_files:
        if not os.path.exists(fp):
            logger.warning(f"文件不存在: {fp}")
            continue
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
            logger.info(f"加载: {fp}，行数: {len(df)}")
        except Exception as e:
            logger.error(f"读取失败 {fp}: {e}")
    if not dfs:
        raise RuntimeError("无可用预测结果文件！")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['ending_date'] = pd.to_datetime(all_df['ending_date'])
    return all_df


def _compute_decile_returns(df: pd.DataFrame,
                             ret_col: str = 'ret_val',
                             up_col: str = 'up_prob',
                             weight_col: str = None,
                             n_decile: int = 10) -> pd.DataFrame:
    """
    计算各期末日期、各十分位的等权或市值加权收益。
    当股票数不足 n_decile 时自动降低分位数。

    Args:
        df: 含 ending_date, ret_col, up_col, [weight_col] 的 DataFrame
        ret_col: 收益列名
        up_col: 排序概率列名（越大越看涨）
        weight_col: 市值加权列名（None 表示等权）
        n_decile: 最大分位数数量

    Returns:
        DataFrame (date × decile)，每格为该期该分位的收益
    """
    df = df.copy()
    df = df.dropna(subset=[ret_col, up_col])

    # 自动确定合适的分位数（至少每组1个）
    min_grp_size = df.groupby('ending_date').size().min() if len(df) > 0 else 0
    actual_n = min(n_decile, max(2, min_grp_size))
    if actual_n < n_decile:
        logger.warning(f"每期股票数不足 {n_decile}，自动调整分位数为 {actual_n}")

    # 按日期分组，对 up_prob 做十分位
    results = []
    for date, grp in df.groupby('ending_date'):
        if len(grp) < 2:
            continue
        local_n = min(actual_n, len(grp))
        try:
            grp = grp.copy()
            grp['decile'] = pd.qcut(grp[up_col], local_n,
                                    labels=range(1, local_n + 1),
                                    duplicates='drop')
        except ValueError:
            # 若 qcut 失败（数值相同），用 rank 代替
            try:
                grp = grp.copy()
                grp['decile'] = pd.cut(grp[up_col].rank(method='first'),
                                       bins=local_n,
                                       labels=range(1, local_n + 1))
            except Exception:
                continue

        for d in range(1, n_decile + 1):
            sub = grp[grp['decile'] == d]
            if len(sub) == 0:
                continue
            if weight_col and weight_col in sub.columns:
                w = sub[weight_col].values
                w = np.where(w > 0, w, 0)
                w_sum = w.sum()
                if w_sum > 0:
                    ret = np.average(sub[ret_col].values, weights=w)
                else:
                    ret = sub[ret_col].mean()
            else:
                ret = sub[ret_col].mean()
            results.append({'date': date, 'decile': d, 'ret': ret})

    if not results:
        return pd.DataFrame()

    pivot = pd.DataFrame(results).pivot(index='date', columns='decile', values='ret')
    pivot.columns = [f'D{c}' for c in pivot.columns]
    return pivot.sort_index()


def _compute_stats(ret_series: pd.Series, ann_factor: int = 12) -> dict:
    """计算收益序列的统计量"""
    ret = ret_series.dropna()
    if len(ret) == 0:
        return {'mean_monthly': np.nan, 'ann_return': np.nan,
                'ann_vol': np.nan, 'sharpe': np.nan, 'n': 0}
    mean_m = float(ret.mean())
    ann_r  = (1 + mean_m) ** ann_factor - 1
    ann_v  = float(ret.std() * np.sqrt(ann_factor))
    sharpe = ann_r / ann_v if ann_v > 1e-10 else 0.0
    return {'mean_monthly': mean_m, 'ann_return': ann_r,
            'ann_vol': ann_v, 'sharpe': sharpe, 'n': len(ret)}


def run_portfolio_analysis(result_files: list,
                            output_dir: str,
                            n_ensembles: int = 5,
                            n_decile: int = 10):
    """
    完整投资组合分析流程。

    Args:
        result_files: ensem*_res_*.csv 文件路径列表
        output_dir: 输出目录（报告、CSV、图表）
        n_ensembles: 集成模型数（用于命名）
        n_decile: 十分位数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    df = _load_results(result_files)
    logger.info(f"总样本数: {len(df)}，覆盖期末日期: {df['ending_date'].nunique()} 个")

    # 2. 计算等权与市值加权十分位收益
    ew_pivot = _compute_decile_returns(df, ret_col='ret_val', up_col='up_prob',
                                        weight_col=None, n_decile=n_decile)
    vw_pivot = _compute_decile_returns(df, ret_col='ret_val', up_col='up_prob',
                                        weight_col='MarketCap', n_decile=n_decile)

    if ew_pivot.empty:
        logger.error("十分位收益计算结果为空，请检查数据（每期股票数可能不足）。")
        return

    # 实际分位数（可能因样本少而调整）
    actual_deciles = [c for c in ew_pivot.columns if c.startswith('D')]
    if actual_deciles:
        n_decile = int(actual_deciles[-1][1:])
        logger.info(f"实际使用分位数: {n_decile}")

    # 3. 添加 H-L 列
    d1 = f'D1'
    d10 = f'D{n_decile}'
    ew_pivot['H-L'] = ew_pivot.get(d10, 0) - ew_pivot.get(d1, 0)
    if not vw_pivot.empty:
        vw_pivot['H-L'] = vw_pivot.get(d10, 0) - vw_pivot.get(d1, 0)

    # 4. 计算统计量
    cols = [f'D{i}' for i in range(1, n_decile + 1)] + ['H-L']
    ew_stats = {}
    vw_stats = {}
    for c in cols:
        if c in ew_pivot.columns:
            ew_stats[c] = _compute_stats(ew_pivot[c])
        if not vw_pivot.empty and c in vw_pivot.columns:
            vw_stats[c] = _compute_stats(vw_pivot[c])

    # 5. 输出 CSV
    ew_stats_df = pd.DataFrame(ew_stats).T
    vw_stats_df = pd.DataFrame(vw_stats).T if vw_stats else pd.DataFrame()

    ew_stats_df.to_csv(os.path.join(output_dir, 'ew_portfolio.csv'))
    if not vw_stats_df.empty:
        vw_stats_df.to_csv(os.path.join(output_dir, 'vw_portfolio.csv'))

    ew_pivot.to_csv(os.path.join(output_dir, 'ew_monthly_returns.csv'))
    if not vw_pivot.empty:
        vw_pivot.to_csv(os.path.join(output_dir, 'vw_monthly_returns.csv'))

    # 6. 写报告文本
    _write_report(ew_stats, vw_stats, output_dir, n_decile)

    # 7. 绘图
    _plot_decile_returns(ew_stats, vw_stats, output_dir, n_decile)
    _plot_sharpe_ratios(ew_stats, vw_stats, output_dir, n_decile)
    _plot_hl_comparison(ew_pivot, vw_pivot, output_dir)
    _plot_monthly_timeseries(ew_pivot, output_dir, n_decile)

    logger.info(f"投资组合分析完成，结果保存至: {output_dir}")


# ─────────────────────────────────────────────
# 报告文本
# ─────────────────────────────────────────────
def _write_report(ew_stats: dict, vw_stats: dict, output_dir: str, n_decile: int):
    """写文本报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("         CNN 股票收益预测 - 投资组合分析报告")
    lines.append("=" * 60)

    for title, stats in [("等权 (Equal-Weight)", ew_stats),
                          ("市值加权 (Value-Weight)", vw_stats)]:
        if not stats:
            continue
        lines.append(f"\n{title}：")
        lines.append(f"{'组合':>8} {'月均收益':>10} {'年化收益':>10} {'年化波动':>10} {'Sharpe':>10} {'样本数':>8}")
        lines.append("-" * 60)
        for c in [f'D{i}' for i in range(1, n_decile + 1)] + ['H-L']:
            if c not in stats:
                continue
            s = stats[c]
            lines.append(f"{c:>8} "
                          f"{s['mean_monthly']:>10.4f} "
                          f"{s['ann_return']:>10.4f} "
                          f"{s['ann_vol']:>10.4f} "
                          f"{s['sharpe']:>10.4f} "
                          f"{s['n']:>8}")

    report_path = os.path.join(output_dir, 'portfolio_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"报告已保存: {report_path}")


# ─────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────
def _plot_decile_returns(ew_stats, vw_stats, output_dir, n_decile):
    """图1: 十分位年化收益柱状图"""
    dcols = [f'D{i}' for i in range(1, n_decile + 1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('各十分位年化收益', fontsize=14)

    for ax, (title, stats) in zip(axes, [('等权', ew_stats), ('市值加权', vw_stats)]):
        if not stats:
            ax.set_visible(False)
            continue
        x = range(1, n_decile + 1)
        vals = [stats.get(c, {}).get('ann_return', 0) * 100 for c in dcols]
        colors = ['red' if v < 0 else 'steelblue' for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('十分位')
        ax.set_ylabel('年化收益 (%)')
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    plt.tight_layout()
    path = os.path.join(output_dir, '1_decile_returns.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存: {path}")


def _plot_sharpe_ratios(ew_stats, vw_stats, output_dir, n_decile):
    """图2: 十分位 Sharpe 比率柱状图"""
    dcols = [f'D{i}' for i in range(1, n_decile + 1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('各十分位 Sharpe 比率', fontsize=14)

    for ax, (title, stats) in zip(axes, [('等权', ew_stats), ('市值加权', vw_stats)]):
        if not stats:
            ax.set_visible(False)
            continue
        x = range(1, n_decile + 1)
        vals = [stats.get(c, {}).get('sharpe', 0) for c in dcols]
        colors = ['red' if v < 0 else 'steelblue' for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('十分位')
        ax.set_ylabel('Sharpe')
        ax.set_title(title)
        ax.set_xticks(list(x))

    plt.tight_layout()
    path = os.path.join(output_dir, '2_sharpe_ratios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存: {path}")


def _plot_hl_comparison(ew_pivot, vw_pivot, output_dir):
    """图3: H-L 策略净值对比"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('H-L 策略累积净值')

    if 'H-L' in ew_pivot.columns:
        cum_ew = (1 + ew_pivot['H-L'].fillna(0)).cumprod()
        ax.plot(cum_ew.index, cum_ew.values, label='等权 H-L', color='steelblue')

    if not vw_pivot.empty and 'H-L' in vw_pivot.columns:
        cum_vw = (1 + vw_pivot['H-L'].fillna(0)).cumprod()
        ax.plot(cum_vw.index, cum_vw.values, label='市值加权 H-L',
                color='darkorange', linestyle='--')

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('日期')
    ax.set_ylabel('累积净值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, '3_hl_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存: {path}")


def _plot_monthly_timeseries(ew_pivot, output_dir, n_decile):
    """图4: D1 / D5 / D10 / H-L 月度收益时间序列"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('各十分位月度收益时间序列', fontsize=13)

    mid = n_decile // 2
    plot_cols = [('D1', 'D1（最低概率）', 'red'),
                 (f'D{mid}', f'D{mid}（中位数）', 'gray'),
                 (f'D{n_decile}', f'D{n_decile}（最高概率）', 'steelblue'),
                 ('H-L', 'H-L', 'green')]

    for ax, (col, title, color) in zip(axes, plot_cols):
        if col in ew_pivot.columns:
            ret = ew_pivot[col].fillna(0)
            ax.bar(ret.index, ret.values * 100, color=color, alpha=0.6)
            ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('月收益 (%)')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, '4_monthly_returns_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存: {path}")

