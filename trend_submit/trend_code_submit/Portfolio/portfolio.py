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
PORT_INPUT_COLS = ['StockID', 'ending_date', 'up_prob', 'ret_val', 'MarketCap']


def write_port_input(df: pd.DataFrame, out_dir: str) -> str:
    """组合分析的输入只保留必要列，存为 parquet（旧版每个组合目录写一份完整 CSV，占用数百 GB）。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pred_for_port.parquet')
    out = df[[c for c in PORT_INPUT_COLS if c in df.columns]].copy()
    if 'StockID' in out.columns:
        out['StockID'] = out['StockID'].astype(str)
    out.to_parquet(path, index=False)
    return path


def _read_result_file(fp: str) -> pd.DataFrame:
    return pd.read_parquet(fp) if fp.endswith('.parquet') else pd.read_csv(fp)


def _load_results(result_files: list) -> pd.DataFrame:
    """读取并合并多个 ensem*_res_*.csv 或 pred_for_port.parquet"""
    dfs = []
    for fp in result_files:
        if not os.path.exists(fp):
            logger.warning(f"文件不存在: {fp}")
            continue
        try:
            df = _read_result_file(fp)
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
    std_m = float(ret.std())
    ann_r  = (1 + mean_m) ** ann_factor - 1
    ann_v  = float(std_m * np.sqrt(ann_factor))
    # Sharpe uses the annualized arithmetic mean, not compounded annual return.
    # Keep ann_r as the separately reported compounded return.
    sharpe = (mean_m * np.sqrt(ann_factor) / std_m) if std_m > 1e-10 else 0.0
    return {'mean_monthly': mean_m, 'ann_return': ann_r,
            'ann_vol': ann_v, 'sharpe': sharpe, 'n': len(ret)}


def _compute_topn_returns(df: pd.DataFrame,
                         ret_col: str = 'ret_val',
                         up_col: str = 'up_prob',
                         weight_col: str = None,
                         top_n_list: list = [5, 10, 20]) -> tuple:
    """
    计算各期末日期、各TopN组合的等权或市值加权收益。
    选股明细只记录一次（以 max(top_n_list) 为上限），用 rank 列区分层级：
      rank <= 5  → Top5
      rank <= 10 → Top10
      rank <= 20 → Top20

    Args:
        df: 含 ending_date, ret_col, up_col, [weight_col] 的 DataFrame
        ret_col: 收益列名
        up_col: 排序概率列名（越大越看涨）
        weight_col: 市值加权列名（None 表示等权）
        top_n_list: TopN股票数量列表，如[5, 10, 20]

    Returns:
        (pivot, selections_df)
        - pivot: DataFrame (date × topN)，每格为该期该TopN组合的收益
        - selections_df: 每行为某月 Top{max_n} 中的一只股票，含 rank / in_top5 / in_top10 等标记列
    """
    df = df.copy()
    df = df.dropna(subset=[ret_col, up_col])

    max_n = max(top_n_list)
    results = []
    selections = []

    for date, grp in df.groupby('ending_date'):
        if len(grp) < 1:
            continue

        # 按up_prob降序排序
        grp_sorted = grp.sort_values(up_col, ascending=False).reset_index(drop=True)
        grp_sorted['rank'] = grp_sorted.index + 1  # 1-based

        # ── 计算各 TopN 的组合收益 ──
        for n in top_n_list:
            top_n_stocks = grp_sorted.head(n)
            if len(top_n_stocks) == 0:
                continue

            if weight_col and weight_col in top_n_stocks.columns:
                w = top_n_stocks[weight_col].values
                w = np.where(w > 0, w, 0)
                w_sum = w.sum()
                ret = (np.average(top_n_stocks[ret_col].values, weights=w)
                       if w_sum > 0 else top_n_stocks[ret_col].mean())
            else:
                ret = top_n_stocks[ret_col].mean()

            results.append({'date': date, 'top_n': f'Top{n}', 'ret': ret,
                            'n_stocks': len(top_n_stocks)})

        # ── 选股明细：只写 max_n 条，附带 in_topX 标记 ──
        top_max = grp_sorted.head(max_n)
        for _, row in top_max.iterrows():
            rk = int(row['rank'])
            rec = {
                'date':    date,
                'rank':    rk,
                'StockID': row.get('StockID', ''),
                'up_prob': round(float(row[up_col]), 6),
                ret_col:   round(float(row[ret_col]), 6),
            }
            for n in top_n_list:
                rec[f'in_top{n}'] = (rk <= n)
            if 'MarketCap' in row:
                rec['MarketCap'] = row['MarketCap']
            if 'EWMA_vol' in row:
                rec['EWMA_vol'] = row['EWMA_vol']
            selections.append(rec)

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    pivot = pd.DataFrame(results).pivot(index='date', columns='top_n', values='ret')
    selections_df = pd.DataFrame(selections).sort_values(['date', 'rank']).reset_index(drop=True)

    return pivot.sort_index(), selections_df


def run_topn_portfolio_analysis(result_files: list = None,
                                 output_dir: str = None,
                                 n_ensembles: int = 5,
                                 top_n_list: list = [5, 10, 20],
                                 df: pd.DataFrame = None,
                                 ann_factor: int = 12,
                                 note: str = None):
    """
    TopN投资组合分析流程。
    选择up_prob最高的N只股票构建组合，计算等权和市值加权的收益。

    Args:
        result_files: ensem*_res_*.csv 文件路径列表（与 df 二选一）
        df: 直接传入已加载/过滤好的 DataFrame（与 result_files 二选一）
        output_dir: 输出目录（报告、CSV、图表）
        n_ensembles: 集成模型数（用于命名）
        top_n_list: TopN股票数量列表，如[5, 10, 20]
        note: 写入报告的额外说明（宇宙、剔涨停等）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据（支持直接传入 DataFrame 或文件路径列表）
    if df is None:
        df = _load_results(result_files)
    else:
        df = df.copy()
        df['ending_date'] = pd.to_datetime(df['ending_date'])
    logger.info(f"总样本数: {len(df)}，覆盖期末日期: {df['ending_date'].nunique()} 个")

    # 2. 计算等权与市值加权TopN收益（选股名单相同，仅权重不同，明细只需保存一份）
    ew_pivot, selections = _compute_topn_returns(
        df, ret_col='ret_val', up_col='up_prob',
        weight_col=None, top_n_list=top_n_list)
    vw_pivot, _ = _compute_topn_returns(
        df, ret_col='ret_val', up_col='up_prob',
        weight_col='MarketCap', top_n_list=top_n_list)

    if ew_pivot.empty:
        logger.error("TopN收益计算结果为空，请检查数据。")
        return

    # 3. 计算统计量
    cols = [f'Top{n}' for n in top_n_list]
    ew_stats = {}
    vw_stats = {}
    for c in cols:
        if c in ew_pivot.columns:
            ew_stats[c] = _compute_stats(ew_pivot[c], ann_factor=ann_factor)
        if not vw_pivot.empty and c in vw_pivot.columns:
            vw_stats[c] = _compute_stats(vw_pivot[c], ann_factor=ann_factor)

    # 4. 输出汇总 CSV
    ew_stats_df = pd.DataFrame(ew_stats).T
    vw_stats_df = pd.DataFrame(vw_stats).T if vw_stats else pd.DataFrame()

    ew_stats_df.to_csv(os.path.join(output_dir, 'topN_ew_portfolio.csv'))
    if not vw_stats_df.empty:
        vw_stats_df.to_csv(os.path.join(output_dir, 'topN_vw_portfolio.csv'))

    ew_pivot.to_csv(os.path.join(output_dir, 'topN_ew_monthly_returns.csv'))
    if not vw_pivot.empty:
        vw_pivot.to_csv(os.path.join(output_dir, 'topN_vw_monthly_returns.csv'))

    # 4b. 输出选股明细 CSV（EW/VW 选股名单相同，只保存一份）
    max_n = max(top_n_list)
    if not selections.empty:
        sel_path = os.path.join(output_dir, 'topN_selections.csv')
        selections.to_csv(sel_path, index=False)
        logger.info(f"选股明细已保存: topN_selections.csv "
                    f"({len(selections)} 条，每月 Top{max_n}，用 in_topX 列过滤)")

    # 5. 写报告文本
    _write_topn_report(ew_stats, vw_stats, output_dir, top_n_list, note=note)

    # 6. 绘图
    _plot_topn_returns(ew_stats, vw_stats, output_dir, top_n_list)
    _plot_topn_sharpe(ew_stats, vw_stats, output_dir, top_n_list)
    _plot_topn_cumulative(ew_pivot, vw_pivot, output_dir, top_n_list)
    _plot_topn_timeseries(ew_pivot, output_dir, top_n_list)

    logger.info(f"TopN投资组合分析完成，结果保存至: {output_dir}")


def run_portfolio_analysis(result_files: list,
                            output_dir: str,
                            n_ensembles: int = 5,
                            n_decile: int = 10,
                            ann_factor: int = 12):
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
            ew_stats[c] = _compute_stats(ew_pivot[c], ann_factor=ann_factor)
        if not vw_pivot.empty and c in vw_pivot.columns:
            vw_stats[c] = _compute_stats(vw_pivot[c], ann_factor=ann_factor)

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


def _write_topn_report(ew_stats: dict, vw_stats: dict, output_dir: str,
                       top_n_list: list, note: str = None):
    """写TopN投资组合文本报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("      CNN 股票收益预测 - TopN 投资组合分析报告")
    lines.append("=" * 60)
    lines.append(f"\n分析参数: TopN = {top_n_list}")
    lines.append("策略说明: 每期按 up_prob 最高的 N 只股票构建组合")
    if note:
        lines.append(f"设定说明: {note}")

    for title, stats in [("等权 (Equal-Weight)", ew_stats),
                          ("市值加权 (Value-Weight)", vw_stats)]:
        if not stats:
            continue
        lines.append(f"\n{'─' * 60}")
        lines.append(f"{title}：")
        lines.append(f"{'组合':>10} {'月均收益':>12} {'年化收益':>12} {'年化波动':>12} {'Sharpe':>10} {'样本数':>8}")
        lines.append("-" * 70)
        for c in [f'Top{n}' for n in top_n_list]:
            if c not in stats:
                continue
            s = stats[c]
            lines.append(f"{c:>10} "
                          f"{s['mean_monthly']:>12.4f} "
                          f"{s['ann_return']:>12.4f} "
                          f"{s['ann_vol']:>12.4f} "
                          f"{s['sharpe']:>10.4f} "
                          f"{s['n']:>8}")

    report_path = os.path.join(output_dir, 'topN_portfolio_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"TopN报告已保存: {report_path}")


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


# ─────────────────────────────────────────────
# TopN 绘图函数
# ─────────────────────────────────────────────
def _plot_topn_returns(ew_stats, vw_stats, output_dir, top_n_list):
    """TopN: 各组合年化收益柱状图"""
    cols = [f'Top{n}' for n in top_n_list]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TopN 组合年化收益', fontsize=14)

    for ax, (title, stats) in zip(axes, [('等权', ew_stats), ('市值加权', vw_stats)]):
        if not stats:
            ax.set_visible(False)
            continue
        x = range(len(cols))
        vals = [stats.get(c, {}).get('ann_return', 0) * 100 for c in cols]
        colors = ['red' if v < 0 else 'steelblue' for v in vals]
        bars = ax.bar(x, vals, color=colors, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('组合')
        ax.set_ylabel('年化收益 (%)')
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cols)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

        # 在柱状图上添加数值标签
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_1_returns.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"TopN图表已保存: {path}")


def _plot_topn_sharpe(ew_stats, vw_stats, output_dir, top_n_list):
    """TopN: 各组合Sharpe比率柱状图"""
    cols = [f'Top{n}' for n in top_n_list]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TopN 组合 Sharpe 比率', fontsize=14)

    for ax, (title, stats) in zip(axes, [('等权', ew_stats), ('市值加权', vw_stats)]):
        if not stats:
            ax.set_visible(False)
            continue
        x = range(len(cols))
        vals = [stats.get(c, {}).get('sharpe', 0) for c in cols]
        colors = ['red' if v < 0 else 'steelblue' for v in vals]
        bars = ax.bar(x, vals, color=colors, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('组合')
        ax.set_ylabel('Sharpe')
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cols)

        # 在柱状图上添加数值标签
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_2_sharpe.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"TopN图表已保存: {path}")


def _plot_topn_cumulative(ew_pivot, vw_pivot, output_dir, top_n_list):
    """TopN: 各组合累积净值对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('TopN 组合累积净值对比', fontsize=14)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, col in enumerate([f'Top{n}' for n in top_n_list]):
        if col in ew_pivot.columns:
            cum_ret = (1 + ew_pivot[col].fillna(0)).cumprod()
            ax.plot(cum_ret.index, cum_ret.values,
                    label=f'{col} (EW)', color=colors[i % len(colors)], linewidth=2)

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('日期')
    ax.set_ylabel('累积净值')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_3_cumulative.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"TopN图表已保存: {path}")


def _plot_topn_timeseries(ew_pivot, output_dir, top_n_list):
    """TopN: 各组合月度收益时间序列"""
    n_plots = len(top_n_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    fig.suptitle('TopN 组合月度收益时间序列', fontsize=13)

    if n_plots == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'green']

    for ax, n, color in zip(axes, top_n_list, colors):
        col = f'Top{n}'
        if col in ew_pivot.columns:
            ret = ew_pivot[col].fillna(0)
            ax.bar(ret.index, ret.values * 100, color=color, alpha=0.6)
            ax.axhline(0, color='black', linewidth=0.6)
            # 添加均线
            rolling_mean = ret.rolling(window=3).mean() * 100
            ax.plot(ret.index, rolling_mean, color='red', linewidth=1.5, label='3期移动平均')
        ax.set_title(f'{col}（等权）', fontsize=10)
        ax.set_ylabel('月收益 (%)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_4_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"TopN图表已保存: {path}")
