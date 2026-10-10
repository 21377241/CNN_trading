# -*- coding: utf-8 -*-
"""
5_analyze_ashare_topk_robust.py

A 股 TopK 稳健性：
  1) 每期按流通市值取 Top500 / 1000 / 2000 / 3000（及全市场对照）
  2) 剔除信号日涨停股（实际很难买到）后再按 up_prob 选 Top5/10/20

用法:
  python ashare_train/5_analyze_ashare_topk_robust.py --spec I20R20
  python ashare_train/5_analyze_ashare_topk_robust.py --spec I5R5
  python ashare_train/5_analyze_ashare_topk_robust.py --spec both
"""
import sys
import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    ASHARE_EXP_DIR_I20, ASHARE_EXP_DIR_I5, ASHARE_EXP_DIR_I20R5, ASHARE_OOS_YEARS,
    ASHARE_PROCESSED_DATA_DIR, N_ENSEMBLES
)
from Portfolio.portfolio import _load_results, run_topn_portfolio_analysis

logger = logging.getLogger(__name__)

SPECS = {
    'I20R20': dict(exp_dir=ASHARE_EXP_DIR_I20,    ann_factor=12, freq_name='月频'),
    'I5R5':   dict(exp_dir=ASHARE_EXP_DIR_I5,     ann_factor=52, freq_name='周频'),
    'I20R5':  dict(exp_dir=ASHARE_EXP_DIR_I20R5,  ann_factor=52, freq_name='周频'),
}

MCAP_UNIVERSES = [500, 1000, 2000, 3000]  # None 另作全市场
TOP_N_LIST = [5, 10, 20]
LIMIT_TOL = 0.005  # 相对涨跌停幅度的容差（复权舍入）
FEATHER_PATH = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_ret.feather')
CACHE_DAILY = os.path.join(ASHARE_PROCESSED_DATA_DIR, 'ashare_daily_ret_raw.feather')


def _norm_stock_id(x) -> str:
    s = str(x).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    s = s.split('.')[0]
    if s.isdigit():
        return s.zfill(6)
    return s


def _limit_pct_series(stock_ids: pd.Series, dates: pd.Series) -> np.ndarray:
    sid = stock_ids.map(_norm_stock_id)
    d = pd.to_datetime(dates)
    limit = np.full(len(sid), 0.10, dtype=float)
    s688 = sid.str.startswith('688')
    chi = sid.str.startswith('300') | sid.str.startswith('301')
    bj = sid.str.startswith('8') | sid.str.startswith('4')
    limit = np.where(s688, 0.20, limit)
    limit = np.where(chi & (d >= pd.Timestamp('2020-08-24')), 0.20, limit)
    limit = np.where(bj & (d >= pd.Timestamp('2023-09-04')), 0.30, limit)
    return limit


def _build_daily_ret_cache() -> pd.DataFrame:
    """用复权收盘价还原未 clip 的日收益，供涨停判定。"""
    if os.path.exists(CACHE_DAILY):
        logger.info(f"加载日收益缓存: {CACHE_DAILY}")
        out = pd.read_feather(CACHE_DAILY)
        out['Date'] = pd.to_datetime(out['Date'])
        out['StockID'] = out['StockID'].map(_norm_stock_id)
        return out

    logger.info(f"从 {FEATHER_PATH} 计算未裁剪日收益…")
    raw = pd.read_feather(FEATHER_PATH, columns=['Date', 'StockID', 'Close'])
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw['StockID'] = raw['StockID'].map(_norm_stock_id)
    raw = raw.sort_values(['StockID', 'Date'])
    raw['daily_ret_raw'] = raw.groupby('StockID')['Close'].pct_change()
    out = raw[['Date', 'StockID', 'daily_ret_raw']].dropna(subset=['daily_ret_raw'])
    os.makedirs(os.path.dirname(CACHE_DAILY), exist_ok=True)
    out.reset_index(drop=True).to_feather(CACHE_DAILY)
    logger.info(f"已缓存 {len(out):,} 行 → {CACHE_DAILY}")
    return out


def _attach_limit_up(pred: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    df = pred.copy()
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df['StockID'] = df['StockID'].map(_norm_stock_id)
    merged = df.merge(
        daily, left_on=['ending_date', 'StockID'],
        right_on=['Date', 'StockID'], how='left'
    )
    ret = merged['daily_ret_raw']
    if 'Ret' in merged.columns:
        ret = ret.fillna(merged['Ret'])
    merged['limit_pct'] = _limit_pct_series(merged['StockID'], merged['ending_date'])
    merged['is_limit_up'] = ret >= (merged['limit_pct'] - LIMIT_TOL)
    n_up = int(merged['is_limit_up'].sum())
    logger.info(f"信号日涨停: {n_up:,} / {len(merged):,} "
                f"({n_up / max(len(merged), 1):.2%})")
    return merged.drop(columns=['Date'], errors='ignore')


def _filter_mcap_universe(df: pd.DataFrame, top_m: int = None) -> pd.DataFrame:
    if top_m is None:
        return df
    parts = []
    for _, g in df.groupby('ending_date'):
        g = g[g['MarketCap'] > 0]
        parts.append(g.nlargest(min(top_m, len(g)), 'MarketCap'))
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def find_files(exp_dir, years):
    pattern = os.path.join(exp_dir, '**', 'ensem_res', f'ensem{N_ENSEMBLES}_res_*.csv')
    all_f = glob.glob(pattern, recursive=True)
    found = []
    for y in years:
        ys = [f for f in all_f if f'_res_{y}.csv' in f]
        if ys:
            ys.sort(key=os.path.getmtime, reverse=True)
            found.append(ys[0])
    return found


def _universe_label(top_m):
    return 'all' if top_m is None else f'mcap{top_m}'


def run_one_spec(spec_name: str) -> pd.DataFrame:
    spec = SPECS[spec_name]
    exp_dir = spec['exp_dir']
    ann = spec['ann_factor']
    files = find_files(exp_dir, ASHARE_OOS_YEARS)
    if not files:
        logger.error(f"{spec_name}: 未找到预测文件")
        return pd.DataFrame()

    pred = _load_results(files)
    daily = _build_daily_ret_cache()
    pred = _attach_limit_up(pred, daily)

    root = os.path.join(exp_dir, 'topN_robust')
    os.makedirs(root, exist_ok=True)

    rows = []
    universes = [None] + MCAP_UNIVERSES
    for top_m in universes:
        uni = _universe_label(top_m)
        tradable = pred.loc[~pred['is_limit_up']].copy()
        tradable = _filter_mcap_universe(tradable, top_m)
        n_dates = tradable['ending_date'].nunique()
        n_avg = tradable.groupby('ending_date').size().mean() if len(tradable) else 0
        n_limit_dropped = int(pred['is_limit_up'].sum()) if top_m is None else np.nan

        logger.info("=" * 60)
        logger.info(f"  {spec_name}  universe={uni}  可买样本={len(tradable):,}  "
                    f"期数={n_dates}  每期约{n_avg:.0f}只")
        logger.info("=" * 60)

        out_dir = os.path.join(root, uni)
        note = (
            f"{spec['freq_name']}；宇宙={('全市场' if top_m is None else f'市值Top{top_m}')}；"
            f"已剔除信号日涨停（板块涨停幅度-0.5pct）；TopK={TOP_N_LIST}"
        )
        run_topn_portfolio_analysis(
            df=tradable,
            output_dir=out_dir,
            n_ensembles=N_ENSEMBLES,
            top_n_list=TOP_N_LIST,
            ann_factor=ann,
            note=note,
        )

        ew_path = os.path.join(out_dir, 'topN_ew_portfolio.csv')
        vw_path = os.path.join(out_dir, 'topN_vw_portfolio.csv')
        if not os.path.exists(ew_path):
            continue
        ew = pd.read_csv(ew_path, index_col=0)
        vw = pd.read_csv(vw_path, index_col=0) if os.path.exists(vw_path) else pd.DataFrame()
        for k in TOP_N_LIST:
            name = f'Top{k}'
            if name not in ew.index:
                continue
            rec = {
                'spec': spec_name,
                'universe': uni,
                'mcap_n': top_m if top_m is not None else 0,
                'top_k': k,
                'n_dates': n_dates,
                'avg_names': n_avg,
                'ew_mean': ew.loc[name, 'mean_monthly'],
                'ew_ann': ew.loc[name, 'ann_return'],
                'ew_vol': ew.loc[name, 'ann_vol'],
                'ew_sharpe': ew.loc[name, 'sharpe'],
            }
            if name in vw.index:
                rec.update({
                    'vw_mean': vw.loc[name, 'mean_monthly'],
                    'vw_ann': vw.loc[name, 'ann_return'],
                    'vw_vol': vw.loc[name, 'ann_vol'],
                    'vw_sharpe': vw.loc[name, 'sharpe'],
                })
            rows.append(rec)

    summary = pd.DataFrame(rows)
    sum_path = os.path.join(root, 'topk_robust_summary.csv')
    summary.to_csv(sum_path, index=False, float_format='%.6f')
    logger.info(f"汇总: {sum_path}")
    return summary


def _write_combined_report(all_sum: pd.DataFrame, path: str):
    lines = [
        "# A股 TopK 稳健性：市值宇宙 × 剔除涨停",
        "",
        "规则：",
        "- 涨停：信号日（ending_date）未裁剪日收益 ≥ 该股当日涨停幅度 − 0.5pct；"
        "主板 10%，创业板 2020-08-24 后 20%，科创板 20%，北交所 2023-09-04 后 30%。",
        "- 市值宇宙：每期按流通市值取最大的 500/1000/2000/3000 只，再在可买集合内按 up_prob 选 Top5/10/20。",
        "- 年化：I20R20 用 (1+μ)^12−1；I5R5 / I20R5 用 (1+μ)^52−1（周频复利，绝对年化仍会偏大，优先看 Sharpe / 期均）。",
        "",
    ]
    for spec in all_sum['spec'].unique():
        sub = all_sum[all_sum['spec'] == spec]
        lines.append(f"## {spec}")
        lines.append("")
        lines.append("| 宇宙 | K | 期数 | EW期均 | EW年化 | EW Sharpe | VW年化 | VW Sharpe |")
        lines.append("|------|---|------|--------|--------|-----------|--------|-----------|")
        for _, r in sub.sort_values(['mcap_n', 'top_k']).iterrows():
            uni = '全市场' if r['universe'] == 'all' else r['universe']
            lines.append(
                f"| {uni} | {int(r['top_k'])} | {int(r['n_dates'])} | "
                f"{r['ew_mean']:.4f} | {r['ew_ann']:.4f} | {r['ew_sharpe']:.3f} | "
                f"{r.get('vw_ann', float('nan')):.4f} | {r.get('vw_sharpe', float('nan')):.3f} |"
            )
        lines.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"对照报告: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', choices=list(SPECS.keys()) + ['both', 'all'], default='both')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    specs = list(SPECS.keys()) if args.spec in ('both', 'all') else [args.spec]
    frames = []
    for s in specs:
        frames.append(run_one_spec(s))
    all_sum = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True)
    if all_sum.empty:
        return 1

    docs = os.path.join(PROJECT_ROOT, 'docs', 'repro')
    os.makedirs(docs, exist_ok=True)
    csv_path = os.path.join(docs, 'ashare_topk_robust_summary.csv')
    if os.path.exists(csv_path) and args.spec not in ('both', 'all'):
        prev = pd.read_csv(csv_path)
        prev = prev[prev['spec'] != args.spec]
        all_sum = pd.concat([prev, all_sum], ignore_index=True)
    all_sum.to_csv(csv_path, index=False, float_format='%.6f')
    _write_combined_report(
        all_sum, os.path.join(docs, 'ashare_topk_robust.md')
    )
    logger.info("[OK] TopK 稳健性分析完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())
