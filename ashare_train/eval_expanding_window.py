# -*- coding: utf-8 -*-
"""评估训练期扩展窗口 OOS：逐年拼预测 → 截面 Rank IC → 与三模型对照。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.metrics import (  # noqa: E402
    cross_section_ic,
    newey_west_t,
    summarize_ic,
    yearly_rank_ic_mean,
)

EW_ROOT = PROJECT_ROOT / 'WORK_SPACE/paper/expanding_window'
OOS_YEARS = list(range(2008, 2026))
KEEP = ['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']


def _load_chart_year(y: int) -> pd.DataFrame:
    matches = list((EW_ROOT / 'chart' / f'y{y}').glob('**/ensem_res/ensem5_res_*.csv'))
    if not matches:
        raise FileNotFoundError(f'chart ensem missing for {y}')
    df = pd.read_csv(matches[0])
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    # CNN 输出常把股票代码读成 int，需补零到 6 位以对齐 features140 / 数值模型
    df['StockID'] = df['StockID'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    return df[KEEP].copy()


def _load_oned_year(y: int) -> pd.DataFrame:
    path = EW_ROOT / 'oned' / f'y{y}' / 'preds' / f'pred_1dcnn_oos{y}.parquet'
    df = pd.read_parquet(path)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df['StockID'] = df['StockID'].astype(str)
    return df[KEEP].copy()


def _load_mlp_year(y: int) -> pd.DataFrame:
    path = EW_ROOT / 'mlp' / f'y{y}' / 'preds' / f'pred_mlp_oos{y}.parquet'
    df = pd.read_parquet(path)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df['StockID'] = df['StockID'].astype(str)
    return df[KEEP].copy()


LOADERS = {'chart': _load_chart_year, 'oned': _load_oned_year, 'mlp': _load_mlp_year}


def stitch(model: str) -> pd.DataFrame:
    parts = []
    for y in OOS_YEARS:
        df = LOADERS[model](y)
        df['oos_year'] = y
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    # 键唯一
    dup = out.duplicated(subset=['StockID', 'ending_date']).sum()
    if dup:
        raise RuntimeError(f'{model}: {dup} duplicate keys after stitch')
    return out


def eval_panel(df: pd.DataFrame, score_col: str = 'up_prob') -> dict:
    ic = cross_section_ic(df, score_col)
    summ = summarize_ic(ic)
    t, p = newey_west_t(ic['rank_ic'].to_numpy(), lag=4) if len(ic) else (float('nan'), float('nan'))
    yric = yearly_rank_ic_mean(df, score_col)
    return {
        'rank_ic_mean': summ['rank_ic_mean'],
        'ic_mean': summ['ic_mean'],
        'n_dates': summ['n_dates'],
        'nw_t': t,
        'nw_p': p,
        'yearly': yric,
        'weekly_ic': ic,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(EW_ROOT / 'eval'))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    yearly = {}
    for model in ['chart', 'oned', 'mlp']:
        print(f'stitch {model} ...', flush=True)
        panel = stitch(model)
        panel_path = out_dir / f'pred_{model}_ew_oos.parquet'
        panel.to_parquet(panel_path, index=False)
        print(f'  rows={len(panel):,} -> {panel_path}', flush=True)
        res = eval_panel(panel)
        yearly[model] = res['yearly']
        res['weekly_ic'].to_csv(out_dir / f'weekly_rank_ic_{model}.csv', index=False)
        summary_rows.append({
            'model': model,
            'rank_ic_mean': res['rank_ic_mean'],
            'ic_mean': res['ic_mean'],
            'n_dates': res['n_dates'],
            'nw_t': res['nw_t'],
            'nw_p': res['nw_p'],
            'n_rows': len(panel),
            'rank_ic_2008': float(res['yearly'].get(2008, np.nan)),
        })
        print(f"  RankIC={res['rank_ic_mean']:.4f}  NW-t={res['nw_t']:.2f}  "
              f"n_dates={res['n_dates']}  y2008={summary_rows[-1]['rank_ic_2008']:.4f}", flush=True)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / 'summary.csv', index=False)

    ydf = pd.DataFrame({m: yearly[m] for m in yearly})
    ydf.index.name = 'year'
    ydf.to_csv(out_dir / 'yearly_rank_ic.csv')

    # pairwise Spearman of scores on common keys (optional sanity)
    print('pairwise weekly rank-corr of scores ...', flush=True)
    panels = {m: pd.read_parquet(out_dir / f'pred_{m}_ew_oos.parquet') for m in ['chart', 'oned', 'mlp']}
    base = panels['chart'][['StockID', 'ending_date', 'up_prob']].rename(columns={'up_prob': 'chart'})
    for m in ['oned', 'mlp']:
        base = base.merge(
            panels[m][['StockID', 'ending_date', 'up_prob']].rename(columns={'up_prob': m}),
            on=['StockID', 'ending_date'], how='inner')
    from scipy.stats import spearmanr
    corr_rows = []
    for dt, g in base.groupby('ending_date'):
        if len(g) < 10:
            continue
        for a, b in [('chart', 'oned'), ('chart', 'mlp'), ('oned', 'mlp')]:
            m = np.isfinite(g[a]) & np.isfinite(g[b])
            if m.sum() < 10:
                continue
            r, _ = spearmanr(g.loc[m, a], g.loc[m, b])
            corr_rows.append({'ending_date': dt, 'pair': f'{a}-{b}', 'rho': r})
    cdf = pd.DataFrame(corr_rows)
    pair_mean = cdf.groupby('pair')['rho'].mean().reset_index()
    pair_mean.to_csv(out_dir / 'score_rank_corr_mean.csv', index=False)

    md = []
    md.append('# 训练期扩展窗口 OOS 评估\n')
    md.append('切分：每年 y 用 train 2000…y−3 / val y−2…y−1 预测 y；超参固定为 registry 选定组合。\n')
    md.append('指标：周截面 Spearman Rank IC 均值；NW-t lag=4。\n')
    md.append('\n## 全样本汇总（2008–2025）\n')
    md.append('| 模型 | Rank IC | NW-t | NW-p | n_dates | 2008 年 Rank IC |\n|---|---:|---:|---:|---:|---:|\n')
    for _, r in summ.iterrows():
        md.append(f"| {r['model']} | {r['rank_ic_mean']:.4f} | {r['nw_t']:.2f} | {r['nw_p']:.4g} | "
                  f"{int(r['n_dates'])} | {r['rank_ic_2008']:.4f} |\n")
    md.append('\n## 逐年 Rank IC\n\n')
    md.append('| year | chart | oned | mlp |\n|---:|---:|---:|---:|\n')
    for y, row in ydf.iterrows():
        md.append(f"| {int(y)} | {row['chart']:.4f} | {row['oned']:.4f} | {row['mlp']:.4f} |\n")
    md.append('\n## 模型得分截面秩相关（周均值）\n\n')
    md.append('| pair | rho |\n|---|---:|\n')
    for _, r in pair_mean.iterrows():
        md.append(f"| {r['pair']} | {r['rho']:.4f} |\n")
    md.append('\n说明：主规格 registry 的 OOS parquet 尚未落盘；'
              'y=2008 窗与主切分相同，可作设定对齐参考。'
              '本结果只作稳健性报告，不用于改超参。\n')
    (out_dir / 'summary.md').write_text(''.join(md), encoding='utf-8')
    (out_dir / 'meta.json').write_text(json.dumps({
        'oos_years': OOS_YEARS,
        'models': ['chart', 'oned', 'mlp'],
        'note': 'expanding-window OOS Rank IC; fixed registry HPs',
    }, indent=2), encoding='utf-8')
    print(f'wrote {out_dir / "summary.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
