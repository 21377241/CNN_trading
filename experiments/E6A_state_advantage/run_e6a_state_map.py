# -*- coding: utf-8 -*-
"""E6A: state-dependent comparative advantage of raw CNN versus MLP-4 scores.

This experiment intentionally does not train models, optimize a portfolio, or learn a
gate. It asks whether observable, pre-return states predict relative cross-sectional
Rank-IC performance of the two already-frozen base forecasts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.E3_shared_specific.utils import newey_west_t

BASE_SCORES = PROJECT_ROOT / 'WORK_SPACE/paper/E4_recent_path_chrono_ric_mlp4/baseline_scores.parquet'
STATE_PANEL = PROJECT_ROOT / 'WORK_SPACE/paper/E5_mechanism_mlp4/pressure_panel.parquet'
OUT_ROOT = PROJECT_ROOT / 'WORK_SPACE/paper/E6A_state_advantage'
DISCOVERY_YEARS = range(2008, 2016)
TEST_YEARS = range(2016, 2026)
NW_LAG = 4
MIN_CELL_N = 20


def tercile(values: pd.Series) -> pd.Series:
    """Date-level terciles with deterministic handling of ties."""
    pct = values.rank(method='average', pct=True)
    return pd.cut(pct, bins=[-np.inf, 1 / 3, 2 / 3, np.inf], labels=[1, 2, 3]).astype('Int64')


def load_panel() -> pd.DataFrame:
    scores = pd.read_parquet(
        BASE_SCORES,
        columns=['StockID', 'ending_date', 'S_cnn', 'S_mlp', 'ret_val'],
    )
    states = pd.read_parquet(
        STATE_PANEL,
        columns=['StockID', 'ending_date', 'CumRet5', 'Pressure'],
    )
    panel = scores.merge(states, on=['StockID', 'ending_date'], how='inner', validate='one_to_one')
    panel['ending_date'] = pd.to_datetime(panel['ending_date'])
    panel['year'] = panel['ending_date'].dt.year
    panel = panel.dropna(subset=['S_cnn', 'S_mlp', 'ret_val', 'CumRet5', 'Pressure']).copy()
    panel['recent_return_tercile'] = panel.groupby('ending_date')['CumRet5'].transform(tercile)
    panel['pressure_tercile'] = panel.groupby('ending_date')['Pressure'].transform(tercile)
    return panel.dropna(subset=['recent_return_tercile', 'pressure_tercile'])


def daily_cell_ic(panel: pd.DataFrame, sample: str, years: range) -> pd.DataFrame:
    rows = []
    sub = panel.loc[panel['year'].isin(years)]
    for dt, day in sub.groupby('ending_date', sort=True):
        for rr in (1, 2, 3):
            for pr in (1, 2, 3):
                g = day.loc[
                    (day['recent_return_tercile'].astype(int) == rr)
                    & (day['pressure_tercile'].astype(int) == pr)
                ]
                if len(g) < MIN_CELL_N:
                    continue
                cnn_ic = spearmanr(g['S_cnn'], g['ret_val']).statistic
                mlp_ic = spearmanr(g['S_mlp'], g['ret_val']).statistic
                if not (np.isfinite(cnn_ic) and np.isfinite(mlp_ic)):
                    continue
                rows.append({
                    'sample': sample,
                    'ending_date': dt,
                    'recent_return_tercile': rr,
                    'pressure_tercile': pr,
                    'n_stocks': len(g),
                    'ic_cnn': float(cnn_ic),
                    'ic_mlp4': float(mlp_ic),
                    'delta_ic_cnn_minus_mlp4': float(cnn_ic - mlp_ic),
                })
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, rr, pr), g in daily.groupby(['sample', 'recent_return_tercile', 'pressure_tercile'], sort=True):
        delta = g['delta_ic_cnn_minus_mlp4'].to_numpy()
        nw_t, nw_p = newey_west_t(delta, lag=NW_LAG)
        rows.append({
            'sample': sample,
            'recent_return_tercile': rr,
            'pressure_tercile': pr,
            'recent_return_label': ['Negative', 'Neutral', 'Positive'][rr - 1],
            'pressure_label': ['Low', 'Mid', 'High'][pr - 1],
            'mean_ic_cnn': g['ic_cnn'].mean(),
            'mean_ic_mlp4': g['ic_mlp4'].mean(),
            'mean_delta_ic': g['delta_ic_cnn_minus_mlp4'].mean(),
            'nw_t_delta_ic': nw_t,
            'nw_p_delta_ic': nw_p,
            'n_rebalances': len(g),
            'mean_n_stocks': g['n_stocks'].mean(),
        })
    return pd.DataFrame(rows)


def heatmap(summary: pd.DataFrame, sample: str, out_path: Path) -> None:
    sub = summary.loc[summary['sample'] == sample]
    vals = sub.pivot(index='recent_return_tercile', columns='pressure_tercile', values='mean_delta_ic')
    ts = sub.pivot(index='recent_return_tercile', columns='pressure_tercile', values='nw_t_delta_ic')
    vals = vals.reindex(index=[1, 2, 3], columns=[1, 2, 3])
    ts = ts.reindex(index=[1, 2, 3], columns=[1, 2, 3])
    vmax = max(0.001, float(np.nanmax(np.abs(vals.to_numpy()))))
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    image = ax.imshow(vals, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(3), ['Low pressure', 'Mid pressure', 'High pressure'])
    ax.set_yticks(range(3), ['Negative recent return', 'Neutral recent return', 'Positive recent return'])
    ax.set_title(f'E6A: CNN minus MLP-4 Rank IC ({sample})')
    for i in range(3):
        for j in range(3):
            value = vals.iloc[i, j]
            t_stat = ts.iloc[i, j]
            text = 'NA' if not np.isfinite(value) else f'{value:+.4f}\\nt={t_stat:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=10)
    fig.colorbar(image, ax=ax, label='Mean delta Rank IC')
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    discovery = summary.loc[summary['sample'] == 'Discovery (2008-2015)'].copy()
    test = summary.loc[summary['sample'] == 'Frozen test (2016-2025)'].copy()
    merged = discovery.merge(
        test,
        on=['recent_return_tercile', 'pressure_tercile', 'recent_return_label', 'pressure_label'],
        suffixes=('_discovery', '_test'),
    )
    cols = [
        'recent_return_label', 'pressure_label',
        'mean_ic_cnn_discovery', 'mean_ic_mlp4_discovery', 'mean_delta_ic_discovery', 'nw_t_delta_ic_discovery',
        'mean_ic_cnn_test', 'mean_ic_mlp4_test', 'mean_delta_ic_test', 'nw_t_delta_ic_test',
    ]
    table = merged[cols].copy()
    table.columns = [
        'Recent return', 'Pressure', 'CNN IC (disc.)', 'MLP-4 IC (disc.)', 'Delta IC (disc.)', 'NW t (disc.)',
        'CNN IC (test)', 'MLP-4 IC (test)', 'Delta IC (test)', 'NW t (test)',
    ]
    lines = [
        '# E6A: CNN--MLP-4 State-dependent Comparative Advantage',
        '',
        '## Purpose',
        '',
        'This experiment uses frozen raw CNN and MLP-4 forecasts. It does not retrain either model, optimize a portfolio, or estimate a gate.',
        'It tests whether the observable state `RecentReturn x Pressure` predicts their relative cross-sectional Rank-IC performance.',
        '',
        '## Fixed Design',
        '',
        '- Discovery/calibration: 2008--2015.',
        '- Frozen test: 2016--2025.',
        '- At each weekly rebalance, CumRet5 and Pressure are independently split into cross-sectional terciles.',
        '- Pressure is the pre-existing composite of turnover proxy, drawdown stress, and end-of-window volume intensity.',
        '- Each cell reports the mean date-level Spearman Rank IC; delta is CNN minus MLP-4.',
        '- Newey--West inference is applied to the weekly delta-IC time series with lag 4.',
        '',
        '## Cell Results',
        '',
        table.to_markdown(index=False, floatfmt='.5f'),
        '',
        '## Interpretation Rule',
        '',
        'The discovery heatmap may motivate a later, pre-fixed combination rule only if its broad sign and location pattern replicates in the frozen test. This report does not evaluate a combination strategy.',
    ]
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    discovery = daily_cell_ic(panel, 'Discovery (2008-2015)', DISCOVERY_YEARS)
    test = daily_cell_ic(panel, 'Frozen test (2016-2025)', TEST_YEARS)
    daily = pd.concat([discovery, test], ignore_index=True)
    summary = summarize(daily)
    daily.to_csv(OUT_ROOT / 'e6a_daily_cell_rank_ic.csv', index=False, float_format='%.8f')
    summary.to_csv(OUT_ROOT / 'e6a_cell_summary.csv', index=False, float_format='%.8f')
    heatmap(summary, 'Discovery (2008-2015)', OUT_ROOT / 'e6a_delta_ic_discovery.png')
    heatmap(summary, 'Frozen test (2016-2025)', OUT_ROOT / 'e6a_delta_ic_test.png')
    write_report(summary, OUT_ROOT / 'e6a_state_advantage_report.md')
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.6f}'))


if __name__ == '__main__':
    main()
