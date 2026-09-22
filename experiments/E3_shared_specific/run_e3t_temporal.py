# -*- coding: utf-8 -*-
"""E3-T: pairwise shared/specific decomposition for CNN and 1D-CNN-120."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E2C_temporal.eval_e2c import (
    ann_stats,
    nw_t_mean,
    residualize_by_date,
    score_summary,
    z_by_date,
)

ANN = 52
E2C_COMMON = os.path.join(
    PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E2C_temporal_1dcnn_120', 'results', 'common_scores.csv'
)
OUT_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E3T_cnn_1d_temporal')


def main() -> int:
    os.makedirs(OUT_ROOT, exist_ok=True)
    df = pd.read_csv(E2C_COMMON, parse_dates=['ending_date'])
    df = df.dropna(subset=['cnn', 'temporal', 'ret_val', 'MarketCap']).copy()
    df = residualize_by_date(df, 'cnn', 'temporal', 'S_cnn_sp_1d')
    df = residualize_by_date(df, 'temporal', 'cnn', 'S_1d_sp_cnn')
    df['z_cnn'] = z_by_date(df, 'cnn')
    df['z_1d'] = z_by_date(df, 'temporal')
    df['S_shared'] = 0.5 * (df['z_cnn'] + df['z_1d'])

    scores = {
        'S_shared': 'S_shared',
        'S_cnn_specific_given_1d': 'S_cnn_sp_1d',
        'S_1d_specific_given_cnn': 'S_1d_sp_cnn',
        'S_cnn_raw': 'cnn',
        'S_1d_raw': 'temporal',
    }
    rows = []
    for label, col in scores.items():
        s = score_summary(df.rename(columns={col: 'score'}), 'score', ANN)
        rank_ic = []
        for _, g in df.groupby('ending_date', sort=True):
            rank_ic.append(g[col].rank().corr(g['ret_val'].rank()))
        s['rank_ic_nw_t'] = nw_t_mean(pd.Series(rank_ic).values)
        rows.append({'score': label, **s})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_ROOT, 'e3t_score_summary.csv'), index=False, float_format='%.6f')
    df.to_parquet(os.path.join(OUT_ROOT, 'scores.parquet'), index=False)

    report = [
        '# E3-T: CNN versus 1D-CNN Shared/Specific Decomposition',
        '',
        'The decomposition uses the 120D E2C common panel only (922 dates; 2,687,822 stock-date observations).',
        'Specific scores are daily cross-sectional OLS residuals with an intercept.',
        '',
        summary.to_markdown(index=False),
        '',
        'Interpretation: `S_cnn_specific_given_1d` measures chart-CNN information not spanned by 1D-CNN; `S_1d_specific_given_cnn` is the converse.',
    ]
    with open(os.path.join(OUT_ROOT, 'e3t_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    with open(os.path.join(OUT_ROOT, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'input': E2C_COMMON, 'n_rows': len(df), 'n_dates': int(df.ending_date.nunique())}, f, indent=2)
    print(summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
