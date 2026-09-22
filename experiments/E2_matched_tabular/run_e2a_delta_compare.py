# -*- coding: utf-8 -*-
"""
E2A-A1: Chrono 120 vs 180 delta comparison (no retrain).

Usage:
  python experiments/E2_matched_tabular/run_e2a_delta_compare.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.metrics import yearly_rank_ic_mean
from experiments.E2_matched_tabular.e2a_layout import CHRONO_120, CHRONO_180, ws

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2a_delta')

MODELS = ['logit', 'histgb', 'mlp']
METRIC_COLS = ['auc', 'ic_mean', 'rank_ic_mean', 'ew_hl_ann', 'ew_hl_sharpe', 'vw_hl_sharpe']


def _load_summary(root: str) -> pd.DataFrame:
    path = os.path.join(root, 'e2_summary.csv')
    df = pd.read_csv(path)
    return df.set_index('model')


def _load_pred(root: str, model: str) -> pd.DataFrame:
    p = os.path.join(root, 'preds', f'pred_{model}_oos.csv')
    df = pd.read_csv(p)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    return df


def main():
    out_dir = ws('WORK_SPACE', 'paper', 'E2A_delta_chrono')
    os.makedirs(out_dir, exist_ok=True)

    s120 = _load_summary(CHRONO_120)
    s180 = _load_summary(CHRONO_180)

    rows = []
    for m in MODELS:
        r = {'model': m}
        for c in METRIC_COLS:
            v120 = float(s120.loc[m, c]) if m in s120.index and c in s120.columns else np.nan
            v180 = float(s180.loc[m, c]) if m in s180.index and c in s180.columns else np.nan
            r[f'{c}_120'] = v120
            r[f'{c}_180'] = v180
            r[f'delta_{c}'] = v180 - v120
        rows.append(r)
    delta_df = pd.DataFrame(rows)
    delta_path = os.path.join(out_dir, 'delta_summary.csv')
    delta_df.to_csv(delta_path, index=False, float_format='%.6f')
    logger.info(f'wrote {delta_path}\n{delta_df.to_string(index=False)}')

    # yearly Rank IC deltas + Wilcoxon
    year_rows = []
    tests = {}
    for m in MODELS:
        y120 = yearly_rank_ic_mean(_load_pred(CHRONO_120, m), 'up_prob')
        y180 = yearly_rank_ic_mean(_load_pred(CHRONO_180, m), 'up_prob')
        common = sorted(set(y120.index) & set(y180.index))
        d = pd.Series({y: float(y180.loc[y] - y120.loc[y]) for y in common}, name=m)
        for y in common:
            year_rows.append({'model': m, 'year': int(y), 'rank_ic_120': float(y120.loc[y]),
                              'rank_ic_180': float(y180.loc[y]), 'delta_rank_ic': float(d.loc[y])})
        # vs logit: paired on years
        tests[m] = {
            'n_years': len(common),
            'mean_delta_rank_ic': float(d.mean()) if len(d) else np.nan,
            'frac_pos': float((d > 0).mean()) if len(d) else np.nan,
        }
        if len(d) >= 5 and np.any(d != 0):
            try:
                stat, p = wilcoxon(d.values, alternative='greater')
                tests[m]['wilcoxon_greater_stat'] = float(stat)
                tests[m]['wilcoxon_greater_p'] = float(p)
            except Exception as e:
                tests[m]['wilcoxon_error'] = str(e)

    year_df = pd.DataFrame(year_rows)
    year_path = os.path.join(out_dir, 'delta_yearly_rank_ic.csv')
    year_df.to_csv(year_path, index=False, float_format='%.6f')

    # logit delta vs others (paired year deltas)
    d_logit = year_df[year_df['model'] == 'logit'].set_index('year')['delta_rank_ic']
    for m in ['histgb', 'mlp']:
        d_m = year_df[year_df['model'] == m].set_index('year')['delta_rank_ic']
        common = sorted(set(d_logit.index) & set(d_m.index))
        diff = pd.Series({y: float(d_logit.loc[y] - d_m.loc[y]) for y in common})
        key = f'logit_minus_{m}'
        tests[key] = {
            'mean_diff': float(diff.mean()) if len(diff) else np.nan,
            'frac_logit_larger': float((diff > 0).mean()) if len(diff) else np.nan,
        }
        if len(diff) >= 5 and np.any(diff != 0):
            try:
                stat, p = wilcoxon(diff.values, alternative='greater')
                tests[key]['wilcoxon_greater_stat'] = float(stat)
                tests[key]['wilcoxon_greater_p'] = float(p)
            except Exception as e:
                tests[key]['wilcoxon_error'] = str(e)

    # Sharpe delta comparison (descriptive)
    sharpe_note = {}
    for m in MODELS:
        row = delta_df[delta_df['model'] == m].iloc[0]
        sharpe_note[m] = float(row['delta_ew_hl_sharpe'])
    tests['delta_ew_hl_sharpe'] = sharpe_note
    tests['hypothesis'] = (
        'Delta_logit > Delta_histgb and Delta_logit > Delta_mlp on Rank IC (Wilcoxon) '
        'and descriptively on EW H-L Sharpe'
    )
    hyp_ok = (
        sharpe_note.get('logit', 0) > sharpe_note.get('histgb', 0)
        and sharpe_note.get('logit', 0) > sharpe_note.get('mlp', 0)
    )
    tests['sharpe_logit_largest'] = bool(hyp_ok)

    test_path = os.path.join(out_dir, 'delta_tests.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(tests, f, indent=2)
    logger.info(f'wrote {test_path}')
    logger.info(json.dumps(tests, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
