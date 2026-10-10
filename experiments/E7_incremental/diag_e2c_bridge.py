# -*- coding: utf-8 -*-
"""Exploratory diagnostic (outside the frozen E7 protocol): bridge from the E2C chart|1D result to E7 M2.

Starting from the E2C-style regression y ~ s_chart + s_1d, controls are added step by step (rev_1w, Layer A,
Layer B, industry demeaning), and each Layer A variable is also added alone, for the original label ret_val and the
E7 main label ret_oo. Main sample, weekly rank transform, Fama-MacBeth with Newey-West lag 4.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import CTRL_A, CTRL_B, OUT, write_manifest  # noqa: E402
from experiments.E7_incremental.run_fm import summarize, weekly_fm  # noqa: E402
from experiments.E7_incremental.sample import load_sample, prepare  # noqa: E402

PAIR = ['s_chart', 's_1d']
LABELS = ['ret_val', 'ret_oo']


def ladder_specs() -> dict[str, list[str]]:
    specs = {'1_pair': PAIR, '2_pair+rev_1w': PAIR + ['rev_1w'], '3_pair+A': PAIR + CTRL_A,
             '4_pair+A+B': PAIR + CTRL_A + CTRL_B}
    specs.update({f'single_A:{c}': PAIR + [c] for c in CTRL_A})
    return specs


def main() -> int:
    t0 = time.time()
    df = load_sample()
    df = df[df['in_main']]
    x_cols = PAIR + CTRL_A + CTRL_B
    rows = []
    for y in LABELS:
        for fe, specs in [(False, ladder_specs()), (True, {'5_pair+A+B+industry_FE': PAIR + CTRL_A + CTRL_B})]:
            prep = prepare(df, x_cols, y=y, transform='rank', fe=fe)
            betas, r2 = weekly_fm(prep, specs)
            coefs, _, _ = summarize(y, specs, betas, r2)
            rows += [{**c, 'industry_fe': fe} for c in coefs if c['var'] in PAIR]
            print(f'[{y} fe={fe}] rows={len(prep):,} {time.time() - t0:.0f}s')
    out = pd.DataFrame(rows).rename(columns={'variant': 'y'})
    out.to_csv(OUT / 'diag_e2c_bridge.csv', index=False)
    write_manifest('diag_e2c_bridge', {'seconds': round(time.time() - t0)})
    piv = out.pivot_table(index='spec', columns=['y', 'var'], values=['coef_ann', 't_nw'], sort=False)
    pd.set_option('display.width', 200)
    print(piv.round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
