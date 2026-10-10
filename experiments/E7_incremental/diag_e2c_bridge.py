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

LABELS = ['ret_val', 'ret_oo']


def ladder_specs(pair: list[str]) -> dict[str, list[str]]:
    specs = {'1_pair': pair, '2_pair+rev_1w': pair + ['rev_1w'], '3_pair+A': pair + CTRL_A,
             '4_pair+A+B': pair + CTRL_A + CTRL_B}
    specs.update({f'single_A:{c}': pair + [c] for c in CTRL_A})
    return specs


def bridge(df: pd.DataFrame, pair: list[str]) -> pd.DataFrame:
    t0 = time.time()
    x_cols = pair + CTRL_A + CTRL_B
    rows = []
    for y in LABELS:
        for fe, specs in [(False, ladder_specs(pair)), (True, {'5_pair+A+B+industry_FE': x_cols})]:
            prep = prepare(df, x_cols, y=y, transform='rank', fe=fe)
            betas, r2 = weekly_fm(prep, specs)
            coefs, _, _ = summarize(y, specs, betas, r2)
            rows += [{**c, 'industry_fe': fe} for c in coefs if c['var'] in pair]
            print(f'[bridge {pair} {y} fe={fe}] rows={len(prep):,} {time.time() - t0:.0f}s')
    return pd.DataFrame(rows).rename(columns={'variant': 'y'})


def show(out: pd.DataFrame) -> None:
    piv = out.pivot_table(index='spec', columns=['y', 'var'], values=['coef_ann', 't_nw'], sort=False)
    pd.set_option('display.width', 200)
    print(piv.round(4).to_string())


def main() -> int:
    t0 = time.time()
    df = load_sample()
    out = bridge(df[df['in_main']], ['s_chart', 's_1d'])
    out.to_csv(OUT / 'diag_e2c_bridge.csv', index=False)
    write_manifest('diag_e2c_bridge', {'seconds': round(time.time() - t0)})
    show(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
