# -*- coding: utf-8 -*-
"""E7 conditional nonlinear-control benchmark (protocol section 9).

HistGB on rank-transformed Layer A + B controls (no model scores), trained on 2000-2005, selected on 2006-2007 by
weekly Rank IC with a one-week embargo at each boundary; target is ret_cc winsorized weekly at 1%/99%.
The OOS score `ctrl_gbm` is added to the FM specifications as an extra control (variant r10_ctrl_gbm: M1G/M2G).
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import (  # noqa: E402
    CTRL_A, CTRL_B, KEYS, OUT, OUT_DATA, WEEKLY_CONTROLS, WEEKLY_RETURNS, cs_rank, cs_winsor, write_manifest,
)
from experiments.E7_incremental.run_fm import run_variant  # noqa: E402
from experiments.E7_incremental.sample import load_sample  # noqa: E402

CTRLS = CTRL_A + CTRL_B
GRID = list(itertools.product([3, 5], [0.05, 0.1]))


def weekly_rank_ic(df: pd.DataFrame, pred: np.ndarray, y: str) -> float:
    d = df[['ending_date', y]].assign(p=pred)
    ic = d.groupby('ending_date').apply(lambda g: g['p'].rank().corr(g[y].rank()), include_groups=False)
    return float(ic.mean())


def main() -> int:
    t0 = time.time()
    c = pd.read_feather(WEEKLY_CONTROLS, columns=KEYS + CTRLS + ['is_ST'])
    r = pd.read_feather(WEEKLY_RETURNS, columns=KEYS + ['ret_cc', 'next_signal'])
    d = c.merge(r, on=KEYS, how='inner')
    d = d[(d['is_ST'] == 0) & d[CTRLS].notna().all(axis=1) & d['ret_cc'].notna()]
    d = d[d['ending_date'] < '2008-01-01'].reset_index(drop=True)
    yr, nyr = d['ending_date'].dt.year, d['next_signal'].dt.year
    train = d[(yr <= 2005) & (nyr <= 2005)].copy()
    val = d[(yr.between(2006, 2007)) & (nyr <= 2007)].copy()
    for part in (train, val):
        part[CTRLS] = cs_rank(part, CTRLS)
        part['y'] = cs_winsor(part['ret_cc'], part['ending_date'])
    print(f'train {len(train):,} rows {train["ending_date"].nunique()} wks; val {len(val):,} rows')

    grid = []
    best = None
    for depth, lr in GRID:
        m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=500, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=42)
        m.fit(train[CTRLS].to_numpy(), train['y'].to_numpy())
        ic = weekly_rank_ic(val, m.predict(val[CTRLS].to_numpy()), 'ret_cc')
        grid.append({'max_depth': depth, 'learning_rate': lr, 'n_iter': m.n_iter_, 'val_rank_ic': ic})
        print(grid[-1])
        if best is None or ic > best[0]:
            best = (ic, m, depth, lr)
    pd.DataFrame(grid).to_csv(OUT / 'ctrl_gbm_grid.csv', index=False)

    df = load_sample()
    main_df = df[df['in_main']].copy()
    X = cs_rank(main_df, CTRLS).to_numpy()
    main_df['ctrl_gbm'] = best[1].predict(X)
    oos_ic = weekly_rank_ic(main_df, main_df['ctrl_gbm'].to_numpy(), 'ret_oo')
    print(f'selected depth={best[2]} lr={best[3]} val_ic={best[0]:.4f}; OOS Rank IC vs ret_oo={oos_ic:.4f}')
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    main_df[KEYS + ['ctrl_gbm']].reset_index(drop=True).to_feather(OUT_DATA / 'ctrl_gbm_oos.feather')

    acc = {'coefs': [], 'r2': [], 'inc': []}
    run_variant('r10_ctrl_gbm', main_df, acc, extra=['ctrl_gbm'])
    for name, key in (('fm_coefs.csv', 'coefs'), ('fm_r2.csv', 'r2'), ('fm_r2_increment.csv', 'inc')):
        old = pd.read_csv(OUT / name)
        old = old[old['variant'] != 'r10_ctrl_gbm']
        pd.concat([old, pd.DataFrame(acc[key])], ignore_index=True).to_csv(OUT / name, index=False)
    write_manifest('run_ctrl_gbm', {'selected': {'max_depth': best[2], 'learning_rate': best[3],
                                                 'val_rank_ic': best[0]}, 'oos_rank_ic_ret_oo': oos_ic,
                                    'seconds': round(time.time() - t0)})
    co = pd.DataFrame(acc['coefs'])
    print(co[co['spec'].str.match(r'^(M1|M2|M3)') & co['var'].isin(['s_chart', 's_1d', 's_mlp4', 'ctrl_gbm'])]
          [['spec', 'var', 'coef', 't_nw', 'coef_ann']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
