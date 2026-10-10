# -*- coding: utf-8 -*-
"""E7 residual Rank IC (protocol section 6.2).

Each week, every score is regressed on Layer A + Layer B controls with industry demeaning (ranks, as in the main FM
specification); the residual e_k is correlated (Spearman) with the raw next-week return ret_oo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import (  # noqa: E402
    CTRL_A, CTRL_B, OUT, OUT_DATA, SCORES, newey_west_t, write_manifest,
)
from experiments.E7_incremental.sample import load_sample, prepare  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    df = load_sample()
    df = df[df['in_main']]
    ctrls = CTRL_A + CTRL_B
    prep = prepare(df, ctrls + SCORES, transform='rank', fe=True, keep=('ret_oo',))
    rows = []
    for dt, idx in prep.groupby('ending_date', sort=True).indices.items():
        g = prep.iloc[idx]
        X = np.column_stack([np.ones(len(g)), g[ctrls].to_numpy()])
        ret = g['ret_oo'].to_numpy()
        row = {'ending_date': dt}
        res = {}
        for k in SCORES:
            s = g[k].to_numpy()
            b, *_ = np.linalg.lstsq(X, s, rcond=None)
            res[k] = s - X @ b
            row[f'ric_raw_{k}'] = spearman(s, ret)
            row[f'ric_resid_{k}'] = spearman(res[k], ret)
            row[f'resid_share_var_{k}'] = float(res[k].var() / s.var())
        row['rho_raw_chart_1d'] = spearman(g['s_chart'].to_numpy(), g['s_1d'].to_numpy())
        row['rho_resid_chart_1d'] = spearman(res['s_chart'], res['s_1d'])
        row['rho_resid_chart_mlp4'] = spearman(res['s_chart'], res['s_mlp4'])
        row['rho_resid_1d_mlp4'] = spearman(res['s_1d'], res['s_mlp4'])
        rows.append(row)
    w = pd.DataFrame(rows).set_index('ending_date')
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    w.reset_index().to_feather(OUT_DATA / 'resid_ric_weekly.feather')
    summ = []
    for c in w.columns:
        t, p = newey_west_t(w[c].to_numpy())
        summ.append({'metric': c, 'mean': float(w[c].mean()), 't_nw': t, 'p': p,
                     'share_pos_weeks': float((w[c] > 0).mean()), 'n_weeks': len(w)})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / 'resid_ric.csv', index=False)
    write_manifest('run_resid_ic', {'rows': len(prep), 'weeks': len(w)})
    print(s.round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
