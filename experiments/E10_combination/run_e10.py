# -*- coding: utf-8 -*-
"""E10：组合层面信号增量（主分析 *_ew；C1=1d+chart 等权秩平均）。

毛口径用 E7 ret_oo 十分位多头；净口径复用 E9 成本函数（冲击 10 bp）。
判定：C1 相对最好单一成分（OOS Sharpe 较高者）的净超额之差 t≥3，且 spanning α（含简单控制）t≥3。

  E7_USE_EW=1 python experiments/E10_combination/run_e10.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.E7_incremental.common import KEYS, SCORES, annualized_sharpe, newey_west_t  # noqa: E402
from experiments.E9_tradable.run_e9 import (  # noqa: E402
    IMPACT_MAIN, build_series, load_panel,
)

import os  # noqa: E402
OUT = (Path(os.environ['E10_OUT_DIR']) if os.environ.get('E10_OUT_DIR')
       else PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E10_combination')


def add_combo_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def rank01(s: pd.Series) -> pd.Series:
        return s.rank(method='average', pct=True)

    rows = []
    for _, g in out.groupby('ending_date', sort=False):
        g = g.copy()
        rc, r1, rm = rank01(g['s_chart']), rank01(g['s_1d']), rank01(g['s_mlp4'])
        g['c1_1d_chart'] = 0.5 * (r1 + rc)
        g['c2_1d_mlp'] = 0.5 * (r1 + rm)
        g['c3_all'] = (r1 + rc + rm) / 3.0
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def spanning_alpha(y: pd.Series, x: pd.Series) -> dict:
    """y = a + b x + e；报告 a 的 NW-t（把残差序列 + 截距等价为对常数回归有偏，这里用 y-b*x 的均值 t）。"""
    d = pd.concat([y, x], axis=1).dropna()
    d.columns = ['y', 'x']
    if len(d) < 30:
        return {'alpha': np.nan, 't_alpha': np.nan, 'beta': np.nan}
    X = np.column_stack([np.ones(len(d)), d['x'].to_numpy()])
    b, *_ = np.linalg.lstsq(X, d['y'].to_numpy(), rcond=None)
    resid = d['y'].to_numpy() - X @ b
    # Newey-West on residual mean ≈ alpha test when x includes constant already absorbed
    # Standard: use NW on the OLS score for intercept via residual + correction; simple: NW on y - b1*x
    a_series = d['y'].to_numpy() - b[1] * d['x'].to_numpy()
    t, p = newey_west_t(a_series)
    return {'alpha': float(b[0]), 't_alpha': t, 'p_alpha': p, 'beta': float(b[1])}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_combo_scores(load_panel())
    signals = ['s_chart', 's_1d', 's_mlp4', 'c1_1d_chart', 'c2_1d_mlp', 'c3_all']

    weekly = []
    for sig in signals:
        s = build_series(df, sig, 'ew', IMPACT_MAIN)
        weekly.append(s)
    w = pd.concat(weekly, ignore_index=True)
    w.to_parquet(OUT / 'portfolio_weekly.parquet', index=False)

    # best single among chart/1d by OOS sharpe of net excess
    singles = {}
    for sig in ('s_chart', 's_1d', 's_mlp4'):
        s = w[w['score'] == sig]
        singles[sig] = annualized_sharpe(s['ret_excess_net'].to_numpy())
    best = max(('s_1d', 's_chart'), key=lambda k: singles[k])
    print('single Sharpes', {k: round(v, 3) for k, v in singles.items()}, 'best=', best, flush=True)

    rows = []
    for combo in ('c1_1d_chart', 'c2_1d_mlp', 'c3_all'):
        sc = w[w['score'] == combo].set_index('ending_date')
        # vs best single
        for base in (best, 's_1d'):
            sb = w[w['score'] == base].set_index('ending_date')
            d = (sc['ret_excess_net'] - sb['ret_excess_net']).dropna()
            t, p = newey_west_t(d.to_numpy())
            sp = spanning_alpha(sc['ret_excess_net'], sb['ret_excess_net'])
            # controls: market = bench excess already in series; add size/rev factors from long-short of controls — skip for v1 minimal
            # use H-L gross of combo vs base as spanning on HL
            d_hl = (sc['ret_hl_gross'] - sb['ret_hl_gross']).dropna()
            t_hl, _ = newey_west_t(d_hl.to_numpy())
            judge = '组合增量成立' if (t >= 3 and sp['t_alpha'] >= 3) else (
                '增量不稳健' if (t >= 1.96 or sp['t_alpha'] >= 1.96) else '无组合增量')
            rows.append({
                'combo': combo, 'base': base,
                'mean_diff_net': float(d.mean()), 't_diff_net': t, 'p_diff_net': p,
                'sharpe_combo': annualized_sharpe(sc['ret_excess_net'].to_numpy()),
                'sharpe_base': annualized_sharpe(sb['ret_excess_net'].to_numpy()),
                'alpha': sp['alpha'], 't_alpha': sp['t_alpha'], 'beta': sp['beta'],
                't_hl_diff': t_hl, 'judgment': judge if base == best else '对照',
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'sharpe_diff.csv', index=False)
    # main C1 vs best
    main = out[(out['combo'] == 'c1_1d_chart') & (out['base'] == best)].iloc[0]
    manifest = {
        'best_single': best, 'single_sharpes': singles,
        'c1_judgment': main['judgment'],
        'c1_t_diff': float(main['t_diff_net']), 'c1_t_alpha': float(main['t_alpha']),
        'seconds': round(time.time() - t0),
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=float), encoding='utf-8')
    print(out.round(4).to_string(index=False), flush=True)
    print('C1 vs best:', main['judgment'], 't_diff=', round(main['t_diff_net'], 2),
          't_alpha=', round(main['t_alpha'], 2), flush=True)
    print(f'done → {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
