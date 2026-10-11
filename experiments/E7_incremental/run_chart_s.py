# -*- coding: utf-8 -*-
"""E7 protocol section 14: ChartCNN-S capacity robustness (added after the E7 main results, before this run).

The full chart-CNN is replaced by the E2G width-scaled ChartCNN-S (`s_chartS`) in the core FM specifications, all
section 8 robustness variants plus r10_ctrl_gbm, residual Rank IC, double sorts and the E2C bridge decomposition.
The full chart-CNN specifications are estimated on the same rows for side-by-side comparison.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import (  # noqa: E402
    CHART_S, KEYS, registry_entry, OUT, OUT_DATA, load_chart_s, write_manifest,
)
from experiments.E7_incremental.diag_e2c_bridge import bridge, show  # noqa: E402
from experiments.E7_incremental.run_double_sort import double_sort  # noqa: E402
from experiments.E7_incremental.run_fm import run_robustness, run_variant  # noqa: E402
from experiments.E7_incremental.run_resid_ic import resid_ric  # noqa: E402
from experiments.E7_incremental.sample import load_sample  # noqa: E402

OUT_S = OUT / 'chart_s'
INCREMENTS_S = [('M1(s_chartS)', 'M0-AB'), ('MA(s_chartS)', 'M0-A'), ('M2-S', 'M1(s_1d)'), ('M2-S', 'M1(s_chartS)'),
                ('M2', 'M1(s_1d)')]


def make_specs_s(ctrl_a: list[str], ctrl_b: list[str], extra: list[str] | None = None) -> dict[str, list[str]]:
    ab = ctrl_a + ctrl_b + (extra or [])
    return {'M0-A': list(ctrl_a), 'M0-AB': ab, 'S(s_chartS)': ['s_chartS'], 'MA(s_chartS)': ctrl_a + ['s_chartS'],
            'M1(s_chartS)': ab + ['s_chartS'], 'M1(s_1d)': ab + ['s_1d'], 'M1(s_chart)': ab + ['s_chart'],
            'M2-S': ab + ['s_chartS', 's_1d'], 'M2': ab + ['s_chart', 's_1d']}


def main() -> int:
    t0 = time.time()
    OUT_S.mkdir(parents=True, exist_ok=True)
    df = load_sample().merge(load_chart_s(), on=KEYS, how='left')
    main_all = df[df['in_main']]
    coverage = float(main_all['s_chartS'].notna().mean())
    df = df[df['s_chartS'].notna()].reset_index(drop=True)
    main_df = df[df['in_main']].copy()
    print(f'main sample rows {len(main_all):,}; s_chartS coverage {coverage:.6f}; used {len(main_df):,}')

    kw = {'spec_fn': make_specs_s, 'increments': INCREMENTS_S}
    acc = {'coefs': [], 'r2': [], 'inc': []}
    run_variant('main', main_df, acc, **kw)
    run_robustness(df, main_df, acc, **kw)
    gbm = pd.read_feather(OUT_DATA / 'ctrl_gbm_oos.feather')
    run_variant('r10_ctrl_gbm', main_df.merge(gbm, on=KEYS, how='inner'), acc, extra=['ctrl_gbm'], **kw)
    coefs = pd.DataFrame(acc['coefs'])
    coefs.to_csv(OUT_S / 'fm_coefs.csv', index=False)
    pd.DataFrame(acc['r2']).to_csv(OUT_S / 'fm_r2.csv', index=False)
    pd.DataFrame(acc['inc']).to_csv(OUT_S / 'fm_r2_increment.csv', index=False)

    pairs = [('rho_raw_chartS_chart', 's_chartS', 's_chart'), ('rho_raw_chartS_1d', 's_chartS', 's_1d'),
             ('rho_resid_chartS_chart', 's_chartS', 's_chart'), ('rho_resid_chartS_1d', 's_chartS', 's_1d')]
    _, ric = resid_ric(main_df, ['s_chartS', 's_chart', 's_1d'], pairs)
    ric.to_csv(OUT_S / 'resid_ric.csv', index=False)
    ds = double_sort(main_df, ['s_chartS'])
    ds.to_csv(OUT_S / 'double_sort.csv', index=False)
    br = bridge(main_df, ['s_chartS', 's_1d'])
    br.to_csv(OUT_S / 'diag_bridge.csv', index=False)

    write_manifest('run_chart_s', {'chart_s_sha1': registry_entry(CHART_S)[1], 'coverage_main': coverage,
                                   'rows_main_used': len(main_df), 'seconds': round(time.time() - t0)})

    pd.set_option('display.width', 220)
    sc = coefs[coefs['var'].isin(['s_chartS', 's_chart', 's_1d'])]
    print(sc[sc['variant'] == 'main'][['spec', 'var', 'coef_ann', 't_nw', 'share_pos_weeks']].round(4)
          .to_string(index=False))
    piv = sc[sc['spec'].isin(['M1(s_chartS)', 'M1(s_chart)', 'M2-S', 'M2'])].copy()
    piv['cell'] = piv['spec'] + ':' + piv['var']
    piv = piv.pivot_table(index='variant', columns='cell', values='t_nw', sort=False)
    print(piv.round(2).to_string())
    m2s = piv['M2-S:s_chartS']
    print(f'M2-S s_chartS: t>=3 in {(m2s >= 3).sum()}/{len(m2s)} variants; t>=1.96 in {(m2s >= 1.96).sum()}')
    print(pd.DataFrame(acc['inc']).query("variant == 'main'").round(5).to_string(index=False))
    print(ric.round(4).to_string(index=False))
    avg = ds[ds['quintile'] == 'avg']
    print(avg[['cond', 'hl_ann', 't_nw', 'sharpe_ann']].round(3).to_string(index=False))
    show(br)
    print(f'total {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
