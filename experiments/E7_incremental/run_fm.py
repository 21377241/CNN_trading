# -*- coding: utf-8 -*-
"""E7 Fama-MacBeth regressions: main specifications, incremental R^2, robustness (protocol sections 5, 6.1, 8)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.E7_incremental.common import (  # noqa: E402
    CTRL_A, CTRL_B, CTRL_VALUATION, OUT, OUT_DATA, SCORES, newey_west_t, write_manifest,
)
from experiments.E7_incremental.sample import build_sample, prepare  # noqa: E402

SUBPERIODS = {'2008_2013': ('2008-01-01', '2013-12-31'), '2014_2019': ('2014-01-01', '2019-12-31'),
              '2020_2025': ('2020-01-01', '2025-12-31')}
INCREMENTS = [(f'M1({k})', 'M0-AB') for k in SCORES] + [(f'MA({k})', 'M0-A') for k in SCORES] + [
    ('M2', 'M1(s_chart)'), ('M2', 'M1(s_1d)'), ('M3', 'M2')]


def make_specs(ctrl_a: list[str], ctrl_b: list[str], extra: list[str] | None = None) -> dict[str, list[str]]:
    ab = ctrl_a + ctrl_b + (extra or [])
    specs = {'M0-A': list(ctrl_a), 'M0-AB': ab}
    for k in SCORES:
        specs[f'S({k})'] = [k]
        specs[f'MA({k})'] = ctrl_a + [k]
        specs[f'M1({k})'] = ab + [k]
    specs['M2'] = ab + ['s_chart', 's_1d']
    specs['M3'] = ab + SCORES
    return specs


def weekly_fm(prep: pd.DataFrame, specs: dict[str, list[str]], weight: str | None = None) -> tuple[dict, dict]:
    """Per-week OLS (or WLS) for every spec. Returns {spec: DataFrame of weekly betas} and {spec: weekly R^2 frame}."""
    betas = {s: [] for s in specs}
    r2 = {s: [] for s in specs}
    for dt, idx in prep.groupby('ending_date', sort=True).indices.items():
        g = prep.iloc[idx]
        y = g['y'].to_numpy()
        w = np.sqrt(g[weight].to_numpy()) if weight else None
        for s, cols in specs.items():
            X = g[cols].to_numpy()
            n, k = X.shape
            if n < k + 10:
                continue
            Xc = np.column_stack([np.ones(n), X])
            if w is not None:
                b, *_ = np.linalg.lstsq(Xc * w[:, None], y * w, rcond=None)
                yw = y * w
                resid = yw - (Xc * w[:, None]) @ b
                sst = np.sum((yw - np.average(y, weights=w ** 2) * w) ** 2)
            else:
                b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
                resid = y - Xc @ b
                sst = np.sum((y - y.mean()) ** 2)
            rsq = 1 - resid @ resid / sst if sst > 0 else np.nan
            betas[s].append({'ending_date': dt, **dict(zip(cols, b[1:])),
                             **{f'sd_{c}': v for c, v in zip(cols, X.std(axis=0))}})
            r2[s].append({'ending_date': dt, 'r2': rsq, 'adj_r2': 1 - (1 - rsq) * (n - 1) / (n - k - 1), 'n': n})
    return ({s: pd.DataFrame(v).set_index('ending_date') for s, v in betas.items()},
            {s: pd.DataFrame(v).set_index('ending_date') for s, v in r2.items()})


def summarize(variant: str, specs: dict, betas: dict, r2: dict) -> tuple[list, list, list]:
    coefs, r2rows, inc = [], [], []
    for s, cols in specs.items():
        B = betas[s]
        if B.empty:
            continue
        for c in cols:
            x = B[c].to_numpy()
            t, p = newey_west_t(x)
            m = float(np.nanmean(x))
            se = m / t if t and np.isfinite(t) and t != 0 else np.nan
            coefs.append({'variant': variant, 'spec': s, 'var': c, 'coef': m, 't_nw': t, 'p': p,
                          'ci_lo': m - 1.96 * abs(se), 'ci_hi': m + 1.96 * abs(se), 'coef_ann': m * 52,
                          'coef_std': float(np.nanmean(B[c] * B[f'sd_{c}'])),
                          'share_pos_weeks': float(np.mean(x > 0)), 'n_weeks': len(x),
                          'avg_n': float(r2[s]['n'].mean())})
        r2rows.append({'variant': variant, 'spec': s, 'r2_mean': float(r2[s]['r2'].mean()),
                       'adj_r2_mean': float(r2[s]['adj_r2'].mean()), 'n_weeks': len(r2[s])})
    for a, b in INCREMENTS:
        if a in r2 and b in r2 and not r2[a].empty and not r2[b].empty:
            d = (r2[a]['r2'] - r2[b]['r2']).dropna()
            da = (r2[a]['adj_r2'] - r2[b]['adj_r2']).dropna()
            t, p = newey_west_t(d.to_numpy())
            ta, _ = newey_west_t(da.to_numpy())
            inc.append({'variant': variant, 'spec': a, 'base': b, 'delta_r2_mean': float(d.mean()),
                        't_nw': t, 'p': p, 'share_pos_weeks': float((d > 0).mean()),
                        'delta_adj_r2_mean': float(da.mean()), 't_nw_adj': ta,
                        'share_pos_weeks_adj': float((da > 0).mean())})
    return coefs, r2rows, inc


def run_variant(name: str, df: pd.DataFrame, acc: dict, *, ctrl_a=CTRL_A, ctrl_b=CTRL_B, extra=None, y='ret_oo',
                transform='rank', fe=True, y_winsor=False, weight=None, weeks: tuple | None = None) -> dict:
    t0 = time.time()
    specs = make_specs(list(ctrl_a), list(ctrl_b), extra)
    x_cols = list(dict.fromkeys(sum(specs.values(), [])))
    keep = (weight,) if weight else ()
    prep = prepare(df, x_cols, y=y, transform=transform, fe=fe, y_winsor=y_winsor, keep=keep)
    if weeks:
        prep = prep[(prep['ending_date'] >= weeks[0]) & (prep['ending_date'] <= weeks[1])]
    betas, r2 = weekly_fm(prep, specs, weight)
    c, r, i = summarize(name, specs, betas, r2)
    acc['coefs'] += c
    acc['r2'] += r
    acc['inc'] += i
    print(f'[{name}] rows={len(prep):,} weeks={prep["ending_date"].nunique()} {time.time() - t0:.0f}s')
    return betas


def main() -> int:
    t0 = time.time()
    df = build_sample()
    main_df = df[df['in_main']].copy()
    print(pd.read_csv(OUT / 'sample_flow.csv').to_string(index=False))
    acc = {'coefs': [], 'r2': [], 'inc': []}

    betas = run_variant('main', main_df, acc)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    pd.concat({s: b for s, b in betas.items()}, names=['spec']).reset_index().to_feather(
        OUT_DATA / 'fm_weekly_betas_main.feather')

    # Protocol section 8 robustness, in order
    run_variant('r1_y_ret_cc', main_df, acc, y='ret_cc')
    run_variant('r1_y_ret_val', main_df, acc, y='ret_val')
    run_variant('r2_y_winsor', main_df, acc, y_winsor=True)
    run_variant('r3_winsor_z', main_df, acc, transform='wz')
    run_variant('r4_no_valuation', main_df, acc, ctrl_b=[c for c in CTRL_B if c not in CTRL_VALUATION])
    run_variant('r4_no_industry_fe', main_df, acc, fe=False)
    for name, (a, b) in SUBPERIODS.items():
        run_variant(f'r5_sub_{name}', main_df, acc, weeks=(pd.Timestamp(a), pd.Timestamp(b)))
    mv_rank = main_df.groupby('ending_date')['FloatMV'].rank(pct=True)
    run_variant('r6_drop_small30', main_df[mv_rank > 0.30], acc)
    run_variant('r6_drop_age_lt_1y', main_df[main_df['age_days'] >= 365], acc)
    run_variant('r6_drop_limit_sig', main_df[~(main_df['limit_up_sig'] | main_df['limit_dn_sig'])], acc)
    run_variant('r6_drop_entry_lock_up', main_df[~main_df['entry_lock_up'].astype(bool)], acc)
    run_variant('r7_delist_minus50', main_df, acc, y='ret_oo_d50')

    full = df[df['ret_oo'].notna() & df[CTRL_A].notna().all(axis=1) & (df['industry'] != 'nan')].copy()
    miss = []
    for c in CTRL_B:
        if full[c].isna().any():
            full[f'miss_{c}'] = full[c].isna().astype(float)
            full[c] = full[c].fillna(full.groupby('ending_date')[c].transform('median'))
            miss.append(f'miss_{c}')
    full_specs_extra = miss
    run_variant('r8_full_sample_impute', full, acc, extra=full_specs_extra)
    run_variant('r9_wls_floatmv', main_df, acc, weight='FloatMV')

    coefs = pd.DataFrame(acc['coefs'])
    coefs.to_csv(OUT / 'fm_coefs.csv', index=False)
    pd.DataFrame(acc['r2']).to_csv(OUT / 'fm_r2.csv', index=False)
    pd.DataFrame(acc['inc']).to_csv(OUT / 'fm_r2_increment.csv', index=False)
    coefs[coefs['variant'].str.startswith('r5_sub_')].to_csv(OUT / 'subperiod_fm.csv', index=False)
    write_manifest('run_fm', {'variants': sorted(coefs['variant'].unique()), 'seconds': round(time.time() - t0)})

    show = coefs[(coefs['var'].isin(SCORES)) & coefs['spec'].str.match(r'^(S|MA|M1|M2|M3)')]
    pd.set_option('display.width', 200)
    print(show[show['variant'] == 'main'][['spec', 'var', 'coef', 't_nw', 'coef_ann', 'share_pos_weeks']]
          .round(4).to_string(index=False))
    print(pd.DataFrame(acc['inc']).query("variant == 'main'").round(5).to_string(index=False))
    piv = show[show['spec'].isin(['M1(s_chart)', 'M1(s_1d)', 'M1(s_mlp4)', 'M2'])].copy()
    piv['cell'] = piv['spec'] + ':' + piv['var']
    print(piv.pivot_table(index='variant', columns='cell', values='t_nw', sort=False).round(2).to_string())
    print(f'total {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
