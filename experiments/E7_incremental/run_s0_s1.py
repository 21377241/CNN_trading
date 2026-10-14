# -*- coding: utf-8 -*-
"""E7 §15：S0 / S1 得分下重复主规格 M1/M2 与残差 Rank IC（同一套 returns/controls）。

  E7_USE_EW=1 python experiments/E7_incremental/run_s0_s1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ashare_train.eval_s0_s1 import load_ew, load_s0, load_s1  # noqa: E402
from experiments.E7_incremental.common import KEYS, OUT, OUT_DATA, SCORES, write_manifest  # noqa: E402
from experiments.E7_incremental.run_fm import run_variant  # noqa: E402
from experiments.E7_incremental.run_resid_ic import resid_ric  # noqa: E402

ROBUST = OUT / 'robustness_s0_s1'
MODEL_TO_SCORE = {'chart': 's_chart', 'oned': 's_1d', 'mlp': 's_mlp4'}
T_MAIN = 3.0
T_WEAK = 1.96
COMMON = (pd.Timestamp('2014-01-01'), pd.Timestamp('2026-09-30'))


def _load_base() -> pd.DataFrame:
    """主分析 e7_sample 去掉得分列，保留收益与控制。"""
    path = OUT_DATA / 'e7_sample.feather'
    if not path.is_file():
        raise FileNotFoundError(f'缺 {path}：先跑完主分析 E7')
    df = pd.read_feather(path)
    drop = [c for c in SCORES if c in df.columns]
    return df.drop(columns=drop)


def _attach(df: pd.DataFrame, loader) -> pd.DataFrame:
    parts = []
    for model, col in MODEL_TO_SCORE.items():
        p = loader(model)[['StockID', 'ending_date', 'up_prob']].rename(columns={'up_prob': col})
        parts.append(p)
    scores = parts[0]
    for p in parts[1:]:
        scores = scores.merge(p, on=KEYS, how='inner')
    out = df.merge(scores, on=KEYS, how='inner')
    return out


def _h1_label(t_beta: float, t_ric: float) -> str:
    if t_beta >= T_MAIN and t_ric >= T_MAIN:
        return '成立'
    if (t_beta >= T_WEAK and t_ric >= T_WEAK) and (t_beta < T_MAIN or t_ric < T_MAIN):
        return '部分成立'
    return '不成立'


def _h2_label(t_chart: float, t_1d: float) -> str:
    c_ok, d_ok = t_chart >= T_MAIN, t_1d >= T_MAIN
    if c_ok and d_ok:
        return '成立'
    if d_ok and not c_ok:
        return '不成立-仅1d'
    if c_ok and not d_ok:
        return '不成立-仅chart'
    return '不成立-均否'


def _coef_t(coefs: pd.DataFrame, spec: str, var: str) -> float:
    r = coefs[(coefs['spec'] == spec) & (coefs['var'] == var)]
    if r.empty:
        return float('nan')
    return float(r.iloc[0]['t_nw'])


def _ric_t(ric: pd.DataFrame, metric: str) -> float:
    r = ric[ric['metric'] == metric]
    return float(r.iloc[0]['t_nw']) if len(r) else float('nan')


def run_scheme(name: str, base: pd.DataFrame, loader, weeks: tuple | None = None) -> dict:
    t0 = time.time()
    df = _attach(base, loader)
    if weeks:
        df = df[(df['ending_date'] >= weeks[0]) & (df['ending_date'] <= weeks[1])].copy()
    main_df = df[df['in_main']].copy()
    print(f'[{name}] merged={len(df):,} main={len(main_df):,} weeks={main_df["ending_date"].nunique()}',
          flush=True)

    acc = {'coefs': [], 'r2': [], 'inc': []}
    run_variant(name, main_df, acc)
    coefs = pd.DataFrame(acc['coefs'])
    coefs.to_csv(ROBUST / f'fm_coefs_{name.lower()}.csv', index=False)

    _, summary = resid_ric(main_df, SCORES, [('rho_resid_chart_1d', 's_chart', 's_1d')])
    summary.to_csv(ROBUST / f'resid_ric_{name.lower()}.csv', index=False)

    h1 = {k: _h1_label(_coef_t(coefs, f'M1({k})', k),
                       _ric_t(summary, f'ric_resid_{k}')) for k in SCORES}
    h2 = _h2_label(_coef_t(coefs, 'M2', 's_chart'), _coef_t(coefs, 'M2', 's_1d'))
    out = {
        'scheme': name,
        'n_main': len(main_df),
        'n_weeks': int(main_df['ending_date'].nunique()),
        'h1': h1,
        'h2': h2,
        't_m1': {k: _coef_t(coefs, f'M1({k})', k) for k in SCORES},
        't_m2': {'s_chart': _coef_t(coefs, 'M2', 's_chart'), 's_1d': _coef_t(coefs, 'M2', 's_1d')},
        't_resid_ric': {k: _ric_t(summary, f'ric_resid_{k}') for k in SCORES},
        'seconds': round(time.time() - t0),
    }
    print(json.dumps({k: out[k] for k in ('scheme', 'h1', 'h2', 't_m1', 't_m2')}, ensure_ascii=False, indent=2),
          flush=True)
    return out


def compare(a: dict, b: dict, label: str) -> dict:
    """§15.2：H2 定性标签相同，且各 k 的 H1「成立」一致。"""
    h2_same = a['h2'] == b['h2']
    h1_same = {k: a['h1'][k] == b['h1'][k] for k in SCORES}
    h1_all = all(h1_same.values())
    boundary = []
    for k in SCORES:
        ta, tb = a['t_m1'][k], b['t_m1'][k]
        for side, t in ((a['scheme'], ta), (b['scheme'], tb)):
            pass
        if (ta >= T_MAIN) != (tb >= T_MAIN) and min(ta, tb) >= T_WEAK:
            boundary.append(k)
    if h2_same and h1_all:
        verdict = 'E7结论稳健'
    elif (not h2_same) or any(a['h1'][k] != b['h1'][k] and '成立' in (a['h1'][k], b['h1'][k]) for k in SCORES):
        if boundary and h2_same:
            verdict = 'H1边界敏感'
        else:
            verdict = '不稳健'
    else:
        verdict = 'H1边界敏感' if boundary else '部分不一致'
    return {
        'comparison': label,
        'h2_a': a['h2'], 'h2_b': b['h2'], 'h2_same': h2_same,
        'h1_same': h1_same, 'boundary_ks': boundary, 'verdict': verdict,
    }


def main() -> int:
    t0 = time.time()
    ROBUST.mkdir(parents=True, exist_ok=True)
    base = _load_base()

    # EW 全样本（与主分析对照；应用同一套 §15 判定标签）
    ew_full = run_scheme('EW', base, load_ew)
    s0_full = run_scheme('S0', base, load_s0)
    # 共同期 2014–2026
    ew_c = run_scheme('EW_common', base, load_ew, weeks=COMMON)
    s0_c = run_scheme('S0_common', base, load_s0, weeks=COMMON)
    s1_c = run_scheme('S1', base, load_s1, weeks=COMMON)

    comps = [
        compare(s0_full, ew_full, 'S0 vs EW (2008–2026)'),
        compare(s1_c, ew_c, 'S1 vs EW (2014–2026 common)'),
        compare(s0_c, ew_c, 'S0 vs EW (2014–2026 common)'),
    ]
    pd.DataFrame(comps).to_csv(ROBUST / 'comparison_summary.csv', index=False)
    summary = {'schemes': [ew_full, s0_full, ew_c, s0_c, s1_c], 'comparisons': comps,
               'seconds': round(time.time() - t0)}
    (ROBUST / 'judgment.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    write_manifest('run_s0_s1', {'out': str(ROBUST), 'seconds': summary['seconds']})
    print('=== §15 comparisons ===', flush=True)
    for c in comps:
        print(f"{c['comparison']}: {c['verdict']} (H2 same={c['h2_same']})", flush=True)
    print(f'done {summary["seconds"]}s → {ROBUST}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
