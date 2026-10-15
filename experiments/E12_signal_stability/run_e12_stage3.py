# -*- coding: utf-8 -*-
"""E12 第三阶段：当前分数 + 一周前分数的联合预测（逐周 Fama-MacBeth；不训练、不改组合）。

Z(Rank R_{i,t+1}) = a_t + β0_t Z(S_{i,t}) + β1_t Z(S_{i,t−1}) [+ 控制] + ε，逐模型、逐周截面 OLS。
Z = 当周回归样本内的截面平均秩再标准化（均值 0、标准差 1）；收益同样处理。
两个回归变量都标准化时，每周 β1 = (IC1 − ρ·IC0) / (1 − ρ²)，ρ 为 S_t 与 S_{t−1} 的秩相关。

规格（按顺序报告，前者为主）：
  base         沿用 E12 第二阶段共同样本（955 周），只放 S_t、S_{t−1}
  others       base + 另外两个模型的当期分数
  base_e7      E7 主样本（控制变量完整，in_main）上的 base，用于分离样本变化
  ctrl         base_e7 + E7 的 15 个控制（Layer A + B）+ 行业去均值（与 E7 主 FM 相同处理）
  ctrl_others  ctrl + 另外两个模型的当期分数
统计：FM 均值、NW t（lag 4）、移动区块 bootstrap 95% CI；β1 的模型间逐周配对差；分 2008–2013、2014–2026。

  python experiments/E12_signal_stability/run_e12_stage3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.common.metrics import newey_west_t  # noqa: E402
from experiments.E7_incremental.common import CTRL_A, CTRL_B  # noqa: E402
from experiments.E12_signal_stability.run_e12_stage2 import E7_SAMPLE, KEYS, build_sample, col  # noqa: E402

PAPER = PROJECT_ROOT / 'WORK_SPACE' / 'paper'
OUT = PAPER / 'E12_signal_stability' / 'stage3'
SCORES = {'s_chart': 'Chart-CNN', 's_1d': '1D-CNN', 's_mlp4': 'MLP'}
PAIRS = [('s_1d', 's_chart'), ('s_1d', 's_mlp4'), ('s_mlp4', 's_chart')]
CTRLS = CTRL_A + CTRL_B
SPECS = ['base', 'others', 'base_e7', 'ctrl', 'ctrl_others']
SPEC_LABEL = {'base': '主回归', 'others': '+ 其他模型当期分数', 'base_e7': '主回归（E7 主样本）',
              'ctrl': '+ 15 个控制 + 行业去均值', 'ctrl_others': '+ 控制 + 行业 + 其他模型'}
PERIODS = {'2008-2026': (2008, 2026), '2008-2013': (2008, 2013), '2014-2026': (2014, 2026)}
BOOT_REPS, BOOT_BLOCK, SEED = 2000, 8, 20261015
COLORS = {'s_chart': (224, 123, 57), 's_1d': (47, 111, 219), 's_mlp4': (42, 157, 143)}


def load() -> pd.DataFrame:
    common, _ = build_sample()
    extra = pd.read_feather(E7_SAMPLE, columns=KEYS + CTRLS + ['industry', 'in_main'])
    extra['StockID'] = extra['StockID'].astype(str).str.zfill(6)
    extra['ending_date'] = pd.to_datetime(extra['ending_date'])
    df = common.merge(extra, on=KEYS, how='left', validate='one_to_one')
    df['in_main'] = df['in_main'].fillna(False).astype(bool)
    return df


def zrank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    r = df.groupby('ending_date')[cols].rank(method='average')
    g = r.groupby(df['ending_date'])
    return (r - g.transform('mean')) / g.transform('std')


def spec_design(df: pd.DataFrame, sc: str, spec: str) -> tuple[pd.DataFrame, list[str]]:
    others = [o for o in SCORES if o != sc]
    cur, old = col(sc, 0), col(sc, 1)
    x = [cur, old]
    if spec in ('others', 'ctrl_others'):
        x += [col(o, 0) for o in others]
    if spec in ('ctrl', 'ctrl_others'):
        x += CTRLS
    d = df if spec in ('base', 'others') else df[df['in_main']]
    out = d[['ending_date', 'industry']].copy()
    out[x + ['ret_val']] = zrank(d, x + ['ret_val'])
    if spec in ('ctrl', 'ctrl_others'):
        cols = x + ['ret_val']
        out[cols] = out[cols] - out.groupby(['ending_date', 'industry'], sort=False)[cols].transform('mean')
    return out, x


def weekly_ols(d: pd.DataFrame, x: list[str]) -> pd.DataFrame:
    rows = {}
    for dt, g in d.groupby('ending_date', sort=True):
        X = np.column_stack([np.ones(len(g)), g[x].to_numpy(float)])
        y = g['ret_val'].to_numpy(float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rows[dt] = beta[1:]
    return pd.DataFrame.from_dict(rows, orient='index', columns=x)


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(x)
    nb = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n - BOOT_BLOCK + 1, size=(BOOT_REPS, nb))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)).reshape(BOOT_REPS, -1)[:, :n]
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def stats(x: pd.Series, rng: np.random.Generator) -> dict:
    v = x.dropna().to_numpy(float)
    lo, hi = boot_ci(v, rng)
    return {'n_weeks': len(v), 'mean': float(v.mean()), 'nw_t': newey_west_t(v, lag=4)[0],
            'boot_lo': lo, 'boot_hi': hi}


def in_period(idx: pd.DatetimeIndex, period: str) -> np.ndarray:
    lo, hi = PERIODS[period]
    return (idx.year >= lo) & (idx.year <= hi)


def plot(summary: pd.DataFrame, path: Path) -> None:
    specs = ['base', 'others', 'ctrl', 'ctrl_others']
    labels = {'base': 'base', 'others': '+other models', 'ctrl': '+controls+ind.', 'ctrl_others': '+ctrl+ind.+others'}
    W, H, L, R, T, B, gap = 1650, 620, 80, 30, 90, 90, 50
    pw = (W - L - R - 2 * gap) // 3
    ph = H - T - B
    lo, hi = -0.012, 0.024
    img = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(img)
    f, fs, ft = (ImageFont.load_default(size=s) for s in (16, 13, 18))
    dr.text((L, 18), 'Coefficient on the one-week-old score, controlling for the current score (beta1, Fama-MacBeth)',
            fill='black', font=ImageFont.load_default(size=20))
    s = summary[(summary['kind'] == 'coef') & (summary['var'] == 'old')]
    for j, period in enumerate(PERIODS):
        x0 = L + j * (pw + gap)
        ys = lambda v: T + ph - (v - lo) / (hi - lo) * ph
        dr.text((x0 + pw / 2, T - 22), period, fill='black', font=ft, anchor='mm')
        for gv in np.arange(-0.010, 0.0201, 0.005):
            dr.line([(x0, ys(gv)), (x0 + pw, ys(gv))], fill=(60, 60, 60) if abs(gv) < 1e-12 else (232, 232, 232))
            if j == 0:
                dr.text((x0 - 8, ys(gv)), f'{gv:.3f}', fill=(90, 90, 90), font=fs, anchor='rm')
        dr.rectangle([x0, T, x0 + pw, T + ph], outline=(160, 160, 160))
        cw = pw / len(specs)
        for si, spec in enumerate(specs):
            cx = x0 + cw * (si + 0.5)
            dr.text((cx, T + ph + 10), labels[spec], fill=(90, 90, 90), font=fs, anchor='mt')
            for i, sc in enumerate(SCORES):
                r = s[(s['period'] == period) & (s['spec'] == spec) & (s['model'] == SCORES[sc])].iloc[0]
                px = cx + (i - 1) * 22
                dr.line([(px, ys(r['boot_lo'])), (px, ys(r['boot_hi']))], fill=COLORS[sc], width=3)
                dr.ellipse([px - 5, ys(r['mean']) - 5, px + 5, ys(r['mean']) + 5], fill=COLORS[sc])
    lx = W - R - 470
    for i, sc in enumerate(SCORES):
        dr.line([(lx + i * 160, 30), (lx + i * 160 + 34, 30)], fill=COLORS[sc], width=4)
        dr.text((lx + i * 160 + 42, 30), SCORES[sc], fill='black', font=f, anchor='lm')
    dr.text((L, H - 18), 'Points: FM mean of weekly beta1; bars: 95% moving-block bootstrap CI (8-week blocks). '
            'Regressors and returns are standardized cross-sectional ranks.', fill=(90, 90, 90), font=fs, anchor='lm')
    img.save(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    weekly, sizes = [], {}
    for spec in SPECS:
        for sc in SCORES:
            d, x = spec_design(df, sc, spec)
            b = weekly_ols(d, x).rename(columns={col(sc, 0): 'cur', col(sc, 1): 'old'})
            b.index.name = 'ending_date'
            long = b.reset_index().melt(id_vars='ending_date', var_name='var', value_name='coef')
            long['spec'], long['score'], long['model'] = spec, sc, SCORES[sc]
            weekly.append(long)
            sizes[f'{spec}|{SCORES[sc]}'] = {'rows': len(d), 'weeks': int(d['ending_date'].nunique()),
                                             'n_regressors': len(x)}
    w = pd.concat(weekly, ignore_index=True)
    w.to_parquet(OUT / 'weekly_fm_coefs.parquet', index=False)

    # 恒等式核对：base 规格中 β1 = (IC1 − ρ·IC0)/(1 − ρ²)
    ident = {}
    for sc in SCORES:
        d, _ = spec_design(df, sc, 'base')
        g = d.groupby('ending_date')
        n = g.size() - 1
        ic0 = g.apply(lambda z: (z[col(sc, 0)] * z['ret_val']).sum(), include_groups=False) / n
        ic1 = g.apply(lambda z: (z[col(sc, 1)] * z['ret_val']).sum(), include_groups=False) / n
        rho = g.apply(lambda z: (z[col(sc, 0)] * z[col(sc, 1)]).sum(), include_groups=False) / n
        implied = (ic1 - rho * ic0) / (1 - rho ** 2)
        b1 = w[(w['spec'] == 'base') & (w['score'] == sc) & (w['var'] == 'old')].set_index('ending_date')['coef']
        ident[SCORES[sc]] = {'max_abs_diff_beta1_vs_formula': float((b1 - implied).abs().max()),
                             'mean_D=IC1-rho*IC0': float((ic1 - rho * ic0).mean()),
                             'mean_rho': float(rho.mean()), 'mean_IC0': float(ic0.mean()), 'mean_IC1': float(ic1.mean())}

    rng = np.random.default_rng(SEED)
    rows = []
    for spec in SPECS:
        for period in PERIODS:
            sub = w[w['spec'] == spec]
            sub = sub[in_period(pd.DatetimeIndex(sub['ending_date']), period)]
            for sc in SCORES:
                for var in ('cur', 'old'):
                    v = sub[(sub['score'] == sc) & (sub['var'] == var)]['coef']
                    rows.append({'kind': 'coef', 'spec': spec, 'period': period, 'model': SCORES[sc], 'var': var,
                                 **stats(v, rng)})
            old = sub[sub['var'] == 'old'].pivot(index='ending_date', columns='score', values='coef')
            for a, b in PAIRS:
                rows.append({'kind': 'beta1_diff', 'spec': spec, 'period': period,
                             'model': f'{SCORES[a]} − {SCORES[b]}', 'var': 'old', **stats(old[a] - old[b], rng)})
    summary = pd.DataFrame(rows)
    summary['spec_label'] = summary['spec'].map(SPEC_LABEL)
    summary.to_csv(OUT / 'fm_summary.csv', index=False)
    plot(summary, OUT / 'beta1_by_spec.png')

    manifest = {'design_sizes': sizes, 'identity_check_base': ident,
                'bootstrap': {'reps': BOOT_REPS, 'block_weeks': BOOT_BLOCK, 'seed': SEED},
                'controls': CTRLS}
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(summary[['kind', 'spec', 'period', 'model', 'var', 'mean', 'nw_t', 'boot_lo', 'boot_hi']]
          .round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
