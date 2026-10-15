# -*- coding: utf-8 -*-
"""E12 第二阶段：滞后预测信号的有效性（不训练、不改组合、不重跑 E9）。

IC_m(k) = Spearman_i( S^m_{i,t−k}, R_{i,t+1} )，k ∈ {0, 1, 2, 4}（事先固定，不再增加）。
- 得分：E7 v2 冻结得分 s_chart / s_1d / s_mlp4；滞后分数直接取 t−k 周当时实际生成的预测（跨训练窗口也不回填）。
- 收益：E7 的未来周收益标签 ret_val（与主分析 Rank IC 相同）。
- 共同样本：(i, t) 需同时有 S_t、S_{t−1}、S_{t−2}、S_{t−4} 与 ret_val；四个年龄、三个模型用同一组股票-周。
- 统计：逐周 Rank IC 均值、NW t（lag 4）、95% CI；IC(0) − IC(k) 配对差；模型间同龄差与衰减差；
  IC(k)/IC(0) 的移动区块 bootstrap CI；分 2008–2013、2014–2026。
- 参照：ρ_k × IC(0)，ρ_k 为同一样本上 S_{t−k} 与 S_t 的排名相关（旧分数的信息若只经由与当期分数的相关传递时的近似值）。

  python experiments/E12_signal_stability/run_e12_stage2.py
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

PAPER = PROJECT_ROOT / 'WORK_SPACE' / 'paper'
E7_SAMPLE = PAPER / 'E7_incremental_v2' / 'data' / 'e7_sample.feather'
MAIN_SUMMARY = PAPER / 'ew_main' / 'eval' / 'summary.csv'
OUT = PAPER / 'E12_signal_stability' / 'stage2'
KEYS = ['StockID', 'ending_date']
SCORES = {'s_chart': 'Chart-CNN', 's_1d': '1D-CNN', 's_mlp4': 'MLP'}
AGES = (0, 1, 2, 4)
PAIRS = [('s_1d', 's_chart'), ('s_1d', 's_mlp4'), ('s_mlp4', 's_chart')]
PERIODS = {'2008-2026': (2008, 2026), '2008-2013': (2008, 2013), '2014-2026': (2014, 2026)}
BOOT_REPS, BOOT_BLOCK, SEED = 2000, 8, 20261015
COLORS = {'s_chart': (224, 123, 57), 's_1d': (47, 111, 219), 's_mlp4': (42, 157, 143)}


def col(sc: str, k: int) -> str:
    return f'{sc}_lag{k}'


def build_sample() -> tuple[pd.DataFrame, dict]:
    df = pd.read_feather(E7_SAMPLE, columns=KEYS + list(SCORES) + ['ret_val'])
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    weeks = np.sort(df['ending_date'].unique())
    widx = pd.Series(np.arange(len(weeks)), index=pd.DatetimeIndex(weeks))
    df['w'] = widx.reindex(df['ending_date']).to_numpy()

    base = df[KEYS + ['w', 'ret_val'] + list(SCORES)].rename(columns={sc: col(sc, 0) for sc in SCORES})
    for k in AGES[1:]:
        lag = df[['StockID', 'w'] + list(SCORES)].copy()
        lag['w'] = lag['w'] + k
        lag = lag.rename(columns={sc: col(sc, k) for sc in SCORES})
        base = base.merge(lag, on=['StockID', 'w'], how='left', validate='one_to_one')
    need = ['ret_val'] + [col(sc, k) for sc in SCORES for k in AGES]
    common = base.dropna(subset=need).reset_index(drop=True)
    info = {'weeks_total': len(weeks), 'rows_total': len(df),
            'rows_with_label': int(df['ret_val'].notna().sum()),
            'rows_common': len(common), 'weeks_common': int(common['ending_date'].nunique()),
            'first_common_week': str(common['ending_date'].min().date()),
            'last_common_week': str(common['ending_date'].max().date()),
            'common_share_of_labelled_rows_in_common_weeks':
                float(len(common) / df.loc[df['w'] >= max(AGES), 'ret_val'].notna().sum())}
    full_ic0 = weekly_spearman(df.dropna(subset=['ret_val']), list(SCORES), 'ret_val').mean()
    info['ic0_full_sample'] = {SCORES[sc]: float(full_ic0[sc]) for sc in SCORES}
    return common, info


def weekly_spearman(d: pd.DataFrame, xs: list[str], y: str) -> pd.DataFrame:
    """逐周截面 Spearman：组内平均秩后的 Pearson 相关。"""
    cols = list(dict.fromkeys(xs + [y]))
    r = d.groupby('ending_date')[cols].rank()
    r = r - r.groupby(d['ending_date']).transform('mean')
    g = d['ending_date']
    vy = (r[y] ** 2).groupby(g).sum()
    out = {}
    for x in xs:
        out[x] = (r[x] * r[y]).groupby(g).sum() / np.sqrt((r[x] ** 2).groupby(g).sum() * vy)
    return pd.DataFrame(out)


def nw_stats(x: pd.Series) -> dict:
    x = x.dropna()
    m = float(x.mean())
    t = newey_west_t(x.to_numpy(), lag=4)[0]
    se = abs(m / t) if t and np.isfinite(t) and t != 0 else np.nan
    return {'n_weeks': len(x), 'mean': m, 'nw_t': t, 'ci_lo': m - 1.96 * se, 'ci_hi': m + 1.96 * se}


def block_bootstrap_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - BOOT_BLOCK + 1, size=int(np.ceil(n / BOOT_BLOCK)))
    return (starts[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel()[:n]


def ratio_ci(ic: pd.DataFrame, rng: np.random.Generator) -> dict:
    """IC(k)/IC(0) 及模型间比值差的移动区块 bootstrap 95% CI（同一组重抽样周，保留截面与模型间相关）。"""
    a = ic.to_numpy()
    n = len(a)
    names = list(ic.columns)
    boots = np.empty((BOOT_REPS, len(names)))
    for b in range(BOOT_REPS):
        boots[b] = a[block_bootstrap_idx(n, rng)].mean(axis=0)
    m = pd.DataFrame(boots, columns=names)
    out = {}
    for sc in SCORES:
        for k in AGES[1:]:
            r = m[col(sc, k)] / m[col(sc, 0)]
            out[('ratio', sc, k)] = (float(ic[col(sc, k)].mean() / ic[col(sc, 0)].mean()),
                                     float(r.quantile(0.025)), float(r.quantile(0.975)))
    for a_, b_ in PAIRS:
        for k in AGES[1:]:
            r = m[col(a_, k)] / m[col(a_, 0)] - m[col(b_, k)] / m[col(b_, 0)]
            pt = ic[col(a_, k)].mean() / ic[col(a_, 0)].mean() - ic[col(b_, k)].mean() / ic[col(b_, 0)].mean()
            out[('ratio_diff', f'{SCORES[a_]} − {SCORES[b_]}', k)] = (float(pt), float(r.quantile(0.025)),
                                                                       float(r.quantile(0.975)))
    return out


def summarize(ic: pd.DataFrame, rho: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for period, (lo, hi) in PERIODS.items():
        sel = ic.index.year.to_series(index=ic.index).between(lo, hi)
        x, rx = ic[sel], rho[sel]
        for sc in SCORES:
            for k in AGES:
                rows.append({'kind': 'ic', 'period': period, 'key': SCORES[sc], 'age': k, **nw_stats(x[col(sc, k)])})
            for k in AGES[1:]:
                rows.append({'kind': 'ic0_minus_ick', 'period': period, 'key': SCORES[sc], 'age': k,
                             **nw_stats(x[col(sc, 0)] - x[col(sc, k)])})
                rows.append({'kind': 'rank_persistence_rho', 'period': period, 'key': SCORES[sc], 'age': k,
                             **nw_stats(rx[col(sc, k)])})
                rows.append({'kind': 'rho_times_ic0', 'period': period, 'key': SCORES[sc], 'age': k,
                             'n_weeks': len(x), 'mean': float(rx[col(sc, k)].mean() * x[col(sc, 0)].mean())})
        for a, b in PAIRS:
            name = f'{SCORES[a]} − {SCORES[b]}'
            for k in AGES:
                rows.append({'kind': 'model_diff_ic', 'period': period, 'key': name, 'age': k,
                             **nw_stats(x[col(a, k)] - x[col(b, k)])})
            for k in AGES[1:]:
                da = x[col(a, 0)] - x[col(a, k)]
                db = x[col(b, 0)] - x[col(b, k)]
                rows.append({'kind': 'model_diff_decay', 'period': period, 'key': name, 'age': k, **nw_stats(da - db)})
        for (kind, key, k), (pt, lo_, hi_) in ratio_ci(x, rng).items():
            rows.append({'kind': f'{kind}_ick_over_ic0', 'period': period, 'key': SCORES.get(key, key), 'age': k,
                         'n_weeks': len(x), 'mean': pt, 'ci_lo': lo_, 'ci_hi': hi_,
                         'note': f'moving-block bootstrap, block={BOOT_BLOCK}, reps={BOOT_REPS}'})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, path: Path) -> None:
    W, H, L, R, T, B = 1650, 640, 80, 30, 90, 80
    gap = 60
    pw = (W - L - R - 2 * gap) // 3
    ph = H - T - B
    lo, hi = -0.01, 0.10
    img = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(img)
    f, fs, ft = (ImageFont.load_default(size=s) for s in (16, 14, 18))
    dr.text((L, 18), 'Rank IC of a k-week-old score for the same next-week return: IC(k) = Spearman(S[t-k], R[t+1])',
            fill='black', font=ImageFont.load_default(size=20))
    for j, period in enumerate(PERIODS):
        x0 = L + j * (pw + gap)
        xs = lambda k: x0 + 30 + k / 4 * (pw - 60)
        ys = lambda v: T + ph - (v - lo) / (hi - lo) * ph
        dr.text((x0 + pw / 2, T - 22), period, fill='black', font=ft, anchor='mm')
        for gv in np.arange(0.0, 0.1001, 0.02):
            dr.line([(x0, ys(gv)), (x0 + pw, ys(gv))], fill=(232, 232, 232) if gv else (60, 60, 60))
            if j == 0:
                dr.text((x0 - 8, ys(gv)), f'{gv:.2f}', fill=(90, 90, 90), font=fs, anchor='rm')
        dr.rectangle([x0, T, x0 + pw, T + ph], outline=(160, 160, 160))
        for k in AGES:
            dr.text((xs(k), T + ph + 10), f'k={k}', fill=(90, 90, 90), font=fs, anchor='mt')
        dr.text((x0 + pw / 2, T + ph + 34), 'signal age (weeks)', fill=(90, 90, 90), font=fs, anchor='mt')
        s = summary[(summary['kind'] == 'ic') & (summary['period'] == period)]
        for i, sc in enumerate(SCORES):
            g = s[s['key'] == SCORES[sc]].set_index('age')
            pts = [(xs(k) + (i - 1) * 5, ys(g.loc[k, 'mean'])) for k in AGES]
            dr.line(pts, fill=COLORS[sc], width=3)
            for k, (px, py) in zip(AGES, pts):
                dr.line([(px, ys(g.loc[k, 'ci_lo'])), (px, ys(g.loc[k, 'ci_hi']))], fill=COLORS[sc], width=2)
                dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=COLORS[sc])
    lx = W - R - 520
    for i, sc in enumerate(SCORES):
        dr.line([(lx + i * 170, 52), (lx + i * 170 + 34, 52)], fill=COLORS[sc], width=4)
        dr.text((lx + i * 170 + 42, 52), SCORES[sc], fill='black', font=f, anchor='lm')
    dr.text((L, H - 18), 'Same stock-weeks for all ages and models. Bars: 95% CI (Newey-West, lag 4).',
            fill=(90, 90, 90), font=fs, anchor='lm')
    img.save(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    common, info = build_sample()
    xs = [col(sc, k) for sc in SCORES for k in AGES]
    ic = weekly_spearman(common, xs, 'ret_val')
    rho = pd.concat([weekly_spearman(common, [col(sc, k) for k in AGES[1:]], col(sc, 0)) for sc in SCORES], axis=1)
    rho = rho.drop(columns=[col(sc, 0) for sc in SCORES], errors='ignore')

    n_stocks = common.groupby('ending_date').size().rename('n_stocks')
    weekly = ic.add_prefix('ic_').join(rho.add_prefix('rho_')).join(n_stocks).reset_index()
    weekly.to_parquet(OUT / 'weekly_lag_ic.parquet', index=False)

    summary = summarize(ic, rho)
    summary.to_csv(OUT / 'lag_ic_summary.csv', index=False)
    plot(summary, OUT / 'ic_by_age.png')

    main_ref = pd.read_csv(MAIN_SUMMARY).set_index('model')['rank_ic_mean']
    info['ic0_main_table'] = {'Chart-CNN': float(main_ref['chart']), '1D-CNN': float(main_ref['oned']),
                              'MLP': float(main_ref['mlp'])}
    info['bootstrap'] = {'reps': BOOT_REPS, 'block_weeks': BOOT_BLOCK, 'seed': SEED}
    (OUT / 'run_manifest.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    for kind in ['ic', 'ic0_minus_ick', 'rank_persistence_rho', 'rho_times_ic0', 'ratio_ick_over_ic0',
                 'model_diff_ic', 'model_diff_decay', 'ratio_diff_ick_over_ic0']:
        s = summary[summary['kind'] == kind]
        print(f'== {kind}')
        print(s[['period', 'key', 'age', 'mean', 'nw_t', 'ci_lo', 'ci_hi']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
