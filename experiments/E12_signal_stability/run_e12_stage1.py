# -*- coding: utf-8 -*-
"""E12 第一阶段：预测信号稳定性的基础诊断（不训练、不用未来收益、不改组合规则）。

输入：E7 v2 的 e7_sample（三模型冻结得分 s_chart / s_1d / s_mlp4，与 ew_main 预测逐行相同），
      ew_main 预测文件中的 retrain_year（标记扩展窗口模型更新边界），E9 的 portfolio_weekly（换手对照）。
样本：E9 的 958 个信号周 → 957 对相邻周 (t−1, t)。
逐对、逐模型：
  rank_corr          共同股票上 S_{t−1} 与 S_t 的 Spearman 相关
  abs_pct_change     共同股票在共同池内百分位排名的平均绝对变化
  ret_raw            原始股票池口径：t−1 周 Top10%（E9 十分位规则，在当周全部样本股票上分组）中本周仍在 Top10% 的比例
  ret_common         共同股票池口径：两周都只用共同股票重新分组后的 Top10% 留存
  top_exit_share     t−1 周 Top10% 中本周不在样本内的比例
  prev_top_pct       t−1 周 Top10%（共同池）在本周共同池中的平均百分位
  held_retention     E9 实际持仓（Top10% 且可交易）留存，对照 1 − 周买入换手
  implied_ret        若排名服从相关为 rank_corr 的高斯 copula，Top10% 应有的留存（基准值）

  python experiments/E12_signal_stability/run_e12_stage1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import multivariate_normal, norm, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.common.metrics import newey_west_t  # noqa: E402

PAPER = PROJECT_ROOT / 'WORK_SPACE' / 'paper'
E7_SAMPLE = PAPER / 'E7_incremental_v2' / 'data' / 'e7_sample.feather'
EW_PRED = PAPER / 'ew_main' / 'eval' / 'pred_chart_ew_oos.parquet'
E9_WEEKLY = PAPER / 'E9_execution' / 'portfolio_weekly.parquet'
OUT = PAPER / 'E12_signal_stability' / 'stage1'
KEYS = ['StockID', 'ending_date']
SCORES = {'s_chart': 'Chart-CNN', 's_1d': '1D-CNN', 's_mlp4': 'MLP'}
PAIRS = [('s_1d', 's_chart'), ('s_1d', 's_mlp4'), ('s_mlp4', 's_chart')]
REFERENCE = {'rev_1w': '上周收益', 'ret_20d': '20 日收益', 'vol_20d': '20 日波动', 'mom_12_1': '12-1 动量'}
PERIODS = {'2008-2026': (2008, 2026), '2008-2013': (2008, 2013), '2014-2026': (2014, 2026)}
METRICS = ['rank_corr', 'abs_pct_change', 'ret_raw', 'ret_common', 'top_exit_share', 'prev_top_pct',
           'held_retention', 'implied_ret', 'ret_gap']
COLORS = {'s_chart': (224, 123, 57), 's_1d': (47, 111, 219), 's_mlp4': (42, 157, 143)}


def load() -> pd.DataFrame:
    df = pd.read_feather(E7_SAMPLE, columns=KEYS + list(SCORES) + list(REFERENCE) + ['ret_oo', 'entry_lock_up'])
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    ry = pd.read_parquet(EW_PRED, columns=KEYS + ['retrain_year'])
    ry['StockID'] = ry['StockID'].astype(str).str.zfill(6)
    ry['ending_date'] = pd.to_datetime(ry['ending_date'])
    df = df.merge(ry, on=KEYS, how='left', validate='one_to_one')
    assert df['retrain_year'].notna().all()
    assert df[list(SCORES)].notna().all().all()
    df['tradable'] = df['ret_oo'].notna() & ~df['entry_lock_up'].fillna(False).astype(bool)
    return df


def top_decile(v: pd.Series) -> pd.Index:
    """与 run_e9.assign_decile 相同：rank(method='first') 后 ceil(r/n*10)==10。"""
    r = v.rank(method='first', ascending=True)
    return v.index[np.ceil(r / len(v) * 10).clip(1, 10).to_numpy() == 10]


def gauss_implied_retention(rho_s: float, q: float = 0.9) -> float:
    r = 2 * np.sin(np.pi * rho_s / 6)
    z = norm.ppf(q)
    joint = 1 - 2 * q + multivariate_normal.cdf([z, z], mean=[0, 0], cov=[[1, r], [r, 1]])
    return float(joint / (1 - q))


def pair_metrics(a: pd.DataFrame, b: pd.DataFrame, score: str) -> dict:
    sa, sb = a[score].dropna(), b[score].dropna()
    ca = sa[sa.index.isin(sb.index)]
    cb = sb[sb.index.isin(sa.index)]
    cb_al = cb.reindex(ca.index)
    n = len(ca)
    ra, rb = rankdata(ca.to_numpy()), rankdata(cb_al.to_numpy())
    rho = float(np.corrcoef(ra, rb)[0, 1])
    pa, pb = (ra - 0.5) / n, (rb - 0.5) / n

    ta, tb = top_decile(sa), top_decile(sb)
    tca, tcb = top_decile(ca), top_decile(cb)
    pb_s = pd.Series(pb, index=ca.index)
    held_a = ta[a.loc[ta, 'tradable'].to_numpy()]
    held_b = tb[b.loc[tb, 'tradable'].to_numpy()]
    return {
        'n_prev': len(sa), 'n_cur': len(sb), 'n_common': n,
        'rank_corr': rho,
        'abs_pct_change': float(np.abs(pa - pb).mean()),
        'ret_raw': len(ta.intersection(tb)) / len(ta),
        'ret_common': len(tca.intersection(tcb)) / len(tca),
        'top_exit_share': float((~ta.isin(sb.index)).mean()),
        'prev_top_pct': float(pb_s.loc[tca].mean()),
        'held_retention': len(held_a.intersection(held_b)) / len(held_a),
    }


def weekly_table(df: pd.DataFrame, weeks: list[pd.Timestamp], cols: dict[str, str]) -> pd.DataFrame:
    keep = set(weeks)
    by_week = {d: g.set_index('StockID') for d, g in df.groupby('ending_date', sort=True) if d in keep}
    rows = []
    for prev, cur in zip(weeks[:-1], weeks[1:]):
        a, b = by_week[prev], by_week[cur]
        ry_prev, ry_cur = int(a['retrain_year'].iloc[0]), int(b['retrain_year'].iloc[0])
        assert a['retrain_year'].nunique() == 1 and b['retrain_year'].nunique() == 1
        for sc in cols:
            m = pair_metrics(a, b, sc)
            m.update({'ending_date': cur, 'prev_date': prev, 'score': sc, 'model': cols[sc],
                      'retrain_year_prev': ry_prev, 'retrain_year': ry_cur, 'boundary': ry_prev != ry_cur})
            rows.append(m)
    w = pd.DataFrame(rows)
    w['implied_ret'] = w['rank_corr'].map(gauss_implied_retention)
    w['ret_gap'] = w['ret_common'] - w['implied_ret']
    cols = ['ending_date', 'prev_date', 'score', 'model', 'retrain_year_prev', 'retrain_year', 'boundary',
            'n_prev', 'n_cur', 'n_common'] + METRICS
    return w[cols]


def in_period(s: pd.Series, period: str) -> pd.Series:
    lo, hi = PERIODS[period]
    return s.dt.year.between(lo, hi)


def summarize(w: pd.DataFrame, e9_turnover: pd.DataFrame) -> pd.DataFrame:
    w = w.merge(e9_turnover, on=['ending_date', 'score'], how='left')
    w['one_minus_turnover'] = 1 - w['turnover_buy']
    out = []
    subsets = {'all': lambda x: x, 'non_boundary': lambda x: x[~x['boundary']], 'boundary': lambda x: x[x['boundary']]}
    level_metrics = METRICS + ['one_minus_turnover']
    for period in PERIODS:
        for sub, f in subsets.items():
            x = f(w[in_period(w['ending_date'], period)])
            if x.empty:
                continue
            for metric in level_metrics:
                for sc in SCORES:
                    v = x.loc[x['score'] == sc, metric].dropna()
                    rec = {'kind': 'level', 'metric': metric, 'period': period, 'subset': sub, 'key': SCORES[sc],
                           'n_pairs': len(v), 'mean': v.mean(), 'median': v.median(),
                           'p10': v.quantile(0.10), 'min': v.min(), 'std': v.std(), 'nw_t': np.nan}
                    out.append(rec)
            wide = x.pivot(index='ending_date', columns='score', values=METRICS)
            for metric in METRICS:
                for a, b in PAIRS:
                    d = (wide[(metric, a)] - wide[(metric, b)]).dropna()
                    t = newey_west_t(d.to_numpy(), lag=4)[0] if sub != 'boundary' else np.nan
                    out.append({'kind': 'paired_diff', 'metric': metric, 'period': period, 'subset': sub,
                                'key': f'{SCORES[a]} − {SCORES[b]}', 'n_pairs': len(d), 'mean': d.mean(),
                                'median': d.median(), 'p10': d.quantile(0.10), 'min': d.min(), 'std': d.std(),
                                'nw_t': t})
    return pd.DataFrame(out)


def boundary_table(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCORES:
        x = w[w['score'] == sc]
        nb = x.loc[~x['boundary'], 'rank_corr'].to_numpy()
        for _, r in x[x['boundary']].iterrows():
            rows.append({'ending_date': r['ending_date'].date(), 'model': SCORES[sc],
                         'retrain': f"R{r['retrain_year_prev']}→R{r['retrain_year']}",
                         'rank_corr': r['rank_corr'],
                         'pctile_vs_non_boundary': float((nb < r['rank_corr']).mean()),
                         'ret_common': r['ret_common'], 'abs_pct_change': r['abs_pct_change']})
        rows.append({'ending_date': None, 'model': SCORES[sc], 'retrain': 'non-boundary mean',
                     'rank_corr': nb.mean(), 'pctile_vs_non_boundary': np.nan,
                     'ret_common': x.loc[~x['boundary'], 'ret_common'].mean(),
                     'abs_pct_change': x.loc[~x['boundary'], 'abs_pct_change'].mean()})
    return pd.DataFrame(rows)


def low_week_census(w: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """每个模型最低 5% 的 rank_corr 周中，有多少是模型更新边界。"""
    rows = []
    for sc in SCORES:
        x = w[w['score'] == sc]
        cut = x['rank_corr'].quantile(q)
        low = x[x['rank_corr'] <= cut]
        rows.append({'model': SCORES[sc], 'threshold': cut, 'n_low_weeks': len(low),
                     'n_boundary_in_low': int(low['boundary'].sum()),
                     'n_low_2008_2013': int((low['ending_date'].dt.year <= 2013).sum())})
    return pd.DataFrame(rows)


def plot(w: pd.DataFrame, path: Path) -> None:
    W, H, L, R, T, B, GAP = 1600, 980, 90, 30, 60, 60, 70
    ph = (H - T - B - GAP) // 2
    img = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(img)
    f, fs, ft = (ImageFont.load_default(size=s) for s in (16, 14, 20))
    dates = np.array(sorted(w['ending_date'].unique()), dtype='datetime64[ns]')
    x0, x1 = dates[0].astype('int64'), dates[-1].astype('int64')
    xs = lambda d: L + (np.asarray(d, dtype='datetime64[ns]').astype('int64') - x0) / (x1 - x0) * (W - L - R)
    bounds = sorted(w.loc[w['boundary'], 'ending_date'].unique())
    panels = [('rank_corr', 'Adjacent-week Spearman rank correlation of predicted scores (thin: weekly; thick: 13-week mean)',
               (-0.5, 1.0), np.arange(-0.4, 1.01, 0.2), 0.0),
              ('ret_common', 'Top-10% retention, common-stock pool (thin: weekly; thick: 13-week mean; 0.1 = random)',
               (0.0, 0.6), np.arange(0.0, 0.61, 0.1), 0.1)]
    for k, (metric, title, (lo, hi), ticks, ref) in enumerate(panels):
        top = T + k * (ph + GAP)
        ys = lambda v: top + ph - (np.asarray(v) - lo) / (hi - lo) * ph
        dr.text((L, top - 30), title, fill='black', font=ft)
        for gv in ticks:
            y = float(ys(gv))
            dr.line([(L, y), (W - R, y)], fill=(232, 232, 232))
            dr.text((L - 10, y), f'{gv:.1f}', fill=(90, 90, 90), font=fs, anchor='rm')
        dr.line([(L, float(ys(ref))), (W - R, float(ys(ref)))], fill=(60, 60, 60), width=1)
        for b in bounds:
            x = float(xs(b))
            for yy in range(top, top + ph, 10):
                dr.line([(x, yy), (x, yy + 5)], fill=(150, 150, 150))
        for yr in range(2008, 2027, 2):
            x = float(xs(np.datetime64(f'{yr}-01-01')))
            dr.text((x, top + ph + 8), str(yr), fill=(90, 90, 90), font=fs, anchor='mt')
        dr.rectangle([L, top, W - R, top + ph], outline=(160, 160, 160))
        for sc in SCORES:
            s = w[w['score'] == sc].set_index('ending_date')[metric].sort_index()
            px = xs(s.index.values)
            c = COLORS[sc]
            light = tuple(int(v + (255 - v) * 0.65) for v in c)
            dr.line(list(zip(px, ys(s.to_numpy().clip(lo, hi)))), fill=light, width=1)
            roll = s.rolling(13, min_periods=7, center=True).mean()
            dr.line(list(zip(px, ys(roll.to_numpy().clip(lo, hi)))), fill=c, width=3)
    lx = W - R - 520
    for i, sc in enumerate(SCORES):
        dr.line([(lx + i * 170, 22), (lx + i * 170 + 34, 22)], fill=COLORS[sc], width=4)
        dr.text((lx + i * 170 + 42, 22), SCORES[sc], fill='black', font=f, anchor='lm')
    dr.text((L, H - 22), 'Dashed vertical lines: expanding-window model updates (first signal week of 2011, 2014, 2017, 2020, 2023).',
            fill=(90, 90, 90), font=fs, anchor='lm')
    img.save(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    e9 = pd.read_parquet(E9_WEEKLY)
    e9 = e9[(e9['weight'] == 'ew') & np.isclose(e9['impact'], 0.001)]
    weeks = sorted(e9['ending_date'].unique())
    all_weeks = sorted(df['ending_date'].unique())
    assert len(weeks) == 958 and weeks == all_weeks[:-1], 'E9 weeks must be all E7 weeks but the last'

    w = weekly_table(df, weeks, SCORES)
    assert len(w) == 957 * len(SCORES)
    w.to_parquet(OUT / 'weekly_rank_stability.parquet', index=False)

    ref = weekly_table(df, weeks, REFERENCE)
    ref_summary = pd.concat([
        ref.groupby('model')[['rank_corr', 'ret_common', 'abs_pct_change']].mean(),
        w.groupby('model')[['rank_corr', 'ret_common', 'abs_pct_change']].mean(),
        pd.DataFrame({'rank_corr': [0.0], 'ret_common': [0.1], 'abs_pct_change': [1 / 3]}, index=['随机排序（理论值）']),
    ])
    ref_summary.index.name = 'signal'
    ref_summary.to_csv(OUT / 'reference_persistence.csv')

    summary = summarize(w, e9[['ending_date', 'score', 'turnover_buy']])
    summary.to_csv(OUT / 'stability_summary.csv', index=False)
    bt = boundary_table(w)
    bt.to_csv(OUT / 'boundary_weeks.csv', index=False)
    lw = low_week_census(w)
    lw.to_csv(OUT / 'low_corr_weeks_census.csv', index=False)
    nb = w[~w['boundary']]
    win = nb.pivot_table(index='retrain_year', columns='model', values=['rank_corr', 'ret_common'], aggfunc='mean')
    win.columns = [f'{m}|{k}' for k, m in win.columns]
    win.to_csv(OUT / 'by_window.csv')
    plot(w, OUT / 'rank_stability.png')

    held = w.merge(e9[['ending_date', 'score', 'turnover_buy']], on=['ending_date', 'score'])
    check = {SCORES[sc]: {'corr(held_retention, 1-turnover_buy)':
                          float(np.corrcoef(g['held_retention'], 1 - g['turnover_buy'])[0, 1]),
                          'mean_held_retention': float(g['held_retention'].mean()),
                          'mean_1_minus_turnover': float((1 - g['turnover_buy']).mean())}
             for sc, g in held.groupby('score')}
    manifest = {'n_weeks': len(weeks), 'n_pairs': len(weeks) - 1, 'first_pair': str(weeks[1].date()),
                'last_pair': str(weeks[-1].date()), 'n_boundary_pairs': int(w.loc[w['score'] == 's_1d', 'boundary'].sum()),
                'held_retention_vs_e9_turnover': check}
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 400)
    lv = summary[(summary['kind'] == 'level') & (summary['subset'] != 'boundary')]
    print(lv.pivot_table(index=['metric', 'period', 'subset'], columns='key', values='mean').round(4).to_string())
    print(lv[(lv['metric'] == 'rank_corr') & (lv['period'] == '2008-2026')][['subset', 'key', 'median', 'p10', 'min', 'std']]
          .round(4).to_string(index=False))
    pdiff = summary[(summary['kind'] == 'paired_diff') & (summary['subset'] != 'boundary')]
    print(pdiff[['metric', 'period', 'subset', 'key', 'mean', 'nw_t']].round(4).to_string(index=False))
    print(bt.round(4).to_string(index=False))
    print(lw.round(4).to_string(index=False))
    print(ref_summary.round(4).to_string())
    print(win.round(4).to_string())
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
