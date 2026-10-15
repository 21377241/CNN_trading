# -*- coding: utf-8 -*-
"""E11 第一阶段：E9 结果口径审计与收益损耗分解（不训练、不重跑回测）。

输入：E7 v2 的 e7_sample / weekly_returns（ret_oo 与执行标记），E9 的 portfolio_weekly（ew，冲击 0 与 10 bp）。
1. 样本审计：三模型是否同一股票-周样本、同一基准。
2. 复核：独立向量化重算 E9 主口径（ew、10 bp）的周序列，与 portfolio_weekly 逐周比对。
3. 分解：毛超额、佣金、印花税、冲击、净超额（bp/周）；买入 / 卖出换手。
4. 模型间：净差 = 毛超额差 + 成本差（成本再拆为费税差与冲击差），逐周配对 NW t（lag 4）。
5. 多头持仓的执行标记占比（描述性，不做反事实）。
6. 印花税修正前后对比：与 E9_execution/_superseded_20261014（按信号日、仅卖出征税）逐模型比较。

  python experiments/E11_attribution/run_e11_stage1.py            # 修订版（v2），对照修正后的 E9
  python experiments/E11_attribution/run_e11_stage1.py --legacy   # 第一版（v1），复现原 E9 印花税规则（commit 8222e7d），
                                                                  # 对照 E9_execution/_superseded_20261014，输出到 stage1/_v1_superseded_20261014
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.common.metrics import newey_west_t  # noqa: E402
from experiments.E9_tradable.run_e9 import COMMISSION, stamp_tax, trade_dates  # noqa: E402

PAPER = PROJECT_ROOT / 'WORK_SPACE' / 'paper'
E7_DATA = PAPER / 'E7_incremental_v2' / 'data'
E9 = PAPER / 'E9_execution'
E9_V1 = E9 / '_superseded_20261014'
OUT = PAPER / 'E11_attribution' / 'stage1'
OUT_V1 = OUT / '_v1_superseded_20261014'
KEYS = ['StockID', 'ending_date']
SCORES = {'s_chart': 'Chart-CNN', 's_1d': '1D-CNN', 's_mlp4': 'MLP'}
IMPACT = 0.0010
BP = 1e4


def legacy_stamp_sell(signal_date: pd.Timestamp) -> float:
    """原 E9（commit 8222e7d）的印花税：只对卖出征收，按信号日判断税率。"""
    d = pd.Timestamp(signal_date)
    if d <= pd.Timestamp('2008-04-23'):
        return 0.0030
    if d <= pd.Timestamp('2023-08-27'):
        return 0.0010
    return 0.0005


def nw(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)), newey_west_t(x, lag=4)[0]


def load_sample() -> pd.DataFrame:
    df = pd.read_feather(E7_DATA / 'e7_sample.feather', columns=KEYS + list(SCORES) + ['ret_oo', 'entry_lock_up'])
    flags = pd.read_feather(E7_DATA / 'weekly_returns.feather',
                            columns=KEYS + ['not_enterable', 'oo_exit_delay', 'oo_exit_lock_dn', 'oo_stale',
                                            'delisted_oo'])
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    flags['StockID'] = flags['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    flags['ending_date'] = pd.to_datetime(flags['ending_date'])
    df = df.merge(flags, on=KEYS, how='left', validate='one_to_one')
    df['tradable'] = df['ret_oo'].notna() & ~df['entry_lock_up'].fillna(False).astype(bool)
    return df


def audit_sample(df: pd.DataFrame) -> dict:
    ew = pd.read_parquet(PAPER / 'ew_main' / 'eval' / 'pred_chart_ew_oos.parquet', columns=KEYS)
    ew['StockID'] = ew['StockID'].astype(str).str.zfill(6)
    ew['ending_date'] = pd.to_datetime(ew['ending_date'])
    m = ew.merge(df[KEYS], on=KEYS, how='outer', indicator=True)
    last_week = df['ending_date'].max()
    is_last = df['ending_date'] == last_week
    lock_up = df['entry_lock_up'].fillna(False).astype(bool)
    not_ent = df['not_enterable'].astype(bool)
    untradable = {
        'last_signal_week_no_exit': is_last,
        'suspended_t1': not_ent & ~is_last,
        'one_price_limit_up_t1': lock_up & ~not_ent & ~is_last,
        'exit_censored_at_data_end': df['ret_oo'].isna() & ~not_ent & ~lock_up & ~is_last,
    }
    assert int(sum(v.sum() for v in untradable.values())) == int((~df['tradable']).sum())
    return {
        'e9_sample_rows': len(df), 'e9_sample_weeks': int(df['ending_date'].nunique()),
        'e9_sample_stocks': int(df['StockID'].nunique()),
        'score_nonnull': {k: int(df[k].notna().sum()) for k in SCORES},
        'rows_with_all_three_scores': int(df[list(SCORES)].notna().all(axis=1).sum()),
        'ew_main_pred_rows': len(ew),
        'keys_only_in_ew_main': int((m['_merge'] == 'left_only').sum()),
        'keys_only_in_e9_sample': int((m['_merge'] == 'right_only').sum()),
        'tradable_share': float(df['tradable'].mean()),
        'not_enterable_share': float(df['not_enterable'].mean()),
        'entry_lock_up_share': float(df['entry_lock_up'].fillna(False).astype(bool).mean()),
        'last_signal_week': str(last_week.date()),
        'last_week_ret_oo_available_share': float(df.loc[is_last, 'ret_oo'].notna().mean()),
        'untradable_rows_by_reason': {k: int(v.sum()) for k, v in untradable.items()},
        'untradable_share_by_reason': {k: float(v.mean()) for k, v in untradable.items()},
        'tradable_share_excluding_last_week': float(df.loc[~is_last, 'tradable'].mean()),
    }


def rebuild_series(df: pd.DataFrame, score: str, legacy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """与 run_e9.week_portfolio / turnover_cost 相同规则的向量化实现（ew，冲击 IMPACT）。"""
    g = df[KEYS + [score, 'ret_oo', 'tradable', 'not_enterable', 'entry_lock_up', 'oo_exit_delay',
                   'oo_exit_lock_dn', 'oo_stale', 'delisted_oo']].copy()
    r = g.groupby('ending_date')[score].rank(method='first')
    n = g.groupby('ending_date')[score].transform('size')
    g['decile'] = np.ceil(r / n * 10).clip(1, 10).astype(int)
    top = g[g['decile'] == 10]
    held = top[top['tradable']].copy()
    held['w'] = 1.0 / held.groupby('ending_date')['StockID'].transform('size')

    bench = g[g['tradable']].groupby('ending_date')['ret_oo'].agg(ret_bench='mean', n_bench='size')
    long = held.groupby('ending_date').agg(ret_long_gross=('ret_oo', 'mean'), n_long=('StockID', 'size'))
    s = long.join(bench, how='inner').reset_index()

    dates = s['ending_date'].tolist()
    prev_of = dict(zip(dates[1:], dates[:-1]))
    cur = held[['ending_date', 'StockID', 'w']]
    prev = cur.copy()
    prev['ending_date'] = prev['ending_date'].map({v: k for k, v in prev_of.items()})
    prev = prev.dropna(subset=['ending_date'])
    both = cur.merge(prev, on=['ending_date', 'StockID'], how='outer', suffixes=('', '_prev')).fillna(
        {'w': 0.0, 'w_prev': 0.0})
    both['buy'] = (both['w'] - both['w_prev']).clip(lower=0)
    both['sell'] = (both['w_prev'] - both['w']).clip(lower=0)
    to = both.groupby('ending_date')[['buy', 'sell']].sum()
    s = s.merge(to, left_on='ending_date', right_index=True, how='left')

    s['trade_date'] = trade_dates(s['ending_date']).to_numpy()
    s['commission'] = (s['buy'] + s['sell']) * COMMISSION
    if legacy:
        s['stamp_buy'] = 0.0
        s['stamp_sell'] = s['sell'] * s['ending_date'].map(legacy_stamp_sell)
    else:
        s['stamp_buy'] = s['buy'] * s['trade_date'].map(lambda d: stamp_tax(d, 'buy'))
        s['stamp_sell'] = s['sell'] * s['trade_date'].map(lambda d: stamp_tax(d, 'sell'))
    s['stamp'] = s['stamp_buy'] + s['stamp_sell']
    s['impact'] = (s['buy'] + s['sell']) * IMPACT
    s['cost'] = s['commission'] + s['stamp'] + s['impact']
    s['gross_ex'] = s['ret_long_gross'] - s['ret_bench']
    s['net_ex'] = s['gross_ex'] - s['cost']
    s['score'] = score

    top_flags = pd.DataFrame({
        'ending_date': top['ending_date'],
        'excluded_not_enterable': top['not_enterable'].astype(float),
        'excluded_entry_lock_up': top['entry_lock_up'].fillna(False).astype(float),
    }).groupby('ending_date').mean()
    held_flags = pd.DataFrame({
        'ending_date': held['ending_date'],
        'exit_deferred': (held['oo_exit_delay'] > 0).astype(float),
        'exit_lock_dn': held['oo_exit_lock_dn'].astype(float),
        'exit_stale': held['oo_stale'].astype(float),
        'delisted': held['delisted_oo'].astype(float),
    }).groupby('ending_date').mean()
    flags = top_flags.join(held_flags, how='inner')
    flags['score'] = score
    return s, flags.reset_index()


def check_against_e9(s: pd.DataFrame, score: str, e9_dir: Path) -> dict:
    w = pd.read_parquet(e9_dir / 'portfolio_weekly.parquet')
    w = w[(w['score'] == score) & (w['weight'] == 'ew')]
    main = w[np.isclose(w['impact'], IMPACT)].set_index('ending_date')
    zero = w[np.isclose(w['impact'], 0.0)].set_index('ending_date')
    x = s.set_index('ending_date')
    common = x.index.intersection(main.index)
    sell_from_e9 = (main.loc[common, 'cost'] - zero.loc[common, 'cost']) / IMPACT - main.loc[common, 'turnover_buy']
    res = {
        'weeks_rebuilt': len(x), 'weeks_e9': len(main), 'weeks_common': len(common),
        'max_abs_diff_ret_long_gross': float((x.loc[common, 'ret_long_gross'] - main.loc[common, 'ret_long_gross']).abs().max()),
        'max_abs_diff_ret_bench': float((x.loc[common, 'ret_bench'] - main.loc[common, 'ret_bench']).abs().max()),
        'max_abs_diff_turnover_buy': float((x.loc[common, 'buy'] - main.loc[common, 'turnover_buy']).abs().max()),
        'max_abs_diff_sell_implied': float((x.loc[common, 'sell'] - sell_from_e9).abs().max()),
        'max_abs_diff_cost': float((x.loc[common, 'cost'] - main.loc[common, 'cost']).abs().max()),
        'max_abs_diff_net_ex': float((x.loc[common, 'net_ex'] - main.loc[common, 'ret_excess_net']).abs().max()),
        'max_abs_diff_cost_impact0_vs_fees': float((x.loc[common, 'commission'] + x.loc[common, 'stamp']
                                                    - zero.loc[common, 'cost']).abs().max()),
    }
    if 'trade_date' in main.columns:
        res['trade_date_mismatch_weeks'] = int((pd.to_datetime(x.loc[common, 'trade_date'])
                                                != pd.to_datetime(main.loc[common, 'trade_date'])).sum())
    return res


def core_table(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        ('多头组合毛周收益', 'ret_long_gross'), ('基准周收益', 'ret_bench'), ('毛超额周收益', 'gross_ex'),
        ('佣金', 'commission'), ('印花税（买入）', 'stamp_buy'), ('印花税（卖出）', 'stamp_sell'),
        ('佣金及税费合计', 'fees'),
        ('冲击成本', 'impact'), ('总成本', 'cost'), ('净超额周收益', 'net_ex'),
    ]
    out = []
    for label, col in rows:
        rec = {'指标': label, '单位': 'bp/周'}
        for sc, name in SCORES.items():
            m, t = nw(series[sc][col])
            rec[name] = round(m * BP, 2)
            rec[f'{name} t'] = round(t, 2)
        out.append(rec)
    for label, col in [('周买入换手', 'buy'), ('周卖出换手', 'sell')]:
        rec = {'指标': label, '单位': '占组合比例'}
        for sc, name in SCORES.items():
            rec[name] = round(float(series[sc][col].mean()), 4)
            rec[f'{name} t'] = np.nan
        out.append(rec)
    rec = {'指标': '多头持仓数（均值）', '单位': '只'}
    for sc, name in SCORES.items():
        rec[name] = round(float(series[sc]['n_long'].mean()), 1)
        rec[f'{name} t'] = np.nan
    out.append(rec)
    return pd.DataFrame(out)


def decompose(series: dict[str, pd.DataFrame], a: str, b: str) -> dict:
    """a 相对 b：净差 = 毛超额差 + (成本_b − 成本_a)。"""
    x = series[a].set_index('ending_date')
    y = series[b].set_index('ending_date')
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]
    comp = {
        'net_diff': x['net_ex'] - y['net_ex'],
        'gross_diff': x['gross_ex'] - y['gross_ex'],
        'cost_diff': y['cost'] - x['cost'],
        'fees_diff': y['fees'] - x['fees'],
        'impact_diff': y['impact'] - x['impact'],
        'bench_diff': x['ret_bench'] - y['ret_bench'],
    }
    res = {'pair': f'{SCORES[a]} − {SCORES[b]}', 'weeks': len(idx)}
    for k, v in comp.items():
        m, t = nw(v)
        res[f'{k}_bp'] = round(m * BP, 2)
        res[f'{k}_t'] = round(t, 2)
    net = res['net_diff_bp']
    res['share_from_gross'] = round(res['gross_diff_bp'] / net, 3) if net else np.nan
    res['share_from_cost'] = round(res['cost_diff_bp'] / net, 3) if net else np.nan
    res['identity_max_abs_err'] = float((comp['net_diff'] - comp['gross_diff'] - comp['cost_diff']).abs().max())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy', action='store_true', help='复现 v1：原 E9 印花税规则，对照归档的 E9')
    legacy = ap.parse_args().legacy
    out, e9_dir = (OUT_V1, E9_V1) if legacy else (OUT, E9)
    out.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    audit = audit_sample(df)
    audit['version'] = 'v1（原 E9 印花税：仅卖出、按信号日）' if legacy else 'v2（修正后：2008-09-18 前双边、按成交日）'
    audit['e9_compared'] = str(e9_dir.relative_to(PAPER))
    series, flags, checks = {}, [], {}
    for sc in SCORES:
        s, f = rebuild_series(df, sc, legacy)
        s['fees'] = s['commission'] + s['stamp']
        series[sc] = s
        flags.append(f)
        checks[sc] = check_against_e9(s, sc, e9_dir)
    bench_same = all(np.allclose(series['s_chart']['ret_bench'].to_numpy(), series[sc]['ret_bench'].to_numpy())
                     and series['s_chart']['ending_date'].equals(series[sc]['ending_date']) for sc in SCORES)
    audit['benchmark_identical_across_models'] = bool(bench_same)

    weekly = pd.concat(series.values(), ignore_index=True)
    weekly.to_parquet(out / 'weekly_components.parquet', index=False)

    table = core_table(series)
    table.to_csv(out / 'core_table.csv', index=False)

    pairs = pd.DataFrame([decompose(series, 's_1d', 's_chart'), decompose(series, 's_1d', 's_mlp4'),
                          decompose(series, 's_mlp4', 's_chart')])
    pairs.to_csv(out / 'pairwise_decomposition.csv', index=False)

    cf = []
    for sc in SCORES:
        s = series[sc]
        for name, other in [('用 1D 的成本', 's_1d'), ('用 Chart 的成本', 's_chart')]:
            o = series[other].set_index('ending_date')['cost'].reindex(s['ending_date']).to_numpy()
            m, t = nw(s['gross_ex'].to_numpy() - o)
            cf.append({'model': SCORES[sc], 'counterfactual': name, 'net_ex_bp': round(m * BP, 2), 't': round(t, 2)})
    pd.DataFrame(cf).to_csv(out / 'cost_swap_counterfactual.csv', index=False)

    sens = []
    if not legacy:
        old = pd.read_parquet(E9_V1 / 'portfolio_weekly.parquet')
        old = old[(old['weight'] == 'ew') & np.isclose(old['impact'], IMPACT)]
        for sc in SCORES:
            s = series[sc].set_index('ending_date')
            o = old[old['score'] == sc].set_index('ending_date')['ret_excess_net'].reindex(s.index)
            m0, t0 = nw(o)
            m1, t1 = nw(s['net_ex'])
            d = s['net_ex'] - o
            sens.append({'model': SCORES[sc], 'weeks_changed': int((d.abs() > 1e-12).sum()),
                         'net_ex_bp_before': round(m0 * BP, 2), 't_before': round(t0, 2),
                         'net_ex_bp_after': round(m1 * BP, 2), 't_after': round(t1, 2),
                         'stamp_buy_bp': round(float(s['stamp_buy'].mean()) * BP, 2)})
        pd.DataFrame(sens).to_csv(out / 'stamp_tax_correction.csv', index=False)

    fl = pd.concat(flags, ignore_index=True)
    fl_summary = fl.groupby('score').mean(numeric_only=True).rename(index=SCORES)
    fl_summary.to_csv(out / 'long_execution_flags.csv')

    (out / 'audit.json').write_text(json.dumps({'sample': audit, 'rebuild_vs_e9': checks}, indent=2,
                                               ensure_ascii=False, default=float), encoding='utf-8')

    pd.set_option('display.width', 250)
    print(json.dumps(audit, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(checks, indent=1, default=float))
    print(table.to_string(index=False))
    print(pairs.T.to_string())
    print(pd.DataFrame(cf).to_string(index=False))
    print(pd.DataFrame(sens).to_string(index=False))
    print((fl_summary * 100).round(3).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
