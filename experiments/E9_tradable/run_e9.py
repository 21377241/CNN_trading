# -*- coding: utf-8 -*-
"""E9：A 股执行约束下的可实现收益（protocol E9；主分析 *_ew）。

执行收益复用 E7 `ret_oo`（与协议第 4 节一致）。成本：佣金、印花税、冲击；换手按权重变化。
主判定：多头相对 universe EW 基准的净超额，冲击 10 bp，|t|≥3。

  E7_USE_EW=1 python experiments/E9_tradable/run_e9.py
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
from experiments.E7_incremental.common import (  # noqa: E402
    EXT_PANEL, KEYS, OUT_DATA as E7_DATA, SCORES, annualized_sharpe, load_canonical_preds, newey_west_t,
)

OUT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E9_execution'
COMMISSION = 0.00025  # 单边 2.5 bp
IMPACT_MAIN = 0.0010  # 单边 10 bp
IMPACT_GRID = (0.0, 0.0005, 0.0010, 0.0020, 0.0030)


def stamp_tax(trade_date: pd.Timestamp, side: str) -> float:
    """印花税（单边，小数），按成交日适用税率（协议第 5 节）：2008-09-18 及之前买卖双边征收，之后只对卖出征收。"""
    d = pd.Timestamp(trade_date)
    if d <= pd.Timestamp('2008-04-23'):
        return 0.0030
    if d <= pd.Timestamp('2008-09-18'):
        return 0.0010
    if side == 'buy':
        return 0.0
    if d <= pd.Timestamp('2023-08-27'):
        return 0.0010
    return 0.0005


def buy_cost_bp(trade_date: pd.Timestamp, impact: float) -> float:
    return COMMISSION + impact + stamp_tax(trade_date, 'buy')


def sell_cost_bp(trade_date: pd.Timestamp, impact: float) -> float:
    return COMMISSION + impact + stamp_tax(trade_date, 'sell')


def trade_dates(signal_dates: pd.Series) -> pd.Series:
    """信号日 t → 调仓成交日 t+1（下一个市场交易日；上一期持仓在同一开盘卖出）。"""
    cal = pd.DatetimeIndex(np.sort(pd.read_feather(EXT_PANEL, columns=['Date'])['Date'].unique()))
    sig = pd.DatetimeIndex(signal_dates.unique()).sort_values()
    pos = cal.searchsorted(sig, side='right')
    if (pos >= len(cal)).any():
        raise RuntimeError('signal date without a following market day')
    return signal_dates.map(dict(zip(sig, cal[pos])))


def load_panel() -> pd.DataFrame:
    sample = E7_DATA / 'e7_sample.feather'
    if sample.is_file():
        cols = KEYS + SCORES + ['ret_oo', 'ret_cc', 'FloatMV', 'entry_lock_up', 'not_enterable', 'in_main',
                                'board']
        import pyarrow.ipc
        with pyarrow.ipc.open_file(str(sample)) as f:
            names = set(f.schema.names)
        df = pd.read_feather(sample, columns=[c for c in cols if c in names])
    else:
        preds = load_canonical_preds()
        rets = pd.read_feather(E7_DATA / 'weekly_returns.feather')
        ctrls = pd.read_feather(E7_DATA / 'weekly_controls.feather',
                                columns=KEYS + ['FloatMV', 'board'])
        df = preds.merge(rets, on=KEYS, how='left').merge(ctrls, on=KEYS, how='left')
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df['trade_date'] = trade_dates(df['ending_date'])
    # 可实现：有 ret_oo 且非一字涨停锁仓（协议：锁仓主规格保留进样本但不进组合——此处用 notna ret_oo 且 ~entry_lock）
    if 'entry_lock_up' in df.columns:
        df['tradable'] = df['ret_oo'].notna() & ~df['entry_lock_up'].fillna(False).astype(bool)
    else:
        df['tradable'] = df['ret_oo'].notna()
    return df


def assign_decile(g: pd.DataFrame, score: str) -> pd.Series:
    r = g[score].rank(method='first', ascending=True)
    return np.ceil(r / len(g) * 10).clip(1, 10).astype(int)


def week_portfolio(g: pd.DataFrame, score: str, weight: str) -> dict:
    """单周：多头(D10)、空头(D1)、全市场基准；仅 tradable 股票进组合与基准。"""
    g = g.copy()
    g['decile'] = assign_decile(g, score)
    trad = g[g['tradable']]
    if trad.empty:
        return {}
    long = trad[trad['decile'] == 10]
    short = trad[trad['decile'] == 1]
    if long.empty:
        return {}

    def wret(x: pd.DataFrame) -> tuple[float, pd.Series]:
        if x.empty:
            return np.nan, pd.Series(dtype=float)
        if weight == 'ew':
            w = pd.Series(1.0 / len(x), index=x.index)
        else:
            mv = x['FloatMV'].to_numpy(dtype=float)
            mv = np.where(np.isfinite(mv) & (mv > 0), mv, np.nan)
            if not np.isfinite(mv).any():
                w = pd.Series(1.0 / len(x), index=x.index)
            else:
                mv = np.where(np.isfinite(mv), mv, 0.0)
                w = pd.Series(mv / mv.sum(), index=x.index)
        return float((w.to_numpy() * x['ret_oo'].to_numpy()).sum()), w

    r_long, w_long = wret(long)
    r_short, _ = wret(short)
    r_bench, _ = wret(trad)
    # 权重映射到 StockID（下一期换手用）
    w_map = pd.Series(w_long.to_numpy(), index=long['StockID'].to_numpy())
    return {
        'ret_long': r_long, 'ret_short': r_short, 'ret_bench': r_bench,
        'ret_hl': (r_long - r_short) if np.isfinite(r_short) else np.nan,
        'n_long': int(len(long)), 'n_bench': int(len(trad)),
        'w_long': w_map,
    }


def turnover_cost(prev_w: pd.Series | None, cur_w: pd.Series, trade_date: pd.Timestamp,
                  impact: float) -> tuple[float, float]:
    """返回 (单边换手, 成本占组合)。买付 buy_cost，卖付 sell_cost，税率按成交日 trade_date。"""
    ids = sorted(set(prev_w.index if prev_w is not None else []) | set(cur_w.index))
    p = (prev_w.reindex(ids).fillna(0.0) if prev_w is not None else pd.Series(0.0, index=ids))
    c = cur_w.reindex(ids).fillna(0.0)
    buy = (c - p).clip(lower=0.0).sum()
    sell = (p - c).clip(lower=0.0).sum()
    cost = buy * buy_cost_bp(trade_date, impact) + sell * sell_cost_bp(trade_date, impact)
    return float(buy), float(cost)  # 单边换手用买入权重（≈卖出）


def build_series(df: pd.DataFrame, score: str, weight: str, impact: float) -> pd.DataFrame:
    rows = []
    prev_w = None
    for dt, g in df.groupby('ending_date', sort=True):
        if g[score].isna().all():
            continue
        port = week_portfolio(g, score, weight)
        if not port:
            continue
        td = g['trade_date'].iloc[0]
        to, cost = turnover_cost(prev_w, port['w_long'], td, impact)
        prev_w = port['w_long']
        gross_ex = port['ret_long'] - port['ret_bench']
        net_long = port['ret_long'] - cost
        net_ex = net_long - port['ret_bench']  # 基准无成本（协议：基准同执行规则；简化：基准毛收益）
        rows.append({
            'ending_date': dt, 'trade_date': td, 'score': score, 'weight': weight, 'impact': impact,
            'ret_long_gross': port['ret_long'], 'ret_bench': port['ret_bench'],
            'ret_hl_gross': port['ret_hl'], 'turnover_buy': to, 'cost': cost,
            'ret_long_net': net_long, 'ret_excess_gross': gross_ex, 'ret_excess_net': net_ex,
            'n_long': port['n_long'], 'n_bench': port['n_bench'],
        })
    return pd.DataFrame(rows)


def summarize_series(s: pd.DataFrame) -> dict:
    x = s['ret_excess_net'].to_numpy()
    g = s['ret_excess_gross'].to_numpy()
    t_net, p_net = newey_west_t(x)
    t_gross, p_gross = newey_west_t(g)
    # max drawdown on net long
    wealth = (1 + s['ret_long_net']).cumprod()
    dd = float((wealth / wealth.cummax() - 1).min())
    # 盈亏平衡冲击：线性近似，在网格上插值使 mean(net_ex)=0
    return {
        'n_weeks': len(s),
        'mean_excess_gross': float(np.mean(g)),
        't_excess_gross': t_gross, 'p_excess_gross': p_gross,
        'mean_excess_net': float(np.mean(x)),
        't_excess_net': t_net, 'p_excess_net': p_net,
        'sharpe_excess_net': annualized_sharpe(x),
        'mean_turnover': float(s['turnover_buy'].mean()),
        'mean_cost': float(s['cost'].mean()),
        'max_dd_long_net': dd,
        'ann_compound_long_net': float(wealth.iloc[-1] ** (52 / len(s)) - 1) if len(s) else np.nan,
    }


def judge(row: dict) -> str:
    t = row['t_excess_net']
    tg = row['t_excess_gross']
    if t >= 3.0:
        return '可实现'
    if tg >= 3.0 and t < 1.96:
        return '不可实现'
    if t >= 1.96:
        return '弱证据'
    return '不可实现'


def breakeven_impact(df: pd.DataFrame, score: str, weight: str) -> float:
    means = []
    for imp in IMPACT_GRID:
        s = build_series(df, score, weight, imp)
        means.append((imp, float(s['ret_excess_net'].mean())))
    # 找穿过 0 的区间
    for (i0, m0), (i1, m1) in zip(means, means[1:]):
        if m0 == 0:
            return i0
        if m0 * m1 < 0:
            return float(i0 + (0 - m0) * (i1 - i0) / (m1 - m0))
    if means[-1][1] > 0:
        return float('inf')
    return float(means[0][0])


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    print(f'panel rows={len(df):,} weeks={df["ending_date"].nunique()} tradable_share={df["tradable"].mean():.4f}',
          flush=True)

    weekly_blocks = []
    summaries = []
    for score in SCORES:
        for weight in ('ew', 'vw'):
            for impact in IMPACT_GRID:
                s = build_series(df, score, weight, impact)
                weekly_blocks.append(s)
                if impact == IMPACT_MAIN and weight == 'ew':
                    sm = summarize_series(s)
                    sm.update({'score': score, 'weight': weight, 'impact': impact,
                               'judgment': judge(sm),
                               'breakeven_impact': breakeven_impact(df, score, weight)})
                    summaries.append(sm)
                    print(f'{score} {weight} impact={impact}: t_net={sm["t_excess_net"]:.2f} '
                          f'judge={sm["judgment"]}', flush=True)

    weekly = pd.concat(weekly_blocks, ignore_index=True)
    weekly.to_parquet(OUT / 'portfolio_weekly.parquet', index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(OUT / 'summary.csv', index=False)

    # pairwise net excess differences (main impact, ew)
    pairs = []
    main_w = weekly[(weekly['impact'] == IMPACT_MAIN) & (weekly['weight'] == 'ew')]
    for a, b in [('s_1d', 's_chart'), ('s_1d', 's_mlp4'), ('s_chart', 's_mlp4')]:
        sa = main_w[main_w['score'] == a].set_index('ending_date')['ret_excess_net']
        sb = main_w[main_w['score'] == b].set_index('ending_date')['ret_excess_net']
        d = (sa - sb).dropna()
        t, p = newey_west_t(d.to_numpy())
        pairs.append({'a': a, 'b': b, 'mean_diff': float(d.mean()), 't_nw': t, 'p': p,
                      'diff_significant': abs(t) >= 3.0})
    pd.DataFrame(pairs).to_csv(OUT / 'pairwise.csv', index=False)

    manifest = {
        'script': 'run_e9', 'seconds': round(time.time() - t0),
        'scores': SCORES, 'impact_main': IMPACT_MAIN,
        'judgments': {r['score']: r['judgment'] for r in summaries},
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summ[['score', 'mean_excess_net', 't_excess_net', 'sharpe_excess_net', 'judgment',
                'breakeven_impact']].round(4).to_string(index=False), flush=True)
    print(f'done → {OUT} ({manifest["seconds"]}s)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
