# -*- coding: utf-8 -*-
"""E2D liquidity / implementability audit on existing OOS predictions."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ANN = 52
MODELS = {
    'CNN': 'WORK_SPACE/paper/E2_matched_is_chrono/preds/pred_cnn_oos.csv',
    'MLP': 'WORK_SPACE/paper/E2_matched_is_chrono/preds/pred_mlp_oos.csv',
    '1D-CNN': 'WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv',
}


def sid_norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.split('.').str[0].str.zfill(6)


def ann_stats(x: pd.Series) -> dict:
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return {'mean_period': np.nan, 'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 'n': int(len(x))}
    mean = float(x.mean())
    vol = float(x.std(ddof=1))
    return {
        'mean_period': mean,
        'ann_return': float((1.0 + mean) ** ANN - 1.0),
        'ann_vol': float(vol * np.sqrt(ANN)),
        'sharpe': float(mean / vol * np.sqrt(ANN)) if vol > 1e-12 else np.nan,
        'n': int(len(x)),
    }


def rank_ic(df: pd.DataFrame, score: str) -> dict:
    vals = []
    for _, g in df.groupby('ending_date', sort=True):
        if len(g) < 10:
            continue
        vals.append(g[score].rank().corr(g['ret_val'].rank()))
    s = pd.Series(vals).dropna()
    return {'rank_ic': float(s.mean()) if len(s) else np.nan, 'n_dates': int(len(s))}


def portfolio(df: pd.DataFrame, score: str, mask: pd.Series | None = None) -> dict:
    d = df if mask is None else df.loc[mask]
    rows = []
    for dt, g in d.groupby('ending_date', sort=True):
        if len(g) < 20:
            continue
        g = g.copy()
        q = g[score].rank(method='first', pct=True)
        hi, lo = q >= 0.9, q <= 0.1
        if hi.sum() == 0 or lo.sum() == 0:
            continue
        rows.append({
            'ending_date': dt,
            'top': float(g.loc[hi, 'ret_val'].mean()),
            'bottom': float(g.loc[lo, 'ret_val'].mean()),
            'hl': float(g.loc[hi, 'ret_val'].mean() - g.loc[lo, 'ret_val'].mean()),
            'market': float(g['ret_val'].mean()),
        })
    r = pd.DataFrame(rows)
    if r.empty:
        return {'rank_ic': np.nan, 'top_sharpe': np.nan, 'bottom_sharpe': np.nan, 'hl_sharpe': np.nan,
                'top_market_sharpe': np.nan, 'n_dates': 0}
    out = {'rank_ic': rank_ic(d, score)['rank_ic'], 'n_dates': int(len(r))}
    for col in ['top', 'bottom', 'hl']:
        out[f'{col}_sharpe'] = ann_stats(r[col])['sharpe']
    out['top_market_sharpe'] = ann_stats(r['top'] - r['market'])['sharpe']
    return out


def cost_test(df: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    prev = {}
    for dt, g in df.groupby('ending_date', sort=True):
        g = g.copy()
        q = g[score].rank(method='first', pct=True)
        long_ids = set(g.loc[q >= .9, 'sid'])
        short_ids = set(g.loc[q <= .1, 'sid'])
        cur = {sid: (0.5 / max(1, len(long_ids)) if sid in long_ids else (-0.5 / max(1, len(short_ids)) if sid in short_ids else 0.0))
               for sid in set(g['sid'])}
        all_ids = set(prev) | set(cur)
        turnover = 0.5 * sum(abs(cur.get(s, 0.0) - prev.get(s, 0.0)) for s in all_ids)
        top = g.loc[q >= .9, 'ret_val'].mean() if long_ids else np.nan
        bottom = g.loc[q <= .1, 'ret_val'].mean() if short_ids else np.nan
        rows.append({'ending_date': dt, 'gross': top - bottom, 'turnover': turnover})
        prev = cur
    r = pd.DataFrame(rows)
    out = []
    for bps in [0, 10, 25, 50, 100]:
        net = r['gross'] - r['turnover'] * bps / 10000.0
        s = ann_stats(net)
        out.append({'cost_bps_one_way': bps, **s, 'mean_turnover': float(r['turnover'].mean())})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='WORK_SPACE/paper/E2D_liquidity_audit')
    ap.add_argument('--mlp-pred', default=None,
                    help='Optional project-relative or absolute MLP prediction CSV.')
    ap.add_argument('--mlp-label', default='MLP',
                    help='Display label for the MLP prediction input.')
    args = ap.parse_args()
    out_root = os.path.join(PROJECT_ROOT, args.out.replace('/', os.sep))
    os.makedirs(out_root, exist_ok=True)

    models = dict(MODELS)
    if args.mlp_pred:
        models[args.mlp_label] = models.pop('MLP')
        models[args.mlp_label] = args.mlp_pred

    frames = []
    for name, rel in models.items():
        p = rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel.replace('/', os.sep))
        d = pd.read_csv(p, usecols=['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob'])
        d['ending_date'] = pd.to_datetime(d['ending_date'])
        d['sid'] = sid_norm(d['StockID'])
        d['model'] = name
        frames.append(d)
    base = frames[0][['sid', 'ending_date', 'ret_val', 'MarketCap']].copy()
    for d in frames[1:]:
        base = base.merge(d[['sid', 'ending_date']], on=['sid', 'ending_date'], how='inner')
    base = base.drop_duplicates(['sid', 'ending_date'])

    daily_path = os.path.join(PROJECT_ROOT, 'WORK_SPACE/data/ashare_processed_data/ashare_ret.feather')
    daily = pd.read_feather(daily_path, columns=['Date', 'StockID', 'Close', 'Vol'])
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily['sid'] = sid_norm(daily['StockID'])
    daily['amount_proxy'] = daily['Close'].abs() * daily['Vol'].abs()
    liq = daily[['sid', 'Date', 'amount_proxy']].rename(columns={'Date': 'ending_date'})
    base = base.merge(liq, on=['sid', 'ending_date'], how='left')
    base['mcap_rank'] = base.groupby('ending_date')['MarketCap'].rank(pct=True)
    base['liq_rank'] = base.groupby('ending_date')['amount_proxy'].rank(pct=True)
    base.to_parquet(os.path.join(out_root, 'audit_panel.parquet'), index=False)

    summary, strata, costs = [], [], []
    for d, name in zip(frames, models):
        x = d.merge(base[['sid', 'ending_date', 'mcap_rank', 'liq_rank', 'amount_proxy']], on=['sid', 'ending_date'], how='inner')
        for label, mask in {
            'Full': pd.Series(True, index=x.index),
            'Largest50': x.mcap_rank >= .5,
            'Largest30': x.mcap_rank >= .7,
            'Liquid50': x.liq_rank >= .5,
            'Liquid30': x.liq_rank >= .7,
            'LargeLiquid': (x.mcap_rank >= .5) & (x.liq_rank >= .5),
        }.items():
            z = portfolio(x, 'up_prob', mask)
            summary.append({'model': name, 'universe': label, **z, 'mean_mcap': float(x.loc[mask, 'MarketCap'].groupby(x.loc[mask, 'ending_date']).mean().mean()),
                            'mean_amount_proxy': float(x.loc[mask, 'amount_proxy'].groupby(x.loc[mask, 'ending_date']).mean().mean())})
        x['bucket'] = pd.cut(x['mcap_rank'], [0, 1/3, 2/3, 1], labels=['SmallCap', 'MidCap', 'LargeCap'], include_lowest=True)
        x['liq_bucket'] = pd.cut(x['liq_rank'], [0, 1/3, 2/3, 1], labels=['Illiquid', 'MidLiquid', 'Liquid'], include_lowest=True)
        for var in ['bucket', 'liq_bucket']:
            for bucket, g in x.groupby(var, observed=True):
                z = portfolio(g, 'up_prob')
                strata.append({'model': name, 'dimension': var, 'bucket': str(bucket), **z})
        costs.append(cost_test(x, 'up_prob').assign(model=name))

    pd.DataFrame(summary).to_csv(os.path.join(out_root, 'universe_summary.csv'), index=False)
    pd.DataFrame(strata).to_csv(os.path.join(out_root, 'strata_summary.csv'), index=False)
    pd.concat(costs, ignore_index=True).to_csv(os.path.join(out_root, 'cost_sensitivity.csv'), index=False)
    report = {'models': list(models), 'n_common_rows': int(len(base)), 'n_common_dates': int(base.ending_date.nunique()),
              'date_min': str(base.ending_date.min().date()), 'date_max': str(base.ending_date.max().date()),
              'liquidity_proxy': 'abs(Close) * abs(Vol)', 'annualization': '52 weekly periods'}
    with open(os.path.join(out_root, 'audit_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
