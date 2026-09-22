# -*- coding: utf-8 -*-
"""Revalue existing E4 portfolios without retraining or generating plots."""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'WORK_SPACE', 'paper', 'E4_recent_path_chrono_ric')
ANN = 52


def stats(x):
    x = pd.Series(x).dropna()
    m, s = float(x.mean()), float(x.std(ddof=1))
    return {'mean': m, 'ann_return': (1 + m) ** ANN - 1, 'ann_vol': s * np.sqrt(ANN),
            'sharpe': m / s * np.sqrt(ANN) if s > 1e-12 else np.nan, 'n': len(x)}


def value(path, make_plots=True):
    d = pd.read_csv(path, usecols=['ending_date', 'ret_val', 'MarketCap', 'up_prob'])
    d = d.dropna(subset=['ending_date', 'ret_val', 'up_prob'])
    d['ending_date'] = pd.to_datetime(d['ending_date'])
    ew, vw = [], []
    for dt, g in d.groupby('ending_date', sort=True):
        if len(g) < 10:
            continue
        g = g.copy()
        g['decile'] = pd.qcut(g['up_prob'].rank(method='first'), 10, labels=False) + 1
        lo, hi = g[g.decile == 1], g[g.decile == 10]
        if len(lo) == 0 or len(hi) == 0:
            continue
        ew.append((hi.ret_val.mean() - lo.ret_val.mean()))
        vw.append((np.average(hi.ret_val, weights=np.maximum(hi.MarketCap, 0)) -
                   np.average(lo.ret_val, weights=np.maximum(lo.MarketCap, 0))))
    ew_s, vw_s = stats(ew), stats(vw)
    if make_plots:
        plot_dir = os.path.dirname(path)
        # Reconstruct decile returns once for lightweight figure regeneration.
        d['decile'] = d.groupby('ending_date')['up_prob'].rank(method='first', pct=True)
        d['decile'] = np.ceil(d['decile'] * 10).clip(1, 10).astype(int)
        dec = d.groupby(['ending_date', 'decile'])['ret_val'].mean().unstack()
        dec['H-L'] = dec[10] - dec[1]
        dec.to_csv(os.path.join(plot_dir, 'ew_standard_returns.csv'))
        # Keep the existing figure names so downstream LaTeX links remain valid.
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(1, 11), [stats(dec[i].dropna())['ann_return'] for i in range(1, 11)], color='steelblue')
        ax.set_xlabel('Prediction decile'); ax.set_ylabel('Compounded annual return')
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, '1_decile_returns.png'), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(1, 11), [stats(dec[i].dropna())['sharpe'] for i in range(1, 11)], color='darkorange')
        ax.axhline(0, color='black', linewidth=.8); ax.set_xlabel('Prediction decile'); ax.set_ylabel('Standard Sharpe')
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, '2_sharpe_ratios.png'), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dec.index, dec['H-L'].cumsum(), label='EW H-L cumulative sum')
        ax.axhline(0, color='black', linewidth=.8); ax.legend(); ax.set_ylabel('Cumulative period return')
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, '3_hl_comparison.png'), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dec.index, (1 + dec['H-L'].fillna(0)).cumprod(), label='EW H-L wealth index')
        ax.legend(); ax.set_ylabel('Wealth index'); fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, '4_monthly_returns_timeseries.png'), dpi=150); plt.close(fig)
    return ew_s, vw_s


def write_summary(out_dir, ew, vw):
    lines = ['=' * 60, '         Standard Sharpe portfolio audit', '=' * 60]
    for title, s in [('等权 (Equal-Weight)', ew), ('市值加权 (Value-Weight)', vw)]:
        lines += ['', title + '：', f"{'组合':>8} {'周期均值':>10} {'年化收益':>10} {'年化波动':>10} {'Sharpe':>10} {'样本数':>8}", '-' * 60]
        lines.append(f"{'H-L':>8} {s['mean']:>10.4f} {s['ann_return']:>10.4f} {s['ann_vol']:>10.4f} {s['sharpe']:>10.4f} {s['n']:>8}")
    with open(os.path.join(out_dir, 'portfolio_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    files = glob.glob(os.path.join(OUT, '**', 'pred_for_port.csv'), recursive=True)
    rows = []
    for i, path in enumerate(files, 1):
        out_dir = os.path.dirname(path)
        ew, vw = value(path)
        write_summary(out_dir, ew, vw)
        rel = os.path.relpath(out_dir, OUT).replace(os.sep, '/')
        rows.append({'directory': rel, 'ew_hl_sharpe': ew['sharpe'], 'vw_hl_sharpe': vw['sharpe'],
                     'ew_hl_ann': ew['ann_return'], 'vw_hl_ann': vw['ann_return'], 'n_dates': ew['n']})
        print(f'[{i}/{len(files)}] {rel} ew={ew["sharpe"]:.4f}', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e4_standard_sharpe_audit.csv'), index=False)
    print(f'wrote {len(rows)} E4 portfolio summaries', flush=True)


if __name__ == '__main__':
    main()
