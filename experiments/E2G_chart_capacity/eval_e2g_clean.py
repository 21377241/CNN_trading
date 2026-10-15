# -*- coding: utf-8 -*-
"""E2G（干净协议）：参数量对标的 ChartCNN-S/M 与完整 chart、1D-CNN、MLP 在 S0 口径下对比。

S0：train 2000–2005 / val 2006–2007 选定，固定权重预测 2008–2026（= R2008）。
选定规则同主分析 chart：phase2 best_val_rank_ic 最大；平局取较小 patience、再取较大 lr。

  python experiments/E2G_chart_capacity/eval_e2g_clean.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ashare_train.eval_ew_main import eval_model  # noqa: E402
from ashare_train.eval_s0_s1 import load_s0  # noqa: E402

E2G = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E2G_clean' / 'R2008'
OUT = E2G / 'eval'
ARCHS = {
    'chartcnn_s': {'channels': '24-48-96', 'n_parameters': 121826, 'target': 'oned', 'target_parameters': 125825},
    'chartcnn_m': {'channels': '10-20-40', 'n_parameters': 29762, 'target': 'mlp', 'target_parameters': 28929},
}
N_PARAMS_BASE = {'chart': 708866, 'oned': 125825, 'mlp': 28929}
COLS = ['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']
TOL = 1e-4


def select(arch: str) -> dict:
    p2 = pd.read_csv(E2G / arch / 'search_phase2.csv')
    best = p2['best_val_rank_ic'].max()
    tied = p2[p2['best_val_rank_ic'] >= best - TOL].sort_values(['patience', 'lr'], ascending=[True, False])
    row = tied.iloc[0]
    p3 = pd.read_csv(E2G / arch / 'phase3_full_train.csv')
    hit = p3[np.isclose(p3['lr'], row['lr']) & np.isclose(p3['dropout'], row['dropout'])
             & (p3['patience'] == row['patience'])]
    if hit.empty:
        raise RuntimeError(f'{arch}: selected HP not full-trained')
    return {'lr': float(row['lr']), 'dropout': float(row['dropout']), 'patience': int(row['patience']),
            'val_rank_ic': float(row['best_val_rank_ic']), 'n_tied_within_tol': int(len(tied)),
            'exp_dir': hit.iloc[0]['exp_dir']}


def load_e2g(exp_dir: str) -> pd.DataFrame:
    parts = []
    for y in range(2008, 2027):
        f = Path(exp_dir) / 'ensem_res' / f'ensem5_res_{y}.csv'
        df = pd.read_csv(f, dtype={'StockID': str}, usecols=COLS)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out['StockID'] = out['StockID'].str.zfill(6)
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    return out


def train_diag(exp_dir: str) -> dict:
    rows = []
    for f in sorted(Path(exp_dir).glob('history_seed*.csv')):
        h = pd.read_csv(f)
        b = h.loc[h['val_rank_ic'].idxmax()]
        rows.append({'best_epoch': int(b['epoch']), 'stop_epoch': int(h['epoch'].max()),
                     'gap': float(b['val_loss'] - b['train_loss'])})
    d = pd.DataFrame(rows)
    return {'best_epoch_mean': d['best_epoch'].mean(), 'stop_epoch_mean': d['stop_epoch'].mean(),
            'val_minus_train_loss': d['gap'].mean()}


def mean_spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    m = a[['StockID', 'ending_date', 'up_prob']].merge(
        b[['StockID', 'ending_date', 'up_prob']], on=['StockID', 'ending_date'], suffixes=('_a', '_b'))
    rhos = [g['up_prob_a'].corr(g['up_prob_b'], method='spearman')
            for _, g in m.groupby('ending_date') if len(g) >= 10]
    return float(np.nanmean(rhos))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    preds: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for m in ('chart', 'oned', 'mlp'):
        preds[m] = load_s0(m)[COLS]
        meta[m] = {'n_parameters': N_PARAMS_BASE[m]}
    for arch, info in ARCHS.items():
        sel = select(arch)
        preds[arch] = load_e2g(sel['exp_dir'])
        meta[arch] = {**info, **sel, **train_diag(sel['exp_dir'])}
    (OUT / 'selection.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    rows, yearly = [], []
    for name, df in preds.items():
        r = eval_model(name, df)
        rows.append({**{k: v for k, v in r.items() if not k.startswith('_')},
                     'n_parameters': meta[name]['n_parameters']})
        yearly.append(r['_yearly'].rename(name))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / 'summary.csv', index=False)
    yr = pd.concat(yearly, axis=1)
    yr.to_csv(OUT / 'yearly_rank_ic.csv')

    corr = pd.DataFrame([
        {'pair': f'{a}-{b}', 'rho': mean_spearman(preds[a], preds[b])}
        for a, b in [('chartcnn_s', 'chart'), ('chartcnn_m', 'chart'), ('chartcnn_s', 'chartcnn_m'),
                     ('chartcnn_s', 'oned'), ('chartcnn_m', 'mlp'), ('chart', 'oned')]
    ])
    corr.to_csv(OUT / 'score_rank_corr.csv', index=False)

    cols = ['model', 'n_parameters', 'rank_ic_mean', 'nw_t', 'ew_hl_sharpe', 'ew_hl_ann', 'vw_hl_sharpe', 'n_dates']
    print(summ[cols].to_markdown(index=False, floatfmt='.4f'))
    print()
    print(corr.to_markdown(index=False, floatfmt='.3f'))
    print()
    print(yr.to_markdown(floatfmt='.4f'))
    print()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'exp_dir'} for k, v in meta.items()},
                     indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
