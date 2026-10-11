# -*- coding: utf-8 -*-
"""
eval_clean_cnn_impact.py — 阶段 4a：图像缺陷与数据修正对 chart-CNN 的影响（clean_spec 第 6 节）。

  M0：旧模型（E1b_cnn_hp_I20R5_ric，lr 1e-4 / dropout 0.5 / patience 5，5 模型）
  M1：旧数据流程 + Jiang 渲染器，旧超参数与旧训练流程（E1c_cnn_M1_rendfix）
  M2：完整干净版（E1c_cnn_hp_I20R5_clean 第三阶段按验证 Rank IC 选中的组合）

三个样本口径：
  A  旧样本（M0 的 OOS 样本键）、旧标签 ret_val、旧 MarketCap：只比较 M0 与 M1，即图像缺陷单独的影响
  B  共同样本：A 与干净 universe（OOS 且有标签）的交集，统一用干净标签 ret_label 和流通市值：三个模型
  C  各自原生样本：M2 用干净 universe 的全部 OOS 样本（有标签）

用法:
  python experiments/E1_image_baseline/eval_clean_cnn_impact.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
for p in (CODE_PKG, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from Misc.config import (ASHARE_STOCKS_SAVEPATH, ASHARE_COUNTRY, ASHARE_UNIVERSE_PATH, OLD_DATA_WORK_DIR,
                         PRE_CLEAN_WORK_DIR, MA_LAGS, VOLUME_BAR)
from Data.dgp_config import get_dataset_dir
from Experiments.cnn_experiment import _dataset, _loader, _predict, _mean_rank_ic, _get_device
from Model.cnn_model import init_model, get_full_model_name
from experiments.common.metrics import newey_west_t, annualized_sharpe

OOS_YEARS_OLD = list(range(2008, 2026))    # M0、M1：旧数据止于 2025
OOS_YEARS_CLEAN = list(range(2008, 2027))  # M2：CSMAR 数据，止于 SAMPLE_END
VAL_YEARS = [2006, 2007]
LEAF = 'D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C256'
M0_DIR = os.path.join(PRE_CLEAN_WORK_DIR, 'paper', 'E1b_cnn_hp_I20R5_ric', LEAF,
                      '20d5p-lr1E-4-dp0.50-maTrue-vbTrue-weekTrained-noDelayedReturn-ric-full-p5')
M1_DIR = os.path.join(OLD_DATA_WORK_DIR, 'paper', 'E1c_cnn_M1_rendfix', LEAF,
                      '20d5p-lr1E-4-dp0.50-maTrue-vbTrue-weekTrained-noDelayedReturn-ric-full-p5')
M2_ROOT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E1c_cnn_hp_I20R5_clean')
OUT_DIR = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'paper', 'E1c_clean_cnn_impact')
DATASETS = {
    'M0': (get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(PRE_CLEAN_WORK_DIR, 'data', 'ashare_stocks_dataset', 'I20R5')), 'legacy'),
    'M1': (get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(OLD_DATA_WORK_DIR, 'data', 'ashare_stocks_dataset',
                                                                      'I20R5_rendfix')), 'legacy'),
    'M2': (get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, 'I20R5_clean')), 'train'),
}


def m2_dir() -> tuple[str, dict]:
    p3 = pd.read_csv(os.path.join(M2_ROOT, 'phase3_full_train.csv'))
    p2 = pd.read_csv(os.path.join(M2_ROOT, 'search_phase2.csv'))
    best = p2.sort_values('best_val_rank_ic', ascending=False).iloc[0]
    row = p3[(np.isclose(p3['lr'], best['lr'])) & (np.isclose(p3['dropout'], best['dropout']))
             & (p3['patience'] == best['patience'])].iloc[0]
    return row['exp_dir'], {'lr': float(row['lr']), 'dropout': float(row['dropout']), 'patience': int(row['patience'])}


def load_preds(exp_dir: str, years: list[int], n: int = 5) -> pd.DataFrame:
    files = [os.path.join(exp_dir, 'ensem_res', f'ensem{n}_res_{y}.csv') for y in years]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f'缺少 {len(missing)} 个预测文件，例如 {missing[0]}')
    df = pd.concat([pd.read_csv(f, dtype={'StockID': str}) for f in files], ignore_index=True)
    df['StockID'] = df['StockID'].str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    return df


def weekly_stats(df: pd.DataFrame, score: str, ret: str, mcap: str) -> pd.DataFrame:
    """每周：Rank IC、十分位 EW/VW H-L 收益。"""
    d = df[['ending_date', score, ret, mcap]].dropna(subset=[score, ret]).copy()
    d['rs'] = d.groupby('ending_date')[score].rank()
    d['rr'] = d.groupby('ending_date')[ret].rank()
    first = d.groupby('ending_date')[score].rank(method='first')
    n = d.groupby('ending_date')['rs'].transform('size')
    d['dec'] = np.minimum((first - 1) * 10 // n, 9).astype(int)
    g = d.groupby('ending_date')
    ric = g[['rs', 'rr']].corr().xs('rs', level=1)['rr']
    d['w'] = d[mcap].where(d[mcap] > 0)
    d['wr'] = d['w'] * d[ret]
    ext = d[d['dec'].isin([0, 9])]
    gg = ext.groupby(['ending_date', 'dec'])
    ew = gg[ret].mean().unstack()
    vw = (gg['wr'].sum() / gg['w'].sum()).unstack()
    out = pd.DataFrame({'rank_ic': ric, 'ew_hl': ew[9] - ew[0], 'vw_hl': vw[9] - vw[0], 'n': g.size()})
    return out[out['n'] >= 10]


def summarize(w: pd.DataFrame) -> dict:
    ic = w['rank_ic'].to_numpy()
    return {
        'n_weeks': int(len(w)),
        'n_obs': int(w['n'].sum()),
        'rank_ic': float(np.nanmean(ic)),
        'rank_ic_nw_t': newey_west_t(ic)[0],
        'icir_ann': float(np.nanmean(ic) / np.nanstd(ic, ddof=1) * np.sqrt(52)),
        'ew_hl_mean': float(w['ew_hl'].mean()),
        'ew_hl_sharpe': annualized_sharpe(w['ew_hl'].to_numpy()),
        'ew_hl_nw_t': newey_west_t(w['ew_hl'].to_numpy())[0],
        'vw_hl_mean': float(w['vw_hl'].mean()),
        'vw_hl_sharpe': annualized_sharpe(w['vw_hl'].to_numpy()),
        'vw_hl_nw_t': newey_west_t(w['vw_hl'].to_numpy())[0],
    }


def pred_rank_corr(df: pd.DataFrame, a: str, b: str) -> float:
    d = df[['ending_date', a, b]].dropna()
    r = d.groupby('ending_date')[[a, b]].rank()
    r['ending_date'] = d['ending_date']
    c = r.groupby('ending_date')[[a, b]].corr().xs(a, level=1)[b]
    return float(c.mean())


def diff_t(wa: pd.DataFrame, wb: pd.DataFrame, col: str) -> tuple[float, float]:
    j = wa[[col]].join(wb[[col]], lsuffix='_a', rsuffix='_b', how='inner')
    x = (j[f'{col}_a'] - j[f'{col}_b']).to_numpy()
    return float(np.nanmean(x)), newey_west_t(x)[0]


def plot_yearly(yearly: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(yearly))
    cols = ['M0', 'M1', 'M2']
    labels = {'M0': 'M0 old', 'M1': 'M1 renderer fixed', 'M2': 'M2 clean'}
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1) * 0.27, yearly[c], width=0.27, label=labels[c])
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index.astype(str), rotation=45)
    ax.set_ylabel('mean weekly Rank IC')
    ax.set_title('Yearly OOS Rank IC on the common sample (clean labels)')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def val_rank_ic(exp_dir: str, model_key: str, n: int = 5) -> dict:
    ds_dir, mode = DATASETS[model_key]
    ds = _dataset(VAL_YEARS, 20, 5, 'week', VOLUME_BAR, MA_LAGS, ASHARE_COUNTRY, ds_dir, mode)
    loader = _loader(ds, 2048, shuffle=False)
    device = _get_device()
    ups, per_seed = [], []
    for i in range(n):
        ck = torch.load(os.path.join(exp_dir, f'checkpoint{i}.pth.tar'), map_location='cpu', weights_only=False)
        m = init_model(get_full_model_name(20), img_h=64, img_w=60, dropout=0.0, n_classes=2, window_size=20).to(device)
        m.load_state_dict(ck['model_state_dict'])
        up, _ = _predict(m, loader, device)
        ups.append(up)
        per_seed.append({'seed': i, 'epoch': ck.get('epoch'), 'val_loss': ck.get('val_loss'),
                         'val_rank_ic': _mean_rank_ic(ds.meta, up)})
    return {'ensemble_val_rank_ic': _mean_rank_ic(ds.meta, np.mean(ups, axis=0)),
            'n_val': int(len(ds)), 'val_up_share': ds.up_share, 'per_seed': per_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-val', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    m2, m2_hp = m2_dir()
    dirs = {'M0': M0_DIR, 'M1': M1_DIR, 'M2': m2}
    preds = {k: load_preds(v, OOS_YEARS_CLEAN if k == 'M2' else OOS_YEARS_OLD) for k, v in dirs.items()}
    key = ['StockID', 'ending_date']

    u = pd.read_feather(ASHARE_UNIVERSE_PATH, columns=['StockID', 'ending_date', 'split', 'ret_label', 'label',
                                                     'board', 'is_bj', 'MarketCap'])
    u['StockID'] = u['StockID'].astype(str)
    u['ending_date'] = pd.to_datetime(u['ending_date'])
    u = u[(u['split'] == 'oos') & u['ret_label'].notna() & ~u['is_bj']]
    u = u.drop(columns='is_bj').rename(columns={'MarketCap': 'mcap_clean'})

    # A：旧样本、旧标签
    a = preds['M0'][key + ['ret_val', 'MarketCap', 'up_prob']].rename(columns={'up_prob': 'M0'})
    a = a.merge(preds['M1'][key + ['up_prob']].rename(columns={'up_prob': 'M1'}), on=key, how='inner')
    if len(a) != len(preds['M0']) or len(a) != len(preds['M1']):
        raise RuntimeError(f'M0/M1 样本键不一致：{len(preds["M0"])} / {len(preds["M1"])} / 交集 {len(a)}')
    # B：共同样本、干净标签
    b = a.drop(columns=['ret_val', 'MarketCap']).merge(u, on=key, how='inner')
    b = b.merge(preds['M2'][key + ['up_prob']].rename(columns={'up_prob': 'M2'}), on=key, how='inner')
    # C：M2 原生样本
    c = preds['M2'][key + ['up_prob']].rename(columns={'up_prob': 'M2'}).merge(u, on=key, how='inner')

    weekly, rows = {}, []
    for panel, df, models, ret, mcap in [('A', a, ['M0', 'M1'], 'ret_val', 'MarketCap'),
                                          ('B', b, ['M0', 'M1', 'M2'], 'ret_label', 'mcap_clean'),
                                          ('C', c, ['M2'], 'ret_label', 'mcap_clean')]:
        for m in models:
            w = weekly_stats(df, m, ret, mcap)
            weekly[(panel, m)] = w
            rows.append({'panel': panel, 'model': m, **summarize(w)})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, float_format='%.6g')

    diffs = []
    for panel, pairs in [('A', [('M1', 'M0')]), ('B', [('M1', 'M0'), ('M2', 'M1'), ('M2', 'M0')])]:
        for x, y in pairs:
            for col in ('rank_ic', 'ew_hl', 'vw_hl'):
                d, t = diff_t(weekly[(panel, x)], weekly[(panel, y)], col)
                diffs.append({'panel': panel, 'diff': f'{x}-{y}', 'metric': col, 'mean_diff': d, 'nw_t': t})
    pd.DataFrame(diffs).to_csv(os.path.join(OUT_DIR, 'differences.csv'), index=False, float_format='%.6g')

    corr = {'A_M0_M1': pred_rank_corr(a, 'M0', 'M1'), 'B_M0_M1': pred_rank_corr(b, 'M0', 'M1'),
            'B_M1_M2': pred_rank_corr(b, 'M1', 'M2'), 'B_M0_M2': pred_rank_corr(b, 'M0', 'M2')}

    yearly = pd.concat({m: weekly[('B', m)]['rank_ic'].groupby(weekly[('B', m)].index.year).mean()
                        for m in ('M0', 'M1', 'M2')}, axis=1)
    yearly['M2_native'] = weekly[('C', 'M2')]['rank_ic'].groupby(weekly[('C', 'M2')].index.year).mean()
    yearly.to_csv(os.path.join(OUT_DIR, 'yearly_rank_ic.csv'), float_format='%.6g')
    plot_yearly(yearly, os.path.join(PROJECT_ROOT, 'docs', 'figures', 'clean_cnn_impact_yearly_ric.png'))

    sub_rows = []
    for name, df, models in [('B', b, ['M0', 'M1', 'M2']), ('C', c, ['M2'])]:
        for board in ('main', 'chinext', 'star'):
            s = df[(df['ending_date'] >= '2020-01-01') & (df['board'] == board)]
            for m in models:
                w = weekly_stats(s, m, 'ret_label', 'mcap_clean')
                sub_rows.append({'panel': name, 'board': board, 'model': m, 'n_obs': int(len(s)),
                                 **{k: v for k, v in summarize(w).items() if k in
                                    ('n_weeks', 'rank_ic', 'rank_ic_nw_t', 'ew_hl_sharpe', 'vw_hl_sharpe')}})
    pd.DataFrame(sub_rows).to_csv(os.path.join(OUT_DIR, 'boards_post2020.csv'), index=False, float_format='%.6g')

    out = {'dirs': dirs, 'm2_hp': m2_hp, 'n_obs': {'A': len(a), 'B': len(b), 'C': len(c)},
           'pred_rank_corr': corr}
    if not args.skip_val:
        out['val'] = {k: val_rank_ic(v, k) for k, v in dirs.items()}
    with open(os.path.join(OUT_DIR, 'impact.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    pd.set_option('display.width', 200)
    print(summary.to_string(index=False))
    print(pd.DataFrame(diffs).to_string(index=False))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(yearly.round(4).to_string())
    print(pd.DataFrame(sub_rows).round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
