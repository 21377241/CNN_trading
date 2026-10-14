# -*- coding: utf-8 -*-
"""主分析 EW OOS：锁定 chart 选定 → 拼接三模型预测 → Rank IC / H-L Sharpe。

  python ashare_train/eval_ew_main.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ashare_train.ew_splits import RETRAIN_YEARS, ew_split  # noqa: E402
from experiments.common.metrics import (  # noqa: E402
    annualized_sharpe,
    cross_section_ic,
    newey_west_t,
    summarize_ic,
    yearly_rank_ic_mean,
)

EW_ROOT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'ew_main'
OUT = EW_ROOT / 'eval'
TIE_TOL = 1e-4
KEEP = ['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob', 'retrain_year']


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def select_chart(R: int) -> dict:
    """phase2：5 成员验证 Rank IC 均值最高；差 < 1e-4 时取更小 patience，再取更大 lr。"""
    root = EW_ROOT / f'R{R}' / 'chart'
    p2 = pd.read_csv(root / 'search_phase2.csv')
    col = 'best_val_rank_ic'
    best = float(p2[col].max())
    tied = p2[p2[col] >= best - TIE_TOL].copy()
    tied = tied.sort_values(['patience', 'lr', col], ascending=[True, False, False])
    pick = tied.iloc[0]
    lr, dp, pat = float(pick['lr']), float(pick['dropout']), int(pick['patience'])

    p3 = pd.read_csv(root / 'phase3_full_train.csv')
    match = p3[(np.isclose(p3['lr'], lr)) & (np.isclose(p3['dropout'], dp)) & (p3['patience'] == pat)]
    if match.empty:
        raise RuntimeError(f'R{R} chart phase3 missing selected lr={lr} dp={dp} pat={pat}')
    exp_dir = Path(match.iloc[0]['exp_dir'])
    split = ew_split(R)
    sel = {
        'model': 'chart',
        'retrain_year': R,
        'hp': {'lr': lr, 'dropout': dp, 'patience': pat},
        'val_rank_ic': float(pick[col]),
        'n_tied_within_tol': int(len(tied)),
        'rule': 'max phase2 best_val_rank_ic (5-member mean); ties (<1e-4) -> smaller patience, then larger lr',
        'exp_dir': str(exp_dir),
        'train_years': split['train_years'],
        'val_years': split['val_years'],
        'pred_years': split['pred_years'],
        'ensem_sha1': {},
    }
    for y in split['pred_years']:
        f = exp_dir / 'ensem_res' / f'ensem5_res_{y}.csv'
        if not f.is_file():
            raise FileNotFoundError(f)
        sel['ensem_sha1'][str(y)] = sha1_file(f)
    path = root / 'selection.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sel, f, indent=2)
    return sel


def load_chart_window(R: int, sel: dict) -> pd.DataFrame:
    years = sel['pred_years']
    exp_dir = Path(sel['exp_dir'])
    parts = []
    for y in years:
        df = pd.read_csv(exp_dir / 'ensem_res' / f'ensem5_res_{y}.csv', dtype={'StockID': str})
        df['StockID'] = df['StockID'].astype(str).str.zfill(6)
        df['ending_date'] = pd.to_datetime(df['ending_date'])
        df = df[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']].copy()
        df['retrain_year'] = R
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_numeric_window(R: int, model: str) -> pd.DataFrame:
    root = EW_ROOT / f'R{R}' / model
    split = ew_split(R)
    preds = list((root / 'preds').glob('*.parquet'))
    if not preds:
        raise FileNotFoundError(f'no preds under {root / "preds"}')
    # R2008 链接的是全段 OOS；其余窗文件名已带年份
    df = pd.read_parquet(preds[0])
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    years = set(split['pred_years'])
    df = df[df['ending_date'].dt.year.isin(years)].copy()
    if 'up_prob' not in df.columns:
        raise KeyError('up_prob')
    out = df[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']].copy()
    out['retrain_year'] = R
    # 记录 selection sha
    sel_path = root / 'selection.json'
    meta = {'retrain_year': R, 'model': model, 'pred_file': str(preds[0].name),
            'pred_sha1': sha1_file(preds[0]), 'n_rows': int(len(out)),
            'pred_years': split['pred_years']}
    if sel_path.is_file():
        meta['selection'] = json.loads(sel_path.read_text(encoding='utf-8'))
    with open(root / 'oos_segment.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return out


def stitch_model(model: str) -> pd.DataFrame:
    parts, segments = [], []
    for R in RETRAIN_YEARS:
        if model == 'chart':
            sel = select_chart(R)
            part = load_chart_window(R, sel)
            segments.append({
                'retrain_year': R, 'hp': sel['hp'], 'val_rank_ic': sel['val_rank_ic'],
                'exp_dir': sel['exp_dir'], 'n_rows': len(part),
                'pred_years': sel['pred_years'], 'ensem_sha1': sel['ensem_sha1'],
            })
        else:
            part = load_numeric_window(R, model)
            segments.append(json.loads((EW_ROOT / f'R{R}' / model / 'oos_segment.json').read_text()))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    # 键唯一：同一 (StockID, ending_date) 只应来自一个窗
    dup = out.duplicated(['StockID', 'ending_date'], keep=False)
    if dup.any():
        raise RuntimeError(f'{model}: duplicate keys {dup.sum()}')
    out = out.sort_values(['ending_date', 'StockID']).reset_index(drop=True)
    tag = {'chart': 'chartcnn_ew', 'oned': 'onedcnn_140_ew', 'mlp': 'mlp_140_ew'}[model]
    pred_path = OUT / f'pred_{model}_ew_oos.parquet'
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pred_path, index=False)
    manifest = {
        'registry_tag': tag,
        'n_rows': len(out),
        'date_min': str(out['ending_date'].min().date()),
        'date_max': str(out['ending_date'].max().date()),
        'n_dates': int(out['ending_date'].nunique()),
        'pred_sha1': sha1_file(pred_path),
        'segments': segments,
    }
    with open(OUT / f'manifest_{model}.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'{model}: wrote {pred_path} rows={len(out):,} dates={manifest["n_dates"]}')
    return out


def hl_weekly(df: pd.DataFrame, weighted: bool = False) -> pd.Series:
    rows = []
    for dt, g in df.groupby('ending_date'):
        g = g.dropna(subset=['up_prob', 'ret_val'])
        if weighted:
            g = g.dropna(subset=['MarketCap'])
            g = g[g['MarketCap'] > 0]
        if len(g) < 30:
            continue
        rnk = g['up_prob'].rank(method='first', pct=True)
        d = np.ceil(rnk * 10).clip(1, 10).astype(int)
        hi, lo = g[d == 10], g[d == 1]
        if len(hi) < 2 or len(lo) < 2:
            continue
        if weighted:
            wh = hi['MarketCap'] / hi['MarketCap'].sum()
            wl = lo['MarketCap'] / lo['MarketCap'].sum()
            spread = float((hi['ret_val'] * wh).sum() - (lo['ret_val'] * wl).sum())
        else:
            spread = float(hi['ret_val'].mean() - lo['ret_val'].mean())
        rows.append((dt, spread))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({t: v for t, v in rows}).sort_index()
    return s


def eval_model(name: str, df: pd.DataFrame) -> dict:
    ic = cross_section_ic(df, 'up_prob', 'ret_val')
    summ = summarize_ic(ic)
    t, p = newey_west_t(ic['rank_ic'].to_numpy(), lag=4)
    ew = hl_weekly(df, weighted=False)
    vw = hl_weekly(df, weighted=True)
    yearly = yearly_rank_ic_mean(df, 'up_prob', 'ret_val')
    return {
        'model': name,
        'rank_ic_mean': summ['rank_ic_mean'],
        'ic_mean': summ['ic_mean'],
        'n_dates': summ['n_dates'],
        'nw_t': t,
        'nw_p': p,
        'n_rows': len(df),
        'ew_hl_sharpe': annualized_sharpe(ew.to_numpy()),
        'ew_hl_ann': float(ew.mean() * 52) if len(ew) else float('nan'),
        'vw_hl_sharpe': annualized_sharpe(vw.to_numpy()),
        'vw_hl_ann': float(vw.mean() * 52) if len(vw) else float('nan'),
        'n_weeks_hl': int(len(ew)),
        '_weekly_ic': ic,
        '_yearly': yearly,
        '_ew_hl': ew,
    }


def score_corr(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ['chart', 'oned', 'mlp']
    merged = None
    for k in keys:
        d = dfs[k][['StockID', 'ending_date', 'up_prob']].rename(columns={'up_prob': k})
        merged = d if merged is None else merged.merge(d, on=['StockID', 'ending_date'], how='inner')
    rows = []
    for a, b in [('chart', 'oned'), ('chart', 'mlp'), ('oned', 'mlp')]:
        rhos = []
        for _, g in merged.groupby('ending_date'):
            if len(g) < 10:
                continue
            rhos.append(g[a].corr(g[b], method='spearman'))
        rows.append({'pair': f'{a}-{b}', 'rho': float(np.nanmean(rhos)), 'n_dates': len(rhos)})
    return pd.DataFrame(rows)


def write_summary(results: list[dict], yearly: pd.DataFrame, corr: pd.DataFrame) -> None:
    (OUT / 'portfolio').mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results])
    summ.to_csv(OUT / 'summary.csv', index=False)
    yearly.to_csv(OUT / 'yearly_rank_ic.csv')
    corr.to_csv(OUT / 'score_rank_corr_mean.csv', index=False)
    for r in results:
        r['_weekly_ic'].to_csv(OUT / f"weekly_rank_ic_{r['model']}.csv", index=False)
        r['_ew_hl'].to_csv(OUT / 'portfolio' / f"ew_hl_{r['model']}.csv", header=['hl'])

    lines = [
        '# 主分析扩展窗口 OOS（三年一窗 + 每窗对称搜参）',
        '',
        '切分：R∈{2008,2011,2014,2017,2020,2023}；train 2000…R−3 / val R−2…R−1 / pred R…R+2（末窗至 2026）。',
        '三模型每窗独立对称搜参；本表为拼接 OOS，选参只用验证集。',
        '',
        '## 全样本汇总',
        '',
        '| 模型 | Rank IC | NW-t | EW H-L Sharpe | EW H-L 年化 | VW H-L Sharpe | n_dates |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['rank_ic_mean']:.4f} | {r['nw_t']:.2f} | "
            f"{r['ew_hl_sharpe']:.3f} | {r['ew_hl_ann']:.3f} | {r['vw_hl_sharpe']:.3f} | {r['n_dates']} |"
        )
    lines += ['', '## 逐年 Rank IC', '', yearly.to_markdown(), '',
              '## 模型得分截面秩相关（周均值）', '', corr.to_markdown(index=False), '']
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'portfolio').mkdir(parents=True, exist_ok=True)
    dfs = {m: stitch_model(m) for m in ('chart', 'oned', 'mlp')}
    results = [eval_model(m, dfs[m]) for m in ('chart', 'oned', 'mlp')]
    yearly = pd.DataFrame({r['model']: r['_yearly'] for r in results})
    corr = score_corr(dfs)
    write_summary(results, yearly, corr)
    # 选参一览
    rows = []
    for R in RETRAIN_YEARS:
        ch = json.loads((EW_ROOT / f'R{R}' / 'chart' / 'selection.json').read_text())
        rows.append({'R': R, 'model': 'chart', 'hp': json.dumps(ch['hp']), 'val_ric': ch['val_rank_ic']})
        for m in ('oned', 'mlp'):
            s = json.loads((EW_ROOT / f'R{R}' / m / 'selection.json').read_text())
            rows.append({'R': R, 'model': m, 'hp': s.get('name', s.get('hp')), 'val_ric': s.get('mean_member_val_rank_ic')})
    pd.DataFrame(rows).to_csv(OUT / 'selections_by_window.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
