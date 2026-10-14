# -*- coding: utf-8 -*-
"""稳健性 S0 / S1 对照主分析 EW。

S0：R=2008 选定 × 预测 2008–2026（固定短训练）
S1：R=2014 选定 × 预测 2014–2026（较长训练，不重搜；切分同 R2014）
EW：主分析拼接（每窗重搜）

  python ashare_train/eval_s0_s1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ashare_train.eval_ew_main import eval_model, hl_weekly  # noqa: E402
from experiments.common.metrics import annualized_sharpe  # noqa: E402

EW = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'ew_main'
OUT = EW / 'robustness' / 'eval'
MODELS = ('chart', 'oned', 'mlp')


def load_s0(model: str) -> pd.DataFrame:
    if model == 'chart':
        sel = json.loads((EW / 'R2008' / 'chart' / 'selection.json').read_text())
        exp = Path(sel['exp_dir'])
        parts = []
        for y in range(2008, 2027):
            f = exp / 'ensem_res' / f'ensem5_res_{y}.csv'
            if not f.is_file():
                continue
            df = pd.read_csv(f, dtype={'StockID': str})
            df['StockID'] = df['StockID'].str.zfill(6)
            df['ending_date'] = pd.to_datetime(df['ending_date'])
            parts.append(df[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']])
        out = pd.concat(parts, ignore_index=True)
    else:
        preds = list((EW / 'R2008' / model / 'preds').glob('*.parquet'))
        out = pd.read_parquet(preds[0])
        out['StockID'] = out['StockID'].astype(str).str.zfill(6)
        out['ending_date'] = pd.to_datetime(out['ending_date'])
        out = out[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']]
    out['scheme'] = 'S0'
    return out


def load_s1(model: str) -> pd.DataFrame:
    root = EW / 'robustness' / 'S1' / model
    if model == 'chart':
        p = root / 'preds' / 'pred_chart_oos2014-2026.parquet'
    else:
        cand = list((root / 'preds').glob('*.parquet'))
        if not cand:
            raise FileNotFoundError(f'S1 {model} preds missing; run run_s1_predict.py first')
        p = cand[0]
    if not p.is_file():
        raise FileNotFoundError(p)
    out = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    # stitch may write parquet via relative path under PROJECT
    if not p.is_file():
        # try project-relative write location from stitch
        alt = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'ew_main' / 'robustness' / 'S1' / model / 'preds' / p.name
        out = pd.read_parquet(alt)
    out['StockID'] = out['StockID'].astype(str).str.zfill(6)
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    out = out[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']].copy()
    out['scheme'] = 'S1'
    return out


def load_ew(model: str) -> pd.DataFrame:
    p = EW / 'eval' / f'pred_{model}_ew_oos.parquet'
    out = pd.read_parquet(p)
    out['StockID'] = out['StockID'].astype(str).str.zfill(6)
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    out = out[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']].copy()
    out['scheme'] = 'EW'
    return out


def slice_years(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    y = df['ending_date'].dt.year
    return df[(y >= y0) & (y <= y1)].copy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    yearly_blocks = []

    # S0 full + EW full
    for model in MODELS:
        for scheme, loader, y0, y1 in [
            ('S0', load_s0, 2008, 2026),
            ('EW', load_ew, 2008, 2026),
        ]:
            df = slice_years(loader(model), y0, y1)
            r = eval_model(f'{scheme}/{model}', df)
            rows.append({**{k: v for k, v in r.items() if not k.startswith('_')},
                         'scheme': scheme, 'model': model, 'year_from': y0, 'year_to': y1})
            yric = r['_yearly'].rename(f'{scheme}/{model}')
            yearly_blocks.append(yric)

    # common window 2014-2026: S0, S1, EW
    s1_ok = True
    try:
        for model in MODELS:
            load_s1(model)
    except FileNotFoundError as e:
        print(f'[warn] S1 preds incomplete: {e}')
        s1_ok = False

    if s1_ok:
        for model in MODELS:
            for scheme, loader in [('S0', load_s0), ('S1', load_s1), ('EW', load_ew)]:
                df = slice_years(loader(model), 2014, 2026)
                r = eval_model(f'{scheme}/{model}@2014-2026', df)
                rows.append({**{k: v for k, v in r.items() if not k.startswith('_')},
                             'scheme': scheme, 'model': model, 'year_from': 2014, 'year_to': 2026})

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / 's0_s1_ew_summary.csv', index=False)

    # readable tables
    full = summ[(summ['year_from'] == 2008) & (summ['year_to'] == 2026)]
    common = summ[(summ['year_from'] == 2014) & (summ['year_to'] == 2026)]

    def fmt(df: pd.DataFrame) -> str:
        cols = ['scheme', 'model', 'rank_ic_mean', 'nw_t', 'ew_hl_sharpe', 'ew_hl_ann', 'vw_hl_sharpe', 'n_dates']
        sub = df[cols].copy()
        sub['rank_ic_mean'] = sub['rank_ic_mean'].map(lambda x: f'{x:.4f}')
        sub['nw_t'] = sub['nw_t'].map(lambda x: f'{x:.2f}')
        sub['ew_hl_sharpe'] = sub['ew_hl_sharpe'].map(lambda x: f'{x:.3f}')
        sub['ew_hl_ann'] = sub['ew_hl_ann'].map(lambda x: f'{x:.3f}')
        sub['vw_hl_sharpe'] = sub['vw_hl_sharpe'].map(lambda x: f'{x:.3f}')
        return sub.to_markdown(index=False)

    lines = [
        '# 稳健性对照：S0 / S1 / 主分析 EW',
        '',
        '- **S0**：train 2000–2005 / val 2006–2007，对称搜参选定后固定预测 2008–2026（= R2008 权重）。',
        '- **S1**：train 2000–2011 / val 2012–2013，对称搜参选定后固定预测 2014–2026（= R2014 权重，不重搜）。',
        '- **EW**：三年一窗每窗重搜后拼接（主分析）。',
        '',
        '## A. 全样本 2008–2026（S0 vs EW）',
        '',
        fmt(full),
        '',
    ]
    if s1_ok:
        lines += [
            '## B. 共同样本 2014–2026（S0 vs S1 vs EW）',
            '',
            fmt(common),
            '',
            '读法：若 EW 与 S0/S1 主排序一致（oned ≫ chart≈mlp），则主结论对固定训练期长度稳健；',
            '若 S1 相对 S0 明显改善，说明加长训练有帮助，但仍以每窗重搜的 EW 为主分析。',
            '',
        ]
    else:
        lines += ['## B. 共同样本 2014–2026', '', '（等待 S1 预测完成后再跑本脚本）', '']

    (OUT / 's0_s1_ew_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0 if s1_ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
