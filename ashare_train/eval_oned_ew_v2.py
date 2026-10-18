# -*- coding: utf-8 -*-
"""拼接 onedcnn_140_ew_v2（R2008–R2014 原段 + R2017–R2023 float64 修正段）并评估主表。

不覆盖原 pred_oned_ew_oos / onedcnn_140_ew。输出到 ew_main_f64/eval/，并登记 registry 条目。

  python ashare_train/eval_oned_ew_v2.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ashare_train.eval_ew_main import eval_model, score_corr, sha1_file  # noqa: E402
from ashare_train.ew_splits import RETRAIN_YEARS, ew_split  # noqa: E402
from experiments.common.metrics import yearly_rank_ic_mean  # noqa: E402

EW_OLD = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'ew_main'
EW_F64 = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'ew_main_f64'
OUT = EW_F64 / 'eval'
REGISTRY = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'registry_v2.json'
REG_PRED = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'registry_v2' / 'preds' / 'pred_onedcnn_140_ew_v2_oos.parquet'
UNCHANGED = {2008, 2011, 2014}
CORRECTED = {2017, 2020, 2023}
KEEP = ['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob', 'retrain_year']


def load_window(R: int) -> tuple[pd.DataFrame, dict]:
    if R in UNCHANGED:
        root, src = EW_OLD / f'R{R}' / 'oned', 'ew_main'
    elif R in CORRECTED:
        root, src = EW_F64 / f'R{R}' / 'oned', 'ew_main_f64'
    else:
        raise ValueError(R)
    split = ew_split(R)
    preds = list((root / 'preds').glob('*.parquet'))
    if not preds:
        raise FileNotFoundError(root / 'preds')
    df = pd.read_parquet(preds[0])
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df = df[df['ending_date'].dt.year.isin(set(split['pred_years']))].copy()
    out = df[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'up_prob']].copy()
    out['retrain_year'] = R
    sel = {}
    if (root / 'selection.json').is_file():
        sel = json.loads((root / 'selection.json').read_text(encoding='utf-8'))
    meta = {
        'retrain_year': R, 'source_root': src, 'pred_file': preds[0].name,
        'pred_sha1': sha1_file(preds[0]), 'n_rows': int(len(out)),
        'pred_years': split['pred_years'], 'selection': sel,
    }
    return out, meta


def stitch_v2() -> tuple[pd.DataFrame, dict]:
    parts, segments = [], []
    for R in RETRAIN_YEARS:
        part, meta = load_window(R)
        parts.append(part)
        segments.append(meta)
    out = pd.concat(parts, ignore_index=True)
    dup = out.duplicated(['StockID', 'ending_date'], keep=False)
    if dup.any():
        raise RuntimeError(f'duplicate keys {int(dup.sum())}')
    out = out.sort_values(['ending_date', 'StockID']).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / 'pred_oned_ew_v2_oos.parquet'
    out.to_parquet(pred_path, index=False)
    manifest = {
        'registry_tag': 'onedcnn_140_ew_v2',
        'n_rows': len(out),
        'date_min': str(out['ending_date'].min().date()),
        'date_max': str(out['ending_date'].max().date()),
        'n_dates': int(out['ending_date'].nunique()),
        'pred_sha1': sha1_file(pred_path),
        'segments': segments,
        'note': 'R2008–R2014 from ew_main; R2017–R2023 from ew_main_f64 (float64 channel_stats)',
    }
    (OUT / 'manifest_oned_v2.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'wrote {pred_path} rows={len(out):,} dates={manifest["n_dates"]} sha1={manifest["pred_sha1"][:16]}')
    return out, manifest


def register(manifest: dict, pred_path: Path) -> None:
    REG_PRED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_path, REG_PRED)
    sha = sha1_file(REG_PRED)
    assert sha == manifest['pred_sha1']
    reg = json.loads(REGISTRY.read_text(encoding='utf-8'))
    reg['predictions']['onedcnn_140_ew_v2'] = {
        'path': str(REG_PRED.relative_to(PROJECT_ROOT)),
        'sha1': sha,
        'rows': manifest['n_rows'],
        'date_min': manifest['date_min'],
        'date_max': manifest['date_max'],
        'n_dates': manifest['n_dates'],
        'source': str(pred_path.relative_to(PROJECT_ROOT)),
        'scheme': 'expanding_window_3y_symmetric_hp_f64_norm',
        'registered_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'note': 'Main 1D after float64 channel_stats fix; R2008–R2014 unchanged, R2017–R2023 retrained',
        'predecessor': 'onedcnn_140_ew',
    }
    REGISTRY.write_text(json.dumps(reg, indent=2) + '\n', encoding='utf-8')
    print(f'registered onedcnn_140_ew_v2 sha1={sha}')


def compare_selections() -> None:
    rows = []
    for R in RETRAIN_YEARS:
        old = json.loads((EW_OLD / f'R{R}' / 'oned' / 'selection.json').read_text())
        new_root = EW_F64 if R in CORRECTED else EW_OLD
        new = json.loads((new_root / f'R{R}' / 'oned' / 'selection.json').read_text())
        rows.append({
            'R': R,
            'segment': 'corrected' if R in CORRECTED else 'unchanged',
            'old_hp': old.get('name'),
            'new_hp': new.get('name'),
            'old_val_ric': old.get('mean_member_val_rank_ic'),
            'new_val_ric': new.get('mean_member_val_rank_ic'),
            'hp_changed': old.get('name') != new.get('name'),
        })
    pd.DataFrame(rows).to_csv(OUT / 'selections_oned_old_vs_v2.csv', index=False)


def write_local_summary(results: list[dict], yearly: pd.DataFrame, corr: pd.DataFrame) -> None:
    (OUT / 'portfolio').mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results])
    summ.to_csv(OUT / 'summary.csv', index=False)
    yearly.to_csv(OUT / 'yearly_rank_ic.csv')
    corr.to_csv(OUT / 'score_rank_corr_mean.csv', index=False)
    for r in results:
        r['_weekly_ic'].to_csv(OUT / f"weekly_rank_ic_{r['model']}.csv", index=False)
        r['_ew_hl'].to_csv(OUT / 'portfolio' / f"ew_hl_{r['model']}.csv", header=['hl'])

    lines = [
        '# 主分析扩展窗口 OOS（1D float64 标准化修正后）',
        '',
        '1D 为 `onedcnn_140_ew_v2`（R2008–R2014 原段 + R2017–R2023 修正段）。',
        'chart / mlp 与修正前相同；`oned_old` 为修正前 `onedcnn_140_ew`，仅作并列对照。',
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
              '## 模型得分截面秩相关（周均值；oned=v2）', '', corr.to_markdown(index=False), '']
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v2, manifest = stitch_v2()
    register(manifest, OUT / 'pred_oned_ew_v2_oos.parquet')
    compare_selections()

    chart = pd.read_parquet(EW_OLD / 'eval' / 'pred_chart_ew_oos.parquet')
    mlp = pd.read_parquet(EW_OLD / 'eval' / 'pred_mlp_ew_oos.parquet')
    old = pd.read_parquet(EW_OLD / 'eval' / 'pred_oned_ew_oos.parquet')
    for df in (chart, mlp, old, v2):
        df['StockID'] = df['StockID'].astype(str).str.zfill(6)
        df['ending_date'] = pd.to_datetime(df['ending_date'])

    results = [
        eval_model('chart', chart),
        eval_model('oned_old', old),
        eval_model('oned_v2', v2),
        eval_model('mlp', mlp),
    ]
    yearly = pd.DataFrame({r['model']: r['_yearly'] for r in results})
    corr = score_corr({'chart': chart, 'oned': v2, 'mlp': mlp})
    write_local_summary(results, yearly, corr)

    side = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results])
    side.to_csv(OUT / 'summary_old_vs_v2.csv', index=False)
    ycmp = pd.DataFrame({
        'oned_old': yearly_rank_ic_mean(old, 'up_prob', 'ret_val'),
        'oned_v2': yearly_rank_ic_mean(v2, 'up_prob', 'ret_val'),
    })
    ycmp['delta'] = ycmp['oned_v2'] - ycmp['oned_old']
    ycmp.to_csv(OUT / 'yearly_rank_ic_oned_old_vs_v2.csv')

    print('\n=== oned old vs v2 (full OOS) ===')
    print(side[side['model'].isin(['oned_old', 'oned_v2'])].to_string(index=False))
    print(f'\ndone → {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
