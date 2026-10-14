# -*- coding: utf-8 -*-
"""S1 稳健性：复用 R=2014 选定权重，预测 2014–2026。

S1 切分 = train 2000–2011 / val 2012–2013，与主分析 EW 的 R=2014 窗相同，故不重搜参。

  python ashare_train/run_s1_predict.py --model chart|oned|mlp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S1_YEARS = list(range(2014, 2027))
EW = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'ew_main'
S1_ROOT = EW / 'robustness' / 'S1'


def predict_numeric(model: str) -> Path:
    import importlib
    num = importlib.import_module('ashare_train.3d_numeric_hp_search')
    src = EW / 'R2014' / model
    out = S1_ROOT / model
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((src / 'selection.json').read_text(encoding='utf-8'))
    sel2 = dict(sel)
    trial = Path(sel['trial_dir'])
    if not trial.is_absolute():
        sel2['trial_dir'] = str((PROJECT_ROOT / sel['trial_dir']).resolve().relative_to(PROJECT_ROOT))
    (out / 'selection.json').write_text(json.dumps({
        **sel2,
        'note': 'S1 reuses R2014 selection (same train/val); predicts 2014-2026',
        'source_root': str(src),
    }, indent=2), encoding='utf-8')
    path = num.predict_selected(model, str(out), pred_years=S1_YEARS)
    print(f'S1 {model}: {path}')
    return Path(path)


def predict_chart() -> Path:
    import subprocess
    sel = json.loads((EW / 'R2014' / 'chart' / 'selection.json').read_text(encoding='utf-8'))
    exp_dir = sel['exp_dir']
    out_dir = S1_ROOT / 'chart'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'selection.json').write_text(json.dumps({
        **sel,
        'note': 'S1 reuses R2014 chart selection; predicts 2014-2026 via stitch_cnn_oos',
    }, indent=2), encoding='utf-8')
    rel_out = 'WORK_SPACE/paper/ew_main/robustness/S1/chart/preds/pred_chart_oos2014-2026.parquet'
    out_parquet = PROJECT_ROOT / rel_out
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'ashare_train' / 'stitch_cnn_oos.py'),
        '--exp-dir', exp_dir,
        '--dataset-subdir', 'I20R5_clean',
        '--training', 'clean',
        '--n-ensembles', '5',
        '--years', *[str(y) for y in S1_YEARS],
        '--out', rel_out,
        '--regenerate',
    ]
    print(' '.join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))
    print(f'S1 chart: {out_parquet}')
    return out_parquet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['chart', 'oned', 'mlp'], required=True)
    args = ap.parse_args()
    S1_ROOT.mkdir(parents=True, exist_ok=True)
    if args.model == 'chart':
        predict_chart()
    else:
        predict_numeric(args.model)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
