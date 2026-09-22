# -*- coding: utf-8 -*-
"""
Optional E2 bridge: if E1b Phase3 beats baseline, export tuned CNN preds and re-eval E2 CNN column.

用法:
  python experiments/E1_image_baseline/run_e2_cnn_bridge.py
  python experiments/E1_image_baseline/run_e2_cnn_bridge.py --force
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HP_CONFIG = os.path.join(PROJECT_ROOT, 'experiments', 'E1_image_baseline', 'config_cnn_hp.yaml')
E2_CONFIG = os.path.join(PROJECT_ROOT, 'experiments', 'E2_matched_tabular', 'config.yaml')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp-config', default=HP_CONFIG)
    parser.add_argument('--e2-config', default=E2_CONFIG)
    parser.add_argument('--force', action='store_true', help='即使未达 Sharpe 阈值也导出并评估')
    args = parser.parse_args()

    with open(args.hp_config, 'r', encoding='utf-8') as f:
        hp = yaml.safe_load(f)
    out_root = os.path.join(PROJECT_ROOT, hp['out_root'].replace('/', os.sep))
    summary_path = os.path.join(out_root, 'phase3_portfolio_summary.csv')
    phase3_path = os.path.join(out_root, 'phase3_full_train.csv')

    if not os.path.isfile(summary_path):
        print(f'缺少 {summary_path}，请先完成 E1b Phase 3')
        return 1

    summ = pd.read_csv(summary_path)
    summ = summ[summ['ew_hl_sharpe'].notna()]
    if summ.empty:
        print('无有效 Phase3 portfolio Sharpe')
        return 1

    best_idx = summ['ew_hl_sharpe'].idxmax()
    best = summ.loc[best_idx]
    baseline = float(hp.get('baseline_ew_hl_sharpe', 2.65))
    delta = float(best['ew_hl_sharpe']) - baseline
    threshold = float(hp.get('e2_bridge_sharpe_delta', 0.15))

    if delta < threshold and not args.force:
        print(f'E1b 最优 EW Sharpe {best["ew_hl_sharpe"]:.4f} 未达阈值 +{threshold} vs {baseline}，跳过 E2 bridge')
        return 0

    exp_dir = best.get('exp_dir') or ''
    if not exp_dir or not os.path.isdir(exp_dir):
        if os.path.isfile(phase3_path):
            p3 = pd.read_csv(phase3_path)
            match = p3[
                (p3['lr'] == best['lr']) &
                (p3['dropout'] == best['dropout']) &
                (p3['patience'] == best['patience'])
            ]
            if len(match):
                exp_dir = match.iloc[0]['exp_dir']
    if not exp_dir or not os.path.isdir(exp_dir):
        print('无法定位 Phase3 exp_dir')
        return 1

    n_ensem = int(hp['phases']['phase3']['n_ensembles'])
    oos_years = hp['oos_years']
    files = []
    for y in oos_years:
        pat = os.path.join(exp_dir, 'ensem_res', f'ensem{n_ensem}_res_{y}.csv')
        if not os.path.isfile(pat):
            hits = glob.glob(os.path.join(exp_dir, '**', f'ensem{n_ensem}_res_{y}.csv'), recursive=True)
            pat = hits[0] if hits else pat
        if os.path.isfile(pat):
            files.append(pat)
    if not files:
        print(f'未找到 OOS ensem 文件于 {exp_dir}')
        return 1

    frames = [pd.read_csv(f) for f in files]
    pred = pd.concat(frames, ignore_index=True)
    cols = ['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label', 'up_prob']
    for c in cols:
        if c not in pred.columns:
            if c == 'binary_label':
                pred['binary_label'] = (pred['ret_val'] > 0).astype(int)
            elif c == 'MarketCap' and 'MarketCap' not in pred.columns:
                pred['MarketCap'] = float('nan')
    pred = pred[cols]

    bridge_dir = os.path.join(out_root, 'e2_bridge')
    os.makedirs(bridge_dir, exist_ok=True)
    pred_path = os.path.join(bridge_dir, 'pred_cnn_oos_tuned.csv')
    pred.to_csv(pred_path, index=False)
    print(f'wrote {pred_path} rows={len(pred):,}  delta_sharpe={delta:.4f}')

    eval_script = os.path.join(PROJECT_ROOT, 'experiments', 'E2_matched_tabular', 'run_eval.py')
    subprocess.run(
        [
            sys.executable, eval_script,
            '--config', args.e2_config,
            '--only', 'cnn,histgb',
            '--cnn-pred', pred_path,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    appendix_path = os.path.join(out_root, 'e2_bridge', 'e2_cnn_tuned_summary.md')
    with open(appendix_path, 'w', encoding='utf-8') as f:
        f.write('# E2 附录：E1b 调参 CNN vs HistGB\n\n')
        f.write(f'- tuned EW H-L Sharpe: {best["ew_hl_sharpe"]:.4f}\n')
        f.write(f'- baseline: {baseline:.4f} (delta {delta:+.4f})\n')
        f.write(f'- exp_dir: {exp_dir}\n')
        f.write(f'- pred: `{pred_path}`\n')
    print(f'wrote {appendix_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
