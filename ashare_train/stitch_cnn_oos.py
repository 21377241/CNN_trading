# -*- coding: utf-8 -*-
"""
stitch_cnn_oos.py — 把 chart-CNN 的逐年 OOS 预测（ensem{n}_res_{year}.csv）拼成一个 parquet，并记录 sha1。

--regenerate：由 checkpoint 重新预测每一年，与已有逐年文件逐行比对（键一致、up_prob 最大绝对差），
确认预测可由保存的权重复现；缺失年份直接用重新预测的结果补齐。

用法:
  python ashare_train/stitch_cnn_oos.py --exp-dir <exp_dir> --dataset-subdir I20R5_clean --training clean \
      --out WORK_SPACE/paper/registry_v2/preds/pred_chartcnn_oos.parquet --regenerate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import ASHARE_STOCKS_SAVEPATH, ASHARE_COUNTRY, MA_LAGS, VOLUME_BAR
from Data.dgp_config import get_dataset_dir
from Experiments.cnn_experiment import _dataset, _loader, _predict, _sample_modes, _get_device
from Model.cnn_model import init_model, get_full_model_name

OOS_YEARS = list(range(2008, 2026))
KEEP = ['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_models(exp_dir: str, n: int, device) -> list:
    models = []
    for i in range(n):
        ck = torch.load(os.path.join(exp_dir, f'checkpoint{i}.pth.tar'), map_location='cpu', weights_only=False)
        m = init_model(get_full_model_name(20), img_h=64, img_w=60, dropout=0.0, n_classes=2, window_size=20,
                       channels=ck.get('channels')).to(device)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_year(models, year: int, dataset_dir: str, sample_mode: str, device) -> pd.DataFrame:
    ds = _dataset([year], 20, 5, 'week', VOLUME_BAR, MA_LAGS, ASHARE_COUNTRY, dataset_dir, sample_mode)
    loader = _loader(ds, 2048, shuffle=False)
    ups = [_predict(m, loader, device)[0] for m in models]
    out = ds.meta.copy()
    out['up_prob'] = np.mean(ups, axis=0)
    for i, up in enumerate(ups):
        out[f'up_prob_m{i}'] = up
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-dir', required=True)
    ap.add_argument('--n-ensembles', type=int, default=5)
    ap.add_argument('--dataset-subdir', required=True)
    ap.add_argument('--training', choices=['legacy', 'clean'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--regenerate', action='store_true')
    args = ap.parse_args()

    out_path = os.path.join(PROJECT_ROOT, args.out.replace('/', os.sep))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, args.dataset_subdir))
    _, oos_mode = _sample_modes(args.training)
    device = _get_device()
    models = load_models(args.exp_dir, args.n_ensembles, device) if args.regenerate else None

    parts, checks = [], []
    for y in OOS_YEARS:
        f = os.path.join(args.exp_dir, 'ensem_res', f'ensem{args.n_ensembles}_res_{y}.csv')
        old = pd.read_csv(f, dtype={'StockID': str}) if os.path.exists(f) else None
        if old is not None:
            old['ending_date'] = pd.to_datetime(old['ending_date'])
        if args.regenerate:
            new = predict_year(models, y, dataset_dir, oos_mode, device)
            new['ending_date'] = pd.to_datetime(new['ending_date'])
            if old is not None:
                same_keys = (len(old) == len(new)
                             and (old['StockID'].str.zfill(6).values == new['StockID'].astype(str).values).all()
                             and (old['ending_date'].values == new['ending_date'].values).all())
                diff = float(np.abs(old['up_prob'].to_numpy() - new['up_prob'].to_numpy()).max()) if same_keys else float('nan')
                checks.append({'year': y, 'n': len(new), 'same_keys': bool(same_keys), 'max_abs_diff': diff})
                print(f'{y}: n={len(new)} same_keys={same_keys} max|Δup_prob|={diff:.2e}')
                if not same_keys or diff > 1e-4:
                    raise RuntimeError(f'{y}: 由 checkpoint 重新预测的结果与已有文件不一致')
            else:
                checks.append({'year': y, 'n': len(new), 'same_keys': None, 'max_abs_diff': None})
                print(f'{y}: 无已有文件，使用重新预测结果 n={len(new)}')
            df = new
        else:
            if old is None:
                raise FileNotFoundError(f)
            df = old
        cols = KEEP + ['up_prob'] + sorted(c for c in df.columns if c.startswith('up_prob_m'))
        df = df[[c for c in cols if c in df.columns]].copy()
        df['StockID'] = df['StockID'].astype(str).str.zfill(6)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(out_path, index=False)
    info = {'path': out_path, 'sha1': sha1_of(out_path), 'rows': int(len(out)), 'exp_dir': args.exp_dir,
            'dataset_subdir': args.dataset_subdir, 'training': args.training, 'n_ensembles': args.n_ensembles,
            'regenerated': bool(args.regenerate), 'checks': checks}
    with open(out_path.replace('.parquet', '.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'wrote {out_path} rows={len(out):,} sha1={info["sha1"][:16]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
