# -*- coding: utf-8 -*-
"""
3c_cnn_full_train.py — 按固定超参数全量训练 chart-CNN 集成并生成 OOS 预测（不搜参）。

用法:
  python ashare_train/3c_cnn_full_train.py --config experiments/E1_image_baseline/config_cnn_m1_rendfix.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import ASHARE_STOCKS_SAVEPATH, ASHARE_COUNTRY, MA_LAGS, VOLUME_BAR, ASHARE_LOG_DIR
from Data.dgp_config import get_dataset_dir, get_dataset_filename
from Experiments.cnn_experiment import train_us_model, _build_exp_dir
from Model.cnn_model import get_full_model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--force', action='store_true', help='忽略已有 checkpoint 重训')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out_root = os.path.join(PROJECT_ROOT, cfg['out_root'].replace('/', os.sep))
    os.makedirs(out_root, exist_ok=True)
    shutil.copy2(args.config, os.path.join(out_root, os.path.basename(args.config)))

    os.makedirs(ASHARE_LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(out_root, 'full_train.log'), encoding='utf-8')],
    )

    ws, pw, freq = cfg['window_size'], cfg['predict_window'], cfg['freq']
    dataset_dir = get_dataset_dir(ASHARE_COUNTRY, base_savepath=os.path.join(ASHARE_STOCKS_SAVEPATH, cfg['dataset_subdir']))

    def available(years):
        return [y for y in years
                if os.path.exists(os.path.join(dataset_dir, get_dataset_filename(ws, freq, VOLUME_BAR, MA_LAGS, y, '_images.dat')))
                and os.path.exists(os.path.join(dataset_dir, get_dataset_filename(ws, freq, VOLUME_BAR, MA_LAGS, y, '_labels.feather')))]

    years = {k: available(cfg[k]) for k in ('train_years', 'val_years', 'oos_years')}
    for k in years:
        if len(years[k]) != len(cfg[k]):
            missing = sorted(set(cfg[k]) - set(years[k]))
            logging.error(f'{k} 缺少数据年份 {missing}（{dataset_dir}）')
            return 1

    hp = cfg['hp']
    lr, dp, pat = float(hp['lr']), float(hp['dropout']), int(hp['patience'])
    suffix = cfg.get('dir_suffix', f'full-p{pat}')
    logging.info(f'dataset={dataset_dir}  training={cfg["training"]}  lr={lr:g} dp={dp} patience={pat}  '
                 f'n_ensembles={cfg["n_ensembles"]}  out={out_root}')
    train_us_model(
        ws_list=[ws], pw_list=[pw], freq=freq, ma_lags=MA_LAGS, has_vb=VOLUME_BAR,
        lr=lr, dropout=dp, n_ensembles=int(cfg['n_ensembles']),
        batch_size=cfg.get('batch_size', 128), max_epochs=cfg.get('max_epochs', 50), early_stop_patience=pat,
        is_years=years['train_years'], val_years=years['val_years'], oos_years=years['oos_years'],
        country=cfg.get('country', ASHARE_COUNTRY), base_exp_dir=out_root, dataset_dir=dataset_dir,
        dir_suffix=suffix, skip_oos=False, force_retrain=args.force, training=cfg['training'],
    )
    exp_dir = _build_exp_dir(get_full_model_name(ws), ws, pw, lr, dp, freq, len(MA_LAGS) > 0, VOLUME_BAR,
                             base_exp_dir=out_root, dir_suffix=suffix)
    logging.info(f'完成：{exp_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
