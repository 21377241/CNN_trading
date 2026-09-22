# -*- coding: utf-8 -*-
"""Build and cache E2 matched feature matrices (IS + OOS)."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.features_matched import build_feature_matrix
from experiments.common.panel import load_is_labels, load_oos_ensem

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2_build')


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        help='path to config yaml',
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = cfg['paths']
    include_ind = bool(cfg.get('include_indicators', False))
    out_root = os.path.join(PROJECT_ROOT, paths['out_root'].replace('/', os.sep))
    feat_dir = os.path.join(out_root, 'features')
    os.makedirs(feat_dir, exist_ok=True)

    feather = os.path.join(PROJECT_ROOT, paths['feather'].replace('/', os.sep))
    logger.info(f'config={args.config}')
    logger.info(f'include_indicators={include_ind}  out_root={out_root}')
    logger.info(f'Loading daily feather: {feather}')
    daily = pd.read_feather(
        feather, columns=['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Ret', 'Vol', 'MarketCap']
    )
    logger.info(f'daily rows={len(daily):,}')

    logger.info('Loading IS labels…')
    is_panel = load_is_labels(PROJECT_ROOT, paths['labels_glob'], cfg['is_years'])
    logger.info(f'IS panel={len(is_panel):,}')

    logger.info('Building IS features…')
    X_is, meta_is = build_feature_matrix(
        is_panel, daily,
        window_size=cfg['window_size'],
        ma_lag=cfg['ma_lag'],
        include_indicators=include_ind,
    )
    logger.info(f'IS features X={X_is.shape}')
    np.save(os.path.join(feat_dir, 'X_is.npy'), X_is)
    meta_is.to_feather(os.path.join(feat_dir, 'meta_is.feather'))

    logger.info('Loading OOS ensem…')
    oos_panel = load_oos_ensem(PROJECT_ROOT, paths['ensem_root'], cfg['oos_years'])
    logger.info(f'OOS panel={len(oos_panel):,}')

    logger.info('Building OOS features…')
    X_oos, meta_oos = build_feature_matrix(
        oos_panel, daily,
        window_size=cfg['window_size'],
        ma_lag=cfg['ma_lag'],
        include_indicators=include_ind,
    )
    logger.info(f'OOS features X={X_oos.shape}')
    np.save(os.path.join(feat_dir, 'X_oos.npy'), X_oos)
    meta_oos.to_feather(os.path.join(feat_dir, 'meta_oos.feather'))

    logger.info('[OK] feature cache written to %s', feat_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
