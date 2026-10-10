# -*- coding: utf-8 -*-
"""Build the 140D matched feature cache (IS + OOS) from the universe file.

Sample keys come only from universe_weekly.feather (clean_spec D10); OOS rows are not filtered by label.
`ashare_train/2_generate_ashare_chart_data.py` writes the same features alongside the images
(WORK_SPACE/paper/clean_inputs/features140); this script rebuilds them on their own, e.g. into another out_root.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.features_matched import build_feature_matrix, load_grid  # noqa: E402
from Data.clean_window import universe_meta  # noqa: E402
from Misc.config import ASHARE_PANEL_PATH, ASHARE_UNIVERSE_PATH, CLEAN_INPUTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2_build')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=os.path.join(CLEAN_INPUTS_DIR, 'features140'))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    uni = pd.read_feather(ASHARE_UNIVERSE_PATH)
    grid = load_grid(ASHARE_PANEL_PATH)
    for tag, mask in (('is', uni['split'].isin(['train', 'val'])), ('oos', uni['split'] == 'oos')):
        u = uni[mask].reset_index(drop=True)
        X = build_feature_matrix(u, grid)
        np.save(os.path.join(args.out_dir, f'X_{tag}.npy'), X)
        universe_meta(u).to_feather(os.path.join(args.out_dir, f'meta_{tag}.feather'))
        logger.info(f'{tag}: X={X.shape}')
    logger.info('[OK] feature cache written to %s', args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
