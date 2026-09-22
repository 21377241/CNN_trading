# -*- coding: utf-8 -*-
"""Orchestrate E4 Phase 1+2 (resume-friendly)."""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.E4_recent_path.build_pathshape import build_pathshape
from experiments.E4_recent_path.paths import OUT_ROOT
from experiments.E4_recent_path.scores_pairwise import build_baseline_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e4_all')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='*', default=None,
                   help='subset: baseline path e42 e44 e45 e41_mlp e41_cnn e43_mlp e43_cnn')
    args = p.parse_args()
    only = set(args.only) if args.only else None

    def want(name: str) -> bool:
        return only is None or name in only

    os.makedirs(OUT_ROOT, exist_ok=True)
    if want('baseline'):
        build_baseline_scores()
    if want('path'):
        build_pathshape()
    if want('e42'):
        from experiments.E4_recent_path.run_e42_pathshape import run_e42
        run_e42()
    if want('e44'):
        from experiments.E4_recent_path.run_e44_matched_return import run_e44
        run_e44()
    if want('e45'):
        from experiments.E4_recent_path.run_e45_disagreement import run_e45
        run_e45()
    if want('e41_mlp'):
        from experiments.E4_recent_path.run_e41_lag import run_mlp_lags
        run_mlp_lags(do_histgb=True)
    if want('e43_mlp'):
        from experiments.E4_recent_path.run_e43_blocks import run_mlp_blocks
        run_mlp_blocks(do_histgb=True)
    if want('e41_cnn'):
        from experiments.E4_recent_path.run_e41_lag import run_cnn_lags
        run_cnn_lags()
    if want('e43_cnn'):
        from experiments.E4_recent_path.run_e43_blocks import run_cnn_blocks
        run_cnn_blocks()
    logger.info('E4 orchestration done')


if __name__ == '__main__':
    main()
