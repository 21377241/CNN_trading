# -*- coding: utf-8 -*-
"""Run E3 Phase 1 end-to-end."""
from __future__ import annotations

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E3_shared_specific.build_geometry import build_geometry
from experiments.E3_shared_specific.build_panel import build_panel
from experiments.E3_shared_specific.build_scores import build_scores
from experiments.E3_shared_specific.run_disagreement import run_disagreement
from experiments.E3_shared_specific.run_eval_scores import run_eval
from experiments.E3_shared_specific.run_spec_vs_features import run_spec_vs_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e3_phase1')


def main():
    logger.info('=== E3 Phase 1: panel ===')
    build_panel()
    logger.info('=== E3 Phase 1: scores ===')
    build_scores()
    logger.info('=== E3 Phase 1: geometry ===')
    build_geometry()
    logger.info('=== E3 Phase 1: eval scores ===')
    run_eval()
    logger.info('=== E3 Phase 1: disagreement ===')
    run_disagreement()
    logger.info('=== E3 Phase 1: spec vs features ===')
    run_spec_vs_features()
    logger.info('[OK] E3 Phase 1 done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
