# -*- coding: utf-8 -*-
"""Run E5 Phase 1: build forward → E5-1 → E5-2 → E5-3."""
from __future__ import annotations

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.run_e51_horizon import run_e51
from experiments.E5_mechanism.run_e52_session import run_e52
from experiments.E5_mechanism.run_e53_state import run_e53

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_phase1')


def main():
    panel = build_forward_panel()
    run_e51(panel)
    run_e52(panel)
    run_e53(panel)
    logger.info('E5 Phase 1 done')


if __name__ == '__main__':
    main()
