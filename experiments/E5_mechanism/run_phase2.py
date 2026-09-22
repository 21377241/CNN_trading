# -*- coding: utf-8 -*-
"""E5 Phase 2: Pressure → Interaction → Cont/Rev."""
from __future__ import annotations

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.E5_mechanism.build_forward import build_forward_panel
from experiments.E5_mechanism.run_e54_pressure import run_e54
from experiments.E5_mechanism.run_e55_interaction import run_e55
from experiments.E5_mechanism.run_e56_cont_rev import run_e56

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e5_phase2')


def main():
    panel = build_forward_panel()
    _, panel = run_e54(panel)
    run_e55(panel)
    run_e56(panel)
    logger.info('E5 Phase 2 done')


if __name__ == '__main__':
    main()
