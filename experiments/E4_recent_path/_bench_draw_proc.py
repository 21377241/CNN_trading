# -*- coding: utf-8 -*-
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'trend_submit', 'trend_code_submit')
sys.path[:0] = [ROOT, CODE]
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments.E2_matched_tabular.e2a_layout import FEAT_180


def _worker(payload):
    row, name = payload
    from experiments.E4_recent_path.cnn_forward import _draw_intervened
    return _draw_intervened(row, name)


if __name__ == '__main__':
    X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
    rows = np.asarray(X[:1024], dtype=np.float64).reshape(-1, 20, 9)
    payloads = [(r, 'flatten_day_15') for r in rows]
    from experiments.E4_recent_path.cnn_forward import _draw_intervened
    t0 = time.time()
    [_draw_intervened(r, 'flatten_day_15') for r in rows]
    print('serial1024', round(time.time() - t0, 2))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as p:
        list(p.map(_worker, payloads, chunksize=32))
    print('proc1024', round(time.time() - t0, 2))
