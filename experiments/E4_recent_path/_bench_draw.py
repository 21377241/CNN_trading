# -*- coding: utf-8 -*-
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'trend_submit', 'trend_code_submit')
sys.path[:0] = [ROOT, CODE]
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from experiments.E4_recent_path.cnn_forward import _draw_intervened, DRAW_WORKERS
from experiments.E2_matched_tabular.e2a_layout import FEAT_180

X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
rows = np.asarray(X[:512], dtype=np.float64).reshape(-1, 20, 9)
t0 = time.time()
[_draw_intervened(r, 'flatten_day_15') for r in rows]
print('serial512', round(time.time() - t0, 2), 'workers', DRAW_WORKERS)
t0 = time.time()
with ThreadPoolExecutor(max_workers=DRAW_WORKERS) as p:
    list(p.map(lambda r: _draw_intervened(r, 'flatten_day_15'), rows))
print('thread512', round(time.time() - t0, 2))
