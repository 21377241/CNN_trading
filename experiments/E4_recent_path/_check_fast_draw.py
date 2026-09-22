# -*- coding: utf-8 -*-
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'trend_submit', 'trend_code_submit')
sys.path[:0] = [ROOT, CODE]
import numpy as np
import pandas as pd
from Data.chart_library import DrawOHLC
from experiments.E4_recent_path.fast_draw import fast_draw_bar
from experiments.E2_matched_tabular.e2a_layout import FEAT_180

X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
diffs = []
for i in range(50):
    row = np.asarray(X[i], dtype=np.float64).reshape(20, 9)
    o,h,l,c,v = row[:,0], row[:,1], row[:,2], row[:,3], row[:,6]
    ma = pd.Series(c).rolling(20, min_periods=1).mean().values
    df = pd.DataFrame({'Open':o,'High':h,'Low':l,'Close':c,'Vol':v,'ma20':ma})
    ref = DrawOHLC(ohlcv_df=df, window_size=20, ma_cols=['ma20'], has_volume_bar=True,
                   chart_type='bar', img_width=60, img_height=64).draw_to_array().astype(np.float32).reshape(64,60)/255.0
    fast = fast_draw_bar(o,h,l,c,v)[0]
    diffs.append(np.mean(np.abs(ref-fast)))
print('mean abs', np.mean(diffs), 'max', np.max(diffs), 'exact', np.mean([d<1e-9 for d in diffs]))
