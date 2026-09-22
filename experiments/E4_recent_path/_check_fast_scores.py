# -*- coding: utf-8 -*-
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'trend_submit', 'trend_code_submit')
sys.path[:0] = [ROOT, CODE]
import numpy as np
import pandas as pd
import torch
from Data.chart_library import DrawOHLC
from experiments.E4_recent_path.fast_draw import fast_draw_bar
from experiments.E4_recent_path.cnn_forward import load_cnn_models, draw_one
from experiments.E2_matched_tabular.e2a_layout import FEAT_180

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = load_cnn_models(device)
X = np.load(os.path.join(FEAT_180, 'X_oos.npy'), mmap_mode='r')
n = 1024
rows = np.asarray(X[:n], dtype=np.float64).reshape(-1, 20, 9)

def draw_pil(row):
    o,h,l,c,v = row[:,0], row[:,1], row[:,2], row[:,3], row[:,6]
    return draw_one(o,h,l,c,v)

t0=time.time(); imgs_pil=np.stack([draw_pil(r) for r in rows]); print('pil', time.time()-t0)
t0=time.time(); imgs_f=np.stack([fast_draw_bar(r[:,0],r[:,1],r[:,2],r[:,3],r[:,6]) for r in rows]); print('fast', time.time()-t0)

@torch.no_grad()
def infer(imgs):
    t=torch.from_numpy(imgs).to(device)
    return np.mean([m.predict_proba(t)[:,1].cpu().numpy() for m in models],0)

p=infer(imgs_pil); f=infer(imgs_f)
print('corr', np.corrcoef(p,f)[0,1], 'mae', np.mean(np.abs(p-f)), 'max', np.max(np.abs(p-f)))
