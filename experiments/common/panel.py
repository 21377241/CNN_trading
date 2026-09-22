# -*- coding: utf-8 -*-
"""Load I20R5 labels (IS) and ensem predictions (OOS)."""
from __future__ import annotations

import glob
import os
import re
import pandas as pd


def norm_stock_id(x) -> str:
    s = str(x).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    s = s.split('.')[0]
    if s.isdigit():
        return s.zfill(6)
    return s


def labels_path(labels_template: str, year: int, project_root: str) -> str:
    """labels_template may contain {year}; brackets in filename are literal."""
    rel = labels_template.format(year=year)
    return os.path.join(project_root, rel.replace('/', os.sep))


def load_is_labels(project_root: str, labels_template: str, years: list) -> pd.DataFrame:
    frames = []
    for y in years:
        p = labels_path(labels_template, y, project_root)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_feather(p)
        df['year'] = y
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out['StockID'] = out['StockID'].map(norm_stock_id)
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    out['ret_val'] = out['ret_val'].astype(float)
    if 'binary_label' not in out.columns:
        out['binary_label'] = (out['ret_val'] > 0).astype(int)
    return out


def find_ensem_files(ensem_root: str, years: list, n_ensembles: int = 5) -> list:
    pattern = os.path.join(ensem_root, '**', 'ensem_res', f'ensem{n_ensembles}_res_*.csv')
    all_f = glob.glob(pattern, recursive=True)
    found = []
    for y in years:
        ys = [f for f in all_f if f'_res_{y}.csv' in os.path.basename(f)]
        if not ys:
            raise FileNotFoundError(f'No ensem file for year {y} under {ensem_root}')
        ys.sort(key=os.path.getmtime, reverse=True)
        found.append(ys[0])
    return found


def load_oos_ensem(project_root: str, ensem_root_rel: str, years: list) -> pd.DataFrame:
    root = os.path.join(project_root, ensem_root_rel.replace('/', os.sep))
    files = find_ensem_files(root, years)
    frames = []
    for f, y in zip(files, years):
        df = pd.read_csv(f)
        df['year'] = y
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out['StockID'] = out['StockID'].map(norm_stock_id)
    out['ending_date'] = pd.to_datetime(out['ending_date'])
    out['ret_val'] = out['ret_val'].astype(float)
    out['up_prob'] = out['up_prob'].astype(float)
    if 'binary_label' not in out.columns:
        out['binary_label'] = (out['ret_val'] > 0).astype(int)
    return out
