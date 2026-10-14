# -*- coding: utf-8 -*-
"""扩展窗口共用：按年加载 140 维面板、动态 embargo、chart labels 覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'clean_inputs' / 'features140'
CHART_SRC = (
    PROJECT_ROOT / 'WORK_SPACE' / 'data' / 'ashare_stocks_dataset'
    / 'I20R5_clean' / 'stocks_CHN' / 'dataset_all'
)


def embargo_dates(meta: pd.DataFrame, boundary_years: list[int]) -> set:
    """每个 boundary year 的最后一个 ending_date。"""
    out: set = set()
    ed = pd.to_datetime(meta['ending_date'])
    years = meta['year'].to_numpy() if 'year' in meta.columns else ed.dt.year.to_numpy()
    for y in boundary_years:
        mask = years == int(y)
        if not mask.any():
            raise RuntimeError(f'no rows for embargo year {y}')
        out.add(pd.Timestamp(ed[mask].max()))
    return out


def load_panel_years(years: list[int]):
    """按年拼接 features140/by_year 与 meta（is∪oos）。"""
    meta_is = pd.read_feather(FEATURE_ROOT / 'meta_is.feather')
    meta_oos = pd.read_feather(FEATURE_ROOT / 'meta_oos.feather')
    meta = pd.concat([meta_is, meta_oos], ignore_index=True)
    meta['ending_date'] = pd.to_datetime(meta['ending_date'])
    if 'year' not in meta.columns:
        meta['year'] = meta['ending_date'].dt.year
    xs, ms = [], []
    for y in years:
        path = FEATURE_ROOT / 'by_year' / f'X_{y}.npy'
        if not path.is_file():
            raise FileNotFoundError(path)
        Xy = np.load(path)
        my = meta.loc[meta['year'] == y].reset_index(drop=True)
        if len(my) != len(Xy):
            raise RuntimeError(f'year {y}: meta rows {len(my)} != X rows {len(Xy)}')
        xs.append(Xy)
        ms.append(my)
    return np.concatenate(xs, axis=0), pd.concat(ms, ignore_index=True)


def trainable_mask(meta: pd.DataFrame, ban: set) -> np.ndarray:
    """用 label_raw∈{0,1}，并剔除动态 embargo 周（不用写死的 binary_label/-99）。"""
    raw = meta['label_raw'].to_numpy()
    ed = pd.to_datetime(meta['ending_date'])
    return np.isin(raw, [0, 1]) & ~ed.isin(ban)


def _chart_labels_name(year: int) -> str:
    return f'20d_week_has_vb_[20]_ma_{year}_labels.feather'


def _chart_images_name(year: int) -> str:
    return f'20d_week_has_vb_[20]_ma_{year}_images.dat'


def prepare_chart_overlay(split: dict, overlay: Path) -> Path:
    """镜像图像文件；重写 train/val 边界年的 labels.embargo / binary_label。"""
    overlay = Path(overlay)
    overlay.mkdir(parents=True, exist_ok=True)
    need_years = sorted(set(split['train_years'] + split['val_years'] + split['pred_years']))
    last_of = {}
    for y in need_years:
        src = CHART_SRC / _chart_labels_name(y)
        m = pd.read_feather(src)
        m['ending_date'] = pd.to_datetime(m['ending_date'])
        last_of[y] = pd.Timestamp(m['ending_date'].max())
    ban = {last_of[split['train_years'][-1]], last_of[split['val_years'][-1]]}
    for y in need_years:
        img_src = CHART_SRC / _chart_images_name(y)
        lab_src = CHART_SRC / _chart_labels_name(y)
        img_dst = overlay / _chart_images_name(y)
        lab_dst = overlay / _chart_labels_name(y)
        if not img_dst.exists():
            os.symlink(img_src, img_dst)
        m = pd.read_feather(lab_src)
        m['ending_date'] = pd.to_datetime(m['ending_date'])
        m['embargo'] = m['ending_date'].isin(ban).astype(np.int8)
        raw = m['label_raw'].to_numpy()
        m['binary_label'] = np.where((raw < 0) | (m['embargo'] == 1), -99, raw).astype(np.int64)
        m.to_feather(lab_dst)
    return overlay
