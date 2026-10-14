# -*- coding: utf-8 -*-
"""主分析扩展窗口：三年一窗切分定义。"""
from __future__ import annotations

RETRAIN_YEARS = [2008, 2011, 2014, 2017, 2020, 2023]


def ew_split(retrain_year: int) -> dict:
    """R → train 2000…R-3, val R-2…R-1, predict R…min(R+2, 2026)."""
    if retrain_year not in RETRAIN_YEARS:
        raise ValueError(f'retrain_year must be in {RETRAIN_YEARS}, got {retrain_year}')
    pred_end = 2026 if retrain_year == 2023 else retrain_year + 2
    return {
        'retrain_year': retrain_year,
        'train_years': list(range(2000, retrain_year - 2)),  # 2000 .. R-3
        'val_years': [retrain_year - 2, retrain_year - 1],
        'pred_years': list(range(retrain_year, pred_end + 1)),
    }


def assert_fair_split(split: dict) -> None:
    tr, va, pr = split['train_years'], split['val_years'], split['pred_years']
    assert tr[-1] + 1 == va[0] and va[0] + 1 == va[1] and va[1] + 1 == pr[0]
    assert set(tr) & set(va) == set() and set(tr) & set(pr) == set() and set(va) & set(pr) == set()
