# -*- coding: utf-8 -*-
"""Paper experiment shared helpers: years / train-val split."""
from __future__ import annotations

import numpy as np
import pandas as pd


IS_YEARS = list(range(2000, 2008))
OOS_YEARS = list(range(2008, 2026))


def train_val_indices(n: int, train_frac: float = 0.7, seed: int = 42):
    """Random 70/30 shuffle over rows (Jiang-style IS internal split)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


def chrono_train_val_indices(
    meta_is: pd.DataFrame,
    train_years: list,
    val_years: list,
):
    """
    Chronological IS split by ending_date year.
    Returns integer positions into meta_is (0..n-1).
    """
    years = pd.to_datetime(meta_is['ending_date']).dt.year.values
    train_set = set(int(y) for y in train_years)
    val_set = set(int(y) for y in val_years)
    tr_idx = np.flatnonzero(np.isin(years, list(train_set)))
    va_idx = np.flatnonzero(np.isin(years, list(val_set)))
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise ValueError(
            f'chrono split empty: train_years={train_years} n_tr={len(tr_idx)} '
            f'val_years={val_years} n_va={len(va_idx)}'
        )
    return tr_idx, va_idx
