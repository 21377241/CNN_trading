# -*- coding: utf-8 -*-
"""1D-CNN 训练集逐通道标准化统计量的精度检查。运行：python tests/test_oned_channel_stats.py（也兼容 pytest）。

float32 累加和一旦增长到使单步增量小于半个 ulp（取值 1 时为 2**24，取值 0.5 时为 2**23）就停止增长；
原写法 X.mean(axis=(0, 1)) 因此偏离真实值。
channel_stats 按 float64 累加，须与 float64 参考值一致。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.E2C_temporal.train_1dcnn import channel_stats  # noqa: E402

N, T = 2_000_000, 20


def _data() -> np.ndarray:
    x = np.empty((N, T, 2), dtype=np.float32)
    x[:, :, 0] = 0.5
    x[:, :, 1] = 1.0
    x[::7, :, 1] = 0.0
    return x


def test_float32_reduction_saturates():
    x = _data()
    ref = x.mean(axis=(0, 1), dtype=np.float64)
    assert np.abs(x.mean(axis=(0, 1)) - ref).max() > 0.1


def test_channel_stats_matches_float64():
    x = _data()
    mean, std = channel_stats(x, chunk=1 << 18)
    ref_m = x.mean(axis=(0, 1), dtype=np.float64)
    ref_s = x.std(axis=(0, 1), dtype=np.float64) + 1e-6
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert np.abs(mean - ref_m).max() < 1e-6
    assert np.abs(std - ref_s).max() < 1e-6


def test_channel_stats_small_input_unchanged():
    rng = np.random.default_rng(0)
    x = rng.random((1000, T, 7), dtype=np.float32)
    mean, std = channel_stats(x)
    assert np.allclose(mean, x.mean(axis=(0, 1)), atol=1e-6)
    assert np.allclose(std, x.std(axis=(0, 1)) + 1e-6, atol=1e-6)


if __name__ == '__main__':
    for f in (test_float32_reduction_saturates, test_channel_stats_matches_float64,
              test_channel_stats_small_input_unchanged):
        f()
        print(f'{f.__name__}: ok')
