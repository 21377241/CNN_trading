# -*- coding: utf-8 -*-
"""common.limit_state 的手工用例。运行：python tests/test_limit_state.py（也兼容 pytest）。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.E7_incremental.common import limit_state  # noqa: E402

NAN = np.nan


def _panel(rows):
    """rows: (StockID, day, O, H, L, C raw, adj_factor, PreClose, LimitUp, LimitDown)"""
    d = pd.DataFrame(rows, columns=['StockID', 'day', 'O', 'H', 'L', 'C', 'f', 'PreClose', 'LimitUp', 'LimitDown'])
    d['Date'] = pd.Timestamp('2010-01-01') + pd.to_timedelta(d['day'], unit='D')
    d['StockID'] = d['StockID'].astype('category')
    for k, raw in (('Open', 'O'), ('High', 'H'), ('Low', 'L'), ('Close', 'C')):
        d[k] = d[raw] * d['f']
    return d.rename(columns={'C': 'Close_raw', 'f': 'adj_factor'})


def test_normal_day_limit_up_and_one_price():
    p = _panel([('000001', 0, 10, 10, 10, 10, 1.0, 10, 11, 9),
                ('000001', 1, 11, 11, 11, 11, 1.0, 10, 11, 9),
                ('000001', 2, 11, 12.1, 11, 12.1, 1.0, 11, 12.1, 9.9)])
    s = limit_state(p)
    assert s['close_up'].tolist() == [False, True, True]
    assert s['one_price_up'].tolist() == [False, True, False]
    assert s['limit_src'].tolist() == [1, 1, 1]


def test_touch_without_close():
    p = _panel([('000001', 0, 10, 10, 10, 10, 1.0, 10, 11, 9),
                ('000001', 1, 10.2, 11, 8.995, 10.99, 1.0, 10, 11, 9)])
    s = limit_state(p).iloc[1]
    assert not s['close_up'] and s['touch_up'] and s['touch_dn'] and not s['close_dn']


def test_no_limit_day():
    p = _panel([('000001', 0, 10, 10, 10, 10, 1.0, 10, 11, 9),
                ('000001', 1, 15, 15, 15, 15, 1.0, 10, NAN, NAN)])
    s = limit_state(p).iloc[1]
    assert s['no_limit'] and s['limit_src'] == 0
    assert not (s['close_up'] or s['touch_up'] or s['one_price_up'] or s['close_dn'] or s['touch_dn'])


def test_stale_preclose_on_bonus_issue_recomputed():
    # 10 送 10：前收 20，因子翻倍，CSMAR 的 PreClose 与限价仍按 20 给出
    p = _panel([('000002', 0, 20, 20, 20, 20, 1.0, 20, 22, 18),
                ('000002', 1, 11, 11, 11, 11, 2.0, 20, 22, 18)])
    s = limit_state(p).iloc[1]
    assert s['limit_src'] == 2
    assert np.isclose(s['up_price'], 11.0) and np.isclose(s['dn_price'], 9.0)
    assert s['close_up'] and s['one_price_up']


def test_correct_preclose_on_ex_day_uses_csmar():
    p = _panel([('000002', 0, 20, 20, 20, 20, 1.0, 20, 22, 18),
                ('000002', 1, 9.5, 9.6, 9.0, 9.0, 2.0, 10, 11, 9)])
    s = limit_state(p).iloc[1]
    assert s['limit_src'] == 1 and s['close_dn'] and not s['one_price_dn']


def test_cash_dividend_ex_day():
    # 每股派 0.5：前收 10，除息参考价 9.5，因子变化倍数 10 / 9.5
    p = _panel([('000003', 0, 10, 10, 10, 10, 1.0, 10, 11, 9),
                ('000003', 1, 10.45, 10.45, 10.45, 10.45, 10 / 9.5, 9.5, 10.45, 8.55)])
    s = limit_state(p).iloc[1]
    assert s['limit_src'] == 1 and s['one_price_up']


def test_rounding_half_up_st_5pct():
    # ST 5%，早期未除权 PreClose：ref = 6.66 / 2 = 3.33，3.33 × 1.05 = 3.4965 → 3.50；3.33 × 0.95 = 3.1635 → 3.16
    p = _panel([('600001', 0, 6.66, 6.66, 6.66, 6.66, 1.0, 6.66, 6.99, 6.33),
                ('600001', 1, 3.5, 3.5, 3.5, 3.5, 2.0, 6.66, 6.99, 6.33)])
    s = limit_state(p).iloc[1]
    assert s['limit_src'] == 2
    assert np.isclose(s['up_price'], 3.50) and np.isclose(s['dn_price'], 3.16)
    assert s['one_price_up']


def test_first_row_of_next_stock_not_treated_as_ex_day():
    p = _panel([('000001', 0, 10, 10, 10, 10, 1.0, 10, 11, 9),
                ('000004', 1, 5, 5.5, 5, 5.5, 3.0, 5, 5.5, 4.5)])
    s = limit_state(p).iloc[1]
    assert s['limit_src'] == 1 and s['close_up']


def test_rejects_unsorted_input():
    p = _panel([('000001', 0, 10, 10, 10, 10, 1.0, 10, 11, 9),
                ('000004', 0, 5, 5, 5, 5, 1.0, 5, 5.5, 4.5),
                ('000001', 1, 10, 10, 10, 10, 1.0, 10, 11, 9)])
    for bad in (p, p.iloc[[0, 2, 1]].iloc[::-1].reset_index(drop=True)):
        try:
            limit_state(bad)
        except ValueError:
            continue
        raise AssertionError('unsorted input accepted')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
