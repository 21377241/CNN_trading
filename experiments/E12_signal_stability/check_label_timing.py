# -*- coding: utf-8 -*-
"""E12 最后三个信号周的标签时间核对（只读，不改任何样本）。

对 E12 第二、三阶段共同样本的最后三个信号周逐股检查：
  1. 标签收盘日 next_date 是否为信号周之后下一个 ISO 周的最后一个交易日；
  2. 用面板收盘价独立重算 ret_val = Close(next_date) / Close(t) − 1，与样本中的值逐股比对；
  3. 面板在 SAMPLE_END 截止，最后一周可能被截断：用 CSMAR 原始日度文件（含截止日之后的行）
     确认 next_date 所在 ISO 周在原始数据中没有更晚的真实交易日（Filling == 0）。

  python experiments/E12_signal_stability/check_label_timing.py [--csmar-root DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'trend_submit' / 'trend_code_submit'))
from Misc.config import ASHARE_PANEL_PATH, ASHARE_UNIVERSE_PATH  # noqa: E402
from experiments.E12_signal_stability.run_e12_stage2 import build_sample  # noqa: E402

OUT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E12_signal_stability' / 'stage3' / 'label_timing_check.csv'
N_LAST = 3


def iso_week(d: pd.Timestamp) -> tuple[int, int]:
    y, w, _ = d.isocalendar()
    return int(y), int(w)


def raw_trading_days(csmar_root: Path, start: pd.Timestamp) -> pd.DatetimeIndex:
    days = set()
    paths = [p for p in sorted(csmar_root.glob('日个股回报率文件*.zip')) if str(start.year) in p.name]
    if not paths:
        raise FileNotFoundError(f'no CSMAR daily zip covering {start.year} under {csmar_root}')
    for path in paths:
        with zipfile.ZipFile(path) as z:
            for n in sorted(n for n in z.namelist() if n.startswith('TRDNEW_Dalyr') and n.endswith('.csv')):
                d = pacsv.read_csv(z.open(n), convert_options=pacsv.ConvertOptions(
                    include_columns=['Trddt', 'Filling'])).to_pandas()
                d['Trddt'] = pd.to_datetime(d['Trddt'])
                days.update(d.loc[(d['Trddt'] >= start) & (d['Filling'] == 0), 'Trddt'].unique())
    return pd.DatetimeIndex(sorted(days))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csmar-root', default=os.environ.get('CSMAR_ROOT') or str(PROJECT_ROOT.parent / 'csmar_ashare_daily'))
    args = ap.parse_args()

    common, _ = build_sample()
    last = sorted(common['ending_date'].unique())[-N_LAST:]
    panel = pd.read_feather(ASHARE_PANEL_PATH, columns=['Date', 'StockID', 'Close'])
    panel['StockID'] = panel['StockID'].astype(str).str.zfill(6)
    cal = pd.DatetimeIndex(np.sort(panel['Date'].unique()))
    px = panel.set_index(['StockID', 'Date'])['Close']
    u = pd.read_feather(ASHARE_UNIVERSE_PATH, columns=['StockID', 'ending_date', 'next_date'])
    u['StockID'] = u['StockID'].astype(str).str.zfill(6)
    raw = raw_trading_days(Path(args.csmar_root), pd.Timestamp(last[0]))

    rows = []
    for d in map(pd.Timestamp, last):
        c = common.loc[common['ending_date'] == d, ['StockID', 'ending_date', 'ret_val']]
        m = c.merge(u, on=['StockID', 'ending_date'], how='left', validate='one_to_one')
        nd = m['next_date'].unique()
        if len(nd) != 1:
            raise RuntimeError(f'{d.date()}: next_date not unique: {nd}')
        nd = pd.Timestamp(nd[0])
        nxt_week = iso_week(cal[cal > d][0])
        expected = max(x for x in cal[cal > d] if iso_week(x) == nxt_week)
        raw_same_week = [x for x in raw if iso_week(x) == iso_week(nd)]
        raw_after = raw[raw > nd]
        c0 = px.reindex(list(zip(m['StockID'], [d] * len(m)))).to_numpy()
        c1 = px.reindex(list(zip(m['StockID'], [nd] * len(m)))).to_numpy()
        rows.append({
            'signal_date': d.date(), 'signal_weekday': d.day_name(),
            'label_date': nd.date(), 'label_weekday': nd.day_name(),
            'label_is_next_iso_week_end_in_panel': nd == expected,
            'trading_days_ahead': int(cal.get_loc(nd) - cal.get_loc(d)),
            'raw_last_trading_day_of_label_week': max(raw_same_week).date(),
            'label_week_complete_in_raw': max(raw_same_week) == nd,
            'raw_next_trading_day_after_label': raw_after[0].date() if len(raw_after) else None,
            'panel_last_date': cal[-1].date(),
            'rows': len(m), 'ret_val_nan': int(m['ret_val'].isna().sum()),
            'max_abs_diff_recomputed_ret_val': float(np.nanmax(np.abs(c1 / c0 - 1 - m['ret_val'].to_numpy()))),
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(out.T.to_string())
    ok = (out['label_is_next_iso_week_end_in_panel'] & out['label_week_complete_in_raw']
          & (out['ret_val_nan'] == 0) & (out['max_abs_diff_recomputed_ret_val'] == 0)).all()
    print('ALL CHECKS PASS' if ok else 'CHECK FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
