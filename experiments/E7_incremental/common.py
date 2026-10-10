# -*- coding: utf-8 -*-
"""Shared paths and helpers for E7 (docs/protocol/E7_protocol.md)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.metrics import annualized_sharpe, newey_west_t  # noqa: E402,F401

RAW_DIR = Path(r'F:\BaiduNetdiskDownload\股票4.14\每只股票一个文件\后复权')
DATA = PROJECT_ROOT / 'WORK_SPACE' / 'data' / 'ashare_processed_data'
EXT_PANEL = DATA / 'ashare_extended.feather'
STOCK_INFO = DATA / 'ashare_stock_info.feather'
OUT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E7_incremental'
OUT_DATA = OUT / 'data'
WEEKLY_RETURNS = OUT_DATA / 'weekly_returns.feather'
WEEKLY_CONTROLS = OUT_DATA / 'weekly_controls.feather'

KEYS = ['StockID', 'ending_date']
LIMIT_TOL = 0.005
OOS_START, OOS_END = pd.Timestamp('2008-01-01'), pd.Timestamp('2025-12-31')

# Protocol section 2: canonical models, files and sha1 (first 16 hex chars).
CANONICAL = {
    's_chart': ('WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/preds/pred_cnn_oos.csv', 'be0a89e6b16f522c'),
    's_1d': ('WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv', '6f1982cad36565e9'),
    's_mlp4': ('WORK_SPACE/paper/E2F_mlp_depth/preds/pred_mlp4_oos.csv', '4bf2a2214272f312'),
}
SCORES = list(CANONICAL)

# Protocol section 14: ChartCNN-S (E2G, 4-8-16, 8,306 parameters, 5-seed ensemble), 18 yearly OOS files.
CHART_S_DIR = ('WORK_SPACE/paper/E2G_chart_capacity/chartcnn_s/D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C256/'
               '20d5p-lr1E-4-dp0.60-maTrue-vbTrue-weekTrained-noDelayedReturn-e2g-chartcnn_s-ch4-8-16-p5/ensem_res')
CHART_S_SHA1 = '5ccc6e6cea9fd09c'

CTRL_A = ['rev_1w', 'ret_20d', 'max_20d', 'vol_20d', 'vtrend', 'range_20d', 'clv_20d', 'ma_gap']
CTRL_B = ['mom_12_1', 'size', 'turn_20d', 'illiq_20d', 'ivol_20d', 'bm', 'ep']
CTRL_VALUATION = ['bm', 'ep']


def sha1_16(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()[:16]


def load_canonical_preds() -> pd.DataFrame:
    """Inner join of the three canonical OOS predictions; aborts if any file hash differs from the protocol."""
    out = None
    for name, (rel, expected) in CANONICAL.items():
        path = PROJECT_ROOT / rel
        got = sha1_16(path)
        if got != expected:
            raise RuntimeError(f'{name}: sha1 {got} != protocol {expected} ({rel})')
        cols = KEYS + ['up_prob'] + (['ret_val'] if out is None else [])
        df = pd.read_csv(path, usecols=cols, dtype={'StockID': str})
        df['StockID'] = df['StockID'].str.zfill(6)
        df['ending_date'] = pd.to_datetime(df['ending_date'])
        df = df.rename(columns={'up_prob': name})
        out = df if out is None else out.merge(df, on=KEYS, how='inner')
    return out


def chart_s_files() -> list[Path]:
    files = sorted((PROJECT_ROOT / CHART_S_DIR).glob('ensem5_res_*.csv'))
    if len(files) != 18:
        raise RuntimeError(f'ChartCNN-S: expected 18 OOS years, found {len(files)}')
    return files


def chart_s_sha1() -> str:
    """sha1 (first 16 hex) of the newline-joined per-file sha1s of the 18 yearly files, in file-name order."""
    return hashlib.sha1('\n'.join(sha1_16(f) for f in chart_s_files()).encode()).hexdigest()[:16]


def load_chart_s() -> pd.DataFrame:
    got = chart_s_sha1()
    if got != CHART_S_SHA1:
        raise RuntimeError(f's_chartS: sha1 {got} != protocol {CHART_S_SHA1}')
    df = pd.concat([pd.read_csv(f, usecols=KEYS + ['up_prob'], dtype={'StockID': str}) for f in chart_s_files()],
                   ignore_index=True)
    df['StockID'] = df['StockID'].str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    return df.rename(columns={'up_prob': 's_chartS'})


def week_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week."""
    d = pd.DataFrame({'Date': pd.DatetimeIndex(calendar).sort_values()})
    iso = d['Date'].dt.isocalendar()
    return pd.DatetimeIndex(d.groupby(iso['year'] * 100 + iso['week'])['Date'].max().sort_values().values)


def limit_pct(stock: pd.Series, date: pd.Series) -> np.ndarray:
    s = stock.astype(str)
    lim = np.full(len(s), 0.10)
    chinext = s.str.startswith(('300', '301')) & (pd.to_datetime(date) >= pd.Timestamp('2020-08-24'))
    star = s.str.startswith(('688', '689'))
    bse = s.str.startswith(('4', '8', '92'))
    lim[chinext.values | star.values] = 0.20
    lim[bse.values] = 0.30
    return lim


def cs_rank(df: pd.DataFrame, cols: list[str], by: str = 'ending_date') -> pd.DataFrame:
    """Cross-sectional rank mapped linearly to [-0.5, 0.5]; NaN stays NaN."""
    g = df.groupby(by, sort=False)[cols]
    r = g.rank(method='average')
    n = g.transform('count')
    return (r - 1).div((n - 1).where(n > 1)) - 0.5


def cs_winsor_z(df: pd.DataFrame, cols: list[str], by: str = 'ending_date', q: float = 0.01) -> pd.DataFrame:
    g = df.groupby(by, sort=False)[cols]
    lo, hi = g.transform(lambda x: x.quantile(q)), g.transform(lambda x: x.quantile(1 - q))
    w = df[cols].clip(lo, hi, axis=0)
    gw = w.groupby(df[by], sort=False)
    return (w - gw.transform('mean')) / gw.transform('std')


def cs_winsor(s: pd.Series, by: pd.Series, q: float = 0.01) -> pd.Series:
    g = s.groupby(by, sort=False)
    return s.clip(g.transform(lambda x: x.quantile(q)), g.transform(lambda x: x.quantile(1 - q)))


def industry_demean(df: pd.DataFrame, cols: list[str], by: tuple = ('ending_date', 'industry')) -> pd.DataFrame:
    return df[cols] - df.groupby(list(by), sort=False, observed=True)[cols].transform('mean')


def git_hash() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return 'unknown'


def write_manifest(name: str, extra: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = {'script': name, 'git': git_hash(), 'time': datetime.now().isoformat(timespec='seconds'),
         'canonical_sha1': {k: v[1] for k, v in CANONICAL.items()}}
    m.update(extra or {})
    (OUT / f'run_manifest_{name}.json').write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                                   encoding='utf-8')
