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

_PKG = PROJECT_ROOT / 'trend_submit' / 'trend_code_submit'
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from Misc.config import ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, ASHARE_STOCK_INFO_PATH  # noqa: E402

DATA = Path(ASHARE_PROCESSED_DATA_DIR)
EXT_PANEL = Path(ASHARE_PANEL_PATH)
STOCK_INFO = Path(ASHARE_STOCK_INFO_PATH)
OUT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E7_incremental'
OUT_DATA = OUT / 'data'
WEEKLY_RETURNS = OUT_DATA / 'weekly_returns.feather'
WEEKLY_CONTROLS = OUT_DATA / 'weekly_controls.feather'

KEYS = ['StockID', 'ending_date']
OOS_START, OOS_END = pd.Timestamp('2008-01-01'), pd.Timestamp('2026-09-30')

# Canonical OOS predictions of the clean models (clean_spec 5.4): path relative to PROJECT_ROOT and sha1 (first 16
# hex chars), frozen in registry v2 once training is done. Nothing is read before the registry exists.
REGISTRY_V2 = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'registry_v2.json'
SCORES = ['s_chart', 's_1d', 's_mlp4']
CHART_S = 's_chartS'


def registry_entry(name: str) -> tuple[Path, str]:
    if not REGISTRY_V2.exists():
        raise RuntimeError(f'{REGISTRY_V2} 不存在：干净版本模型尚未训练、冻结（C3）')
    e = json.loads(REGISTRY_V2.read_text(encoding='utf-8'))['predictions'][name]
    return PROJECT_ROOT / e['path'], e['sha1']


def canonical_sha1() -> dict:
    return {k: registry_entry(k)[1] for k in SCORES + [CHART_S]} if REGISTRY_V2.exists() else {}

CTRL_A = ['rev_1w', 'ret_20d', 'max_20d', 'vol_20d', 'vtrend', 'range_20d', 'clv_20d', 'ma_gap']
CTRL_B = ['mom_12_1', 'size', 'turn_20d', 'illiq_20d', 'ivol_20d', 'bm', 'ep']
CTRL_VALUATION = ['bm', 'ep']


def sha1_16(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()[:16]


def load_prediction(name: str) -> pd.DataFrame:
    """One registered OOS prediction file (KEYS, up_prob -> name); aborts if its sha1 differs from registry v2."""
    path, expected = registry_entry(name)
    got = sha1_16(path)
    if got != expected:
        raise RuntimeError(f'{name}: sha1 {got} != registry {expected} ({path})')
    df = pd.read_csv(path, usecols=KEYS + ['up_prob'], dtype={'StockID': str})
    df['StockID'] = df['StockID'].str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    return df.rename(columns={'up_prob': name})


def load_canonical_preds() -> pd.DataFrame:
    """Inner join of the three canonical OOS predictions with the universe label (ret_val)."""
    from Misc.config import ASHARE_UNIVERSE_PATH
    out = pd.read_feather(ASHARE_UNIVERSE_PATH, columns=KEYS + ['split', 'ret_label'])
    out = out[out['split'] == 'oos'].drop(columns='split').rename(columns={'ret_label': 'ret_val'})
    out['StockID'] = out['StockID'].astype(str)
    for name in SCORES:
        out = out.merge(load_prediction(name), on=KEYS, how='inner')
    return out


def load_chart_s() -> pd.DataFrame:
    return load_prediction(CHART_S)


def week_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week."""
    d = pd.DataFrame({'Date': pd.DatetimeIndex(calendar).sort_values()})
    iso = d['Date'].dt.isocalendar()
    return pd.DatetimeIndex(d.groupby(iso['year'] * 100 + iso['week'])['Date'].max().sort_values().values)


PRICE_TOL = 0.005
EX_FACTOR_TOL = 1e-4
EX_PRECLOSE_TOL = 0.005


def _round_cent(x: np.ndarray) -> np.ndarray:
    return np.floor(x * 100 + 0.5 + 1e-9) / 100


def limit_state(p: pd.DataFrame) -> pd.DataFrame:
    """涨跌停状态（clean_spec 第 9 节“数据源更换为 CSMAR”第 3 条），以原始价格精确到分判定。

    p 须为面板中各股票的完整、连续交易行（不可先按日期或 universe 子集过滤），按 (StockID, Date) 排序，含列
    StockID, Open, High, Low, Close, Close_raw, adj_factor, PreClose, LimitUp, LimitDown。
    除权日 = 复权因子相对前一交易行变化 > 1e-4，或 PreClose 与前一交易行收盘相差 > 0.005 元。
    除权日若 |PreClose / ref − 1| > 0.5%（ref = 前收 / 因子变化倍数），按 ref 与限价幅度重算限价，否则用 CSMAR 限价。
    返回与 p 同索引：up_price, dn_price, limit_src（0 无限价 / 1 CSMAR / 2 重算）, no_limit,
    close_up, close_dn（收盘在涨/跌停价）, touch_up, touch_dn（最高/最低价触及）, one_price_up, one_price_dn（开高低收都在限价）。
    """
    s = p['StockID']
    sid = s.cat.codes.to_numpy() if isinstance(s.dtype, pd.CategoricalDtype) else pd.factorize(s)[0]
    new_stock = np.r_[True, sid[1:] != sid[:-1]]
    if new_stock.sum() != len(np.unique(sid)):
        raise ValueError('limit_state: rows of a stock must be contiguous (sort by StockID, Date)')
    d = p['Date'].to_numpy()
    if (~new_stock[1:] & (d[1:] <= d[:-1])).any():
        raise ValueError('limit_state: dates not strictly increasing within a stock')
    f = p['adj_factor'].to_numpy(np.float64)
    c = p['Close_raw'].to_numpy(np.float64)
    o, h, lo = (p[k].to_numpy(np.float64) / f for k in ('Open', 'High', 'Low'))
    pre = p['PreClose'].to_numpy(np.float64)
    up = p['LimitUp'].to_numpy(np.float64).copy()
    dn = p['LimitDown'].to_numpy(np.float64).copy()

    prev_c = np.r_[np.nan, c[:-1]]
    jump = np.r_[np.nan, f[1:] / f[:-1]]
    prev_c[new_stock] = np.nan
    jump[new_stock] = np.nan
    with np.errstate(invalid='ignore'):
        ex = (np.abs(jump - 1) > EX_FACTOR_TOL) | (np.abs(pre - prev_c) > EX_PRECLOSE_TOL)
        ref = prev_c / jump
        stale = ex & (np.abs(pre / ref - 1) > 0.005) & np.isfinite(up)
        pct_up = np.round(up / pre - 1, 2)
        pct_dn = np.round(1 - dn / pre, 2)
    up[stale] = _round_cent(ref[stale] * (1 + pct_up[stale]))
    dn[stale] = _round_cent(ref[stale] * (1 - pct_dn[stale]))

    has = np.isfinite(up) & np.isfinite(dn)
    src = np.where(has, np.where(stale, 2, 1), 0).astype(np.int8)
    with np.errstate(invalid='ignore'):
        at = lambda x, lim: np.abs(x - lim) <= PRICE_TOL  # noqa: E731
        out = pd.DataFrame({
            'up_price': up, 'dn_price': dn, 'limit_src': src, 'no_limit': ~has,
            'close_up': has & at(c, up), 'close_dn': has & at(c, dn),
            'touch_up': has & (h >= up - PRICE_TOL), 'touch_dn': has & (lo <= dn + PRICE_TOL),
        }, index=p.index)
        out['one_price_up'] = out['close_up'] & at(o, up) & at(h, up) & at(lo, up)
        out['one_price_dn'] = out['close_dn'] & at(o, dn) & at(h, dn) & at(lo, dn)
    return out


LIMIT_INPUT_COLS = ['Date', 'StockID', 'Open', 'High', 'Low', 'Close_raw', 'adj_factor', 'PreClose', 'LimitUp',
                    'LimitDown']


def limit_flags_at(keys: pd.DataFrame, date_col: str = 'ending_date') -> pd.DataFrame:
    """limit_state on the full panel, picked at keys (StockID, date_col); returned in the order of keys.
    Keys without a trading row get NaN in every flag column."""
    p = pd.read_feather(EXT_PANEL, columns=LIMIT_INPUT_COLS)
    ls = limit_state(p).drop(columns=['up_price', 'dn_price'])
    ls['StockID'] = p['StockID'].astype(str).to_numpy()
    ls[date_col] = p['Date'].to_numpy()
    del p
    k = pd.DataFrame({'StockID': keys['StockID'].astype(str).str.zfill(6).to_numpy(),
                      date_col: pd.to_datetime(keys[date_col]).to_numpy()})
    out = k.merge(ls, on=['StockID', date_col], how='left', validate='many_to_one')
    out.index = keys.index
    return out.drop(columns=['StockID', date_col])


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
         'canonical_sha1': canonical_sha1()}
    m.update(extra or {})
    (OUT / f'run_manifest_{name}.json').write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                                   encoding='utf-8')
