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
LIMIT_TOL = 0.005
OOS_START, OOS_END = pd.Timestamp('2008-01-01'), pd.Timestamp('2026-09-30')

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


CHINEXT_REFORM = pd.Timestamp('2020-08-24')
MAIN_REGISTRATION = pd.Timestamp('2023-04-10')
MAIN_ST_10PCT = pd.Timestamp('2025-07-07')


def limit_pct(stock: pd.Series, date: pd.Series, is_st=None, age=None, first_trade=None,
              data_start: pd.Timestamp | None = None) -> np.ndarray:
    """当日涨跌幅限制；无涨跌幅限制的交易日为 NaN（与之比较的判定恒为 False）。

    is_st: 当日是否 ST。主板（含改革前创业板）ST 为 5%，2025-07-07 起主板 ST 为 10%；改革后创业板、科创板 ST 仍为 20%。
    age: 上市后第几个交易日（0 = 上市首日）；first_trade: 股票在面板中的首个交易日。
      首个交易日晚于 data_start（即上市日在数据内）时：上市首日不设限；科创板、2020-08-24 起上市的创业板、
      2023-04-10 起上市的主板，前 5 个交易日不设限。
    """
    s = pd.Series(np.asarray(stock).astype(str))
    dt = pd.to_datetime(pd.Series(np.asarray(date)))
    lim = np.full(len(s), 0.10)
    is_chinext = s.str.startswith(('300', '301')).to_numpy()
    chinext_post = is_chinext & (dt >= CHINEXT_REFORM).to_numpy()
    star = s.str.startswith(('688', '689')).to_numpy()
    bse = s.str.startswith(('4', '8', '92')).to_numpy()
    lim[chinext_post | star] = 0.20
    lim[bse] = 0.30
    if is_st is not None:
        st = np.asarray(is_st).astype(bool)
        main_like = st & ~(chinext_post | star | bse)
        lim[main_like] = np.where((dt >= MAIN_ST_10PCT).to_numpy()[main_like], 0.10, 0.05)
    if age is not None and first_trade is not None:
        age = np.asarray(age, dtype=np.float64)
        ft = pd.to_datetime(pd.Series(np.asarray(first_trade)))
        listed_in_data = (ft > (data_start or ft.min())).to_numpy()
        five_day = star | (is_chinext & (ft >= CHINEXT_REFORM).to_numpy()) | \
            (~(is_chinext | star | bse) & (ft >= MAIN_REGISTRATION).to_numpy())
        no_limit = listed_in_data & ((age == 0) | (five_day & (age < 5)))
        lim[no_limit] = np.nan
    return lim


def hit_limit_band(ret, lim, side: str = 'up') -> np.ndarray:
    """旧区间规则（只用于与 limit_state 对账）：收益落在 [限制幅度 − 容差, 限制幅度 + 容差] 内视为触及涨（跌）停。
    lim 为 NaN（无涨跌幅限制）时恒为 False。"""
    r = np.asarray(ret, dtype=np.float64)
    lim = np.asarray(lim, dtype=np.float64)
    if side == 'dn':
        r = -r
    with np.errstate(invalid='ignore'):
        return (r >= lim - LIMIT_TOL) & (r <= lim + LIMIT_TOL)


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
