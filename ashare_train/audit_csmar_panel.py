# -*- coding: utf-8 -*-
"""
audit_csmar_panel.py - CSMAR 面板独立审计（clean_spec 第 9 节“数据源更换为 CSMAR”第 6 条）

不导入构建代码，从原始 zip 独立重算并与构建产物比对；只有截断测试（E1）以子进程方式运行构建脚本。
各阶段结果写入 WORK_SPACE/data/ashare_processed_data/audit/results_<phase>.json，最后汇总为
docs/protocol/csmar_panel_audit.md。

阶段:
  e1    截断重算（面板、财务时点表、市场表在截断点之前逐位相同）
  raw   原始文件重算：A0 逐列重算比对，A1–A6，B1–B10
  fund  财务与市场：E2、E3、F1、F3、F4、M1
  ext   第三方（akshare）抽查：C1、C2
与旧数据商数据的对账（原 D1、D2、F2）不属于审计，移到 reconcile_pre_clean.py；本脚本不读取任何旧数据。

用法:
  python ashare_train/audit_csmar_panel.py                 # 全部阶段
  python ashare_train/audit_csmar_panel.py --phases raw fund
  python ashare_train/audit_csmar_panel.py --report-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
for _p in (PROJECT_ROOT, CODE_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Misc.config import (  # noqa: E402
    ASHARE_FUNDAMENTALS_PATH, ASHARE_MARKET_PATH, ASHARE_PANEL_PATH, ASHARE_PROCESSED_DATA_DIR, CSMAR_ROOT,
    SAMPLE_END, WORK_DIR,
)
from experiments.E7_incremental.common import limit_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('audit')

AUDIT_DIR = Path(ASHARE_PROCESSED_DATA_DIR, 'audit')
REPORT = Path(PROJECT_ROOT, 'docs', 'protocol', 'csmar_panel_audit.md')
TRUNC_END = '2015-06-30'
TRUNC_DIR = Path(WORK_DIR, 'tmp', 'audit_trunc')
SEED = 20261011

A_MARKETS = (1, 4, 16, 32, 64)
MAIN_MARKETS = (1, 4, 16, 32)
ST_SET = (2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16)
CODE_MAP = {300114: 302132, 22: 1872, 43: 1914, 601313: 601360, 600849: 601607}
# candidates found by the A5 scan that are deliberately not merged
CODE_CHANGE_EXCLUDED = {
    (832317, 920405): '不同公司，前收盘价巧合',
}
# same legal entity per the company file, not merged because the return across the gap is unobservable in CSMAR
ENTITY_NOT_MERGED = {
    (600087, 601975): '长航油运 2014 年退市，2019 年重新上市为招商南油，间隔约 4.5 年',
    (765, 1267): '华信股份 2005 年退市，2021 年重新上市为汇绿生态',
    (833874, 301192): '北交所转板创业板，CSMAR 缺转板前后数月数据',
    (833994, 301321): '北交所转板创业板，CSMAR 缺转板前后数月数据',
    (832317, 688287): '北交所转板科创板，CSMAR 缺转板前后数月数据',
}
CODE_CHANGE_MAX_GAP = 20
CODE_REUSE = {600018: '2006-10-26'}
SAMPLE_START = '1999-01-01'

F64 = ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd', 'Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd', 'PreClosePrice',
       'LimitDown', 'LimitUp']
F32 = ['Markettype', 'Filling', 'Trdsta', 'Dnvaltrd', 'Dsmvosd', 'Dsmvtll', 'Dsmvtl2', 'Dsmvtl3', 'LimitStatus']
RET_FIELDS = ['Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd', 'PreClosePrice']

# 中国人民银行一年期整存整取定期存款基准利率（%），生效日
PBOC_1Y_DEPOSIT = [
    ('1999-06-10', 2.25), ('2002-02-21', 1.98), ('2004-10-29', 2.25), ('2006-08-19', 2.52), ('2007-03-18', 2.79),
    ('2007-05-19', 3.06), ('2007-07-21', 3.33), ('2007-08-22', 3.60), ('2007-09-15', 3.87), ('2007-12-21', 4.14),
    ('2008-10-09', 3.87), ('2008-10-30', 3.60), ('2008-11-27', 2.52), ('2008-12-23', 2.25), ('2010-10-20', 2.50),
    ('2010-12-26', 2.75), ('2011-02-09', 3.00), ('2011-04-06', 3.25), ('2011-07-07', 3.50), ('2012-06-08', 3.25),
    ('2012-07-06', 3.00), ('2014-11-22', 2.75), ('2015-03-01', 2.50), ('2015-05-11', 2.25), ('2015-06-28', 2.00),
    ('2015-08-26', 1.75), ('2015-10-24', 1.50),
]


# --------------------------------------------------------------------------------------------- utilities

def day_int(x) -> np.ndarray:
    return np.asarray(x, dtype='datetime64[ns]').astype('datetime64[D]').astype(np.int64)


def day_str(d) -> str:
    return str(np.datetime64(int(d), 'D'))


def result(rid: str, check: str, criterion: str, value, passed, note: str = '') -> dict:
    return {'id': rid, 'check': check, 'criterion': criterion, 'value': value,
            'passed': None if passed is None else bool(passed), 'note': note}


def save_phase(phase: str, results: list, details: dict) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    Path(AUDIT_DIR, f'results_{phase}.json').write_text(
        json.dumps({'results': results, 'details': details, 'time': time.strftime('%Y-%m-%d %H:%M:%S')},
                   indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def zip_member(glob_pat: str, member: str) -> tuple[Path, str]:
    path = sorted(Path(CSMAR_ROOT).glob(glob_pat))
    if len(path) != 1:
        raise RuntimeError(f'expected one {glob_pat}, found {path}')
    return path[0], member


def read_zip_csv(glob_pat: str, member: str, **kw) -> pd.DataFrame:
    p, m = zip_member(glob_pat, member)
    with zipfile.ZipFile(p) as z:
        return pd.read_csv(z.open(m), **kw)


def read_company() -> pd.DataFrame:
    co = read_zip_csv('公司文件*.zip', 'TRDNEW_Co.csv', dtype=str)
    co['code'] = co['Stkcd'].astype(int)
    co['Markettype'] = co['Markettype'].astype(int)
    for c in ['Listdt', 'DelistedDate']:
        co[c] = pd.to_datetime(co[c], errors='coerce')
    return co.set_index('code')


def panel_col(path, col: str) -> np.ndarray:
    t = feather.read_table(path, columns=[col], memory_map=True)
    a = t.column(col)
    if col == 'StockID':
        ch = a.combine_chunks()
        return np.asarray(ch.dictionary.to_pylist(), dtype=np.int64)[ch.indices.to_numpy()]
    return a.to_numpy()


def rel_close(a: np.ndarray, b: np.ndarray, rtol: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid='ignore', divide='ignore'):
        ok = np.abs(a - b) <= rtol * np.maximum(np.abs(a), np.abs(b)) + 1e-12
    return ok | both_nan


# --------------------------------------------------------------------------------------------- phase e1

def phase_e1() -> None:
    results, details = [], {}
    if TRUNC_DIR.exists():
        shutil.rmtree(TRUNC_DIR)
    env = dict(os.environ, CNN_TRADING_WORK_DIR=str(TRUNC_DIR), CNN_TRADING_SAMPLE_END=TRUNC_END,
               PYTHONIOENCODING='utf-8')
    for script in ['1_preprocess_ashare_data.py', 'build_fundamentals.py', 'build_market.py']:
        t0 = time.time()
        r = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(PROJECT_ROOT, 'ashare_train', script)]
                           + (['--force'] if script.startswith('1_') else []),
                           env=env, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding='utf-8')
        details[f'{script}_seconds'] = round(time.time() - t0)
        if r.returncode != 0:
            raise RuntimeError(f'truncated {script} failed:\n{r.stderr[-3000:]}')
        logger.info(f'截断重算 {script} 完成（{time.time() - t0:.0f}s）')
    tdir = Path(TRUNC_DIR, 'data', 'ashare_processed_data')

    end = np.datetime64(TRUNC_END)
    full_date = panel_col(ASHARE_PANEL_PATH, 'Date')
    m = full_date <= end
    del full_date
    tp = Path(tdir, Path(ASHARE_PANEL_PATH).name)
    cols = feather.read_table(tp, memory_map=True).column_names
    full_cols = feather.read_table(ASHARE_PANEL_PATH, memory_map=True).column_names
    diff_cols = {}
    if cols != full_cols:
        diff_cols['__columns__'] = [cols, full_cols]
    # code changes after the cutoff leave the history under the old code in the truncated build
    t_sid = pd.Series(panel_col(tp, 'StockID')).replace(CODE_MAP).to_numpy(np.int64)
    order = np.lexsort((panel_col(tp, 'Date'), t_sid))
    details['E1_relabelled_old_codes'] = sorted(f'{c:06d}' for c in set(panel_col(tp, 'StockID')) & set(CODE_MAP))
    n_trunc = None
    for c in full_cols:
        a = panel_col(ASHARE_PANEL_PATH, c)[m]
        b = (t_sid if c == 'StockID' else panel_col(tp, c))[order]
        n_trunc = len(b)
        if len(a) != len(b):
            diff_cols[c] = f'rows {len(a)} vs {len(b)}'
            continue
        same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)
        if not same:
            diff_cols[c] = int((~(rel_close(a, b, 0) if a.dtype.kind == 'f' else a == b)).sum())
    details['panel_rows_before_cutoff'] = n_trunc
    details['panel_columns_differing'] = diff_cols

    f_full = pd.read_parquet(ASHARE_FUNDAMENTALS_PATH)
    f_tr = pd.read_parquet(Path(tdir, Path(ASHARE_FUNDAMENTALS_PATH).name))
    f_full = f_full[f_full['avail_date'] <= pd.Timestamp(TRUNC_END)].reset_index(drop=True)
    f_tr = f_tr[f_tr['avail_date'] <= pd.Timestamp(TRUNC_END)].reset_index(drop=True)
    fund_same = f_full.equals(f_tr)
    details['fundamentals_rows_before_cutoff'] = [len(f_full), len(f_tr)]
    m_full = pd.read_parquet(ASHARE_MARKET_PATH)
    m_tr = pd.read_parquet(Path(tdir, Path(ASHARE_MARKET_PATH).name))
    m_full = m_full[m_full['Date'] <= pd.Timestamp(TRUNC_END)].reset_index(drop=True)
    m_tr = m_tr.reset_index(drop=True)
    mkt_same = m_full.equals(m_tr)
    details['market_rows_before_cutoff'] = [len(m_full), len(m_tr)]
    ok = not diff_cols and fund_same and mkt_same
    results.append(result('E1', f'截断到 {TRUNC_END} 后重算', '截断点之前的所有列逐位相同（截断点之后才发生的代码变更，旧代码行按新代码对齐）',
                          f'面板 {n_trunc:,} 行 {len(full_cols)} 列不同列 {len(diff_cols)}；财务表相同={fund_same}；市场表相同={mkt_same}',
                          ok))
    save_phase('e1', results, details)
    shutil.rmtree(TRUNC_DIR, ignore_errors=True)


# --------------------------------------------------------------------------------------------- phase raw

def load_raw(end: np.datetime64) -> tuple[dict, pd.DataFrame, list]:
    """A 股、样本截止日前的原始行；返回 Filling = 0 的列数组、Filling 1/2 的键、每个分段的范围。"""
    zips = sorted(Path(CSMAR_ROOT).glob('日个股回报率文件*.zip'))
    cols = {c: [] for c in ['Stkcd', 'day', 'zip'] + F64 + F32}
    fill_keys, zinfo = [], []
    for i, zp in enumerate(zips):
        with zipfile.ZipFile(zp) as z:
            members = sorted(n for n in z.namelist() if n.startswith('TRDNEW_Dalyr') and n.endswith('.csv'))
            for n in members:
                d = pd.read_csv(z.open(n), usecols=['Stkcd', 'Trddt'] + F64 + F32,
                                dtype={'Stkcd': 'int32', 'Trddt': str, **{c: 'float64' for c in F64},
                                       **{c: 'float32' for c in F32}})
                day = day_int(pd.to_datetime(d['Trddt'], format='%Y-%m-%d'))
                zinfo.append({'zip': zp.name, 'member': n, 'rows': len(d), 'first': day_str(day.min()),
                              'last': day_str(day.max()), 'markettypes': sorted(d['Markettype'].dropna().astype(int).unique().tolist())})
                keep = d['Markettype'].isin(A_MARKETS).to_numpy() & (day <= day_int([end])[0])
                fl = d['Filling'].to_numpy()
                for v in (1, 2):
                    k = keep & (fl == v)
                    fill_keys.append(pd.DataFrame({'Stkcd': d['Stkcd'].to_numpy()[k], 'day': day[k], 'Filling': np.int8(v)}))
                k = keep & (fl == 0)
                other = keep & ~np.isin(fl, [0, 1, 2])
                if other.any():
                    raise ValueError(f'{zp.name}/{n}: Filling values outside 0/1/2')
                cols['Stkcd'].append(d['Stkcd'].to_numpy()[k])
                cols['day'].append(day[k])
                cols['zip'].append(np.full(int(k.sum()), i, dtype=np.int8))
                for c in F64 + F32:
                    cols[c].append(d[c].to_numpy()[k])
                del d
        logger.info(f'  读入 {zp.name}')
    R = {}
    for c in list(cols):
        R[c] = np.concatenate(cols.pop(c))
    return R, pd.concat(fill_keys, ignore_index=True), zinfo


def take(R: dict, idx) -> None:
    for c in list(R):
        R[c] = R[c][idx]


def phase_raw() -> None:
    results, details = [], {}
    end = np.datetime64(SAMPLE_END)
    t0 = time.time()
    R, fills, zinfo = load_raw(end)
    details['zips'] = zinfo
    logger.info(f'原始交易行 {len(R["Stkcd"]):,}（{time.time() - t0:.0f}s）')

    take(R, np.lexsort((R['day'], R['Stkcd'])))
    key = (R['Stkcd'].astype(np.int64) << 20) | R['day']
    same = key[1:] == key[:-1]
    conflict = 0
    if same.any():
        i = np.flatnonzero(same)
        neq = np.zeros(len(i), bool)
        for c in F64 + F32:
            a = R[c]
            neq |= ~((a[i] == a[i + 1]) | (np.isnan(a[i]) & np.isnan(a[i + 1])))
        conflict = int(neq.sum())
        details['duplicate_examples'] = [(int(R['Stkcd'][j]), day_str(R['day'][j])) for j in i[:10]]
        take(R, np.r_[True, ~same])
    n_dup = int(same.sum())
    del key, same

    allmiss = np.ones(len(R['Stkcd']), bool)
    for c in RET_FIELDS:
        allmiss &= np.isnan(R[c])
    details['all_return_fields_missing'] = [(int(R['Stkcd'][j]), day_str(R['day'][j])) for j in np.flatnonzero(allmiss)]
    take(R, ~allmiss)
    n = len(R['Stkcd'])

    co = read_company()
    cal = np.unique(R['day'][np.isin(R['Markettype'], [1, 4])])

    # ---- A7 code reuse: listing date (company file) far after the first trading row
    first_tr = pd.Series(R['day']).groupby(R['Stkcd']).min()
    lst = co['Listdt'].reindex(first_tr.index)
    has_l = lst.notna().to_numpy() & ~first_tr.index.isin(list(CODE_MAP.values()))
    lag_r = np.searchsorted(cal, first_tr.to_numpy()) - np.searchsorted(cal, day_int(lst.fillna(pd.Timestamp('2100-01-01'))))
    reuse = {int(c): str(lst[c].date()) for c in first_tr.index[has_l & (lag_r < -5)]}
    rlog = []
    for code, dt in reuse.items():
        cut = day_int([np.datetime64(dt)])[0]
        m = (R['Stkcd'] == code) & (R['day'] < cut)
        rlog.append({'code': f'{code:06d}', 'listdt': dt, 'name_now': co.loc[code, 'Stknme'],
                     'predecessor_rows': int(m.sum()), 'predecessor_span': [day_str(R['day'][m].min()), day_str(R['day'][m].max())]})
        take(R, ~m)
        fills = fills[~((fills['Stkcd'] == code) & (fills['day'] < cut))]
    details['A7_code_reuse'] = rlog
    results.append(result('A7', '代码复用（新上市公司沿用被吸收公司的代码）', '检出情形与规格登记表一致，前身段已剔除',
                          f'检出 {len(rlog)} 例：{[(r["code"], r["listdt"], r["predecessor_rows"]) for r in rlog]}',
                          reuse == CODE_REUSE))
    n = len(R['Stkcd'])
    sid, day = R['Stkcd'], R['day']
    ci = np.searchsorted(cal, day)
    details['rows_on_non_calendar_days'] = int((cal[np.minimum(ci, len(cal) - 1)] != day).sum())

    # ---- A6 zip boundaries
    zr = pd.DataFrame(zinfo).groupby('zip').agg(first=('first', 'min'), last=('last', 'max')).sort_values('first')
    per_day = np.bincount(ci, minlength=len(cal))
    gaps = []
    for (_, a), (_, b) in zip(zr.iloc[:-1].iterrows(), zr.iloc[1:].iterrows()):
        bd = np.searchsorted(cal, day_int([np.datetime64(a['last'])])[0])
        w = per_day[max(0, bd - 5): bd + 6]
        gaps.append({'boundary': a['last'], 'next_first': b['first'], 'rows_per_day_window': w.tolist(),
                     'min_over_median': float(w.min() / np.median(w))})
    details['zip_boundaries'] = gaps
    boundary_ok = all(g['min_over_median'] > 0.9 for g in gaps)
    results.append(result('A6', '分段 zip 拼接', '内容冲突的重复行为 0；分段边界前后无缺日',
                          f'完全相同的重复行 {n_dup:,}，内容冲突 {conflict}；边界窗口最小/中位行数比 '
                          f'{min(g["min_over_median"] for g in gaps):.3f}', conflict == 0 and boundary_ok))

    # ---- per-stock spans on original codes
    new = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:] - 1, n - 1]
    span = pd.DataFrame({'code': sid[starts], 'first': day[starts], 'last': day[ends], 'first_ci': ci[starts],
                         'last_ci': ci[ends], 'first_pre': R['PreClosePrice'][starts], 'last_close': R['Clsprc'][ends],
                         'mt_last': R['Markettype'][ends].astype(int)})

    # ---- A5 code changes
    old = span[span['last_ci'] < len(cal) - 1][['code', 'last_ci', 'last_close']]
    new_ = span[['code', 'first_ci', 'first_pre']]
    pairs = old.assign(k=1).merge(new_.assign(k=1), on='k', suffixes=('_old', '_new')).drop(columns='k')
    pairs = pairs[pairs['code_old'] != pairs['code_new']]
    pairs['gap'] = pairs['first_ci'] - pairs['last_ci']
    pairs['conme_old'] = pairs['code_old'].map(co['Conme'])
    pairs['conme_new'] = pairs['code_new'].map(co['Conme'])
    pairs['same_conme'] = pairs['conme_old'].notna() & (pairs['conme_old'] == pairs['conme_new'])
    pairs['preclose_link'] = (pairs['first_pre'] - pairs['last_close']).abs() < 0.005
    near = (pairs['gap'] >= 1) & (pairs['gap'] <= CODE_CHANGE_MAX_GAP)
    cand = pairs[near & (pairs['preclose_link'] | pairs['same_conme'])].copy()
    far_same = pairs[(pairs['gap'] > CODE_CHANGE_MAX_GAP) & pairs['same_conme']]
    del pairs
    got = set(zip(cand['code_old'], cand['code_new']))
    merged = {(o, n_) for o, n_ in CODE_MAP.items()}
    cand['decision'] = [('合并' if (a, b) in merged else CODE_CHANGE_EXCLUDED.get((a, b), '未登记'))
                        for a, b in zip(cand['code_old'], cand['code_new'])]
    details['A5_candidates'] = cand.astype(str).to_dict('records')
    details['A5_same_company_gap_gt_20'] = far_same.astype(str).to_dict('records')
    # same legal entity in the company file: identical founding date and IPO date, old code delisted before the new listing
    ent = co[co['Estbdt'].notna() & co['Ipodt'].notna() & co['Markettype'].isin(A_MARKETS)].reset_index()[['code', 'Estbdt', 'Ipodt', 'Listdt', 'DelistedDate']]
    ent = ent.merge(ent, on=['Estbdt', 'Ipodt'], suffixes=('_old', '_new'))
    ent = ent[(ent['code_old'] != ent['code_new']) & (ent['DelistedDate_old'] <= ent['Listdt_new'])]
    got_ent = set(zip(ent['code_old'], ent['code_new']))
    details['A5_same_entity_company_file'] = ent.astype(str).to_dict('records')
    ent_unexplained = got_ent - merged - set(ENTITY_NOT_MERGED)
    ok5 = got == merged | set(CODE_CHANGE_EXCLUDED) and not ent_unexplained
    results.append(result('A5', '代码变更（同一法人主体换代码）',
                          '列出全部情形并给出处理（价格扫描：旧代码结束后 20 个交易日内新代码开始，且前收 = 旧收盘或公司全称相同；'
                          '法人扫描：公司文件成立日与 IPO 日相同、旧代码退市不晚于新代码上市）',
                          f'价格扫描候选 {len(cand)} 对：合并 {sorted(got & merged)}；不合并 {sorted(got & set(CODE_CHANGE_EXCLUDED))}；'
                          f'未登记 {sorted(got - merged - set(CODE_CHANGE_EXCLUDED))}；法人扫描 {len(got_ent)} 对：已合并 '
                          f'{len(got_ent & merged)}，登记不合并 {sorted(got_ent & set(ENTITY_NOT_MERGED))}，未解释 {sorted(ent_unexplained)}',
                          ok5))

    # ---- A1 company file
    panel_set = set(span['code']) - set(CODE_MAP)
    co_a = co[co['Markettype'].isin(A_MARKETS)]
    not_in_co = sorted(set(span['code']) - set(co.index))
    missing = []
    fill_codes = set(fills['Stkcd'].unique())
    for code, r in co_a.loc[sorted(set(co_a.index) - panel_set)].iterrows():
        if code in CODE_MAP:
            why = f'代码变更并入 {CODE_MAP[code]}'
        elif pd.notna(r['Listdt']) and r['Listdt'] > pd.Timestamp(SAMPLE_END):
            why = '样本截止日后上市'
        elif code in fill_codes:
            why = '只有停牌/填充行'
        else:
            why = '原始文件中无任何行'
        missing.append({'code': f'{code:06d}', 'name': r['Stknme'], 'markettype': int(r['Markettype']),
                        'listdt': str(r['Listdt'].date()) if pd.notna(r['Listdt']) else None, 'why': why})
    unexplained = [m for m in missing if m['why'] in ('原始文件中无任何行', '只有停牌/填充行')]
    details['A1_company_not_in_panel'] = missing
    details['A1_panel_not_in_company'] = not_in_co
    results.append(result('A1', '面板 A 股集合与公司文件逐只核对', '无法解释的差异为 0',
                          f'面板 {len(panel_set)} 只（并入后）；公司文件 A 股 {len(co_a)} 只；面板有而公司文件无 {len(not_in_co)}；'
                          f'公司文件有而面板无 {len(missing)}（{"；".join(m["code"] + " " + m["why"] for m in missing)}）',
                          not not_in_co and not unexplained))

    # ---- A2 calendar vs 000001
    ix = read_zip_csv('指数文件*.zip', 'TRDNEW_Index.csv', usecols=['Indexcd', 'Trddt'], dtype={'Indexcd': str})
    sh = np.unique(day_int(pd.to_datetime(ix.loc[ix['Indexcd'] == '000001', 'Trddt'])))
    lo = day_int([np.datetime64('1991-01-02')])[0]
    c2, s2 = cal[cal >= lo], sh[sh >= lo]
    only_cal = np.setdiff1d(c2, s2)
    only_sh = np.setdiff1d(s2, c2)
    wd = pd.to_datetime(pd.Series(only_cal.astype('datetime64[D]'))).dt.dayofweek
    details['A2_calendar_not_in_000001'] = [day_str(x) for x in only_cal]
    details['A2_000001_not_in_calendar'] = [day_str(x) for x in only_sh]
    since94 = day_int([np.datetime64('1994-01-01')])[0]
    results.append(result('A2', '市场日历与上证综指交易日（1991-01-02 起）', '完全一致',
                          f'日历有而指数无 {len(only_cal)} 天（其中周六 {int((wd == 5).sum())} 天，最晚 '
                          f'{day_str(only_cal.max()) if len(only_cal) else "-"}）；指数有而日历无 {len(only_sh)} 天；'
                          f'1994 年起差异 {int((only_cal >= since94).sum() + (only_sh >= since94).sum())} 天',
                          len(only_cal) == 0 and len(only_sh) == 0))

    # ---- A3 listing date vs first trade
    sp = span.copy()
    sp['listdt'] = sp['code'].map(co['Listdt'])
    for o, nn in CODE_MAP.items():
        sp.loc[sp['code'] == nn, 'listdt'] = co.loc[o, 'Listdt']
        for c in ('first_ci', 'first'):
            sp.loc[sp['code'] == nn, c] = sp.loc[sp['code'] == o, c].iloc[0]
        sp = sp[sp['code'] != o]
    lst = day_int(sp['listdt'].fillna(pd.Timestamp('1900-01-01')))
    pre_data = (lst < cal[0]) | sp['listdt'].isna().to_numpy()
    lag = sp['first_ci'].to_numpy() - np.searchsorted(cal, lst)
    ev = ~pre_data
    within = (lag[ev] >= 0) & (lag[ev] <= 5)
    bad = sp[ev][~within].assign(lag=lag[ev][~within])
    bad = bad.assign(first=lambda x: [day_str(v) for v in x['first']], listdt=lambda x: x['listdt'].dt.date.astype(str))
    details['A3_beyond_5_days'] = bad[['code', 'listdt', 'first', 'lag', 'mt_last']].astype(str).to_dict('records')
    results.append(result('A3', '首个交易日与上市日之差（交易日）', '≥ 99% 的股票在 5 个交易日内，其余逐只列出',
                          f'{ev.sum()} 只上市日在数据内，{within.mean():.4%} 在 0–5 日内；超出 {len(bad)} 只；'
                          f'上市日早于数据起点 {int(pre_data.sum())} 只', within.mean() >= 0.99))

    # ---- A4 delisting
    sp['delist'] = sp['code'].map(co['DelistedDate'])
    dl = sp[sp['delist'].notna()]
    after = dl[day_int(dl['delist']) < dl['last'].to_numpy()]
    details['A4_trading_after_delist'] = after.assign(last=lambda x: [day_str(v) for v in x['last']])[
        ['code', 'delist', 'last']].astype(str).to_dict('records')
    results.append(result('A4', '已退市股票退市日之后无交易行', '0 只', f'有退市日 {len(dl)} 只；退市日后仍有交易行 {len(after)} 只',
                          len(after) == 0))
    del sp, span

    # ---- code changes: first-day return, comparable-price continuation, relabel, panel order
    handling = []
    for o, nn in CODE_MAP.items():
        io, inn = np.flatnonzero(R['Stkcd'] == o), np.flatnonzero(R['Stkcd'] == nn)
        lo, f0 = io[-1], inn[0]
        h = {'old': f'{o:06d}', 'new': f'{nn:06d}', 'csmar_first_dretwd': float(R['Dretwd'][f0])}
        r_cc = R['Clsprc'][f0] / R['Clsprc'][lo] - 1
        for ret in ('Dretwd', 'Dretnd'):
            if abs(R[ret][f0] - r_cc) > 1e-3:
                if abs(R['PreClosePrice'][f0] - R['Clsprc'][lo]) >= 0.005:
                    raise ValueError(f'{o}->{nn}: first-day {ret} inconsistent and PreClose != old close')
                R[ret][f0] = R['Clsprc'][f0] / R['PreClosePrice'][f0] - 1
                h[f'{ret}_first_overridden'] = float(R[ret][f0])
        for adj, ret in (('Adjprcwd', 'Dretwd'), ('Adjprcnd', 'Dretnd')):
            a = R[adj][inn]
            if np.isnan(a).all():
                R[adj][inn] = R[adj][lo] * np.cumprod(1.0 + R[ret][inn])
                h[adj] = 'chained'
            elif not np.isnan(a).any():
                c = R[adj][lo] * (1.0 + R[ret][f0]) / a[0]
                R[adj][inn] = a * c
                h[adj] = f'rescaled x{c:.6f}'
            else:
                raise ValueError(f'{nn}: {adj} partially missing')
        handling.append(h)
        R['Stkcd'] = np.where(R['Stkcd'] == o, nn, R['Stkcd']).astype(np.int32)
    details['A5_handling'] = handling
    take(R, np.lexsort((R['day'], R['Stkcd'])))
    sid, day = R['Stkcd'], R['day']
    ci = np.searchsorted(cal, day)
    new = np.r_[True, sid[1:] != sid[:-1]]
    consec = np.r_[False, ci[1:] == ci[:-1] + 1] & ~new
    prev = np.r_[0, np.arange(n - 1)]

    # ---- A0 column-by-column reconstruction against the panel
    f = R['Adjprcwd'] / R['Clsprc']
    fnd = R['Adjprcnd'] / R['Clsprc']
    ret = R['Dretwd'].copy()
    ret[new] = np.nan
    ret_nd = R['Dretnd'].copy()
    ret_nd[new] = np.nan
    float_mv = R['Dsmvosd'].astype(np.float64) * 1000.0
    exp = {
        'StockID': sid.astype(np.int64), 'Date': day, 'Close_raw': R['Clsprc'], 'adj_factor': f, 'adj_factor_nd': fnd,
        'Open': R['Opnprc'] * f, 'High': R['Hiprc'] * f, 'Low': R['Loprc'] * f, 'Close': R['Adjprcwd'],
        'Vol': R['Dnshrtrd'] / fnd, 'Vol_raw': R['Dnshrtrd'], 'Amount': R['Dnvaltrd'], 'FloatMV': float_mv,
        'MarketCap': float_mv, 'TotalMV': R['Dsmvtl3'].astype(np.float64) * 1000.0,
        'TotalMV_A': R['Dsmvtll'].astype(np.float64) * 1000.0,
        'FloatShares': float_mv / R['Clsprc'], 'TotalShares': R['Dsmvtl2'].astype(np.float64) * 1000.0 / R['Clsprc'],
        'Turnover': R['Dnshrtrd'] / (float_mv / R['Clsprc']), 'is_ST': np.isin(R['Trdsta'], ST_SET).astype(np.int8),
        'Ret': ret, 'Ret_nd': ret_nd, 'PreClose': R['PreClosePrice'], 'LimitUp': R['LimitUp'],
        'LimitDown': R['LimitDown'], 'LimitStatus': R['LimitStatus'], 'Markettype': R['Markettype'], 'Trdsta': R['Trdsta'],
    }
    ma = np.full(n, np.nan)
    st_idx = np.flatnonzero(new)
    en_idx = np.r_[st_idx[1:], n]
    c_adj = R['Adjprcwd']
    for a, b in zip(st_idx, en_idx):
        if b - a >= 20:
            ma[a + 19:b] = np.convolve(c_adj[a:b], np.ones(20), 'valid') / 20
    exp['MA20'] = ma
    tol = {'Turnover': 1e-6, 'Amount': 1e-6, 'FloatMV': 1e-6, 'MarketCap': 1e-6, 'TotalMV': 1e-6,
           'TotalMV_A': 1e-6, 'FloatShares': 1e-6, 'TotalShares': 1e-6, 'LimitStatus': 0, 'Markettype': 0, 'Trdsta': 0, 'MA20': 1e-9}
    pcols = feather.read_table(ASHARE_PANEL_PATH, memory_map=True).column_names
    a0 = {}
    p_rows = feather.read_table(ASHARE_PANEL_PATH, columns=['Date'], memory_map=True).num_rows
    if p_rows != n:
        a0['__rows__'] = f'panel {p_rows} vs raw {n}'
    else:
        for c in pcols:
            if c not in exp:
                continue
            got = panel_col(ASHARE_PANEL_PATH, c)
            e = exp[c]
            if c == 'Date':
                got = day_int(got)
            if c in ('StockID', 'Date', 'is_ST'):
                bad = int((got != e).sum())
            else:
                bad = int((~rel_close(got, e, tol.get(c, 1e-9))).sum())
            if bad:
                a0[c] = bad
    a0_missing = [c for c in pcols if c not in exp and c != 'EWMA_vol']
    details['A0_mismatch_by_column'] = a0
    details['A0_columns_not_reconstructed'] = a0_missing + ['EWMA_vol（只作参考，不用于样本筛选）']
    results.append(result('A0', '由原始 zip 独立重算面板各列并逐行比对', '全部列一致（浮点相对差 ≤ 1e-9，float32 存储列 ≤ 1e-6）',
                          f'{n:,} 行；不一致列 {a0 if a0 else "无"}', not a0 and not a0_missing))
    del exp, ma

    # suspension_days.parquet
    sus = pd.read_parquet(Path(ASHARE_PROCESSED_DATA_DIR, 'suspension_days.parquet'))
    f1 = fills[fills['Filling'] == 1][['Stkcd', 'day']].drop_duplicates()
    f1['Stkcd'] = f1['Stkcd'].replace(CODE_MAP)
    k_exp = np.sort((f1['Stkcd'].to_numpy(np.int64) << 20) | f1['day'].to_numpy())
    k_got = np.sort((sus['StockID'].astype(int).to_numpy(np.int64) << 20) | day_int(sus['Date']))
    sus_ok = np.array_equal(k_exp, k_got)
    results.append(result('A0s', '停牌日文件与原始 Filling = 1 行（含代码变更）', '键集合完全一致',
                          f'原始 {len(k_exp):,}，文件 {len(k_got):,}，一致={sus_ok}', sus_ok))
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fills.to_parquet(Path(AUDIT_DIR, 'raw_fill_keys.parquet'), index=False)

    # ---- B1 Dretwd vs Adjprcwd ratio
    with np.errstate(invalid='ignore', divide='ignore'):
        r_adj = R['Adjprcwd'] / R['Adjprcwd'][prev] - 1
        d1 = np.abs(r_adj - R['Dretwd'])
    gap = ~new & ~consec
    b1c = float((d1[consec] < 1e-4).mean())
    b1g = float((d1[gap] < 1e-4).mean())
    worst = np.flatnonzero(consec & ~(d1 < 1e-4))
    details['B1_examples'] = [(int(sid[j]), day_str(day[j]), float(r_adj[j]), float(R['Dretwd'][j])) for j in worst[:20]]
    gb = np.flatnonzero(gap & ~(d1 < 1e-4))
    gdiff = R['Dretwd'][gb] - r_adj[gb]
    details['B1_gap_mismatch'] = {
        'n': len(gb), 'by_year': pd.Series(day[gb].astype('datetime64[D]')).dt.year.value_counts().sort_index().to_dict(),
        'abs_diff_quantiles': {str(q): float(np.quantile(np.abs(gdiff), q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(gb) else {},
        'n_abs_diff_gt_1pp': int((np.abs(gdiff) > 0.01).sum()),
        'n_abs_diff_gt_1pp_since_1999': int(((np.abs(gdiff) > 0.01) & (day[gb] >= day_int([np.datetime64(SAMPLE_START)])[0])).sum()),
        'largest': [(int(sid[j]), day_str(day[j]), float(R['Dretwd'][j]), float(r_adj[j])) for j in gb[np.argsort(-np.abs(gdiff))][:20]]}
    results.append(result('B1', '`Dretwd` 与 `Adjprcwd` 比值收益', '绝对差 < 1e-4 的比例 ≥ 99.9%（连续交易、跨停牌分别报告）',
                          f'连续交易 {b1c:.5%}（{consec.sum():,} 行）；跨停牌 {b1g:.5%}（{gap.sum():,} 行）',
                          b1c >= 0.999 and b1g >= 0.999))

    # ---- B2 / B3
    with np.errstate(invalid='ignore', divide='ignore'):
        non_ex = ~new & (np.abs(R['PreClosePrice'] - R['Clsprc'][prev]) <= 0.005)
        r_raw = R['Clsprc'] / R['Clsprc'][prev] - 1
        b2 = float((np.abs(r_raw - R['Dretnd'])[non_ex] < 1e-4).mean())
        jump = f / f[prev] - 1
        jump_nd = fnd / fnd[prev] - 1
        b3 = float((np.abs(jump[non_ex]) > 1e-4).mean())
        b3_nd = float((np.abs(jump_nd[non_ex]) > 1e-4).mean())
    j = np.flatnonzero(non_ex & (np.abs(jump) > 1e-4))
    yrs = pd.Series(day[j].astype('datetime64[D]')).dt.year.value_counts().sort_index()
    details['B3_nonex_factor_change_by_year'] = yrs.to_dict()
    results.append(result('B2', '非除权日 `Dretnd` 与 `Clsprc` 比值收益', '绝对差 < 1e-4 的比例 ≥ 99.9%',
                          f'{b2:.5%}（非除权日 {non_ex.sum():,} 行）', b2 >= 0.999))
    results.append(result('B3', '复权因子只在除权日变化', '非除权日因子变化 > 1e-4 的比例 < 0.1%',
                          f'含红利因子 {b3:.4%}；不含红利因子 {b3_nd:.4%}', b3 < 0.001,
                          '“非除权日”按 PreClosePrice 等于前一交易行收盘判定；早期 CSMAR 除权日的 PreClosePrice 未除权会计入此处'))

    # ---- B4 VWAP
    vol = R['Dnshrtrd']
    pos = vol > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        vwap = R['Dnvaltrd'].astype(np.float64) / vol
        inside = (vwap >= R['Loprc'] - 0.01) & (vwap <= R['Hiprc'] + 0.01)
    b4 = float(inside[pos].mean())
    out_idx = np.flatnonzero(pos & ~inside)
    yo = pd.Series(day[out_idx].astype('datetime64[D]')).dt.year.value_counts().sort_index()
    mo = pd.Series(R['Markettype'][out_idx].astype(int)).value_counts()
    details['B4_outside_by_year'] = yo.to_dict()
    details['B4_outside_by_markettype'] = mo.to_dict()
    b4_post = float(inside[pos & (day >= day_int([np.datetime64(SAMPLE_START)])[0])].mean())
    results.append(result('B4', '成交均价 `Dnvaltrd / Dnshrtrd` 落在 [最低 − 0.01, 最高 + 0.01]', '≥ 99.9%',
                          f'全样本 {b4:.4%}，1999 年起 {b4_post:.4%}；越界 {len(out_idx):,} 行', b4 >= 0.999))

    # ---- B5 bonus / split events
    ev_idx = np.flatnonzero(consec & (jump_nd >= 0.2))
    st_of = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    en_of = np.r_[np.flatnonzero(new)[1:] - 1, n - 1][np.cumsum(new) - 1]
    ok_ev = ev_idx[(ev_idx - 5 >= st_of[ev_idx]) & (ev_idx + 4 <= en_of[ev_idx])]
    vadj = R['Dnshrtrd'] / fnd
    win_b = ok_ev[:, None] + np.arange(-5, 0)[None, :]
    win_a = ok_ev[:, None] + np.arange(0, 5)[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        rat_adj = vadj[win_a].mean(1) / vadj[win_b].mean(1)
        rat_raw = vol[win_a].mean(1) / vol[win_b].mean(1)
        jumps = 1 + jump_nd[ok_ev]
    fin = np.isfinite(rat_adj) & np.isfinite(rat_raw)
    med_adj, med_raw = float(np.median(rat_adj[fin])), float(np.median(rat_raw[fin]))
    details['B5'] = {'events': int(fin.sum()), 'median_vol_ratio': med_adj, 'median_vol_raw_ratio': med_raw,
                     'median_share_multiplier': float(np.median(jumps[fin]))}
    results.append(result('B5', '送转事件（不含红利因子跳升 ≥ 20%）前后 5 日成交量均值之比',
                          '`Vol` 的中位数在 [0.85, 1.15]；`Vol_raw` 明显高于 1',
                          f'{fin.sum():,} 次事件；`Vol` 中位比 {med_adj:.3f}，`Vol_raw` 中位比 {med_raw:.3f}（股本倍数中位 {np.median(jumps[fin]):.2f}）',
                          0.85 <= med_adj <= 1.15 and med_raw > 1.2))

    # ---- B6 turnover
    with np.errstate(invalid='ignore', divide='ignore'):
        turn = R['Dnshrtrd'] / (float_mv / R['Clsprc'])
    med_t = float(np.nanmedian(turn))
    big = np.flatnonzero(turn > 1.0)
    details['B6_turnover_gt_100pct'] = [(int(sid[j]), day_str(day[j]), float(turn[j]), int(R['Markettype'][j]),
                                         bool(new[j]) or int(ci[j] - ci[prev[j]]) > 1) for j in big[:200]]
    first5 = (np.arange(n) - st_of) < 5
    big_first5 = int(first5[big].sum())
    results.append(result('B6', '换手率分布', '中位数在 0.5%–5%；超过 100% 的逐条列出',
                          f'中位数 {med_t:.3%}；> 100% 的 {len(big)} 行（其中上市前 5 个交易行 {big_first5} 行）',
                          0.005 <= med_t <= 0.05))

    # ---- B7 ST double check
    lo7, hi7 = day_int([np.datetime64('2007-01-01'), np.datetime64('2025-06-30')])
    m7 = np.isin(R['Markettype'], [1, 4]) & (day >= lo7) & (day <= hi7) & np.isfinite(R['LimitUp']) & (R['PreClosePrice'] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        pct = np.round(R['LimitUp'] / R['PreClosePrice'] - 1, 2)
    st = np.isin(R['Trdsta'], ST_SET)
    agree = (pct == 0.05) == st
    b7 = float(agree[m7].mean())
    dis = np.flatnonzero(m7 & ~agree)
    cat = pd.DataFrame({'is_ST': st[dis], 'pct': pct[dis], 'Trdsta': R['Trdsta'][dis].astype(int)})
    details['B7_disagreements'] = cat.value_counts().head(20).reset_index().astype(str).to_dict('records')
    results.append(result('B7', 'ST 双重验证（主板 2007 至 2025-06，涨停幅度 5% ⇔ `is_ST`）', '一致率 ≥ 99%',
                          f'{b7:.4%}（{m7.sum():,} 行，不一致 {len(dis):,}）', b7 >= 0.99))

    # ---- B8 OHLC consistency and zero volume
    o, h, l, c = R['Opnprc'], R['Hiprc'], R['Loprc'], R['Clsprc']
    incons = (l > np.minimum(o, c) + 1e-9) | (h < np.maximum(o, c) - 1e-9) | (o <= 0) | (h <= 0) | (l <= 0)
    zero = ~(vol > 0)
    since = day >= day_int([np.datetime64(SAMPLE_START)])[0]
    zy = pd.Series(day[zero].astype('datetime64[D]')).dt.year.value_counts().sort_index()
    details['B8_zero_volume_by_year'] = zy.to_dict()
    details['B8_zero_volume_since_1999'] = [(int(sid[j]), day_str(day[j]), float(c[j]), float(R['PreClosePrice'][j]))
                                            for j in np.flatnonzero(zero & since)]
    details['B8_ohlc_inconsistent'] = [(int(sid[j]), day_str(day[j])) for j in np.flatnonzero(incons)[:50]]
    results.append(result('B8', '开高低收不一致；`Filling = 0` 但成交量为 0', '各 < 0.01%，逐条列出',
                          f'不一致 {incons.sum()} 行；零成交 {zero.sum():,} 行（{zero.mean():.4%}），其中 1999 年起 '
                          f'{int((zero & since).sum())} 行（{(zero & since).sum() / since.sum():.5%}）',
                          incons.mean() < 1e-4 and zero.mean() < 1e-4))

    # ---- B9 largest absolute returns
    age = np.arange(n) - st_of
    abs_r = np.abs(ret)

    def attribute(j: int) -> str:
        if day[j] < day_int([np.datetime64('1994-01-01')])[0] and \
                abs(np.log(R['Clsprc'][j] / R['Clsprc'][prev[j]])) > np.log(50):
            return '1990–1993 原始价格单位错误（收盘价跳变 > 50 倍）'
        if not np.isfinite(R['LimitUp'][j]):
            if age[j] < 5:
                return '上市初期无涨跌幅'
            if not consec[j]:
                return '复牌首日无涨跌幅'
            return '当日无涨跌幅'
        if not consec[j]:
            return '跨停牌'
        lim = np.round(R['LimitUp'][j] / R['PreClosePrice'][j] - 1, 2) if R['PreClosePrice'][j] > 0 else np.nan
        if np.isfinite(lim) and abs(R['Dretwd'][j]) <= lim + 0.02 + abs(jump[j]):
            return '涨跌停内（含除权）'
        return '未归因'

    tabs = {}
    for scope, m in (('全历史', np.ones(n, bool)), ('1999 年起', since)):
        rr = np.nan_to_num(np.where(m, abs_r, -1), nan=-1)
        top = np.argpartition(rr, -50)[-50:]
        top = top[np.argsort(-rr[top])]
        rows = []
        for j in top:
            why = attribute(j)
            rows.append({'code': f'{sid[j]:06d}', 'date': day_str(day[j]), 'ret': float(ret[j]), 'why': why})
        tabs[scope] = rows
    details['B9_top50'] = tabs
    un_all = sum(r['why'] == '未归因' for r in tabs['全历史'])
    un_post = sum(r['why'] == '未归因' for r in tabs['1999 年起'])
    data_err = sum(r['why'].startswith('1990') for r in tabs['全历史'])
    results.append(result('B9', '|日收益| 最大的 50 条', '全部可归因（复牌、上市初期、无涨跌幅日等）',
                          f'全历史未归因 {un_all} 条（另有早期价格单位错误 {data_err} 条）；1999 年起未归因 {un_post} 条；'
                          f'1999 年起最大 |Ret| {tabs["1999 年起"][0]["ret"]:.2%}', un_all == 0 and un_post == 0))

    # ---- B10 limit_state vs LimitStatus
    p = pd.DataFrame({'StockID': sid, 'Date': day.astype('datetime64[D]'), 'Open': R['Opnprc'] * f,
                      'High': R['Hiprc'] * f, 'Low': R['Loprc'] * f, 'Close': R['Adjprcwd'], 'Close_raw': R['Clsprc'],
                      'adj_factor': f, 'PreClose': R['PreClosePrice'], 'LimitUp': R['LimitUp'], 'LimitDown': R['LimitDown']})
    ls = limit_state(p)
    del p
    stat = R['LimitStatus']
    has = np.isfinite(stat)
    pred = np.where(ls['close_up'].to_numpy(), 1, np.where(ls['close_dn'].to_numpy(), -1, 0))
    agree10 = pred == stat
    b10 = float(agree10[has].mean())
    dj = np.flatnonzero(has & ~agree10)
    dd = pd.DataFrame({'pred': pred[dj], 'csmar': stat[dj].astype(int), 'src': ls['limit_src'].to_numpy()[dj],
                       'year': pd.Series(day[dj].astype('datetime64[D]')).dt.year.to_numpy(),
                       'mt': R['Markettype'][dj].astype(int)})
    details['B10_disagreement_by_pred_csmar_src'] = dd.groupby(['pred', 'csmar', 'src']).size().reset_index().astype(str).to_dict('records')
    details['B10_disagreement_by_year'] = dd['year'].value_counts().sort_index().to_dict()
    details['B10_examples'] = [(int(sid[j]), day_str(day[j]), float(R['Clsprc'][j]), float(ls['up_price'].iat[j]),
                                float(ls['dn_price'].iat[j]), float(R['LimitUp'][j]), float(R['LimitDown'][j]),
                                int(stat[j]), int(pred[j])) for j in dj[:40]]
    details['limit_src_counts'] = ls['limit_src'].value_counts().to_dict()
    details['limit_state_shares'] = {k: float(ls[k].mean()) for k in ['close_up', 'close_dn', 'one_price_up', 'one_price_dn', 'no_limit']}
    b10_post = float(agree10[has & since].mean())
    results.append(result('B10', '新涨跌停规则（`limit_state` 收盘在限价）与 `LimitStatus`', '一致率 ≥ 99.9%，不一致的分类报告',
                          f'全样本 {b10:.4%}（{has.sum():,} 行，不一致 {len(dj):,}）；1999 年起 {b10_post:.4%}；'
                          f'重算限价的除权日 {int((ls["limit_src"] == 2).sum()):,} 行', b10 >= 0.999))
    save_phase('raw', results, details)


# --------------------------------------------------------------------------------------------- phase fund

def phase_fund() -> None:
    results, details = [], {}
    fund = pd.read_parquet(ASHARE_FUNDAMENTALS_PATH)
    mkt = pd.read_parquet(ASHARE_MARKET_PATH)
    cal = mkt['Date'].to_numpy()

    # E2 availability
    q = fund['Accper'].dt.month
    y = fund['Accper'].dt.year
    dl = pd.to_datetime(pd.DataFrame({'year': y + (q == 12), 'month': q.map({3: 4, 6: 8, 9: 10, 12: 4}),
                                      'day': q.map({3: 30, 6: 31, 9: 31, 12: 30})}))
    exp_avail = cal[np.searchsorted(cal, dl.to_numpy(), side='right')]
    e2a = float((fund['avail_date'] > fund['Accper']).mean())
    e2b = float((fund['avail_date'].to_numpy() == exp_avail).mean())
    results.append(result('E2', '财务时点表生效日', '生效日晚于会计期末 100%（另：等于法定截止日后首个交易日 100%）',
                          f'晚于会计期末 {e2a:.4%}；等于截止日后首个交易日 {e2b:.4%}；{len(fund):,} 条', e2a == 1 and e2b == 1))

    # E3 TTM recomputed from the raw income statement
    inc = read_zip_csv('利润表*.zip', 'FS_Comins.csv', usecols=['Stkcd', 'Accper', 'Typrep', 'B002000101', 'B002000000',
                                                              'B001100000', 'B001101000'],
                       dtype={'Stkcd': str, 'Accper': str, 'Typrep': str})
    inc = inc[(inc['Typrep'] == 'A') & ~inc['Accper'].str.endswith('01-01')].copy()
    inc['code'] = inc['Stkcd'].astype(int)
    inc['Accper'] = pd.to_datetime(inc['Accper'])
    inc['seg'] = 0
    for o, nn in CODE_MAP.items():
        have = set(inc.loc[inc['code'] == nn, 'Accper'])
        is_o = inc['code'] == o
        inc = inc[~(is_o & inc['Accper'].isin(have))]
        is_o = inc['code'] == o
        inc.loc[is_o, 'seg'] = 1
        inc.loc[is_o, 'code'] = nn
    mo = inc['Accper'].dt.month
    inc_dl = pd.to_datetime(pd.DataFrame({'year': inc['Accper'].dt.year + (mo == 12), 'month': mo.map({3: 4, 6: 8, 9: 10, 12: 4}),
                                          'day': mo.map({3: 30, 6: 31, 9: 31, 12: 30})}))
    for code, dt in CODE_REUSE.items():
        inc = inc[~((inc['code'] == code) & (inc_dl < pd.Timestamp(dt)))]
    inc['ni'] = inc['B002000101'].fillna(inc['B002000000'])
    inc['rev'] = inc['B001100000'].fillna(inc['B001101000'])
    lut = inc.set_index(['code', 'seg', 'Accper'])[['ni', 'rev']]
    fr = fund.assign(code=fund['StockID'].astype(int))
    seg_of = inc.set_index(['code', 'Accper'])['seg']
    fr['seg'] = seg_of.reindex(pd.MultiIndex.from_arrays([fr['code'], fr['Accper']])).to_numpy()
    exists = fr['seg'].notna().to_numpy()
    fr['seg'] = fr['seg'].fillna(0).astype(int)
    prev_ann = pd.to_datetime((fr['Accper'].dt.year - 1).astype(str) + '-12-31')
    prev_same = fr['Accper'] - pd.DateOffset(years=1)
    prev_same = prev_same + pd.offsets.MonthEnd(0)
    cur = lut.reindex(pd.MultiIndex.from_arrays([fr['code'], fr['seg'], fr['Accper']])).to_numpy()
    pa_ = lut.reindex(pd.MultiIndex.from_arrays([fr['code'], fr['seg'], prev_ann])).to_numpy()
    ps_ = lut.reindex(pd.MultiIndex.from_arrays([fr['code'], fr['seg'], prev_same])).to_numpy()
    is_q4 = (fr['Accper'].dt.month == 12).to_numpy()[:, None]
    ttm = np.where(is_q4, cur, cur + pa_ - ps_)
    ttm = np.where(np.isnan(ttm), pa_, ttm)
    ttm[~exists] = np.nan
    e3 = {}
    for k, col in enumerate(['NI_TTM', 'REV_TTM']):
        got = fr[col].to_numpy()
        e3[col] = float(rel_close(got, ttm[:, k], 1e-9).mean())
    results.append(result('E3', '由原始利润表独立重算 TTM', '100% 一致', f'{e3}', all(v == 1 for v in e3.values())))

    # F1 identity
    bs = read_zip_csv('资产负债表*.zip', 'FS_Combas.csv', usecols=['Stkcd', 'Accper', 'Typrep', 'A003000000', 'A003100000',
                                                                'A003200000'], dtype={'Stkcd': str, 'Accper': str, 'Typrep': str})
    bs = bs[(bs['Typrep'] == 'A') & ~bs['Accper'].str.endswith('01-01') & (bs['Accper'] >= '2007-01-01')]
    tot = bs['A003000000']
    with np.errstate(invalid='ignore', divide='ignore'):
        rd = (tot - bs['A003100000'] - bs['A003200000'].fillna(0)).abs() / tot.abs()
    ok = rd.notna()
    f1 = float((rd[ok] < 0.01).mean())
    details['F1_failures'] = bs[ok & (rd >= 0.01)].head(30).astype(str).to_dict('records')
    results.append(result('F1', '会计恒等式：所有者权益合计 = 归母 + 少数股东（2007 年起）', '相对差 < 1% ≥ 99%',
                          f'{f1:.4%}（{ok.sum():,} 条）', f1 >= 0.99))

    # F3 market return
    p = {c: panel_col(ASHARE_PANEL_PATH, c) for c in ['StockID', 'Date', 'Ret', 'FloatMV', 'Markettype']}
    n = len(p['Ret'])
    new = np.r_[True, p['StockID'][1:] != p['StockID'][:-1]]
    ci = np.searchsorted(cal, p['Date'])
    consec = np.r_[False, ci[1:] == ci[:-1] + 1] & ~new
    w = np.r_[np.nan, p['FloatMV'][:-1]]
    m = consec & np.isin(p['Markettype'], MAIN_MARKETS) & np.isfinite(p['Ret']) & (w > 0)
    num = np.bincount(ci[m], weights=w[m] * p['Ret'][m], minlength=len(cal))
    den = np.bincount(ci[m], weights=w[m], minlength=len(cal))
    with np.errstate(invalid='ignore', divide='ignore'):
        vw = num / den
    del p
    s = pd.DataFrame({'own': vw, 'csmar': mkt['mkt_vw'].to_numpy(), 'Date': cal}).dropna()
    s = s[np.isfinite(s['own'])]
    rho = float(s['own'].corr(s['csmar']))
    s99 = s[s['Date'] >= pd.Timestamp(SAMPLE_START)]
    rho99 = float(s99['own'].corr(s99['csmar']))
    mad = float((s99['own'] - s99['csmar']).abs().median())
    results.append(result('F3', '面板自算流通市值加权市场收益与 `Markettype = 53`', '相关系数 ≥ 0.99',
                          f'全样本 {rho:.5f}；1999 年起 {rho99:.5f}（日差绝对值中位 {mad:.2e}）', rho >= 0.99))

    # F4 risk-free rate at PBOC change dates
    rf = read_zip_csv('无风险利率文件*.zip', 'TRDNEW_Nrrate.csv', usecols=['Clsdt', 'Nrrdata'], parse_dates=['Clsdt'])
    rf = rf.set_index('Clsdt')['Nrrdata'].sort_index()
    rows = []
    for d, v in PBOC_1Y_DEPOSIT:
        t = pd.Timestamp(d)
        got = rf.get(t, np.nan)
        before = rf[rf.index < t].iloc[-1] if (rf.index < t).any() else np.nan
        rows.append({'date': d, 'pboc': v, 'csmar_on_date': got, 'csmar_day_before': before, 'match': abs(got - v) < 1e-6})
    details['F4'] = rows
    nm = sum(r['match'] for r in rows)
    results.append(result('F4', '无风险利率在央行一年期存款利率调整日的取值', '与央行公布值一致',
                          f'{nm}/{len(rows)} 个调整日一致', nm == len(rows)))

    # M1 market table vs raw files
    mk = read_zip_csv('综合日*.zip', 'TRDNEW_Cndalym.csv', usecols=['Markettype', 'Trddt', 'Cdretwdos', 'Cdretwdeq'],
                      parse_dates=['Trddt'])
    mm = mk[mk['Markettype'] == 53].set_index('Trddt')['Cdretwdos'].reindex(mkt['Date'])
    rf_prev = rf.reindex(mkt['Date'], method='ffill').shift(1).to_numpy()
    days = mkt['Date'].diff().dt.days.to_numpy()
    rf_exp = pd.Series((1 + rf_prev / 100) ** (days / 365) - 1)
    ix = read_zip_csv('指数文件*.zip', 'TRDNEW_Index.csv', usecols=['Indexcd', 'Trddt', 'Clsindex'], dtype={'Indexcd': str},
                      parse_dates=['Trddt'])
    i300 = ix[ix['Indexcd'] == '000300'].set_index('Trddt')['Clsindex'].reindex(mkt['Date'])
    m1 = {'mkt_vw': bool(rel_close(mkt['mkt_vw'].to_numpy(), mm.to_numpy(), 0).all()),
          'rf': bool(rel_close(mkt['rf'].to_numpy(), rf_exp.to_numpy(), 1e-12).all()),
          'idx_000300_close': bool(rel_close(mkt['idx_000300_close'].to_numpy(), i300.to_numpy(), 0).all())}
    sup = {}
    for code in ['000905', '000852', '399006', '000688']:
        a = mkt[f'idx_{code}_close']
        sup[code] = {'first': str(mkt.loc[a.notna(), 'Date'].min().date()), 'missing_after_first': int(a[a.notna().cummax()].isna().sum())}
    details['M1_supplement'] = sup
    ov = pd.DataFrame({'csi300': mkt['idx_000300_ret'], 'mkt': mkt['mkt_vw']}).dropna()
    details['M1_rf_annual_range'] = [float(mkt['rf_annual'].min()), float(mkt['rf_annual'].max())]
    results.append(result('M1', '市场表与原始文件逐值一致（市场收益、无风险利率、沪深 300）', '全部一致',
                          f'{m1}；补充指数首日后缺失 {{{", ".join(k + ": " + str(v["missing_after_first"]) for k, v in sup.items())}}}；'
                          f'沪深 300 与市场收益相关 {ov["csi300"].corr(ov["mkt"]):.3f}', all(m1.values())))
    save_phase('fund', results, details)


# --------------------------------------------------------------------------------------------- phase ext

def fetch_ak(code: str, start: str, end: str, adjust: str) -> tuple[pd.DataFrame, str]:
    import akshare as ak
    try:
        d = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust=adjust)
        d = d.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
        d['volume'] = d['volume'] * 100.0
        src = 'eastmoney'
    except Exception:
        prefix = 'bj' if code.startswith(('4', '8', '92')) else ('sh' if code.startswith(('6', '9')) else 'sz')
        d = ak.stock_zh_a_daily(symbol=prefix + code, start_date=start, end_date=end, adjust=adjust)
        src = 'sina'
    d['date'] = pd.to_datetime(d['date'])
    return d.set_index('date'), src


def phase_ext() -> None:
    results, details = [], {}
    rng = np.random.default_rng(SEED)
    p = pd.DataFrame({c: panel_col(ASHARE_PANEL_PATH, c) for c in ['StockID', 'Date', 'Markettype', 'Close_raw', 'adj_factor',
                                                                   'Open', 'High', 'Low', 'Vol_raw', 'Ret']})
    p = p[np.isin(p['Markettype'], MAIN_MARKETS) & (p['Date'] >= pd.Timestamp('2005-01-01'))].reset_index(drop=True)
    p['jump'] = p.groupby('StockID')['adj_factor'].pct_change()
    for k in ['Open', 'High', 'Low']:
        p[k + '_raw'] = p[k] / p['adj_factor']

    codes = rng.choice(p['StockID'].unique(), size=50, replace=False)
    rows, srcs, errs = [], {}, []
    for code in codes:
        sub = p[p['StockID'] == code]
        pick = sub.iloc[np.sort(rng.choice(len(sub), size=min(5, len(sub)), replace=False))]
        try:
            ak_d, src = fetch_ak(f'{code:06d}', pick['Date'].min().strftime('%Y%m%d'), pick['Date'].max().strftime('%Y%m%d'), '')
        except Exception as e:  # noqa: BLE001
            errs.append((f'{code:06d}', repr(e)[:200]))
            continue
        srcs[src] = srcs.get(src, 0) + 1
        for _, r in pick.iterrows():
            if r['Date'] not in ak_d.index:
                rows.append({'code': f'{code:06d}', 'date': str(r['Date'].date()), 'found': False})
                continue
            a = ak_d.loc[r['Date']]
            px = all(abs(float(a[k]) - v) < 0.006 for k, v in [('open', r['Open_raw']), ('high', r['High_raw']),
                                                               ('low', r['Low_raw']), ('close', r['Close_raw'])])
            vr = abs(float(a['volume']) / r['Vol_raw'] - 1) if r['Vol_raw'] > 0 else np.nan
            rows.append({'code': f'{code:06d}', 'date': str(r['Date'].date()), 'found': True, 'price_ok': px,
                         'vol_rel_diff': vr, 'src': src})
        time.sleep(0.3)
    c1 = pd.DataFrame(rows)
    details['C1_rows'] = c1.to_dict('records')
    details['C1_errors'] = errs
    if len(c1) and c1['found'].any():
        f_ = c1[c1['found']]
        px_rate = float(f_['price_ok'].mean())
        vol_rate = float((f_['vol_rel_diff'] < 0.01).mean())
        results.append(result('C1', '第三方（akshare）抽查 50 只 × 5 天不复权开高低收与成交量',
                              '价格到分一致 ≥ 99%，成交量相对差 < 1% ≥ 99%',
                              f'取得 {len(f_)} 个股票-日（来源 {srcs}；未取到 {int((~c1["found"]).sum())}，失败股票 {len(errs)}）；'
                              f'价格一致 {px_rate:.2%}；成交量一致 {vol_rate:.2%}', px_rate >= 0.99 and vol_rate >= 0.99))
    else:
        results.append(result('C1', '第三方（akshare）抽查', '—', f'未取得数据：{errs[:3]}', None))

    ev = p[(p['jump'].abs() > 1e-3) & (p['Date'] >= pd.Timestamp('2010-01-01'))]
    ev = ev.iloc[rng.choice(len(ev), size=min(20, len(ev)), replace=False)]
    rows, errs = [], []
    for _, r in ev.iterrows():
        code = f'{r["StockID"]:06d}'
        try:
            ak_d, src = fetch_ak(code, (r['Date'] - pd.Timedelta(days=40)).strftime('%Y%m%d'), r['Date'].strftime('%Y%m%d'), 'hfq')
        except Exception as e:  # noqa: BLE001
            errs.append((code, repr(e)[:200]))
            continue
        ak_d = ak_d.sort_index()
        if r['Date'] not in ak_d.index or ak_d.index.get_loc(r['Date']) == 0:
            rows.append({'code': code, 'date': str(r['Date'].date()), 'found': False})
            continue
        k = ak_d.index.get_loc(r['Date'])
        ret_ak = float(ak_d['close'].iloc[k] / ak_d['close'].iloc[k - 1] - 1)
        rows.append({'code': code, 'date': str(r['Date'].date()), 'found': True, 'ret_csmar': float(r['Ret']), 'ret_ak': ret_ak,
                     'abs_diff': abs(ret_ak - float(r['Ret'])), 'factor_jump': float(r['jump']), 'src': src})
        time.sleep(0.3)
    c2 = pd.DataFrame(rows)
    details['C2_rows'] = c2.to_dict('records')
    details['C2_errors'] = errs
    if len(c2) and c2['found'].any():
        f_ = c2[c2['found']]
        rate = float((f_['abs_diff'] < 0.001).mean())
        results.append(result('C2', '第三方抽查 20 次除权事件的复权收益', '差异 < 0.1 个百分点 ≥ 95%',
                              f'取得 {len(f_)} 次；{rate:.0%} 在 0.1pp 内（中位差 {f_["abs_diff"].median():.2e}）', rate >= 0.95))
    else:
        results.append(result('C2', '第三方抽查除权事件', '—', f'未取得数据：{errs[:3]}', None))
    save_phase('ext', results, details)


# --------------------------------------------------------------------------------------------- report

ORDER = ['A0', 'A0s', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
         'C1', 'C2', 'E1', 'E2', 'E3', 'F1', 'F3', 'F4', 'M1']


def write_report() -> None:
    res, det, times = {}, {}, {}
    for f in sorted(AUDIT_DIR.glob('results_*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        ph = f.stem.replace('results_', '')
        times[ph] = j['time']
        for r in j['results']:
            res[r['id']] = r
        det[ph] = j['details']
    mark = {True: '通过', False: '**未通过**', None: '仅报告'}
    lines = ['# CSMAR 面板审计结果', '',
             f'由 `ashare_train/audit_csmar_panel.py` 生成；标准见 `clean_spec.md` 第 9 节“数据源更换为 CSMAR”第 6 条。'
             f'各阶段运行时间：{times}。详细结果见 `WORK_SPACE/data/ashare_processed_data/audit/results_<phase>.json`。', '',
             '| 编号 | 检查 | 通过标准 | 结果 | 结论 |', '|---|---|---|---|---|']
    for k in ORDER:
        if k in res:
            r = res[k]
            lines.append(f'| {k} | {r["check"]} | {r["criterion"]} | {r["value"]} | {mark[r["passed"]]} |')
        else:
            lines.append(f'| {k} | — | — | 未运行 | — |')
    notes = [f'- {k}：{res[k]["note"]}' for k in ORDER if k in res and res[k]['note']]
    if notes:
        lines += ['', '说明：', ''] + notes
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logger.info(f'写出 {REPORT}')


PHASES = {'e1': phase_e1, 'raw': phase_raw, 'fund': phase_fund, 'ext': phase_ext}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phases', nargs='*', default=list(PHASES), choices=list(PHASES))
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args(argv)
    if not args.report_only:
        for ph in args.phases:
            t0 = time.time()
            logger.info(f'== 阶段 {ph}')
            PHASES[ph]()
            logger.info(f'== 阶段 {ph} 完成（{time.time() - t0:.0f}s）')
    write_report()
    return 0


if __name__ == '__main__':
    sys.exit(main())
