# -*- coding: utf-8 -*-
"""
从 clean 输入到扩展窗口评估的端到端审计（只读，不改训练）。

检查项：
  A 特征/标签/图键一致与形状
  B 扩展窗切分与动态 embargo
  C 超参来自 registry；产出文件齐全
  D 预测键对齐、无训练年泄漏、ret 覆盖
  E 独立重算 Rank IC / 十分位 H-L Sharpe，对照 eval 目录
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ashare_train.run_expanding_window import (  # noqa: E402
    FEATURE_ROOT, CHART_SRC, OUT_ROOT, REGISTRY, OOS_YEARS, SEEDS,
    year_split, embargo_dates, load_panel_years, trainable_mask, selected_hp,
)
from experiments.common.metrics import cross_section_ic, summarize_ic, newey_west_t  # noqa: E402

EVAL = OUT_ROOT / 'eval'
REPORT: list[str] = []
FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg: str) -> None:
    REPORT.append(f'PASS  {msg}')
    print(f'PASS  {msg}', flush=True)


def fail(msg: str) -> None:
    FAILS.append(msg)
    REPORT.append(f'FAIL  {msg}')
    print(f'FAIL  {msg}', flush=True)


def warn(msg: str) -> None:
    WARNS.append(msg)
    REPORT.append(f'WARN  {msg}')
    print(f'WARN  {msg}', flush=True)


def section(title: str) -> None:
    REPORT.append(f'\n## {title}\n')
    print(f'\n===== {title} =====', flush=True)


# ── A: raw clean inputs ──────────────────────────────────────────────
def audit_A() -> dict:
    section('A 干净输入（特征 / meta / 图像标签）')
    meta_is = pd.read_feather(FEATURE_ROOT / 'meta_is.feather')
    meta_oos = pd.read_feather(FEATURE_ROOT / 'meta_oos.feather')
    X_is = np.load(FEATURE_ROOT / 'X_is.npy', mmap_mode='r')
    X_oos = np.load(FEATURE_ROOT / 'X_oos.npy', mmap_mode='r')

    if X_is.shape == (len(meta_is), 140):
        ok(f'X_is shape {X_is.shape} ↔ meta_is {len(meta_is)}')
    else:
        fail(f'X_is shape {X_is.shape} vs meta_is {len(meta_is)}')
    if X_oos.shape == (len(meta_oos), 140):
        ok(f'X_oos shape {X_oos.shape} ↔ meta_oos {len(meta_oos)}')
    else:
        fail(f'X_oos shape {X_oos.shape} vs meta_oos {len(meta_oos)}')

    for name, meta in [('is', meta_is), ('oos', meta_oos)]:
        need = {'StockID', 'ending_date', 'ret_val', 'MarketCap', 'label_raw', 'year'}
        # year may be derived
        miss = need - set(meta.columns) - ({'year'} if 'year' not in meta.columns else set())
        if 'year' not in meta.columns:
            meta = meta.copy()
            meta['year'] = pd.to_datetime(meta['ending_date']).dt.year
        if miss - {'year'}:
            fail(f'meta_{name} missing cols {miss}')
        else:
            ok(f'meta_{name} has required columns')
        if meta.duplicated(subset=['StockID', 'ending_date']).any():
            fail(f'meta_{name} duplicate keys')
        else:
            ok(f'meta_{name} keys unique ({len(meta):,})')

    # by_year vs concat X_is/X_oos
    meta_all = pd.concat([meta_is, meta_oos], ignore_index=True)
    meta_all['ending_date'] = pd.to_datetime(meta_all['ending_date'])
    if 'year' not in meta_all.columns:
        meta_all['year'] = meta_all['ending_date'].dt.year
    years = sorted(meta_all['year'].unique())
    by_dir = FEATURE_ROOT / 'by_year'
    max_abs = 0.0
    for y in years:
        path = by_dir / f'X_{y}.npy'
        if not path.is_file():
            fail(f'missing by_year/X_{y}.npy')
            continue
        Xy = np.load(path, mmap_mode='r')
        n = int((meta_all['year'] == y).sum())
        if Xy.shape[0] != n:
            fail(f'by_year {y}: rows {Xy.shape[0]} != meta {n}')
            continue
        if Xy.ndim != 3 or Xy.shape[1:] != (20, 7):
            fail(f'by_year {y}: expected (N,20,7) got {Xy.shape}')
            continue
        flat = np.asarray(Xy).reshape(n, -1)
        # compare to X_is/X_oos slice
        if y <= 2007:
            src = X_is
            m = meta_is
        else:
            src = X_oos
            m = meta_oos
        m2 = m.copy()
        if 'year' not in m2.columns:
            m2['year'] = pd.to_datetime(m2['ending_date']).dt.year
        idx = np.flatnonzero(m2['year'].to_numpy() == y)
        if len(idx) != n:
            fail(f'year {y}: is/oos year count {len(idx)} != {n}')
            continue
        ref = np.asarray(src[idx])
        diff = np.max(np.abs(flat.astype(np.float64) - ref.astype(np.float64)))
        max_abs = max(max_abs, float(diff))
        if diff > 1e-5:
            fail(f'by_year {y} flat vs X_is/oos max_abs={diff}')
    if max_abs <= 1e-5:
        ok(f'by_year flatten == X_is/X_oos for all years (max_abs={max_abs:.2e})')

    # chart labels vs meta keys per year (sample several years)
    chart_ok = True
    for y in [2000, 2005, 2007, 2008, 2012, 2020, 2025]:
        lab = CHART_SRC / f'20d_week_has_vb_[20]_ma_{y}_labels.feather'
        img = CHART_SRC / f'20d_week_has_vb_[20]_ma_{y}_images.dat'
        if not lab.is_file() or not img.is_file():
            fail(f'chart files missing for {y}')
            chart_ok = False
            continue
        L = pd.read_feather(lab)
        L['ending_date'] = pd.to_datetime(L['ending_date'])
        L['StockID'] = L['StockID'].astype(str)
        sub = meta_all.loc[meta_all['year'] == y, ['StockID', 'ending_date']].copy()
        sub['StockID'] = sub['StockID'].astype(str)
        if len(L) != len(sub):
            fail(f'chart labels {y}: {len(L)} vs meta {len(sub)}')
            chart_ok = False
            continue
        # same multiset of keys
        a = set(zip(L['StockID'], L['ending_date']))
        b = set(zip(sub['StockID'], sub['ending_date']))
        if a != b:
            fail(f'chart labels {y}: key set ≠ meta ({len(a-b)} only-lab, {len(b-a)} only-meta)')
            chart_ok = False
        # image file size: expect N * 20 * something - just check non-empty
        if img.stat().st_size < 1000:
            fail(f'chart images {y} too small')
            chart_ok = False
    if chart_ok:
        ok('chart labels key sets match meta for sampled years; image files present')

    # ret_val coverage
    ret_cov = float(meta_oos['ret_val'].notna().mean())
    if ret_cov < 0.95:
        warn(f'meta_oos ret_val coverage {ret_cov:.3f} < 0.95')
    else:
        ok(f'meta_oos ret_val coverage {ret_cov:.4f}')

    return {'meta_is': meta_is, 'meta_oos': meta_oos, 'meta_all': meta_all}


# ── B: split / embargo ───────────────────────────────────────────────
def audit_B(meta_all: pd.DataFrame) -> None:
    section('B 扩展窗切分与动态 embargo')
    # y=2008 == main split
    s = year_split(2008)
    if s['train_years'] == list(range(2000, 2006)) and s['val_years'] == [2006, 2007] and s['oos_years'] == [2008]:
        ok('y=2008 split == main (train 2000-2005, val 2006-2007, oos 2008)')
    else:
        fail(f'y=2008 split unexpected: {s}')

    s12 = year_split(2012)
    if s12['train_years'] == list(range(2000, 2010)) and s12['val_years'] == [2010, 2011]:
        ok('y=2012 split: train 2000-2009, val 2010-2011')
    else:
        fail(f'y=2012 split unexpected: {s12}')

    # embargo: last ending_date of boundary years
    for y in [2008, 2012, 2020, 2025]:
        split = year_split(y)
        years = split['train_years'] + split['val_years'] + split['oos_years']
        X, meta = load_panel_years(years)
        ban = embargo_dates(meta, [split['train_years'][-1], split['val_years'][-1]])
        # independently recompute
        ed = pd.to_datetime(meta['ending_date'])
        yrs = meta['year'].to_numpy()
        ban2 = set()
        for by in [split['train_years'][-1], split['val_years'][-1]]:
            ban2.add(pd.Timestamp(ed[yrs == by].max()))
        if ban != ban2:
            fail(f'y={y} embargo mismatch {ban} vs {ban2}')
        else:
            ok(f'y={y} embargo={sorted(str(x.date()) for x in ban)}')

        tr = np.flatnonzero(np.isin(yrs, split['train_years']) & trainable_mask(meta, ban))
        va = np.flatnonzero(np.isin(yrs, split['val_years']) & trainable_mask(meta, ban))
        oos = np.flatnonzero(np.isin(yrs, split['oos_years']))
        # no oos year in train/val
        if set(yrs[tr]) & set(split['oos_years']) or set(yrs[va]) & set(split['oos_years']):
            fail(f'y={y} leakage: oos year in train/val')
        # embargo weeks not in train/val
        if ed[tr].isin(ban).any() or ed[va].isin(ban).any():
            fail(f'y={y} embargo rows still in train/val')
        # label_raw only 0/1 in train/val
        raw_tr = meta['label_raw'].to_numpy()[tr]
        raw_va = meta['label_raw'].to_numpy()[va]
        if not (np.isin(raw_tr, [0, 1]).all() and np.isin(raw_va, [0, 1]).all()):
            fail(f'y={y} non-binary labels in train/val')
        # OOS predicts full year keys (may include label_raw < 0)
        if len(oos) != int((yrs == y).sum()):
            fail(f'y={y} oos count {len(oos)} != year rows {(yrs == y).sum()}')
        else:
            ok(f'y={y} train={len(tr):,} val={len(va):,} oos={len(oos):,} (no leakage)')

        # compare baked binary_label embargo in source chart for main years vs dynamic
        if y == 2008:
            # main baked embargo should be last week of 2005 and 2007
            for by, expect_in_ban in [(2005, True), (2007, True), (2006, False)]:
                last = pd.Timestamp(ed[yrs == by].max()) if (yrs == by).any() else None
                in_ban = last in ban if last is not None else False
                if expect_in_ban and not in_ban:
                    fail(f'main-split embago year {by} last week not in ban')
                if not expect_in_ban and in_ban:
                    fail(f'year {by} unexpectedly in 2008 ban')


# ── C: HP + artifacts ────────────────────────────────────────────────
def audit_C() -> None:
    section('C 超参与产出齐全性')
    reg = json.loads(REGISTRY.read_text(encoding='utf-8'))
    expect = {
        'chart': reg['predictions']['chartcnn_clean']['hp'],
        'oned': reg['predictions']['onedcnn_140']['hp'],
        'mlp': reg['predictions']['mlp_140']['hp'],
    }
    for m, hp in expect.items():
        got = selected_hp(m)
        if got != hp:
            fail(f'{m} selected_hp {got} != registry {hp}')
        else:
            ok(f'{m} HP = registry {hp}')

    for m in ['chart', 'oned', 'mlp']:
        missing = []
        for y in OOS_YEARS:
            if m == 'chart':
                hits = list((OUT_ROOT / 'chart' / f'y{y}').glob('**/ensem_res/ensem5_res_*.csv'))
                if not hits:
                    missing.append(y)
            else:
                tag = '1dcnn' if m == 'oned' else 'mlp'
                p = OUT_ROOT / m / f'y{y}' / 'preds' / f'pred_{tag}_oos{y}.parquet'
                if not p.is_file():
                    missing.append(y)
            # split.json
            sj = OUT_ROOT / m / f'y{y}' / 'split.json'
            if sj.is_file():
                sp = json.loads(sj.read_text(encoding='utf-8'))
                if sp != year_split(y):
                    fail(f'{m} y{y} split.json != year_split()')
        if missing:
            fail(f'{m} missing preds for years {missing}')
        else:
            ok(f'{m}: all 18 OOS prediction files present')

    # chart config hp in one year
    cfg = OUT_ROOT / 'chart' / 'y2012' / 'config.yaml'
    if cfg.is_file():
        import yaml
        c = yaml.safe_load(cfg.read_text(encoding='utf-8'))
        hp = selected_hp('chart')
        if (float(c['hp']['lr']) == float(hp['lr']) and float(c['hp']['dropout']) == float(hp['dropout'])
                and int(c['hp']['patience']) == int(hp['patience'])):
            ok(f'chart y2012 config.yaml HP matches registry')
        else:
            fail(f'chart y2012 config HP {c["hp"]} != {hp}')


# ── D: prediction panel integrity ────────────────────────────────────
def audit_D(meta_oos: pd.DataFrame) -> dict:
    section('D 预测面板：键、泄漏、与 meta 对齐')
    meta = meta_oos.copy()
    meta['ending_date'] = pd.to_datetime(meta['ending_date'])
    meta['StockID'] = meta['StockID'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    if 'year' not in meta.columns:
        meta['year'] = meta['ending_date'].dt.year
    u_keys = set(zip(meta['StockID'], meta['ending_date']))

    panels = {}
    for m in ['chart', 'oned', 'mlp']:
        p = EVAL / f'pred_{m}_ew_oos.parquet'
        if not p.is_file():
            fail(f'missing stitched {p}')
            continue
        df = pd.read_parquet(p)
        df['ending_date'] = pd.to_datetime(df['ending_date'])
        df['StockID'] = df['StockID'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
        panels[m] = df

        if df.duplicated(subset=['StockID', 'ending_date']).any():
            fail(f'{m} stitched duplicate keys')
        else:
            ok(f'{m} stitched keys unique ({len(df):,})')

        # years only 2008-2025
        yrs = pd.to_datetime(df['ending_date']).dt.year
        if sorted(yrs.unique()) != OOS_YEARS:
            fail(f'{m} years {sorted(yrs.unique())} != {OOS_YEARS}')
        else:
            ok(f'{m} covers exactly OOS years 2008-2025')

        # each oos_year column matches ending_date year
        if 'oos_year' in df.columns:
            bad = (df['oos_year'] != yrs).sum()
            if bad:
                fail(f'{m} oos_year ≠ ending_date.year on {bad} rows')
            else:
                ok(f'{m} oos_year == calendar year of ending_date')

        # keys ⊆ universe oos
        keys = set(zip(df['StockID'], df['ending_date']))
        only_pred = keys - u_keys
        only_u = {k for k in u_keys if k[1].year >= 2008 and k[1].year <= 2025} - keys
        # meta_oos may include 2026?
        u_oos_years = {k for k in u_keys if 2008 <= k[1].year <= 2025}
        only_u = u_oos_years - keys
        if only_pred:
            fail(f'{m} {len(only_pred)} keys not in meta_oos')
        elif only_u:
            fail(f'{m} missing {len(only_u)} meta_oos keys in 2008-2025')
        else:
            ok(f'{m} keys == meta_oos ∩ 2008-2025 ({len(keys):,})')

        # ret_val should match meta
        merged = df.merge(meta[['StockID', 'ending_date', 'ret_val', 'MarketCap']],
                          on=['StockID', 'ending_date'], suffixes=('', '_meta'), how='left')
        # compare where both finite
        a = merged['ret_val'].to_numpy()
        b = merged['ret_val_meta'].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            mad = float(np.max(np.abs(a[both] - b[both])))
            if mad > 1e-10:
                fail(f'{m} ret_val ≠ meta max_abs={mad}')
            else:
                ok(f'{m} ret_val matches meta_oos (max_abs={mad:.1e}, n={both.sum():,})')
        cov = float(np.isfinite(a).mean())
        if cov < 0.95:
            warn(f'{m} ret_val coverage {cov:.3f}')
        else:
            ok(f'{m} ret_val coverage {cov:.4f}')

        # up_prob in (0,1)
        up = df['up_prob'].to_numpy()
        if not (np.isfinite(up).all() and up.min() >= 0 and up.max() <= 1):
            fail(f'{m} up_prob out of [0,1] or nonfinite: min={np.nanmin(up)} max={np.nanmax(up)}')
        else:
            ok(f'{m} up_prob in [0,1] finite')

    # pairwise key equality
    if len(panels) == 3:
        k0 = set(zip(panels['chart']['StockID'], panels['chart']['ending_date']))
        for m in ['oned', 'mlp']:
            k1 = set(zip(panels[m]['StockID'], panels[m]['ending_date']))
            if k0 != k1:
                fail(f'chart vs {m} key mismatch')
            else:
                ok(f'chart and {m} share identical keys')

    # per-year file vs stitch consistency (sample)
    for m, y in [('oned', 2018), ('mlp', 2015), ('chart', 2020)]:
        st = panels[m]
        st_y = st.loc[pd.to_datetime(st['ending_date']).dt.year == y]
        if m == 'chart':
            hits = list((OUT_ROOT / 'chart' / f'y{y}').glob('**/ensem_res/ensem5_res_*.csv'))
            raw = pd.read_csv(hits[0])
        else:
            tag = '1dcnn' if m == 'oned' else 'mlp'
            raw = pd.read_parquet(OUT_ROOT / m / f'y{y}' / 'preds' / f'pred_{tag}_oos{y}.parquet')
        raw['ending_date'] = pd.to_datetime(raw['ending_date'])
        raw['StockID'] = raw['StockID'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
        mrg = st_y.merge(raw[['StockID', 'ending_date', 'up_prob']], on=['StockID', 'ending_date'],
                         suffixes=('_st', '_raw'))
        if len(mrg) != len(st_y) or len(mrg) != len(raw):
            fail(f'{m} y{y} stitch rows {len(st_y)} raw {len(raw)} merge {len(mrg)}')
        else:
            mad = float(np.max(np.abs(mrg['up_prob_st'] - mrg['up_prob_raw'])))
            if mad > 1e-12:
                fail(f'{m} y{y} stitch up_prob ≠ file max_abs={mad}')
            else:
                ok(f'{m} y{y} stitch == yearly file (max_abs={mad:.1e})')

    return panels


# ── E: recompute metrics ─────────────────────────────────────────────
def decile_hl_sharpe(df: pd.DataFrame, weight_col=None, ann=52) -> dict:
    df = df.dropna(subset=['ret_val', 'up_prob']).copy()
    rows = []
    for date, grp in df.groupby('ending_date'):
        if len(grp) < 10:
            continue
        g = grp.copy()
        try:
            g['decile'] = pd.qcut(g['up_prob'], 10, labels=range(1, 11), duplicates='drop')
        except ValueError:
            g['decile'] = pd.cut(g['up_prob'].rank(method='first'), bins=10, labels=range(1, 11))
        rets = {}
        for d in range(1, 11):
            sub = g[g['decile'] == d]
            if len(sub) == 0:
                continue
            if weight_col:
                w = np.where(sub[weight_col].values > 0, sub[weight_col].values, 0.0)
                rets[d] = np.average(sub['ret_val'].values, weights=w) if w.sum() > 0 else sub['ret_val'].mean()
            else:
                rets[d] = float(sub['ret_val'].mean())
        if 1 in rets and 10 in rets:
            rows.append({'date': date, 'hl': rets[10] - rets[1]})
    s = pd.DataFrame(rows).set_index('date')['hl']
    mean_m, std_m = float(s.mean()), float(s.std(ddof=1))
    sharpe = mean_m / std_m * np.sqrt(ann) if std_m > 1e-15 else np.nan
    ann_r = (1 + mean_m) ** ann - 1
    return {'sharpe': sharpe, 'ann_return': ann_r, 'ann_vol': std_m * np.sqrt(ann), 'n': len(s), 'series': s}


def audit_E(panels: dict) -> None:
    section('E 独立重算 Rank IC / H–L Sharpe，对照 eval 产物')
    # load prior summary if present
    prev_ic = pd.read_csv(EVAL / 'summary.csv') if (EVAL / 'summary.csv').is_file() else None
    prev_sh = pd.read_csv(EVAL / 'portfolio' / 'sharpe_summary.csv') if (EVAL / 'portfolio' / 'sharpe_summary.csv').is_file() else None

    rows_ic, rows_sh = [], []
    for m, df in panels.items():
        ic = cross_section_ic(df, 'up_prob')
        summ = summarize_ic(ic)
        t, p = newey_west_t(ic['rank_ic'].to_numpy(), lag=4)
        rows_ic.append({'model': m, 'rank_ic_mean': summ['rank_ic_mean'], 'nw_t': t, 'n_dates': summ['n_dates']})
        ew = decile_hl_sharpe(df, weight_col=None)
        vw = decile_hl_sharpe(df, weight_col='MarketCap')
        rows_sh.append({'model': m, 'ew_hl_sharpe': ew['sharpe'], 'ew_hl_ann': ew['ann_return'],
                        'vw_hl_sharpe': vw['sharpe'], 'vw_hl_ann': vw['ann_return'], 'n_weeks': ew['n']})
        ok(f'{m} recompute RankIC={summ["rank_ic_mean"]:.6f} EW-HL Sharpe={ew["sharpe"]:.4f}')

        if prev_ic is not None:
            ref = float(prev_ic.loc[prev_ic['model'] == m, 'rank_ic_mean'].iloc[0])
            if abs(ref - summ['rank_ic_mean']) > 1e-6:
                fail(f'{m} RankIC recompute {summ["rank_ic_mean"]:.6f} ≠ eval {ref:.6f}')
            else:
                ok(f'{m} RankIC matches eval/summary.csv')
        if prev_sh is not None:
            ref = float(prev_sh.loc[prev_sh['model'] == m, 'ew_hl_sharpe'].iloc[0])
            if abs(ref - ew['sharpe']) > 1e-4:
                fail(f'{m} EW-HL Sharpe recompute {ew["sharpe"]:.4f} ≠ eval {ref:.4f}')
            else:
                ok(f'{m} EW-HL Sharpe matches portfolio/sharpe_summary.csv')

        # weekly ic file check
        wpath = EVAL / f'weekly_rank_ic_{m}.csv'
        if wpath.is_file():
            w = pd.read_csv(wpath, parse_dates=['ending_date'])
            if len(w) != len(ic):
                fail(f'{m} weekly_rank_ic rows {len(w)} ≠ recompute {len(ic)}')
            else:
                mad = float(np.max(np.abs(w['rank_ic'].to_numpy() - ic['rank_ic'].to_numpy())))
                if mad > 1e-9:
                    fail(f'{m} weekly rank_ic series mismatch max_abs={mad}')
                else:
                    ok(f'{m} weekly_rank_ic.csv matches recompute')

    pd.DataFrame(rows_ic).to_csv(EVAL / 'audit_rank_ic.csv', index=False)
    pd.DataFrame(rows_sh).to_csv(EVAL / 'audit_sharpe.csv', index=False)

    # ensemble mean check for oned/mlp: up_prob ≈ mean(seeds)
    for m in ['oned', 'mlp']:
        y = 2012
        tag = '1dcnn' if m == 'oned' else 'mlp'
        raw = pd.read_parquet(OUT_ROOT / m / f'y{y}' / 'preds' / f'pred_{tag}_oos{y}.parquet')
        seed_cols = [c for c in raw.columns if c.startswith('up_prob_seed')]
        if len(seed_cols) != 5:
            fail(f'{m} y{y} expected 5 seed cols got {seed_cols}')
        else:
            mean_s = raw[seed_cols].mean(axis=1)
            mad = float(np.max(np.abs(mean_s - raw['up_prob'])))
            if mad > 1e-10:
                fail(f'{m} y{y} up_prob ≠ mean(seeds) max_abs={mad}')
            else:
                ok(f'{m} y{y} up_prob == mean of 5 seeds (max_abs={mad:.1e})')


def main() -> int:
    section('扩展窗口端到端审计')
    REPORT.append(f'project={PROJECT_ROOT}\nfeature_root={FEATURE_ROOT}\nchart_src={CHART_SRC}\nout={OUT_ROOT}\n')
    data = audit_A()
    audit_B(data['meta_all'])
    audit_C()
    panels = audit_D(data['meta_oos'])
    if panels:
        audit_E(panels)

    section('汇总')
    REPORT.append(f'\nPASS={sum(1 for x in REPORT if x.startswith("PASS"))}  '
                  f'WARN={len(WARNS)}  FAIL={len(FAILS)}\n')
    if FAILS:
        REPORT.append('### 失败项\n')
        for f in FAILS:
            REPORT.append(f'- {f}\n')
    if WARNS:
        REPORT.append('### 警告项\n')
        for w in WARNS:
            REPORT.append(f'- {w}\n')

    out = EVAL / 'audit_report.md'
    EVAL.mkdir(parents=True, exist_ok=True)
    out.write_text('# 扩展窗口审计报告\n\n' + '\n'.join(REPORT), encoding='utf-8')
    print(f'\n===== SUMMARY: FAIL={len(FAILS)} WARN={len(WARNS)} =====')
    print(f'wrote {out}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
