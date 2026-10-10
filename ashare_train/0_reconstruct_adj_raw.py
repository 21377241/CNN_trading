# -*- coding: utf-8 -*-
"""
0_reconstruct_adj_raw.py — 由不复权 CSV 重建后复权 CSV，补齐后复权目录缺失的股票。

后复权目录只有 5,000 个文件（按代码排序截断在 688103），不复权目录是完整的。
数据商的后复权以数据首日为基准（首日复权因子为 1）；不复权文件的 `前收盘价` 是交易所的除权参考价，
因此复权因子可以逐日精确递推：
    F_1 = 1,   F_t = F_{t-1} × 收盘价_{t-1} / 前收盘价_t
后复权 OHLC = 不复权 OHLC × F；其余字段两个目录逐位相同，原样保留。
`前收盘价` 写成 F_t × 前收盘价_t（即前一日后复权收盘价）；均线字段下游不使用，按 F 缩放。

先在两边都有的股票上验证重建结果与数据商后复权一致，再为缺失的股票写出重建文件。

用法:
  python ashare_train/0_reconstruct_adj_raw.py                # 验证 + 写出缺失股票
  python ashare_train/0_reconstruct_adj_raw.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import ASHARE_RAW_ADJ_DIR, ASHARE_RAW_UNADJ_DIR, WORK_DIR

OUT_DIR = os.path.join(WORK_DIR, 'data', 'raw_reconstructed', '后复权')
PRICE_COLS = ['开盘价', '最高价', '最低价', '收盘价']
MA_COLS = ['5日线', '10日线', '20日线', '30日线', '60日线', '120日线', '250日线']


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8', dtype={'代码': str})
    df['_date'] = pd.to_datetime(df['日期'], format='mixed')
    for c in PRICE_COLS + ['前收盘价'] + MA_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.sort_values('_date', kind='mergesort').reset_index(drop=True)


def unadj_files() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(os.listdir(ASHARE_RAW_UNADJ_DIR)):
        if f.endswith('.csv'):
            out.setdefault(f.split('(')[0].replace('.csv', ''), []).append(os.path.join(ASHARE_RAW_UNADJ_DIR, f))
    return out


def pick_unadj(paths: list[str]) -> tuple[pd.DataFrame, dict]:
    """重复文件：内容相同取其一；不同则取行数多的，并记录。"""
    dfs = [read_csv(p) for p in paths]
    info = {'n_files': len(paths)}
    if len(dfs) > 1:
        base = dfs[0].drop(columns='_date')
        info['identical'] = all(d.drop(columns='_date').equals(base) for d in dfs[1:])
        info['rows'] = [len(d) for d in dfs]
    return max(dfs, key=len), info


def factor(df: pd.DataFrame) -> np.ndarray:
    c = df['收盘价'].to_numpy(np.float64)
    pc = df['前收盘价'].to_numpy(np.float64)
    step = np.ones(len(df))
    ok = (pc[1:] > 0) & np.isfinite(pc[1:]) & (c[:-1] > 0)
    step[1:][ok] = c[:-1][ok] / pc[1:][ok]
    return np.cumprod(step)


def reconstruct(df: pd.DataFrame) -> pd.DataFrame:
    F = factor(df)
    out = df.copy()
    for col in PRICE_COLS + [c for c in MA_COLS if c in df.columns]:
        out[col] = df[col].to_numpy(np.float64) * F
    out['前收盘价'] = df['前收盘价'].to_numpy(np.float64) * F
    return out


def validate_one(code: str, paths: list[str]) -> dict:
    u, dup = pick_unadj(paths)
    a = read_csv(os.path.join(ASHARE_RAW_ADJ_DIR, f'{code}.csv'))
    r = reconstruct(u)
    m = a[['_date'] + PRICE_COLS].merge(r[['_date'] + PRICE_COLS], on='_date', suffixes=('_a', '_r'))
    rel = np.column_stack([np.abs(m[f'{c}_r'].to_numpy(np.float64) / m[f'{c}_a'].to_numpy(np.float64) - 1)
                           for c in PRICE_COLS])
    rel = np.where(np.isfinite(rel), rel, 0.0)
    ra = m['收盘价_a'].pct_change().to_numpy()[1:]
    rr = m['收盘价_r'].pct_change().to_numpy()[1:]
    dr = np.abs(rr - ra)
    dr = np.where(np.isfinite(dr), dr, 0.0)
    return {'code': code, 'rows_adj': len(a), 'rows_unadj': len(u), 'rows_matched': len(m),
            'max_ret_err': float(dr.max()) if len(dr) else 0.0,
            'n_ret_err_gt_1e-4': int((dr > 1e-4).sum()), 'n_ret_err_gt_1e-3': int((dr > 1e-3).sum()),
            'max_rel_err': float(rel.max()) if len(rel) else float('nan'),
            'share_rel_err_lt_1e-4': float((rel.max(axis=1) < 1e-4).mean()) if len(rel) else float('nan'),
            'share_rel_err_lt_1e-3': float((rel.max(axis=1) < 1e-3).mean()) if len(rel) else float('nan'),
            **{f'dup_{k}': v for k, v in dup.items() if k != 'n_files'}, 'n_unadj_files': dup['n_files']}


def write_one(code: str, paths: list[str]) -> dict:
    u, dup = pick_unadj(paths)
    r = reconstruct(u).drop(columns='_date')
    r.to_csv(os.path.join(OUT_DIR, f'{code}.csv'), index=False, encoding='utf-8')
    return {'code': code, 'rows': len(r), 'first_date': str(u['_date'].iloc[0].date()),
            'n_unadj_files': dup['n_files'], 'dup_identical': dup.get('identical')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    files = unadj_files()
    adj_codes = {f[:-4] for f in os.listdir(ASHARE_RAW_ADJ_DIR) if f.endswith('.csv')}
    both = sorted(set(files) & adj_codes)
    missing = sorted(set(files) - adj_codes)
    only_adj = sorted(adj_codes - set(files))
    print(f'不复权 {len(files)} 个代码，后复权 {len(adj_codes)} 个；两边都有 {len(both)}，后复权缺失 {len(missing)}，'
          f'只有后复权 {len(only_adj)}')

    with ProcessPoolExecutor(args.workers) as ex:
        val = pd.DataFrame(list(ex.map(validate_one, both, [files[c] for c in both], chunksize=20)))
    os.makedirs(OUT_DIR, exist_ok=True)
    val.to_csv(os.path.join(os.path.dirname(OUT_DIR), 'validation.csv'), index=False, encoding='utf-8')
    rows = val['rows_matched'].sum()
    summary = {
        'codes_unadj': len(files), 'codes_adj': len(adj_codes), 'codes_both': len(both),
        'codes_missing_adj': len(missing), 'codes_only_adj': len(only_adj),
        'validation_rows': int(rows),
        'rows_rel_err_lt_1e-4': float((val['share_rel_err_lt_1e-4'] * val['rows_matched']).sum() / rows),
        'rows_rel_err_lt_1e-3': float((val['share_rel_err_lt_1e-3'] * val['rows_matched']).sum() / rows),
        'codes_max_rel_err_gt_1e-3': int((val['max_rel_err'] > 1e-3).sum()),
        'daily_ret_err_gt_1e-4_share': float(val['n_ret_err_gt_1e-4'].sum() / rows),
        'daily_ret_err_gt_1e-3_share': float(val['n_ret_err_gt_1e-3'].sum() / rows),
        'daily_ret_err_max': float(val['max_ret_err'].max()),
        'daily_ret_err_max_p99_by_code': float(val['max_ret_err'].quantile(0.99)),
        'codes_row_mismatch': int((val['rows_adj'] != val['rows_matched']).sum()),
        'missing_by_prefix': pd.Series([c[:3] for c in missing]).value_counts().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(val.sort_values('max_rel_err', ascending=False).head(10).to_string(index=False))

    if not args.validate_only:
        with ProcessPoolExecutor(args.workers) as ex:
            wrote = pd.DataFrame(list(ex.map(write_one, missing, [files[c] for c in missing], chunksize=20)))
        wrote.to_csv(os.path.join(os.path.dirname(OUT_DIR), 'reconstructed.csv'), index=False, encoding='utf-8')
        summary['reconstructed'] = int(len(wrote))
        summary['reconstructed_first_date_min'] = str(wrote['first_date'].min())
        print(f'写出 {len(wrote)} 个重建文件到 {OUT_DIR}')
    with open(os.path.join(os.path.dirname(OUT_DIR), 'reconstruct_check.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
