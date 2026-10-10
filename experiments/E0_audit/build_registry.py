# -*- coding: utf-8 -*-
"""E0-T1: scan every OOS prediction / score panel and build a reproducible registry."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER = PROJECT_ROOT / 'WORK_SPACE' / 'paper'
OUT = PAPER / 'E0_audit'
KEYS = ['StockID', 'ending_date']
INTERVENTION = re.compile(r'e33_perturb|e41_lag|e43_blocks|tab_cache')
EXCLUDE = re.compile(r'_invalid_')


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()[:16]


def read_any(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def header(path: Path) -> list[str]:
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_csv(path, nrows=0).columns)


def load_reference(path: Path) -> pd.DataFrame:
    ref = pd.read_csv(path, usecols=KEYS + ['ret_val', 'binary_label'])
    ref['StockID'] = ref['StockID'].astype(str).str.zfill(6)
    ref['ending_date'] = pd.to_datetime(ref['ending_date'])
    return ref.rename(columns={'ret_val': 'ref_ret', 'binary_label': 'ref_label'})


def describe(path: Path, ref: pd.DataFrame, full: bool) -> dict:
    row = {
        'path': path.relative_to(PROJECT_ROOT).as_posix(),
        'size_mb': round(path.stat().st_size / 2**20, 1),
        'mtime': pd.Timestamp(path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M'),
        'kind': 'intervention' if INTERVENTION.search(path.as_posix()) else 'main',
    }
    if path.stat().st_size == 0:
        row['kind'] = 'empty'
        return row
    if not full:
        return row
    cols = header(path)
    row.update({'sha1': sha1(path), 'columns': '|'.join(cols)})
    if not set(KEYS).issubset(cols):
        return row
    use = KEYS + [c for c in ('ret_val', 'binary_label') if c in cols]
    df = read_any(path, use)
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    row.update({
        'n_rows': len(df),
        'n_weeks': df['ending_date'].nunique(),
        'n_stocks': df['StockID'].nunique(),
        'date_min': df['ending_date'].min().date().isoformat(),
        'date_max': df['ending_date'].max().date().isoformat(),
        'dup_keys': int(df.duplicated(KEYS).sum()),
    })
    m = df.merge(ref, on=KEYS, how='outer', indicator=True)
    row['keys_only_here'] = int((m['_merge'] == 'left_only').sum())
    row['keys_only_ref'] = int((m['_merge'] == 'right_only').sum())
    both = m[m['_merge'] == 'both']
    if 'ret_val' in df.columns:
        diff = (both['ret_val'] - both['ref_ret']).abs()
        row['ret_val_max_abs_diff'] = float(np.nanmax(diff)) if len(diff) else np.nan
    if 'binary_label' in df.columns:
        row['label_mismatch'] = int((both['binary_label'] != both['ref_label']).sum())
    return row


def main() -> int:
    manual = yaml.safe_load((Path(__file__).with_name('registry_manual.yaml')).read_text(encoding='utf-8'))
    ref = load_reference(PROJECT_ROOT / manual['reference_labels'])
    files = sorted(
        p for pattern in ('**/pred_*oos*.csv', '**/*scores*.parquet', '**/common_scores.csv')
        for p in PAPER.glob(pattern) if not EXCLUDE.search(p.as_posix())
    )
    rows = []
    for i, p in enumerate(files, 1):
        full = not INTERVENTION.search(p.as_posix())
        print(f'[{i}/{len(files)}] {p.relative_to(PAPER)}', flush=True)
        rows.append(describe(p, ref, full))
    reg = pd.DataFrame(rows)

    model_of = {}
    for name, spec in manual['models'].items():
        for f in [spec.get('file')] + spec.get('duplicates', []) + spec.get('files', []):
            if f:
                model_of[f] = (name, spec['role'])
    reg['model'] = reg['path'].map(lambda p: model_of.get(p, ('', ''))[0])
    reg['role'] = reg['path'].map(lambda p: model_of.get(p, ('', ''))[1])
    reg['hash_group_size'] = reg.groupby('sha1')['path'].transform('count')

    OUT.mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT / 'registry.csv', index=False)
    missing = [f for f in model_of if not (PROJECT_ROOT / f).exists()]
    if missing:
        print('MISSING registered files:', *missing, sep='\n  ')
        return 1
    print(f'wrote {OUT / "registry.csv"} ({len(reg)} files)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
