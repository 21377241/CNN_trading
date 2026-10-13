# -*- coding: utf-8 -*-
"""
data_manifest.py - 迁移训练数据时的文件清单与校验（docs/migration_gpu_server.md）

  python ashare_train/data_manifest.py --write   # 本机：写 WORK_SPACE/migration_manifest.json
  python ashare_train/data_manifest.py --check   # 新机器：逐个文件核对大小与 sha1
  --set s0：S0 的登记预测、搜参记录与全部权重，清单为 WORK_SPACE/migration_manifest_s0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.environ.get('CNN_TRADING_WORK_DIR', os.path.join(PROJECT_ROOT, 'WORK_SPACE'))
SETS = {
    'train': ('migration_manifest.json', [
        'data/ashare_stocks_dataset/I20R5_clean',
        'paper/clean_inputs/features140',
        'data/ashare_processed_data/universe_weekly.feather',
        'paper/registry_v2.json',
        *[f'paper/hp_numeric_140/{m}/{f}' for m in ('mlp', 'oned') for f in ('selection.json', 'phase1.csv', 'phase2.csv')],
        *[f'paper/E1c_cnn_hp_I20R5_clean/{f}' for f in ('search_phase1.csv', 'search_phase2.csv',
                                                        'phase3_full_train.csv', 'config_cnn_hp.yaml')],
    ]),
    's0': ('migration_manifest_s0.json', [
        'paper/registry_v2',
        'paper/E1c_cnn_hp_I20R5_clean',
        'paper/hp_numeric_140',
    ]),
}


def files(item: str) -> list[str]:
    path = os.path.join(WORK_DIR, item)
    if os.path.isfile(path):
        return [item]
    out = []
    for d, _, fs in os.walk(path):
        out += [os.path.relpath(os.path.join(d, f), WORK_DIR).replace(os.sep, '/') for f in fs]
    return sorted(out)


def sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', action='store_true')
    g.add_argument('--check', action='store_true')
    ap.add_argument('--set', choices=sorted(SETS), default='train')
    args = ap.parse_args()
    name, items = SETS[args.set]
    manifest = os.path.join(WORK_DIR, name)
    if args.write:
        entries = {}
        for item in items:
            fs = files(item)
            if not fs:
                raise RuntimeError(f'{item} 不存在或为空')
            for f in fs:
                p = os.path.join(WORK_DIR, f)
                entries[f] = {'bytes': os.path.getsize(p), 'sha1': sha1(p)}
                print(f, flush=True)
        with open(manifest, 'w', encoding='utf-8') as fh:
            json.dump({'items': items, 'files': entries}, fh, indent=1)
        print(f'wrote {manifest}: {len(entries)} files, {sum(e["bytes"] for e in entries.values()) / 1e9:.2f} GB')
        return 0
    with open(manifest, encoding='utf-8') as fh:
        man = json.load(fh)
    bad = []
    for f, e in man['files'].items():
        p = os.path.join(WORK_DIR, f)
        if not os.path.isfile(p):
            bad.append(f'缺失 {f}')
        elif os.path.getsize(p) != e['bytes'] or sha1(p) != e['sha1']:
            bad.append(f'不一致 {f}')
        else:
            print(f'ok {f}', flush=True)
    for b in bad:
        print(b)
    print(f'{len(man["files"]) - len(bad)}/{len(man["files"])} 个文件一致')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
