# -*- coding: utf-8 -*-
"""clean 分支只使用 CSMAR 数据的守卫检查。运行：python tests/test_clean_independence.py（也兼容 pytest）。

  1. 代码：只有 ALLOW_OLD 中的脚本可以引用旧数据（新旧对账、渲染器缺陷评估 M0/M1）；
     旧原始数据路径、旧涨跌停区间规则、旧面板独有的列在任何 A 股代码中都不出现。
  2. WORK_SPACE：旧数据产物只能在 _olddata 下；日面板的列集合必须是 CSMAR 面板。
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'WORK_SPACE')
SCAN_DIRS = ('ashare_train', 'experiments', 'scripts', 'tests', os.path.join('trend_submit', 'trend_code_submit'))
SCAN_EXT = ('.py', '.yaml', '.yml', '.ps1', '.sh')
SELF = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, '/')

ALLOW_OLD = {
    'trend_submit/trend_code_submit/Misc/config.py',
    'ashare_train/reconcile_pre_clean.py',
    'ashare_train/2m1_rerender_legacy_windows.py',
    'experiments/E1_image_baseline/eval_clean_cnn_impact.py',
    'experiments/E1_image_baseline/config_cnn_m1_rendfix.yaml',
}
OLD_ACCESS = re.compile(r'PRE_CLEAN_WORK_DIR|OLD_DATA_WORK_DIR|_olddata|CNN_TRADING_PRE_CLEAN')
FORBIDDEN = re.compile(r'ASHARE_RAW|股票4\.14|BaiduNetdisk|limit_pct|hit_limit_band|LIMIT_TOL')
OLD_COLUMNS = re.compile(r'\b(PE_TTM|PB|pct_chg|is_limit_up|sec_name)\b')
LEGACY_TRAINING = re.compile(r'^\s*training:\s*legacy\b', re.M)

CSMAR_COLUMNS = {'Close_raw', 'adj_factor', 'PreClose', 'LimitUp', 'LimitDown', 'Markettype', 'Trdsta', 'FloatMV'}
OLD_PANEL_COLUMNS = {'PE_TTM', 'PB', 'pct_chg', 'is_limit_up', 'sec_name'}
OLD_PAPER_DIRS = re.compile(r'^(E1b_|E1c_cnn_M1_rendfix$|E2_matched_is|E2A_|E2B_|E2D_|E2C_temporal_1dcnn_120$|'
                            r'E2F_mlp_depth$)')
OLD_WORK_DIRS = re.compile(r'^(ashare_model_res|ashare_paper_E)')


def _code_files():
    for d in SCAN_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, d)):
            dirnames[:] = [x for x in dirnames if x not in ('__pycache__', 'CACHE_DIR')]
            for fn in filenames:
                if fn.endswith(SCAN_EXT):
                    path = os.path.join(dirpath, fn)
                    rel = os.path.relpath(path, ROOT).replace(os.sep, '/')
                    if rel != SELF:
                        with open(path, encoding='utf-8', errors='replace') as f:
                            yield rel, f.read()


def _hits(pattern: re.Pattern, allow: set[str] = frozenset()) -> list[str]:
    out = []
    for rel, text in _code_files():
        if rel in allow:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                out.append(f'{rel}:{i}: {line.strip()[:120]}')
    return out


def test_old_data_access_only_in_allowlist():
    hits = _hits(OLD_ACCESS, ALLOW_OLD)
    assert not hits, '旧数据只能由对账与渲染器缺陷评估脚本读取：\n' + '\n'.join(hits)


def test_no_old_raw_paths_or_limit_rules():
    hits = _hits(FORBIDDEN)
    assert not hits, '旧原始数据路径或旧涨跌停区间规则：\n' + '\n'.join(hits)


def test_no_old_panel_columns():
    hits = _hits(OLD_COLUMNS, {'ashare_train/reconcile_pre_clean.py'})
    assert not hits, '旧面板独有的列（CSMAR 面板没有）：\n' + '\n'.join(hits)


def test_legacy_training_only_for_m1():
    hits = [rel for rel, text in _code_files()
            if rel.endswith(('.yaml', '.yml')) and LEGACY_TRAINING.search(text) and rel not in ALLOW_OLD]
    assert not hits, f'training: legacy 只用于 M1：{hits}'


def test_workspace_has_no_old_artifacts():
    if not os.path.isdir(WORK):
        print('skip: WORK_SPACE 不存在')
        return
    bad = [n for n in os.listdir(WORK) if OLD_WORK_DIRS.match(n)]
    paper = os.path.join(WORK, 'paper')
    if os.path.isdir(paper):
        bad += [f'paper/{n}' for n in os.listdir(paper) if OLD_PAPER_DIRS.match(n)]
    data = os.path.join(WORK, 'data')
    if os.path.isdir(os.path.join(data, 'raw_reconstructed')):
        bad.append('data/raw_reconstructed')
    stocks = os.path.join(data, 'ashare_stocks_dataset')
    if os.path.isdir(stocks):
        bad += [f'data/ashare_stocks_dataset/{n}' for n in os.listdir(stocks) if not n.endswith('_clean')]
    assert not bad, f'旧数据产物应移到 WORK_SPACE/_olddata：{bad}'


def test_panel_is_csmar():
    path = os.path.join(WORK, 'data', 'ashare_processed_data', 'ashare_ret.feather')
    if not os.path.isfile(path):
        print('skip: 日面板不存在')
        return
    import pyarrow.feather as feather
    cols = set(feather.read_table(path, memory_map=True).schema.names)
    assert CSMAR_COLUMNS <= cols, f'日面板缺少 CSMAR 列：{sorted(CSMAR_COLUMNS - cols)}'
    assert not cols & OLD_PANEL_COLUMNS, f'日面板含旧数据商的列：{sorted(cols & OLD_PANEL_COLUMNS)}'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
