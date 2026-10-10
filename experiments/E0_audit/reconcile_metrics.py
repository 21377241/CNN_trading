# -*- coding: utf-8 -*-
"""E0-T2: recompute every registered model on its native sample and on the common sample.

Reports Rank IC and decile H-L Sharpe under the frozen convention (arithmetic mean * sqrt(52) / std)
and, for reconciliation only, the superseded convention (compounded annual return / annual vol).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.E2C_temporal.eval_e2c import nw_t_mean
from experiments.common.metrics import cross_section_ic

ANN = 52
KEYS = ['StockID', 'ending_date']
OUT = PROJECT_ROOT / 'WORK_SPACE' / 'paper' / 'E0_audit'
ROLES = ('canonical', 'variant', 'legacy')
COMMON_SET = ['chartcnn_e1b_ric', 'onedcnn_120', 'mlp4_120_s5', 'mlp4_180_bag10', 'mlp2_180_bag10',
              'histgb_180', 'logit_180']


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=KEYS + ['ret_val', 'MarketCap', 'up_prob'])
    df['StockID'] = df['StockID'].astype(str).str.zfill(6)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    return df


def hl_series(df: pd.DataFrame, score: str, n_dec: int = 10) -> pd.DataFrame:
    rows = []
    for dt, g in df.dropna(subset=[score, 'ret_val']).groupby('ending_date'):
        if len(g) < n_dec:
            continue
        dec = pd.qcut(g[score].rank(method='first'), n_dec, labels=False)
        lo, hi = g[dec == 0], g[dec == n_dec - 1]
        w_hi, w_lo = np.maximum(hi['MarketCap'], 0), np.maximum(lo['MarketCap'], 0)
        rows.append({
            'ending_date': dt,
            'ew': hi['ret_val'].mean() - lo['ret_val'].mean(),
            'vw': np.average(hi['ret_val'], weights=w_hi) - np.average(lo['ret_val'], weights=w_lo),
        })
    return pd.DataFrame(rows)


def sharpe_pair(r: pd.Series) -> tuple[float, float, float]:
    m, s = r.mean(), r.std(ddof=1)
    ann = (1 + m) ** ANN - 1
    return ann, m / s * np.sqrt(ANN), ann / (s * np.sqrt(ANN))


def summarize(df: pd.DataFrame, score: str) -> dict:
    ic = cross_section_ic(df, score)
    hl = hl_series(df, score)
    ew_ann, ew_sr, ew_sr_old = sharpe_pair(hl['ew'])
    vw_ann, vw_sr, vw_sr_old = sharpe_pair(hl['vw'])
    return {
        'n_rows': len(df), 'n_weeks': df['ending_date'].nunique(),
        'rank_ic': ic['rank_ic'].mean(), 'rank_ic_nw_t': nw_t_mean(ic['rank_ic'].values),
        'ew_hl_ann': ew_ann, 'ew_sharpe': ew_sr, 'ew_sharpe_old': ew_sr_old, 'ew_nw_t': nw_t_mean(hl['ew'].values),
        'vw_hl_ann': vw_ann, 'vw_sharpe': vw_sr, 'vw_sharpe_old': vw_sr_old, 'vw_nw_t': nw_t_mean(hl['vw'].values),
    }


def main() -> int:
    manual = yaml.safe_load((Path(__file__).with_name('registry_manual.yaml')).read_text(encoding='utf-8'))
    models = {}
    for name, spec in manual['models'].items():
        if spec['role'] not in ROLES:
            continue
        if spec.get('file'):
            models[name] = (spec['role'], spec['file'])
        for f in spec.get('files', []):
            models[f'{name}_{Path(f).stem}'] = (spec['role'], f)

    preds = {name: load_pred(PROJECT_ROOT / f) for name, (_, f) in models.items()}
    native = []
    for name, df in preds.items():
        print('native', name, flush=True)
        native.append({'model': name, 'role': models[name][0], 'file': models[name][1], **summarize(df, 'up_prob')})
    native = pd.DataFrame(native)

    base = preds[COMMON_SET[0]][KEYS + ['ret_val', 'MarketCap']]
    common = base.copy()
    for name in COMMON_SET:
        common = common.merge(preds[name][KEYS + ['up_prob']].rename(columns={'up_prob': name}), on=KEYS, how='inner')
    common = common.dropna(subset=['ret_val'])
    rows = []
    for name in COMMON_SET:
        print('common', name, flush=True)
        rows.append({'model': name, **summarize(common, name)})
    common_tbl = pd.DataFrame(rows)

    label_check = []
    for name, df in preds.items():
        m = df[KEYS + ['ret_val']].merge(base[KEYS + ['ret_val']], on=KEYS, suffixes=('', '_ref'))
        label_check.append({'model': name, 'n_overlap': len(m),
                            'ret_val_max_abs_diff': float((m['ret_val'] - m['ret_val_ref']).abs().max())})

    OUT.mkdir(parents=True, exist_ok=True)
    native.to_csv(OUT / 'reconcile_native.csv', index=False)
    common_tbl.to_csv(OUT / 'reconcile_common.csv', index=False)
    pd.DataFrame(label_check).to_csv(OUT / 'reconcile_label_check.csv', index=False)
    print(f'common sample: {len(common):,} rows, {common["ending_date"].nunique()} weeks')
    cols = ['model', 'n_rows', 'rank_ic', 'rank_ic_nw_t', 'ew_sharpe', 'ew_sharpe_old', 'vw_sharpe', 'vw_sharpe_old']
    print(native[cols].to_string(index=False))
    print(common_tbl[cols].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
