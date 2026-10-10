# -*- coding: utf-8 -*-
"""
E2B: CNN–tabular complementarity on fair chrono_ric panel.

Partners: histgb (default) or mlp. Same B1–B6 protocol.

Usage:
  python experiments/E2_matched_tabular/run_e2b_complement.py --step all
  python experiments/E2_matched_tabular/run_e2b_complement.py --partner mlp --step all
  python experiments/E2_matched_tabular/run_e2b_complement.py --partner mlp --step corr,overlap,fm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.common.metrics import cross_section_ic, summarize_ic
from experiments.E2_matched_tabular.e2a_layout import CHRONO_RIC, ws
from experiments.E2_matched_tabular.run_eval import parse_hl_sharpe
from Portfolio.portfolio import run_portfolio_analysis, write_port_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2b')

ANN = 52

PARTNERS = {
    'histgb': {
        'pred_file': 'pred_histgb_oos.csv',
        'label': 'histgb',
        'default_out': 'E2B_complement_chrono_ric',
    },
    'mlp': {
        'pred_file': 'pred_mlp_oos.csv',
        'label': 'mlp',
        'default_out': 'E2B_cnn_mlp_chrono_ric',
    },
}


def load_panel(pred_root: str | None = None, partner: str = 'histgb') -> pd.DataFrame:
    if partner not in PARTNERS:
        raise ValueError(f'partner must be one of {list(PARTNERS)}')
    root = pred_root or os.path.join(CHRONO_RIC, 'preds')
    cnn = pd.read_csv(os.path.join(root, 'pred_cnn_oos.csv'))
    tab = pd.read_csv(os.path.join(root, PARTNERS[partner]['pred_file']))
    keys = ['StockID', 'ending_date']
    cnn = cnn.rename(columns={'up_prob': 'S_cnn'})
    tab = tab.rename(columns={'up_prob': 'S_tab'})
    keep_t = keys + ['S_tab']
    for c in ['ret_val', 'MarketCap', 'binary_label']:
        if c not in cnn.columns and c in tab.columns:
            keep_t.append(c)
    m = cnn.merge(tab[keep_t], on=keys, how='inner')
    for c in ['ret_val', 'MarketCap', 'binary_label']:
        if f'{c}_x' in m.columns:
            m[c] = m[f'{c}_x'].fillna(m.get(f'{c}_y'))
            m.drop(columns=[x for x in m.columns if x.startswith(c + '_')], inplace=True)
    m['ending_date'] = pd.to_datetime(m['ending_date'])
    m['year'] = m['ending_date'].dt.year
    m['month'] = m['ending_date'].dt.to_period('M').astype(str)
    logger.info(f'panel partner={partner} rows={len(m):,} dates={m["ending_date"].nunique()}')
    return m


def _cs_zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return (s - mu) / sd


def newey_west_t(x: np.ndarray, lag: int = 4) -> tuple[float, float]:
    """t-stat for mean(x) with NW HAC."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return float('nan'), float('nan')
    mu = x.mean()
    gamma0 = np.mean((x - mu) ** 2)
    var = gamma0
    for L in range(1, min(lag, T - 1) + 1):
        w = 1.0 - L / (lag + 1)
        gamma = np.mean((x[L:] - mu) * (x[:-L] - mu))
        var += 2 * w * gamma
    se = np.sqrt(max(var, 0.0) / T)
    if se < 1e-15:
        return float('nan'), float('nan')
    t = mu / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=T - 1))
    return float(t), float(p)


def market_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    by_dt = panel.groupby('ending_date').agg(
        mkt_ret=('ret_val', 'mean'),
        cs_vol=('ret_val', 'std'),
    )
    med_ret = by_dt['mkt_ret'].median()
    med_vol = by_dt['cs_vol'].median()
    by_dt['bull'] = by_dt['mkt_ret'] >= med_ret
    by_dt['high_vol'] = by_dt['cs_vol'] >= med_vol
    return by_dt


def step_corr(panel: pd.DataFrame, out_dir: str, partner: str):
    rows = []
    for dt, g in panel.groupby('ending_date'):
        if len(g) < 20:
            continue
        rho = g['S_cnn'].corr(g['S_tab'])
        rows.append({'ending_date': dt, 'corr': rho, 'n': len(g)})
    daily = pd.DataFrame(rows)
    daily['ending_date'] = pd.to_datetime(daily['ending_date'])
    daily['year'] = daily['ending_date'].dt.year
    regimes = market_regimes(panel)
    daily = daily.merge(regimes[['bull', 'high_vol']], left_on='ending_date', right_index=True, how='left')

    def _summ(s):
        s = s.dropna()
        return {
            'mean': float(s.mean()),
            'median': float(s.median()),
            'std': float(s.std()),
            'p25': float(s.quantile(0.25)),
            'p75': float(s.quantile(0.75)),
            'n': int(len(s)),
        }

    summary = {'partner': partner, 'overall': _summ(daily['corr'])}
    summary['by_year'] = {int(y): _summ(g['corr']) for y, g in daily.groupby('year')}
    summary['bull'] = _summ(daily.loc[daily['bull'] == True, 'corr'])
    summary['bear'] = _summ(daily.loc[daily['bull'] == False, 'corr'])
    summary['high_vol'] = _summ(daily.loc[daily['high_vol'] == True, 'corr'])
    summary['low_vol'] = _summ(daily.loc[daily['high_vol'] == False, 'corr'])

    daily.to_csv(os.path.join(out_dir, 'b1_daily_corr.csv'), index=False, float_format='%.6f')
    with open(os.path.join(out_dir, 'b1_corr_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'B1 overall corr mean={summary["overall"]["mean"]:.4f}')
    return summary


def _top_mask(s: pd.Series, q: float, top: bool) -> pd.Series:
    thr = s.quantile(1 - q) if top else s.quantile(q)
    return s >= thr if top else s <= thr


def step_overlap(panel: pd.DataFrame, out_dir: str, partner: str):
    rows = []
    for dt, g in panel.groupby('ending_date'):
        if len(g) < 50:
            continue
        for q, label in [(0.1, '10'), (0.2, '20')]:
            for side, top in [('top', True), ('bottom', False)]:
                mc = _top_mask(g['S_cnn'], q, top)
                mt = _top_mask(g['S_tab'], q, top)
                inter = int((mc & mt).sum())
                union = int((mc | mt).sum())
                n_c, n_t = int(mc.sum()), int(mt.sum())
                overlap = inter / n_c if n_c else np.nan
                jaccard = inter / union if union else np.nan
                rows.append({
                    'ending_date': dt, 'band': f'{side}{label}',
                    'overlap_cnn_in_partner': overlap, 'jaccard': jaccard,
                    'n_inter': inter, 'n_cnn': n_c, 'n_partner': n_t,
                })
    daily = pd.DataFrame(rows)
    daily['ending_date'] = pd.to_datetime(daily['ending_date'])
    regimes = market_regimes(panel)
    daily = daily.merge(regimes[['bull']], left_on='ending_date', right_index=True, how='left')

    summ_rows = []
    for band, g in daily.groupby('band'):
        summ_rows.append({
            'band': band,
            'overlap_mean': float(g['overlap_cnn_in_partner'].mean()),
            'overlap_median': float(g['overlap_cnn_in_partner'].median()),
            'jaccard_mean': float(g['jaccard'].mean()),
            'jaccard_median': float(g['jaccard'].median()),
            'bull_overlap_mean': float(g.loc[g['bull'] == True, 'overlap_cnn_in_partner'].mean()),
            'bear_overlap_mean': float(g.loc[g['bull'] == False, 'overlap_cnn_in_partner'].mean()),
        })
    summary = pd.DataFrame(summ_rows)
    daily.to_csv(os.path.join(out_dir, 'b2_daily_overlap.csv'), index=False, float_format='%.6f')
    summary.to_csv(os.path.join(out_dir, 'b2_overlap_summary.csv'), index=False, float_format='%.6f')
    logger.info(f'B2\n{summary.to_string(index=False)}')
    return summary


def step_fm(panel: pd.DataFrame, out_dir: str, partner: str):
    betas = []
    for dt, g in panel.groupby('ending_date'):
        if len(g) < 30:
            continue
        r = g['ret_val'].values
        sc = _cs_zscore(g['S_cnn']).values
        st = _cs_zscore(g['S_tab']).values
        m = np.isfinite(r) & np.isfinite(sc) & np.isfinite(st)
        if m.sum() < 30:
            continue
        r, sc, st = r[m], sc[m], st[m]
        b1 = LinearRegression().fit(sc.reshape(-1, 1), r).coef_[0]
        b2 = LinearRegression().fit(st.reshape(-1, 1), r).coef_[0]
        b3 = LinearRegression().fit(np.column_stack([sc, st]), r).coef_
        betas.append({
            'ending_date': dt,
            'beta_cnn_m1': b1, f'beta_{partner}_m2': b2,
            'beta_cnn_m3': b3[0], f'beta_{partner}_m3': b3[1],
        })
    bdf = pd.DataFrame(betas)
    summary = {'partner': partner}
    for col in ['beta_cnn_m1', f'beta_{partner}_m2', 'beta_cnn_m3', f'beta_{partner}_m3']:
        t, p = newey_west_t(bdf[col].values, lag=4)
        summary[col] = {
            'mean': float(bdf[col].mean()),
            'nw_t': t,
            'nw_p': p,
            'n_dates': int(len(bdf)),
        }
    bdf.to_csv(os.path.join(out_dir, 'b3_fm_daily_betas.csv'), index=False, float_format='%.8f')
    with open(os.path.join(out_dir, 'b3_fm_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'B3 FM {json.dumps(summary, indent=2)}')
    return summary


def _quintile_labels(s: pd.Series, n: int = 5) -> pd.Series:
    return pd.qcut(s.rank(method='first'), n, labels=False) + 1


def step_doublesort(panel: pd.DataFrame, out_dir: str, partner: str):
    results = {}
    for control, sort_key, name in [
        ('S_tab', 'S_cnn', f'cnn_given_{partner}'),
        ('S_cnn', 'S_tab', f'{partner}_given_cnn'),
    ]:
        hl_by_date = []
        for dt, g in panel.groupby('ending_date'):
            if len(g) < 100:
                continue
            g = g.copy()
            g['q_ctrl'] = _quintile_labels(g[control], 5)
            inner_hls = []
            for q, gsub in g.groupby('q_ctrl'):
                if len(gsub) < 25:
                    continue
                gsub = gsub.copy()
                gsub['q_sort'] = _quintile_labels(gsub[sort_key], 5)
                top = gsub.loc[gsub['q_sort'] == 5, 'ret_val'].mean()
                bot = gsub.loc[gsub['q_sort'] == 1, 'ret_val'].mean()
                if np.isfinite(top) and np.isfinite(bot):
                    inner_hls.append(top - bot)
            if inner_hls:
                hl_by_date.append({'ending_date': dt, 'hl': float(np.mean(inner_hls))})
        hdf = pd.DataFrame(hl_by_date)
        rets = hdf['hl'].values
        ann = float(np.nanmean(rets) * ANN)
        vol = float(np.nanstd(rets, ddof=1) * np.sqrt(ANN))
        sharpe = ann / vol if vol > 1e-12 else np.nan
        results[name] = {
            'ew_hl_ann': ann,
            'ew_hl_vol': vol,
            'ew_hl_sharpe': float(sharpe),
            'n_dates': int(len(hdf)),
            'mean_weekly_hl': float(np.nanmean(rets)),
        }
        hdf.to_csv(os.path.join(out_dir, f'b4_{name}_daily_hl.csv'), index=False, float_format='%.8f')
    with open(os.path.join(out_dir, 'b4_doublesort_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f'B4 {json.dumps(results, indent=2)}')
    return results


def step_resid(panel: pd.DataFrame, out_dir: str, partner: str):
    pieces = []
    for dt, g in panel.groupby('ending_date'):
        if len(g) < 30:
            continue
        g = g.copy()
        sc = g['S_cnn'].values
        st = g['S_tab'].values
        m = np.isfinite(sc) & np.isfinite(st)
        g = g.loc[m].copy()
        sc, st = g['S_cnn'].values, g['S_tab'].values
        lr_c = LinearRegression().fit(st.reshape(-1, 1), sc)
        g['u_cnn_lin'] = sc - lr_c.predict(st.reshape(-1, 1))
        lr_t = LinearRegression().fit(sc.reshape(-1, 1), st)
        g['u_tab_lin'] = st - lr_t.predict(sc.reshape(-1, 1))
        poly = PolynomialFeatures(degree=2, include_bias=True)
        Xt = poly.fit_transform(st.reshape(-1, 1))
        Xc = poly.fit_transform(sc.reshape(-1, 1))
        g['u_cnn_poly'] = sc - LinearRegression().fit(Xt, sc).predict(Xt)
        g['u_tab_poly'] = st - LinearRegression().fit(Xc, st).predict(Xc)
        pieces.append(g)
    full = pd.concat(pieces, ignore_index=True)

    port_root = os.path.join(out_dir, 'b5_portfolio')
    os.makedirs(port_root, exist_ok=True)
    summary_rows = []
    for score_col, label in [
        ('u_cnn_lin', 'resid_cnn_lin'),
        ('u_tab_lin', f'resid_{partner}_lin'),
        ('u_cnn_poly', 'resid_cnn_poly'),
        ('u_tab_poly', f'resid_{partner}_poly'),
        ('S_cnn', 'raw_cnn'),
        ('S_tab', f'raw_{partner}'),
    ]:
        df = full[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
        df['up_prob'] = full[score_col].values
        df['up_prob'] = df.groupby('ending_date')['up_prob'].rank(pct=True)
        mdir = os.path.join(port_root, label)
        os.makedirs(mdir, exist_ok=True)
        tmp = write_port_input(df, mdir)
        run_portfolio_analysis(
            result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=ANN,
        )
        hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
        ic = summarize_ic(cross_section_ic(df, 'up_prob'))
        summary_rows.append({'score': label, **ic, **hl, 'n': len(df)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, 'b5_resid_summary.csv'), index=False, float_format='%.6f')
    logger.info(f'B5\n{summary.to_string(index=False)}')
    return summary


def step_ens(panel: pd.DataFrame, out_dir: str, partner: str):
    df = panel[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    ens = 0.5 * panel['S_cnn'].values + 0.5 * panel['S_tab'].values
    port_root = os.path.join(out_dir, 'b6_portfolio')
    os.makedirs(port_root, exist_ok=True)

    rows = []
    for name, scores in [
        ('cnn', panel['S_cnn'].values),
        (partner, panel['S_tab'].values),
        ('ens_050', ens),
    ]:
        d = df.copy()
        d['up_prob'] = scores
        mdir = os.path.join(port_root, name)
        os.makedirs(mdir, exist_ok=True)
        tmp = write_port_input(d, mdir)
        run_portfolio_analysis(
            result_files=[tmp], output_dir=mdir, n_ensembles=5, n_decile=10, ann_factor=ANN,
        )
        hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
        y = d['binary_label'].values
        s = d['up_prob'].values
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        ic = summarize_ic(cross_section_ic(d, 'up_prob'))
        rows.append({'model': name, 'auc': auc, **ic, **hl, 'n': len(d)})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'b6_ensemble_summary.csv'), index=False, float_format='%.6f')
    logger.info(f'B6\n{summary.to_string(index=False)}')
    return summary


STEPS = {
    'corr': step_corr,
    'overlap': step_overlap,
    'fm': step_fm,
    'doublesort': step_doublesort,
    'resid': step_resid,
    'ens': step_ens,
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default='all', help='all or comma list of corr,overlap,fm,doublesort,resid,ens')
    parser.add_argument('--partner', default='histgb', choices=list(PARTNERS), help='tabular partner vs CNN')
    parser.add_argument('--pred-root', default='', help='override preds dir')
    parser.add_argument('--out-dir', default='', help='override output dir under WORK_SPACE/paper/')
    args = parser.parse_args(argv)

    partner = args.partner
    out_name = args.out_dir or PARTNERS[partner]['default_out']
    out_dir = ws('WORK_SPACE', 'paper', out_name)
    os.makedirs(out_dir, exist_ok=True)
    panel = load_panel(args.pred_root or None, partner=partner)

    if args.step.strip().lower() == 'all':
        names = list(STEPS.keys())
    else:
        names = [s.strip() for s in args.step.split(',') if s.strip()]

    for name in names:
        if name not in STEPS:
            raise ValueError(f'unknown step {name}')
        logger.info(f'=== step {name} (partner={partner}) ===')
        STEPS[name](panel, out_dir, partner)

    with open(os.path.join(out_dir, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'pred_root': args.pred_root or os.path.join(CHRONO_RIC, 'preds'),
            'partner': partner,
            'steps': names,
            'n_rows': int(len(panel)),
            'n_dates': int(panel['ending_date'].nunique()),
            'out_dir': out_dir,
        }, f, indent=2)
    logger.info(f'[OK] E2B done partner={partner} -> {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
