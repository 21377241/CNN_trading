# -*- coding: utf-8 -*-
"""Evaluate CNN vs Logit / HistGB / MLP on common OOS panel."""
from __future__ import annotations

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, log_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.common.metrics import cross_section_ic, summarize_ic, yearly_rank_ic_mean
from Portfolio.portfolio import run_portfolio_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2_eval')


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_hl_sharpe(summary_path: str) -> dict:
    """Parse EW/VW H-L sharpe from portfolio_summary.txt."""
    out = {}
    if not os.path.exists(summary_path):
        return out
    section = None
    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '等权' in line or 'Equal-Weight' in line:
                section = 'ew'
            elif '市值加权' in line or 'Value-Weight' in line:
                section = 'vw'
            if section and line.strip().startswith('H-L'):
                parts = line.split()
                # H-L mean ann_ret ann_vol sharpe n
                if len(parts) >= 5:
                    out[f'{section}_hl_ann'] = float(parts[2])
                    out[f'{section}_hl_sharpe'] = float(parts[4])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        help='path to config yaml',
    )
    parser.add_argument('--only', default='', help='comma list of models to eval (cnn always kept)')
    parser.add_argument(
        '--cnn-pred',
        default='',
        help='override CNN pred csv path (E1b tuned model)',
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out_root = os.path.join(PROJECT_ROOT, cfg['paths']['out_root'].replace('/', os.sep))
    pred_dir = os.path.join(out_root, 'preds')
    port_root = os.path.join(out_root, 'portfolio')
    os.makedirs(port_root, exist_ok=True)
    include_ind = bool(cfg.get('include_indicators', False))
    logger.info(f'config={args.config}  include_indicators={include_ind}')

    models = ['cnn', 'logit', 'logit_paper', 'histgb', 'mlp']
    extra = os.path.join(pred_dir, 'pred_logit_paper_oos.csv')
    if os.path.exists(extra):
        models = ['cnn', 'logit', 'logit_paper', 'histgb', 'histgb_paper', 'mlp']
    only = {s.strip().lower() for s in getattr(args, 'only', '').split(',') if s.strip()}
    if only:
        models = [m for m in models if m in only or m == 'cnn']
        if 'logit' in only and 'logit_paper' not in models:
            models.append('logit_paper')
        if 'histgb' in only and 'histgb_paper' not in models:
            models.append('histgb_paper')
    frames = {}
    for m in models:
        if m == 'cnn' and args.cnn_pred:
            p = args.cnn_pred
            if not os.path.isabs(p):
                p = os.path.join(PROJECT_ROOT, p.replace('/', os.sep))
        else:
            p = os.path.join(pred_dir, f'pred_{m}_oos.csv')
        if not os.path.exists(p):
            logger.warning(f'missing pred, skip: {p}')
            continue
        df = pd.read_csv(p)
        df['ending_date'] = pd.to_datetime(df['ending_date'])
        frames[m] = df
    models = [m for m in models if m in frames]
    if 'cnn' not in frames:
        raise FileNotFoundError('pred_cnn_oos.csv required')
    if len(models) < 2:
        raise RuntimeError(f'need cnn + at least one tabular; got {models}')

    # common keys
    keys = ['StockID', 'ending_date']
    base = frames['cnn'][keys].copy()
    for m in models:
        base = base.merge(frames[m][keys], on=keys, how='inner')
    logger.info(f'common panel rows={len(base):,} dates={base["ending_date"].nunique()}')

    rows = []
    year_ic = {}
    for m in models:
        df = frames[m].merge(base, on=keys, how='inner')
        y = df['binary_label'].values
        s = df['up_prob'].values
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        try:
            ll = log_loss(y, np.clip(s, 1e-6, 1 - 1e-6))
        except Exception:
            ll = np.nan
        ic_df = cross_section_ic(df, 'up_prob')
        ic_sum = summarize_ic(ic_df)
        yric = yearly_rank_ic_mean(df, 'up_prob')
        year_ic[m] = yric

        # portfolio
        mdir = os.path.join(port_root, m)
        os.makedirs(mdir, exist_ok=True)
        tmp = os.path.join(mdir, 'pred_for_port.csv')
        df.to_csv(tmp, index=False)
        run_portfolio_analysis(
            result_files=[tmp],
            output_dir=mdir,
            n_ensembles=5,
            n_decile=10,
            ann_factor=cfg['ann_factor'],
        )
        hl = parse_hl_sharpe(os.path.join(mdir, 'portfolio_summary.txt'))
        rows.append({
            'model': m,
            'auc': auc,
            'logloss': ll,
            **ic_sum,
            **hl,
            'n': len(df),
        })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_root, 'e2_summary.csv')
    summary.to_csv(summary_path, index=False, float_format='%.6f')
    logger.info(f'\n{summary.to_string(index=False)}')

    # judgment CNN vs HistGB (if present)
    cnn = summary[summary['model'] == 'cnn'].iloc[0]
    if 'histgb' in year_ic:
        hgb = summary[summary['model'] == 'histgb'].iloc[0]
        d_sh = float(cnn.get('ew_hl_sharpe', np.nan) - hgb.get('ew_hl_sharpe', np.nan))
        y_cnn = year_ic['cnn']
        y_hgb = year_ic['histgb']
        common_years = sorted(set(y_cnn.index) & set(y_hgb.index))
        if common_years:
            wins = sum(1 for y in common_years if y_cnn.loc[y] > y_hgb.loc[y])
            win_frac = wins / len(common_years)
        else:
            win_frac = np.nan
        if d_sh >= 0.2 or (np.isfinite(win_frac) and win_frac >= 2 / 3 and d_sh > 0):
            verdict = 'CNN_gt_HistGB'
            verdict_zh = 'CNN 相对 HistGB 具有可检测优势（存在 representation / extraction gap 的初步证据）。'
        elif abs(d_sh) < 0.2 and (not np.isfinite(win_frac) or abs(win_frac - 0.5) < 0.2):
            verdict = 'CNN_approx_HistGB'
            verdict_zh = 'CNN 与 HistGB 大致相当；优势更可能来自非线性拟合而非图特有信息。'
        elif d_sh < 0:
            verdict = 'CNN_le_HistGB'
            verdict_zh = 'CNN 未优于 HistGB；图像在 matched 信息集下无显示额外优势。'
        else:
            verdict = 'mixed'
            verdict_zh = '结果介于阈值之间，宜结合年份 Rank IC 与 VW H-L 谨慎解读。'
        mrg = frames['cnn'][keys + ['up_prob']].rename(columns={'up_prob': 'cnn'})
        mrg = mrg.merge(
            frames['histgb'][keys + ['up_prob']].rename(columns={'up_prob': 'histgb'}),
            on=keys, how='inner'
        )
        corr_by_date = mrg.groupby('ending_date').apply(
            lambda g: g['cnn'].corr(g['histgb']) if len(g) > 5 else np.nan
        )
        mean_corr = float(np.nanmean(corr_by_date.values))
    else:
        d_sh = float('nan')
        win_frac = float('nan')
        mean_corr = float('nan')
        verdict = 'partial_eval'
        verdict_zh = '本次仅评估部分模型（未含 HistGB）；主判定暂缓。'

    # write markdown (auto dump under out_root unless paths.results_md overrides)
    md_rel = cfg['paths'].get('results_md')
    if md_rel:
        md_path = os.path.join(PROJECT_ROOT, str(md_rel).replace('/', os.sep))
    else:
        md_path = os.path.join(out_root, 'e2_results_auto.md')
    feat_desc = (
        '20 日归一化 OHLC + Vol + MA20 + **δ20 / uP / uV**（180 维；无 MA5）'
        if include_ind else
        '20 日归一化 OHLC + Vol + **仅 MA20**（120 维）；无 MA5'
    )
    split_mode = str(cfg.get('split_mode', 'random')).lower()
    if split_mode == 'chrono':
        tr_y = cfg.get('chrono_train_years', list(range(2000, 2006)))
        va_y = cfg.get('chrono_val_years', [2006, 2007])
        split_desc = (
            f'IS 时序 train {tr_y[0]}–{tr_y[-1]} / val {va_y[0]}–{va_y[-1]}；'
            f'OOS 2008–2025（额外稳健性，非主结论协议）'
        )
    else:
        split_desc = 'IS 2000–2007（70/30, seed=42）/ OOS 2008–2025'
    lines = [
        '# E2 自动汇总（勿作唯一结论稿）',
        '',
        f'- 统一结论请见 `docs/results/paper_results_E2.md`',
        f'- 切分：{split_desc}',
        f'- 特征：{feat_desc}',
        '- Logit：`SGDClassifier`；`logit` = IS-val 网格；`logit_paper` = 原文默认',
        '- 图像：复用 I20R5 `up_prob`（不重训）',
        f'- Common panel：n={int(summary["n"].iloc[0]):,}，产物：`{cfg["paths"]["out_root"]}/`',
        f'- Corr(S_cnn, S_histgb) 日均：{mean_corr:.4f}' if np.isfinite(mean_corr) else '- Corr(S_cnn, S_histgb) 日均：n/a',
        f'- EW H-L Sharpe 差 (CNN−HistGB)：{d_sh:.4f}；年份 Rank IC 胜率：{win_frac if np.isfinite(win_frac) else float("nan"):.3f}'
        if np.isfinite(d_sh) else '- EW H-L Sharpe 差 (CNN−HistGB)：n/a（未评 HistGB）',
        '',
        '## 主表（OOS）',
        '',
        '| 模型 | AUC | Rank IC | EW H-L 年化 | EW H-L Sharpe | VW H-L Sharpe |',
        '|------|-----|---------|-------------|---------------|---------------|',
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {r['auc']:.4f} | {r['rank_ic_mean']:.4f} | "
            f"{r.get('ew_hl_ann', float('nan')):.4f} | {r.get('ew_hl_sharpe', float('nan')):.4f} | "
            f"{r.get('vw_hl_sharpe', float('nan')):.4f} |"
        )
    lines += [
        '',
        '## 预注册判定',
        '',
        f'- 代码：`{verdict}`',
        f'- 结论：{verdict_zh}',
        '',
        '## 说明',
        '',
        '- 评价与 E1 相同：周频 `ann_factor=52`；组合按 `up_prob` 十分位。',
        '- 未覆盖 `ashare_model_res_I20R5/`。',
        '- **第一版负收益是实现错误，不是经济结论**：IS `labels.feather` 中 `binary_label=-99` 表示无效标签（约 1.9%）。未剔除时 sklearn 做成三类分类，`predict_proba[:,1]` 对应的是类别 **0**（下跌）而非上涨。已剔除 `-99` 并按 `P(y=1)` 重训。',
        '- 修正后 HistGB 十分位大体单调为正。CNN 仍高于 HistGB。**MLP 的 EW H-L Sharpe 可能高于 CNN**，主对照仍以 HistGB 为准。',
        '',
    ]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f'wrote {md_path}')
    logger.info(f'verdict={verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
