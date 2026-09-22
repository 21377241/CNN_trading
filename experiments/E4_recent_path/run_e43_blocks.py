# -*- coding: utf-8 -*-
"""E4-3 matched blocks: CNN OHLCV transforms + MLP/HistGB column drops."""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from experiments.E3_shared_specific.utils import norm_stock_id
from experiments.E4_recent_path.cnn_forward import load_cnn_models, predict_intervention
from experiments.E4_recent_path.eval_specific import attach_deltas, eval_specific_df
from experiments.E4_recent_path.interventions_tabular import TAB_BLOCKS, block_cols
from experiments.E4_recent_path.paths import OUT_ROOT, PRED_ROOT
from experiments.E4_recent_path.scores_pairwise import (
    build_baseline_scores,
    residualize_cnn_on_mlp,
    residualize_mlp_on_cnn,
)
from experiments.E4_recent_path.tabular_train import partner_cnn_df, partner_mlp_df, train_predict_subset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e4_e43')

BLOCKS = list(TAB_BLOCKS.keys())  # volume, wick, direction, return, gap, intraday


def _baseline_row(side: str, scores: pd.DataFrame, out_dir: str) -> dict:
    col = 'S_cnn_sp' if side == 'cnn' else 'S_mlp_sp'
    return eval_specific_df(scores.rename(columns={col: 'S_sp'}), 'baseline', out_dir)


def run_mlp_blocks(do_histgb: bool = True) -> pd.DataFrame:
    out_dir = os.path.join(OUT_ROOT, 'e43_blocks', 'mlp')
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(OUT_ROOT, 'e43_blocks', 'tab_cache')
    scores = pd.read_parquet(os.path.join(OUT_ROOT, 'baseline_scores.parquet'))
    partner_c = partner_cnn_df()
    rows = [_baseline_row('mlp', scores, out_dir)]
    hgb_rows = []
    if do_histgb:
        hgb_dir = os.path.join(OUT_ROOT, 'e43_blocks', 'histgb')
        os.makedirs(hgb_dir, exist_ok=True)
        hgb = pd.read_csv(os.path.join(PRED_ROOT, 'pred_histgb_oos.csv'))
        hgb['StockID'] = norm_stock_id(hgb['StockID'])
        hgb['ending_date'] = pd.to_datetime(hgb['ending_date'])
        for c in ('ret_val', 'MarketCap', 'binary_label'):
            if c not in hgb.columns and c in scores.columns:
                hgb = hgb.merge(scores[['StockID', 'ending_date', c]], on=['StockID', 'ending_date'], how='left')
        base_sp = residualize_mlp_on_cnn(hgb.rename(columns={'up_prob': 'S_mlp'}), partner_c)
        hgb_rows = [eval_specific_df(base_sp, 'baseline', hgb_dir)]

    for name in BLOCKS:
        tag = f'block_{name}'
        logger.info(f'=== MLP block {name} ===')
        cols = block_cols(name)
        prob, meta = train_predict_subset(cols, tag, 'mlp', cache)
        pred = meta[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
        pred['StockID'] = norm_stock_id(pred['StockID'])
        pred['ending_date'] = pd.to_datetime(pred['ending_date'])
        pred['up_prob'] = prob
        sp = residualize_mlp_on_cnn(pred, partner_c)
        row = eval_specific_df(sp, tag, out_dir)
        row['block'] = name
        rows.append(row)
        if do_histgb:
            prob_h, meta_h = train_predict_subset(cols, tag, 'histgb', cache)
            pred_h = meta_h[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
            pred_h['StockID'] = norm_stock_id(pred_h['StockID'])
            pred_h['ending_date'] = pd.to_datetime(pred_h['ending_date'])
            pred_h['up_prob'] = prob_h
            sp_h = residualize_mlp_on_cnn(pred_h, partner_c)
            rh = eval_specific_df(sp_h, tag, os.path.join(OUT_ROOT, 'e43_blocks', 'histgb'))
            rh['block'] = name
            hgb_rows.append(rh)

    summ = attach_deltas(pd.DataFrame(rows))
    # normalized drop vector for H3
    base_sh = float(summ.loc[summ['tag'] == 'baseline', 'ew_hl_sharpe'].iloc[0])
    drops = summ.loc[summ['tag'] != 'baseline', 'delta_ew_sharpe'].values
    if abs(drops).sum() > 1e-12:
        summ.loc[summ['tag'] != 'baseline', 'drop_norm'] = drops / abs(drops).sum()
    else:
        summ['drop_norm'] = 0.0
    summ.to_csv(os.path.join(out_dir, 'e43_mlp_block_summary.csv'), index=False, float_format='%.6f')
    if do_histgb:
        hs = attach_deltas(pd.DataFrame(hgb_rows))
        hs.to_csv(os.path.join(OUT_ROOT, 'e43_blocks', 'histgb', 'e43_histgb_block_summary.csv'), index=False, float_format='%.6f')
    logger.info(f'MLP blocks:\n{summ.to_string(index=False)}')
    return summ


def run_cnn_blocks() -> pd.DataFrame:
    out_dir = os.path.join(OUT_ROOT, 'e43_blocks', 'cnn')
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, 'preds')
    scores = pd.read_parquet(os.path.join(OUT_ROOT, 'baseline_scores.parquet'))
    partner_m = partner_mlp_df()
    rows = [_baseline_row('cnn', scores, out_dir)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_cnn_models(device)
    for name in BLOCKS:
        logger.info(f'=== CNN block {name} ===')
        pred = predict_intervention(name, pred_dir, models, device)
        sp = residualize_cnn_on_mlp(pred, partner_m)
        if 'ret_val' not in sp.columns:
            sp = sp.merge(
                scores[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']],
                on=['StockID', 'ending_date'], how='left',
            )
        row = eval_specific_df(sp, f'block_{name}', out_dir)
        row['block'] = name
        rows.append(row)
    summ = attach_deltas(pd.DataFrame(rows))
    drops = summ.loc[summ['tag'] != 'baseline', 'delta_ew_sharpe'].values
    if abs(drops).sum() > 1e-12:
        summ.loc[summ['tag'] != 'baseline', 'drop_norm'] = drops / abs(drops).sum()
    else:
        summ['drop_norm'] = 0.0
    summ.to_csv(os.path.join(out_dir, 'e43_cnn_block_summary.csv'), index=False, float_format='%.6f')
    logger.info(f'CNN blocks:\n{summ.to_string(index=False)}')
    return summ


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT_ROOT, 'baseline_scores.parquet')):
        build_baseline_scores()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--side', choices=['mlp', 'cnn', 'both'], default='both')
    p.add_argument('--no-histgb', action='store_true')
    args = p.parse_args()
    if args.side in ('mlp', 'both'):
        run_mlp_blocks(do_histgb=not args.no_histgb)
    if args.side in ('cnn', 'both'):
        run_cnn_blocks()


if __name__ == '__main__':
    main()
