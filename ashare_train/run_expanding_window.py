# -*- coding: utf-8 -*-
"""
训练期扩展窗口（expanding window）：沿用 registry 选定超参，按 OOS 年重切 train/val。

协议（见 docs/protocol/clean_spec.md 追加节）：
  对每个 OOS 年 y ∈ {2008,...,2025}：
    train = 2000 .. y-3
    val   = y-2 .. y-1
    predict y
  embargo = train 末年与 val 末年各自最后一个信号周（动态重算，不用主规格写死的 2005/2007）。
  超参固定为 chartcnn_clean / onedcnn_140 / mlp_140 的选定组合。

用法（单窗，供 job 提交）：
  python ashare_train/run_expanding_window.py --model chart --oos-year 2012
  python ashare_train/run_expanding_window.py --model oned  --oos-year 2012
  python ashare_train/run_expanding_window.py --model mlp   --oos-year 2012
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_ROOT = PROJECT_ROOT / 'WORK_SPACE/paper/clean_inputs/features140'
CHART_SRC = PROJECT_ROOT / 'WORK_SPACE/data/ashare_stocks_dataset/I20R5_clean/stocks_CHN/dataset_all'
OUT_ROOT = PROJECT_ROOT / 'WORK_SPACE/paper/expanding_window'
REGISTRY = PROJECT_ROOT / 'WORK_SPACE/paper/registry_v2.json'
SEEDS = [42, 43, 44, 45, 46]
OOS_YEARS = list(range(2008, 2026))  # 2008–2025
MONITOR = 'rank_ic'

logger = logging.getLogger('expanding_window')


def year_split(oos_year: int) -> dict:
    if oos_year not in OOS_YEARS:
        raise ValueError(f'oos_year must be in {OOS_YEARS[0]}..{OOS_YEARS[-1]}, got {oos_year}')
    return {
        'oos_year': oos_year,
        'train_years': list(range(2000, oos_year - 2)),  # 2000 .. y-3
        'val_years': [oos_year - 2, oos_year - 1],
        'oos_years': [oos_year],
    }


def embargo_dates(meta: pd.DataFrame, boundary_years: list[int]) -> set:
    """每个 boundary year 的最后一个 ending_date。"""
    out = set()
    ed = pd.to_datetime(meta['ending_date'])
    years = meta['year'].to_numpy() if 'year' in meta.columns else ed.dt.year.to_numpy()
    for y in boundary_years:
        mask = years == int(y)
        if not mask.any():
            raise RuntimeError(f'no rows for embargo year {y}')
        out.add(pd.Timestamp(ed[mask].max()))
    return out


def load_panel_years(years: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    """按年拼接 features140/by_year 与 meta（is∪oos）。"""
    meta_is = pd.read_feather(FEATURE_ROOT / 'meta_is.feather')
    meta_oos = pd.read_feather(FEATURE_ROOT / 'meta_oos.feather')
    meta = pd.concat([meta_is, meta_oos], ignore_index=True)
    meta['ending_date'] = pd.to_datetime(meta['ending_date'])
    if 'year' not in meta.columns:
        meta['year'] = meta['ending_date'].dt.year
    xs, ms = [], []
    for y in years:
        path = FEATURE_ROOT / 'by_year' / f'X_{y}.npy'
        if not path.is_file():
            raise FileNotFoundError(path)
        Xy = np.load(path)
        my = meta.loc[meta['year'] == y].reset_index(drop=True)
        if len(my) != len(Xy):
            raise RuntimeError(f'year {y}: meta rows {len(my)} != X rows {len(Xy)}')
        xs.append(Xy)
        ms.append(my)
    return np.concatenate(xs, axis=0), pd.concat(ms, ignore_index=True)


def trainable_mask(meta: pd.DataFrame, ban: set) -> np.ndarray:
    """用 label_raw∈{0,1}，并剔除动态 embargo 周（不用写死的 binary_label/-99）。"""
    raw = meta['label_raw'].to_numpy()
    ed = pd.to_datetime(meta['ending_date'])
    ok = np.isin(raw, [0, 1]) & ~ed.isin(ban)
    return ok


def selected_hp(model: str) -> dict:
    reg = json.loads(REGISTRY.read_text(encoding='utf-8'))
    key = {'chart': 'chartcnn_clean', 'oned': 'onedcnn_140', 'mlp': 'mlp_140'}[model]
    return reg['predictions'][key]['hp']


def run_mlp(split: dict, out_dir: Path) -> None:
    from experiments.E2_matched_tabular.mlp_bagging import fit_mlp_chrono_es

    hp = selected_hp('mlp')
    years = split['train_years'] + split['val_years'] + split['oos_years']
    X, meta = load_panel_years(years)
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)  # by_year 为 (N,20,7) → 140
    ban = embargo_dates(meta, [split['train_years'][-1], split['val_years'][-1]])
    years_arr = meta['year'].to_numpy()
    y_raw = meta['label_raw'].to_numpy(dtype=int)
    tr = np.flatnonzero(np.isin(years_arr, split['train_years']) & trainable_mask(meta, ban))
    va = np.flatnonzero(np.isin(years_arr, split['val_years']) & trainable_mask(meta, ban))
    oos = np.flatnonzero(np.isin(years_arr, split['oos_years']))
    logger.info(f"mlp y={split['oos_year']}: X={X.shape} train={len(tr)} val={len(va)} oos={len(oos)} embargo={sorted(ban)}")

    model_root = out_dir / 'models'
    model_root.mkdir(parents=True, exist_ok=True)
    preds = []
    for seed in SEEDS:
        seed_dir = model_root / f'seed{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        info_path = seed_dir / 'info.json'
        if info_path.is_file():
            pipe = __import__('joblib').load(seed_dir / 'model.joblib')
        else:
            pipe, info = fit_mlp_chrono_es(
                X[tr], y_raw[tr], X[va], y_raw[va], meta.iloc[va].reset_index(drop=True),
                seed=seed, hidden_layer_sizes=tuple(hp['hidden']), alpha=hp['alpha'], lr=hp['lr'],
                monitor=MONITOR, max_iter=200, n_iter_no_change=15)
            __import__('joblib').dump(pipe, seed_dir / 'model.joblib')
            info_path.write_text(json.dumps({k: v for k, v in info.items() if k != 'history'}, indent=2), encoding='utf-8')
        clf = pipe.named_steps['clf']
        preds.append(pipe.predict_proba(X[oos])[:, list(clf.classes_).index(1)])

    out = meta.iloc[oos][['StockID', 'ending_date', 'ret_val', 'MarketCap', 'label_raw']].copy()
    out['StockID'] = out['StockID'].astype(str)
    out['up_prob'] = np.mean(preds, axis=0)
    for s, p in zip(SEEDS, preds):
        out[f'up_prob_seed{s}'] = p
    pred_dir = out_dir / 'preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"pred_mlp_oos{split['oos_year']}.parquet"
    out.to_parquet(path, index=False)
    logger.info(f'wrote {path} rows={len(out)}')


def run_oned(split: dict, out_dir: Path) -> None:
    from experiments.E2C_temporal.train_1dcnn import fit_seed, predict_oos_seed

    hp = selected_hp('oned')
    cfg_path = PROJECT_ROOT / 'experiments/E2C_temporal/config_1dcnn_140.yaml'
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['model']['hidden_channels'] = list(hp['channels'])
    cfg['model']['dropout'] = float(hp['dropout'])
    cfg['train']['learning_rate'] = float(hp['lr'])
    cfg['train']['monitor'] = MONITOR
    cfg['model_tag'] = f"1dcnn140_ew{split['oos_year']}"
    cfg['pred_format'] = 'parquet'
    cfg['seeds'] = SEEDS

    years = split['train_years'] + split['val_years'] + split['oos_years']
    X, meta = load_panel_years(years)
    ban = embargo_dates(meta, [split['train_years'][-1], split['val_years'][-1]])
    years_arr = meta['year'].to_numpy()
    y_raw = meta['label_raw'].to_numpy(dtype=int)
    tr = np.flatnonzero(np.isin(years_arr, split['train_years']) & trainable_mask(meta, ban))
    va = np.flatnonzero(np.isin(years_arr, split['val_years']) & trainable_mask(meta, ban))
    oos = np.flatnonzero(np.isin(years_arr, split['oos_years']))
    # fit_seed expects contiguous meta_is indexing; pass full X/meta and index arrays
    X_tv = X  # full panel for this window
    meta_tv = meta
    y_tv = y_raw
    logger.info(f"oned y={split['oos_year']}: train={len(tr)} val={len(va)} oos={len(oos)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'models').mkdir(exist_ok=True)
    (out_dir / 'preds').mkdir(exist_ok=True)
    for seed in SEEDS:
        info_path = out_dir / 'models' / f'seed{seed}' / 'info.json'
        if not info_path.is_file():
            fit_seed(cfg, seed, X_tv, y_tv, meta_tv, tr, va, str(out_dir))

    preds = [predict_oos_seed(cfg, seed, X[oos], meta.iloc[oos].reset_index(drop=True), str(out_dir)) for seed in SEEDS]
    out = meta.iloc[oos][['StockID', 'ending_date', 'ret_val', 'MarketCap', 'label_raw']].copy()
    out['StockID'] = out['StockID'].astype(str)
    out['up_prob'] = np.mean(preds, axis=0)
    for s, p in zip(SEEDS, preds):
        out[f'up_prob_seed{s}'] = p
    path = out_dir / 'preds' / f"pred_1dcnn_oos{split['oos_year']}.parquet"
    out.to_parquet(path, index=False)
    logger.info(f'wrote {path} rows={len(out)}')


def _chart_labels_name(year: int) -> str:
    return f'20d_week_has_vb_[20]_ma_{year}_labels.feather'


def _chart_images_name(year: int) -> str:
    return f'20d_week_has_vb_[20]_ma_{year}_images.dat'


def prepare_chart_overlay(split: dict, overlay: Path) -> None:
    """镜像图像文件；重写 train/val 边界年的 labels.embargo / binary_label。"""
    overlay.mkdir(parents=True, exist_ok=True)
    need_years = sorted(set(split['train_years'] + split['val_years'] + split['oos_years']))
    # collect last dates from source labels
    last_of = {}
    for y in need_years:
        src = CHART_SRC / _chart_labels_name(y)
        m = pd.read_feather(src)
        m['ending_date'] = pd.to_datetime(m['ending_date'])
        last_of[y] = pd.Timestamp(m['ending_date'].max())
    ban = {last_of[split['train_years'][-1]], last_of[split['val_years'][-1]]}

    for y in need_years:
        img_src = CHART_SRC / _chart_images_name(y)
        lab_src = CHART_SRC / _chart_labels_name(y)
        img_dst = overlay / _chart_images_name(y)
        lab_dst = overlay / _chart_labels_name(y)
        if not img_dst.exists():
            os.symlink(img_src, img_dst)
        m = pd.read_feather(lab_src)
        m['ending_date'] = pd.to_datetime(m['ending_date'])
        # clear baked embargo; apply dynamic ban; rebuild binary_label from label_raw
        m['embargo'] = m['ending_date'].isin(ban).astype(np.int8)
        raw = m['label_raw'].to_numpy()
        m['binary_label'] = np.where((raw < 0) | (m['embargo'] == 1), -99, raw).astype(np.int64)
        m.to_feather(lab_dst)


def run_chart(split: dict, out_dir: Path) -> None:
    CODE_PKG = PROJECT_ROOT / 'trend_submit' / 'trend_code_submit'
    if str(CODE_PKG) not in sys.path:
        sys.path.insert(0, str(CODE_PKG))
    from Experiments.cnn_experiment import train_us_model, _build_exp_dir
    from Model.cnn_model import get_full_model_name
    from Misc.config import ASHARE_COUNTRY, MA_LAGS, VOLUME_BAR

    hp = selected_hp('chart')
    overlay = out_dir / 'dataset_overlay'
    prepare_chart_overlay(split, overlay)
    cfg = {
        'window_size': 20, 'predict_window': 5, 'freq': 'week',
        'training': 'clean', 'monitor': 'rank_ic',
        'batch_size': 128, 'max_epochs': 50, 'n_ensembles': 5,
        'hp': {'lr': hp['lr'], 'dropout': hp['dropout'], 'patience': hp['patience']},
        'train_years': split['train_years'], 'val_years': split['val_years'], 'oos_years': split['oos_years'],
        'dir_suffix': f"ew{split['oos_year']}-p{hp['patience']}",
    }
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    logger.info(f"chart y={split['oos_year']}: train={cfg['train_years'][0]}-{cfg['train_years'][-1]} "
                f"val={cfg['val_years']} oos={cfg['oos_years']} hp={cfg['hp']}")
    train_us_model(
        ws_list=[20], pw_list=[5], freq='week', ma_lags=MA_LAGS, has_vb=VOLUME_BAR,
        lr=float(hp['lr']), dropout=float(hp['dropout']), n_ensembles=5,
        batch_size=128, max_epochs=50, early_stop_patience=int(hp['patience']),
        is_years=split['train_years'], val_years=split['val_years'], oos_years=split['oos_years'],
        country=ASHARE_COUNTRY, base_exp_dir=str(out_dir), dataset_dir=str(overlay),
        dir_suffix=cfg['dir_suffix'], skip_oos=False, force_retrain=False,
        training='clean', monitor='rank_ic',
    )
    exp_dir = _build_exp_dir(get_full_model_name(20), 20, 5, float(hp['lr']), float(hp['dropout']),
                             'week', True, True, base_exp_dir=str(out_dir), dir_suffix=cfg['dir_suffix'])
    logger.info(f'chart done: {exp_dir}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['chart', 'oned', 'mlp'], required=True)
    ap.add_argument('--oos-year', type=int, required=True)
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    split = year_split(args.oos_year)
    out_dir = Path(args.out_root) / args.model / f"y{args.oos_year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out_dir / 'run.log', encoding='utf-8')],
    )
    logger.info(f'start model={args.model} split={split}')
    if args.model == 'mlp':
        run_mlp(split, out_dir)
    elif args.model == 'oned':
        run_oned(split, out_dir)
    else:
        run_chart(split, out_dir)
    logger.info('DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
