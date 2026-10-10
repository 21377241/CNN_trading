# -*- coding: utf-8 -*-
"""
Experiments/cnn_experiment.py - CNN 实验训练与推理

主要函数：
  train_us_model(ws_list, pw_list, ...)
    - 对 (window_size, predict_window) 组合训练 N_ENSEMBLES 个模型
    - 在 OOS_YEARS 上做集成预测，输出 ensem5_res_{year}.csv

  Experiment 类（可选，封装单次实验）
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Misc.config import (
    EXP_DIR, LOG_DIR, LATEX_DIR, IS_YEARS, OOS_YEARS,
    BENCHMARK_MODEL_NAME, BENCHMARK_MODEL_WINDOW, BENCHMARK_MODEL_PREDICT_WINDOW,
    LEARNING_RATE, DROPOUT_RATE, N_ENSEMBLES, BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOP_PATIENCE, FREQ, MA_LAGS, VOLUME_BAR, COUNTRY
)
from Misc.utilities import compute_mcc, setup_logger
from Data.chart_dataset import EquityDataset, batch_collate
from Data.dgp_config import IMAGE_WIDTH, IMAGE_HEIGHT
from Model.cnn_model import init_model, get_full_model_name

logger = logging.getLogger(__name__)

# 'legacy'：旧流程（ReduceLROnPlateau；HP trial 保存验证 Rank IC 最高那一轮），仅用于复现旧结果
# 'clean' ：clean_spec 5.1（无学习率调度；按验证 loss 早停并保存 loss 最优 checkpoint；
#           trial 得分为 loss 最优 checkpoint 的验证 Rank IC；训练/验证用 sample_mode='train'，OOS 用 'oos'）
TRAINING_MODES = ('legacy', 'clean')


def _sample_modes(training: str) -> tuple[str, str]:
    if training not in TRAINING_MODES:
        raise ValueError(f'training 须为 {TRAINING_MODES}，得到 {training}')
    return ('legacy', 'legacy') if training == 'legacy' else ('train', 'oos')


def _dataset(years, ws, pw, freq, has_vb, ma_lags, country, dataset_dir, sample_mode) -> EquityDataset:
    return EquityDataset(years=years, window_size=ws, predict_window=pw, freq=freq, has_volume_bar=has_vb,
                         ma_lags=ma_lags, country=country, image_scale='norm255', dataset_dir=dataset_dir,
                         sample_mode=sample_mode, batched=True)


def _loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False,
                      collate_fn=batch_collate)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_exp_dir(model_name: str, window_size: int, predict_window: int,
                   lr: float, dropout: float, freq: str,
                   has_ma: bool, has_vb: bool,
                   base_exp_dir: str = None,
                   dir_suffix: str = None) -> str:
    """
    构建实验目录路径（与文档命名约定一致）。

    Args:
        base_exp_dir: 实验根目录，默认使用 config.EXP_DIR（SPX路径）。
                      CSI800 请传入 config.CSI800_EXP_DIR。

    示例：
      WORK_DIR/spx_model_res/<model_name>/<exp_name>/
      WORK_DIR/csi800_model_res/<model_name>/<exp_name>/
    """
    base = base_exp_dir if base_exp_dir is not None else EXP_DIR
    lr_str = f"lr{lr:.0E}".replace('-0', '-').replace('+0', '')
    dp_str = f"dp{dropout:.2f}"
    ma_str = 'maTrue' if has_ma else 'maFalse'
    vb_str = 'vbTrue' if has_vb else 'vbFalse'
    exp_name = (f"{window_size}d{predict_window}p"
                f"-{lr_str}-{dp_str}-{ma_str}-{vb_str}"
                f"-{freq}Trained-noDelayedReturn")
    if dir_suffix:
        exp_name = f"{exp_name}-{dir_suffix}"
    exp_dir = os.path.join(base, model_name, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'ensem_res'), exist_ok=True)
    return exp_dir


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
    mcc = compute_mcc(np.array(all_labels), np.array(all_preds))
    return avg_loss, acc, mcc


@torch.no_grad()
def _eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
    mcc = compute_mcc(np.array(all_labels), np.array(all_preds))
    return avg_loss, acc, mcc


@torch.no_grad()
def _predict(model, loader, device):
    """返回 (up_prob_list, down_prob_list)"""
    model.eval()
    up_probs = []
    down_probs = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        probs = model.predict_proba(imgs)
        down_probs.extend(probs[:, 0].cpu().numpy())
        up_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(up_probs), np.array(down_probs)


def _mean_rank_ic(meta: pd.DataFrame, scores: np.ndarray, min_n: int = 10) -> float:
    """Cross-sectional Spearman Rank IC averaged over ending_date."""
    from scipy.stats import spearmanr

    df = meta[['ending_date', 'ret_val']].copy()
    df['score'] = scores
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    ics = []
    for _, g in df.groupby('ending_date'):
        s = g['score'].values
        r = g['ret_val'].values
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < min_n:
            continue
        sr, _ = spearmanr(s[m], r[m])
        if np.isfinite(sr):
            ics.append(float(sr))
    if not ics:
        return float('nan')
    return float(np.mean(ics))


@torch.no_grad()
def _eval_val_rank_ic(model, loader, meta, device) -> float:
    up, _ = _predict(model, loader, device)
    if len(up) != len(meta):
        logger.warning(f'Rank IC length mismatch: probs={len(up)} meta={len(meta)}')
        return float('nan')
    return _mean_rank_ic(meta, up)


# ─────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────
def train_us_model(ws_list: list = None,
                   pw_list: list = None,
                   freq: str = FREQ,
                   ma_lags: list = None,
                   has_vb: bool = VOLUME_BAR,
                   lr: float = LEARNING_RATE,
                   dropout: float = DROPOUT_RATE,
                   n_ensembles: int = N_ENSEMBLES,
                   batch_size: int = BATCH_SIZE,
                   max_epochs: int = MAX_EPOCHS,
                   early_stop_patience: int = EARLY_STOP_PATIENCE,
                   is_years: list = None,
                   oos_years: list = None,
                   val_years: list = None,
                   train_val_split: float = 0.7,
                   country: str = COUNTRY,
                   base_exp_dir: str = None,
                   dataset_dir: str = None,
                   dir_suffix: str = None,
                   skip_oos: bool = False,
                   force_retrain: bool = False,
                   training: str = 'legacy'):
    """
    训练多个 (window_size, predict_window) 组合的集成 CNN 模型，并生成 OOS 预测。

    Args:
        ws_list: 窗口大小列表，如 [20]
        pw_list: 预测窗口列表，如 [20]
        freq: 预测频率，如 'month'
        ma_lags: 均线周期列表
        has_vb: 是否含成交量柱
        lr: 学习率
        dropout: Dropout 率
        n_ensembles: 集成模型数量
        batch_size: 批大小
        max_epochs: 最大训练轮数
        early_stop_patience: 早停耐心（验证集 loss 不降）
        is_years: 训练年份列表（若提供 val_years，则仅为 Train 年）
        oos_years: 测试年份列表
        val_years: 若提供，则按年 chronological 验证（不与 Train 随机混切）
        train_val_split: 训练/验证比例（仅当 val_years 为空时，按样本随机划分）
        country: 国家标识
        base_exp_dir: 实验根目录，默认 EXP_DIR（SPX）。CSI800 传入 CSI800_EXP_DIR。
        dataset_dir: 图表数据集目录，默认由 get_dataset_dir(country) 推断。
                     CSI800 传入 get_dataset_dir(CSI800_COUNTRY, CSI800_STOCKS_SAVEPATH)。
        dir_suffix: 追加到实验目录名（HP 搜索时区分 patience 等）。
        skip_oos: 为 True 时不生成 OOS 预测（HP 搜索 Phase 1/2）。
        force_retrain: 为 True 时忽略已有 checkpoint 重新训练。
        training: 'legacy' | 'clean'（见 TRAINING_MODES）。
    """
    fit_mode, oos_mode = _sample_modes(training)
    ws_list  = ws_list  or [BENCHMARK_MODEL_WINDOW]
    pw_list  = pw_list  or [BENCHMARK_MODEL_PREDICT_WINDOW]
    ma_lags  = ma_lags  if ma_lags is not None else MA_LAGS
    is_years = is_years or IS_YEARS
    oos_years= oos_years or OOS_YEARS

    device = _get_device()
    logger.info(f"训练设备: {device}")

    log_dir = os.path.join(base_exp_dir, 'log') if base_exp_dir else LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    setup_logger('train', os.path.join(log_dir, 'train.log'))

    for ws, pw in zip(ws_list, pw_list):
        img_h = IMAGE_HEIGHT.get(ws, 64)
        img_w = IMAGE_WIDTH.get(ws, 60)
        model_name = get_full_model_name(ws)
        exp_dir = _build_exp_dir(model_name, ws, pw, lr, dropout, freq,
                                 len(ma_lags) > 0, has_vb,
                                 base_exp_dir=base_exp_dir,
                                 dir_suffix=dir_suffix)

        logger.info(f"\n{'='*60}")
        logger.info(f"开始实验: ws={ws}, pw={pw}, 实验目录: {exp_dir}")
        if val_years:
            logger.info(f"切分模式: chronological  Train={is_years}  Val={val_years}  Test={oos_years}")
        else:
            logger.info(f"切分模式: IS内随机 {train_val_split:.0%}/{1-train_val_split:.0%}  IS={is_years}  OOS={oos_years}")

        # ── 加载训练集 ──
        all_ckpts_exist = all(os.path.exists(os.path.join(exp_dir, f'checkpoint{i}.pth.tar'))
                              for i in range(n_ensembles))
        is_dataset = val_dataset_fixed = None
        if not all_ckpts_exist or force_retrain:
            try:
                is_dataset = _dataset(is_years, ws, pw, freq, has_vb, ma_lags, country, dataset_dir, fit_mode)
            except RuntimeError as e:
                logger.error(f"训练数据加载失败: {e}")
                continue
            if val_years:
                try:
                    val_dataset_fixed = _dataset(val_years, ws, pw, freq, has_vb, ma_lags, country, dataset_dir,
                                                 fit_mode)
                except RuntimeError as e:
                    logger.error(f"验证数据加载失败: {e}")
                    continue
                logger.info(f"训练集样本: {len(is_dataset)}（上涨比例 {is_dataset.up_share:.4f}），"
                            f"验证集样本: {len(val_dataset_fixed)}（上涨比例 {val_dataset_fixed.up_share:.4f}）")
        if is_dataset is not None and not val_years:
            n_total = len(is_dataset)
            n_train = int(n_total * train_val_split)
            n_val   = n_total - n_train
            logger.info(f"训练集总样本: {n_total}, 训练: {n_train}, 验证: {n_val}")

        # ── 训练 N 个集成模型 ──
        for ens_idx in range(n_ensembles):
            ckpt_path = os.path.join(exp_dir, f'checkpoint{ens_idx}.pth.tar')
            if os.path.exists(ckpt_path) and not force_retrain:
                logger.info(f"Ensemble {ens_idx}: checkpoint 已存在，跳过训练。")
                continue

            logger.info(f"\n--- 训练 Ensemble {ens_idx}（training={training}）---")

            torch.manual_seed(42 + ens_idx)
            np.random.seed(42 + ens_idx)

            if val_dataset_fixed is not None:
                train_loader = _loader(is_dataset, batch_size, shuffle=True)
                val_loader = _loader(val_dataset_fixed, batch_size, shuffle=False)
                val_meta = val_dataset_fixed.meta
            else:
                train_set, val_set = random_split(is_dataset, [n_train, n_val])
                train_loader = _loader(train_set, batch_size, shuffle=True)
                val_loader = _loader(val_set, batch_size, shuffle=False)
                val_meta = is_dataset.meta.iloc[val_set.indices].reset_index(drop=True)

            model = init_model(model_name, img_h=img_h, img_w=img_w,
                                dropout=dropout, n_classes=2,
                                window_size=ws).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            scheduler = None
            if training == 'legacy':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=1, factor=0.5
                )

            best_val_loss = float('inf')
            best_val_mcc = -1.0
            best_epoch = 0
            loss_best_epoch = 0
            no_improve = 0
            best_state = None
            history = []

            for epoch in range(1, max_epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc, tr_mcc = _train_one_epoch(
                    model, train_loader, optimizer, criterion, device)
                va_loss, va_acc, va_mcc = _eval_one_epoch(
                    model, val_loader, criterion, device)
                if scheduler is not None:
                    scheduler.step(va_loss)
                history.append({'seed': 42 + ens_idx, 'epoch': epoch, 'train_loss': tr_loss,
                                'val_loss': va_loss, 'val_mcc': va_mcc})

                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} tr_mcc={tr_mcc:.3f} | "
                    f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} va_mcc={va_mcc:.3f} | "
                    f"{elapsed:.1f}s"
                )

                if va_loss < best_val_loss - 1e-6:
                    best_val_loss = va_loss
                    loss_best_epoch = epoch
                    no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        logger.info(f"早停（{early_stop_patience} 轮无改善）")
                        break
                if va_mcc > best_val_mcc:
                    best_val_mcc = va_mcc
                    best_epoch = epoch

            # 保存 checkpoint（两种 training 都保存 loss 最优的权重）
            if best_state is not None:
                model.load_state_dict(best_state)
                val_ric = _eval_val_rank_ic(model, val_loader, val_meta, device)
                torch.save({'epoch': best_epoch if training == 'legacy' else loss_best_epoch,
                            'loss_best_epoch': loss_best_epoch,
                            'model_state_dict': best_state,
                            'val_loss': best_val_loss,
                            'val_mcc': best_val_mcc,
                            'val_rank_ic': val_ric,
                            'training': training,
                            'seed': 42 + ens_idx,
                            'window_size': ws,
                            'predict_window': pw,
                            'img_h': img_h,
                            'img_w': img_w,
                            'dropout': dropout},
                           ckpt_path)
                pd.DataFrame(history).to_csv(os.path.join(exp_dir, f'history_seed{42 + ens_idx}.csv'),
                                             index=False)
                logger.info(f"Checkpoint 保存至: {ckpt_path}（loss 最优轮 {loss_best_epoch}，"
                            f"验证 Rank IC {val_ric:.4f}）")

        # ── OOS 预测 ──
        if not skip_oos:
            _generate_oos_predictions(
                exp_dir=exp_dir,
                ws=ws, pw=pw,
                img_h=img_h, img_w=img_w,
                freq=freq, has_vb=has_vb, ma_lags=ma_lags,
                n_ensembles=n_ensembles,
                oos_years=oos_years,
                batch_size=batch_size,
                device=device,
                country=country,
                dropout=dropout,
                model_name=model_name,
                dataset_dir=dataset_dir,
                sample_mode=oos_mode,
            )
        else:
            logger.info("skip_oos=True，跳过 OOS 预测。")


def run_cnn_trial(
    ws: int,
    pw: int,
    freq: str,
    lr: float,
    dropout: float,
    is_years: list,
    val_years: list,
    country: str,
    base_exp_dir: str,
    dataset_dir: str,
    ma_lags: list = None,
    has_vb: bool = VOLUME_BAR,
    n_ensembles: int = 1,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    dir_suffix: str = None,
    force_retrain: bool = False,
    ens_seed_base: int = 42,
    search_metric: str = 'val_rank_ic',
    channels: list[int] | tuple[int, ...] | None = None,
    save_history: bool = False,
    training: str = 'legacy',
) -> dict:
    """单次 HP trial：chrono val 训练，不写 OOS。返回 val 指标。

    Early stop 用 val_loss；跨 trial 选参默认用 best_val_rank_ic：
      legacy：各轮验证 Rank IC 的最大值（并保存该轮权重）
      clean ：loss 最优 checkpoint 的验证 Rank IC（多个 seed 时取均值）
    """
    fit_mode, _ = _sample_modes(training)
    ma_lags = ma_lags if ma_lags is not None else MA_LAGS
    device = _get_device()
    img_h = IMAGE_HEIGHT.get(ws, 64)
    img_w = IMAGE_WIDTH.get(ws, 60)
    model_name = get_full_model_name(ws)
    exp_dir = _build_exp_dir(
        model_name, ws, pw, lr, dropout, freq,
        len(ma_lags) > 0, has_vb,
        base_exp_dir=base_exp_dir,
        dir_suffix=dir_suffix,
    )

    t_start = time.time()
    try:
        is_dataset = _dataset(is_years, ws, pw, freq, has_vb, ma_lags, country, dataset_dir, fit_mode)
        val_dataset = _dataset(val_years, ws, pw, freq, has_vb, ma_lags, country, dataset_dir, fit_mode)
    except RuntimeError as e:
        logger.error(f"HP trial 数据加载失败: {e}")
        return {
            'lr': lr, 'dropout': dropout, 'patience': early_stop_patience,
            'best_val_mcc': float('nan'), 'best_val_loss': float('nan'),
            'best_val_rank_ic': float('nan'),
            'best_epoch': 0, 'exp_dir': exp_dir, 'wall_time_sec': 0.0,
            'error': str(e),
        }

    train_loader = _loader(is_dataset, batch_size, shuffle=True)
    val_loader = _loader(val_dataset, batch_size, shuffle=False)
    val_meta = val_dataset.meta
    logger.info(f"trial 训练集 {len(is_dataset)}（上涨比例 {is_dataset.up_share:.4f}），"
                f"验证集 {len(val_dataset)}（上涨比例 {val_dataset.up_share:.4f}）")

    trial_best_mcc = -1.0
    trial_best_loss = float('inf')
    trial_best_rank_ic = float('-inf')
    trial_best_epoch = 0
    seed_rics = []

    for ens_idx in range(n_ensembles):
        ckpt_path = os.path.join(exp_dir, f'checkpoint{ens_idx}.pth.tar')
        if os.path.exists(ckpt_path) and not force_retrain:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            trial_best_mcc = max(trial_best_mcc, float(ckpt.get('val_mcc', -1.0)))
            trial_best_loss = min(trial_best_loss, float(ckpt.get('val_loss', float('inf'))))
            ric = ckpt.get('val_rank_ic', None)
            if ric is None or not np.isfinite(float(ric)):
                # recompute Rank IC for old checkpoints
                model = init_model(model_name, img_h=img_h, img_w=img_w,
                                   dropout=0.0, n_classes=2, window_size=ws,
                                   channels=channels).to(device)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                ric = _eval_val_rank_ic(model, val_loader, val_meta, device)
            trial_best_rank_ic = max(trial_best_rank_ic, float(ric))
            seed_rics.append(float(ric))
            trial_best_epoch = max(trial_best_epoch, int(ckpt.get('epoch', 0)))
            continue

        torch.manual_seed(ens_seed_base + ens_idx)
        np.random.seed(ens_seed_base + ens_idx)

        model = init_model(model_name, img_h=img_h, img_w=img_w,
                           dropout=dropout, n_classes=2, window_size=ws,
                           channels=channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = None
        if training == 'legacy':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=1, factor=0.5
            )

        best_val_loss = float('inf')
        best_val_mcc = -1.0
        best_val_rank_ic = float('-inf')
        best_epoch = 0
        no_improve = 0
        best_state = None  # by early-stop loss
        loss_best_epoch = 0
        loss_best_ric = float('nan')
        best_ric_state = None
        best_ric_epoch = 0
        history = []

        for epoch in range(1, max_epochs + 1):
            tr_loss, _, _ = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc, va_mcc = _eval_one_epoch(model, val_loader, criterion, device)
            va_ric = _eval_val_rank_ic(model, val_loader, val_meta, device)
            if scheduler is not None:
                scheduler.step(va_loss)
            history.append({
                'seed': ens_seed_base + ens_idx,
                'epoch': epoch,
                'train_loss': tr_loss,
                'val_loss': va_loss,
                'val_mcc': va_mcc,
                'val_rank_ic': va_ric,
            })
            logger.info(
                f"trial lr={lr:g} dp={dropout} pat={early_stop_patience} "
                f"epoch={epoch} va_loss={va_loss:.4f} va_mcc={va_mcc:.4f} "
                f"va_rank_ic={va_ric:.4f}"
            )

            if va_loss < best_val_loss - 1e-6:
                best_val_loss = va_loss
                loss_best_epoch = epoch
                loss_best_ric = va_ric
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    break
            if va_mcc > best_val_mcc:
                best_val_mcc = va_mcc
            if training == 'legacy' and np.isfinite(va_ric) and va_ric > best_val_rank_ic:
                best_val_rank_ic = va_ric
                best_ric_epoch = epoch
                best_ric_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if training == 'clean':
            best_val_rank_ic = loss_best_ric
            save_state, save_epoch = best_state, loss_best_epoch
        else:
            # Prefer Rank-IC-best weights for HP search; fall back to loss-best
            save_state = best_ric_state if best_ric_state is not None else best_state
            save_epoch = best_ric_epoch if best_ric_state is not None else epoch
        if save_state is not None:
            # if we saved ric-best, recompute loss/mcc on that state for logging
            if training == 'legacy' and best_ric_state is not None:
                model.load_state_dict(best_ric_state)
                model.to(device)
                best_val_loss, _, best_val_mcc = _eval_one_epoch(
                    model, val_loader, criterion, device)
            torch.save({
                'epoch': save_epoch,
                'model_state_dict': save_state,
                'val_loss': best_val_loss,
                'val_mcc': best_val_mcc,
                'val_rank_ic': best_val_rank_ic if np.isfinite(best_val_rank_ic) else float('nan'),
                'window_size': ws,
                'predict_window': pw,
                'img_h': img_h,
                'img_w': img_w,
                'dropout': dropout,
                'search_metric': search_metric,
                'channels': list(channels) if channels is not None else None,
                'training': training,
                'seed': ens_seed_base + ens_idx,
            }, ckpt_path)
        if save_history or training == 'clean':
            pd.DataFrame(history).to_csv(
                os.path.join(exp_dir, f'history_seed{ens_seed_base + ens_idx}.csv'), index=False)

        trial_best_mcc = max(trial_best_mcc, best_val_mcc)
        if best_val_loss < trial_best_loss:
            trial_best_loss = best_val_loss
        if np.isfinite(best_val_rank_ic):
            trial_best_rank_ic = max(trial_best_rank_ic, best_val_rank_ic)
            seed_rics.append(float(best_val_rank_ic))
        trial_best_epoch = max(trial_best_epoch, save_epoch if save_state is not None else 0)

    if training == 'clean':
        trial_best_rank_ic = float(np.mean(seed_rics)) if seed_rics else float('nan')
    if not np.isfinite(trial_best_rank_ic) or trial_best_rank_ic == float('-inf'):
        trial_best_rank_ic = float('nan')

    return {
        'lr': lr,
        'dropout': dropout,
        'patience': early_stop_patience,
        'best_val_mcc': trial_best_mcc,
        'best_val_loss': trial_best_loss,
        'best_val_rank_ic': trial_best_rank_ic,
        'best_epoch': trial_best_epoch,
        'exp_dir': exp_dir,
        'wall_time_sec': time.time() - t_start,
        'dir_suffix': dir_suffix or '',
        'search_metric': search_metric,
        'training': training,
        'train_up_share': is_dataset.up_share,
        'val_up_share': val_dataset.up_share,
    }


def _generate_oos_predictions(exp_dir, ws, pw, img_h, img_w,
                               freq, has_vb, ma_lags, n_ensembles,
                               oos_years, batch_size, device, country,
                               dropout, model_name, dataset_dir=None, channels=None,
                               sample_mode: str = 'legacy'):
    """加载 N 个 checkpoint，对 OOS 年做集成预测，保存 ensem5_res_{year}.csv

    sample_mode='oos' 时不按标签过滤，样本键与 labels 文件（universe）一致；另存每个 seed 的 up_prob。
    """
    ensem_dir = os.path.join(exp_dir, 'ensem_res')
    os.makedirs(ensem_dir, exist_ok=True)

    # 加载所有 checkpoint
    models = []
    for ens_idx in range(n_ensembles):
        ckpt_path = os.path.join(exp_dir, f'checkpoint{ens_idx}.pth.tar')
        if not os.path.exists(ckpt_path):
            logger.warning(f"找不到 checkpoint {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        m = init_model(model_name, img_h=img_h, img_w=img_w,
                        dropout=0.0, n_classes=2, window_size=ws,
                        channels=channels).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models.append(m)

    if not models:
        logger.error("无可用 checkpoint，跳过 OOS 预测。")
        return

    for year in oos_years:
        out_path = os.path.join(ensem_dir, f'ensem{n_ensembles}_res_{year}.csv')
        if os.path.exists(out_path):
            logger.info(f"OOS 预测已存在: {out_path}，跳过。")
            continue

        try:
            oos_dataset = _dataset([year], ws, pw, freq, has_vb, ma_lags, country, dataset_dir, sample_mode)
        except RuntimeError as e:
            logger.warning(f"OOS 数据加载失败 ({year}): {e}")
            continue

        if len(oos_dataset) == 0:
            logger.warning(f"{year} 年 OOS 数据为空，跳过。")
            continue

        oos_loader = _loader(oos_dataset, max(batch_size, 1024), shuffle=False)

        # 集成预测：取 up_prob 平均
        all_up = []
        all_down = []
        for m in models:
            up, down = _predict(m, oos_loader, device)
            all_up.append(up)
            all_down.append(down)

        avg_up   = np.mean(all_up, axis=0)
        avg_down = np.mean(all_down, axis=0)

        meta = oos_dataset.meta.copy()
        meta['up_prob']   = avg_up
        meta['down_prob'] = avg_down
        if sample_mode == 'oos':
            for i, up in enumerate(all_up):
                meta[f'up_prob_m{i}'] = up

        meta.to_csv(out_path, index=False)
        logger.info(f"OOS 预测已保存: {out_path}，样本数: {len(meta)}")


# ─────────────────────────────────────────────
# Experiment 类（封装单次实验，可选）
# ─────────────────────────────────────────────
class Experiment:
    """
    单次实验封装，方便从脚本或 notebook 调用。

    用法::

        exp = Experiment(ws=20, pw=20)
        exp.train()
        exp.predict_oos()
    """

    def __init__(self,
                 ws: int = 20,
                 pw: int = 20,
                 freq: str = FREQ,
                 ma_lags: list = None,
                 has_vb: bool = VOLUME_BAR,
                 lr: float = LEARNING_RATE,
                 dropout: float = DROPOUT_RATE,
                 n_ensembles: int = N_ENSEMBLES,
                 batch_size: int = BATCH_SIZE,
                 max_epochs: int = MAX_EPOCHS,
                 early_stop_patience: int = EARLY_STOP_PATIENCE,
                 is_years: list = None,
                 oos_years: list = None,
                 country: str = COUNTRY):
        self.ws = ws
        self.pw = pw
        self.params = dict(
            ws_list=[ws], pw_list=[pw], freq=freq,
            ma_lags=ma_lags, has_vb=has_vb, lr=lr, dropout=dropout,
            n_ensembles=n_ensembles, batch_size=batch_size,
            max_epochs=max_epochs, early_stop_patience=early_stop_patience,
            is_years=is_years, oos_years=oos_years, country=country
        )

    def train(self):
        train_us_model(**self.params)

    def predict_oos(self):
        """仅执行 OOS 预测（假设 checkpoint 已存在）"""
        train_us_model(**{**self.params, 'is_years': []})
