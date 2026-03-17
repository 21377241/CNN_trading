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
from Data.chart_dataset import EquityDataset
from Data.dgp_config import IMAGE_WIDTH, IMAGE_HEIGHT
from Model.cnn_model import init_model, get_full_model_name

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_exp_dir(model_name: str, window_size: int, predict_window: int,
                   lr: float, dropout: float, freq: str,
                   has_ma: bool, has_vb: bool) -> str:
    """
    构建实验目录路径（与文档命名约定一致）。

    示例：
      WORK_DIR/spx_model_res/<model_name>/<exp_name>/
    """
    lr_str = f"lr{lr:.0E}".replace('-0', '-').replace('+0', '')
    dp_str = f"dp{dropout:.2f}"
    ma_str = 'maTrue' if has_ma else 'maFalse'
    vb_str = 'vbTrue' if has_vb else 'vbFalse'
    exp_name = (f"{window_size}d{predict_window}p"
                f"-{lr_str}-{dp_str}-{ma_str}-{vb_str}"
                f"-{freq}Trained-noDelayedReturn")
    exp_dir = os.path.join(EXP_DIR, model_name, exp_name)
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
                   train_val_split: float = 0.7,
                   country: str = COUNTRY):
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
        is_years: 训练年份列表
        oos_years: 测试年份列表
        train_val_split: 训练/验证比例（按样本随机划分）
        country: 国家标识
    """
    ws_list  = ws_list  or [BENCHMARK_MODEL_WINDOW]
    pw_list  = pw_list  or [BENCHMARK_MODEL_PREDICT_WINDOW]
    ma_lags  = ma_lags  if ma_lags is not None else MA_LAGS
    is_years = is_years or IS_YEARS
    oos_years= oos_years or OOS_YEARS

    device = _get_device()
    logger.info(f"训练设备: {device}")

    setup_logger('train', os.path.join(LOG_DIR, 'train.log'))

    for ws, pw in zip(ws_list, pw_list):
        img_h = IMAGE_HEIGHT.get(ws, 64)
        img_w = IMAGE_WIDTH.get(ws, 60)
        model_name = get_full_model_name(ws)
        exp_dir = _build_exp_dir(model_name, ws, pw, lr, dropout, freq,
                                 len(ma_lags) > 0, has_vb)

        logger.info(f"\n{'='*60}")
        logger.info(f"开始实验: ws={ws}, pw={pw}, 实验目录: {exp_dir}")

        # ── 加载训练集 ──
        try:
            is_dataset = EquityDataset(
                years=is_years,
                window_size=ws,
                predict_window=pw,
                freq=freq,
                has_volume_bar=has_vb,
                ma_lags=ma_lags,
                country=country,
                image_scale='norm255'
            )
        except RuntimeError as e:
            logger.error(f"训练数据加载失败: {e}")
            continue

        n_total = len(is_dataset)
        n_train = int(n_total * train_val_split)
        n_val   = n_total - n_train

        logger.info(f"训练集总样本: {n_total}, 训练: {n_train}, 验证: {n_val}")

        # ── 训练 N 个集成模型 ──
        for ens_idx in range(n_ensembles):
            ckpt_path = os.path.join(exp_dir, f'checkpoint{ens_idx}.pth.tar')
            if os.path.exists(ckpt_path):
                logger.info(f"Ensemble {ens_idx}: checkpoint 已存在，跳过训练。")
                continue

            logger.info(f"\n--- 训练 Ensemble {ens_idx} ---")

            # 每次用不同随机划分（通过固定 seed + ens_idx）
            torch.manual_seed(42 + ens_idx)
            np.random.seed(42 + ens_idx)

            train_set, val_set = random_split(is_dataset, [n_train, n_val])
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=0, pin_memory=False)
            val_loader   = DataLoader(val_set, batch_size=batch_size,
                                      shuffle=False, num_workers=0, pin_memory=False)

            model = init_model(model_name, img_h=img_h, img_w=img_w,
                                dropout=dropout, n_classes=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=1, factor=0.5
            )

            best_val_loss = float('inf')
            no_improve = 0
            best_state = None

            for epoch in range(1, max_epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc, tr_mcc = _train_one_epoch(
                    model, train_loader, optimizer, criterion, device)
                va_loss, va_acc, va_mcc = _eval_one_epoch(
                    model, val_loader, criterion, device)
                scheduler.step(va_loss)

                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} tr_mcc={tr_mcc:.3f} | "
                    f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} va_mcc={va_mcc:.3f} | "
                    f"{elapsed:.1f}s"
                )

                if va_loss < best_val_loss - 1e-6:
                    best_val_loss = va_loss
                    no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        logger.info(f"早停（{early_stop_patience} 轮无改善）")
                        break

            # 保存 checkpoint
            if best_state is not None:
                torch.save({'epoch': epoch,
                            'model_state_dict': best_state,
                            'val_loss': best_val_loss,
                            'window_size': ws,
                            'predict_window': pw,
                            'img_h': img_h,
                            'img_w': img_w,
                            'dropout': dropout},
                           ckpt_path)
                logger.info(f"Checkpoint 保存至: {ckpt_path}")

        # ── OOS 预测 ──
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
            model_name=model_name
        )


def _generate_oos_predictions(exp_dir, ws, pw, img_h, img_w,
                               freq, has_vb, ma_lags, n_ensembles,
                               oos_years, batch_size, device, country,
                               dropout, model_name):
    """加载 N 个 checkpoint，对 OOS 年做集成预测，保存 ensem5_res_{year}.csv"""
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
                        dropout=0.0, n_classes=2).to(device)
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
            oos_dataset = EquityDataset(
                years=[year],
                window_size=ws,
                predict_window=pw,
                freq=freq,
                has_volume_bar=has_vb,
                ma_lags=ma_lags,
                country=country,
                image_scale='norm255'
            )
        except RuntimeError as e:
            logger.warning(f"OOS 数据加载失败 ({year}): {e}")
            continue

        if len(oos_dataset) == 0:
            logger.warning(f"{year} 年 OOS 数据为空，跳过。")
            continue

        oos_loader = DataLoader(oos_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

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

