# -*- coding: utf-8 -*-
"""Train E2C 1D-CNN temporal benchmark on matched 120D features."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.metrics import cross_section_ic
from experiments.common.splits import chrono_train_val_indices

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('e2c_train_1dcnn')


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 window_size: int, per_day: int):
        self.X = X
        self.y = y.astype(np.int64)
        self.idx = np.asarray(idx, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.window_size = int(window_size)
        self.per_day = int(per_day)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        x = np.asarray(self.X[j], dtype=np.float32).reshape(self.window_size, self.per_day)
        x = (x - self.mean) / self.std
        x = np.transpose(x, (1, 0))  # (C, T)
        return torch.from_numpy(x), torch.tensor(self.y[j], dtype=torch.long)


class Conv1DTemporal(nn.Module):
    """20x6 -> Conv1D(6,32,k=3) -> ReLU -> Conv1D(32,64,k=3) -> GAP -> Linear."""

    def __init__(self, in_channels: int = 6, hidden_channels=(32, 64), kernel_size: int = 3, dropout: float = 0.25):
        super().__init__()
        h1, h2 = hidden_channels
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, h1, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(h1)
        self.conv2 = nn.Conv1d(h1, h2, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(h2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h2, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


@dataclass
class EvalResult:
    loss: float
    auc: float
    rank_ic: float


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs = []
    labels = []
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb.float())
            losses.append(float(loss.item()) * len(yb))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.cpu().numpy())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    loss = float(np.sum(losses) / max(1, len(y)))
    return p, y, loss


def evaluate(model: nn.Module, loader: DataLoader, meta: pd.DataFrame, device: torch.device) -> EvalResult:
    p, y, loss = predict(model, loader, device)
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = float('nan')
    tmp = meta.copy()
    tmp['up_prob'] = p
    ic = cross_section_ic(tmp, 'up_prob')
    rank_ic = float(ic['rank_ic'].mean()) if not ic.empty else float('nan')
    return EvalResult(loss=loss, auc=auc, rank_ic=rank_ic)


def fit_seed(cfg: dict, seed: int, X_is: np.ndarray, y_is: np.ndarray, meta_is: pd.DataFrame,
             tr_idx: np.ndarray, va_idx: np.ndarray, out_root: str):
    set_seed(seed)
    window_size = int(cfg['window_size'])
    per_day = int(cfg['per_day_features'])
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    device_name = str(train_cfg.get('device', 'auto')).lower()
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    X_train_seq = np.asarray(X_is[tr_idx], dtype=np.float32).reshape(len(tr_idx), window_size, per_day)
    mean = X_train_seq.mean(axis=(0, 1), keepdims=False)
    std = X_train_seq.std(axis=(0, 1), keepdims=False) + 1e-6

    train_ds = SeqDataset(X_is, y_is, tr_idx, mean, std, window_size, per_day)
    val_ds = SeqDataset(X_is, y_is, va_idx, mean, std, window_size, per_day)
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg['batch_size']), shuffle=True,
                              num_workers=int(train_cfg.get('num_workers', 0)), pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg['batch_size']) * 2, shuffle=False,
                            num_workers=int(train_cfg.get('num_workers', 0)), pin_memory=(device.type == 'cuda'))

    model = Conv1DTemporal(
        in_channels=per_day,
        hidden_channels=tuple(model_cfg.get('hidden_channels', [32, 64])),
        kernel_size=int(model_cfg.get('kernel_size', 3)),
        dropout=float(model_cfg.get('dropout', 0.25)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get('learning_rate', 1e-3)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)),
    )

    max_epochs = int(train_cfg.get('max_epochs', 80))
    patience = int(train_cfg.get('patience', 10))
    monitor = str(train_cfg.get('monitor', 'rank_ic'))
    best_score = -np.inf
    best_epoch = -1
    bad = 0
    history = []
    seed_dir = os.path.join(out_root, 'models', f'seed{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_path = os.path.join(seed_dir, 'best.pt')

    logger.info(f'seed={seed} device={device} train={len(tr_idx):,} val={len(va_idx):,}')
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb.float())
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

        val = evaluate(model, val_loader, meta_is.iloc[va_idx].reset_index(drop=True), device)
        score = val.rank_ic if monitor == 'rank_ic' else val.auc
        row = {
            'epoch': epoch,
            'train_loss': total_loss / max(1, n),
            'val_loss': val.loss,
            'val_auc': val.auc,
            'val_rank_ic': val.rank_ic,
        }
        history.append(row)
        logger.info(
            f'seed={seed} epoch={epoch:03d} train_loss={row["train_loss"]:.5f} '
            f'val_auc={val.auc:.4f} val_rank_ic={val.rank_ic:.5f}'
        )
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_epoch = epoch
            bad = 0
            torch.save({'model': model.state_dict(), 'mean': mean, 'std': std, 'cfg': cfg, 'seed': seed}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                logger.info(f'seed={seed} early stop at epoch={epoch}, best_epoch={best_epoch}')
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(seed_dir, 'history.csv'), index=False)
    info = {'seed': seed, 'best_epoch': best_epoch, 'best_score': best_score, 'monitor': monitor}
    with open(os.path.join(seed_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return info


def predict_oos_seed(cfg: dict, seed: int, X_oos: np.ndarray, meta_oos: pd.DataFrame, out_root: str):
    window_size = int(cfg['window_size'])
    per_day = int(cfg['per_day_features'])
    train_cfg = cfg['train']
    device_name = str(train_cfg.get('device', 'auto')).lower()
    device = torch.device('cuda' if (device_name == 'auto' and torch.cuda.is_available()) else ('cpu' if device_name == 'auto' else device_name))
    ckpt_path = os.path.join(out_root, 'models', f'seed{seed}', 'best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = cfg['model']
    model = Conv1DTemporal(
        in_channels=per_day,
        hidden_channels=tuple(model_cfg.get('hidden_channels', [32, 64])),
        kernel_size=int(model_cfg.get('kernel_size', 3)),
        dropout=float(model_cfg.get('dropout', 0.25)),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    ds = SeqDataset(
        X_oos,
        meta_oos['binary_label'].values.astype(int),
        np.arange(len(meta_oos)),
        ckpt['mean'],
        ckpt['std'],
        window_size,
        per_day,
    )
    loader = DataLoader(ds, batch_size=int(train_cfg['batch_size']) * 4, shuffle=False,
                        num_workers=int(train_cfg.get('num_workers', 0)), pin_memory=(device.type == 'cuda'))
    p, _, _ = predict(model, loader, device)
    return p


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config_1dcnn_120.yaml'))
    parser.add_argument('--seeds', default='', help='optional comma-separated seeds overriding config')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out_root = os.path.join(PROJECT_ROOT, cfg['paths']['out_root'].replace('/', os.sep))
    feat_dir = os.path.join(PROJECT_ROOT, cfg['paths']['features_root'].replace('/', os.sep))
    os.makedirs(os.path.join(out_root, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'preds'), exist_ok=True)

    X_is = np.load(os.path.join(feat_dir, 'X_is.npy'), mmap_mode='r')
    X_oos = np.load(os.path.join(feat_dir, 'X_oos.npy'), mmap_mode='r')
    meta_is = pd.read_feather(os.path.join(feat_dir, 'meta_is.feather'))
    meta_oos = pd.read_feather(os.path.join(feat_dir, 'meta_oos.feather'))
    meta_is['ending_date'] = pd.to_datetime(meta_is['ending_date'])
    meta_oos['ending_date'] = pd.to_datetime(meta_oos['ending_date'])
    y_is = meta_is['binary_label'].values.astype(int)
    valid = np.isin(y_is, [0, 1])
    if (~valid).sum():
        logger.info(f'drop invalid IS labels: {(~valid).sum():,}')
        # mmap advanced indexing creates in-memory copy, acceptable for 431k x 120
        X_is = np.asarray(X_is[valid], dtype=np.float32)
        meta_is = meta_is.loc[valid].reset_index(drop=True)
        y_is = y_is[valid]

    tr_idx, va_idx = chrono_train_val_indices(meta_is, cfg['chrono_train_years'], cfg['chrono_val_years'])
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()] if args.seeds else [int(s) for s in cfg['seeds']]

    infos = []
    if not args.predict_only:
        for seed in seeds:
            infos.append(fit_seed(cfg, seed, X_is, y_is, meta_is, tr_idx, va_idx, out_root))
        with open(os.path.join(out_root, 'models', 'train_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'infos': infos, 'config': cfg}, f, indent=2)

    pred_cols = []
    for seed in seeds:
        p = predict_oos_seed(cfg, seed, X_oos, meta_oos, out_root)
        pred_cols.append(p)
        out = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
        out['up_prob'] = p
        out_path = os.path.join(out_root, 'preds', f'pred_1dcnn120_seed{seed}_oos.csv')
        out.to_csv(out_path, index=False)
        logger.info(f'wrote {out_path} rows={len(out):,}')

    p_bag = np.mean(np.vstack(pred_cols), axis=0)
    out = meta_oos[['StockID', 'ending_date', 'ret_val', 'MarketCap', 'binary_label']].copy()
    out['up_prob'] = p_bag
    out_path = os.path.join(out_root, 'preds', 'pred_1dcnn120_oos.csv')
    out.to_csv(out_path, index=False)
    logger.info(f'wrote {out_path} rows={len(out):,} seeds={seeds}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

