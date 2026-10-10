# -*- coding: utf-8 -*-
"""
Model/cnn_model.py - CNN 模型定义（对齐 Jiang et al. JF 2023）

论文 Appendix：
  - 卷积核 5×3，MaxPool 2×1，LeakyReLU，BatchNorm，FC Dropout 50%
  - 5 日图：2 个 block，通道 64→128，第 1 层 stride=1 / dilation=1
  - 20 日图：3 个 block，通道 64→128→256，第 1 层 stride=3 / dilation=2
  - 60 日图：4 个 block，通道 64→128→256→512（预留）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Misc.config import BENCHMARK_MODEL_NAME, DROPOUT_RATE


def _xavier_init(module: nn.Module):
    """Xavier 初始化（论文采用 Glorot & Bengio 2010）"""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CNNModel(nn.Module):
    """
    2D CNN 二分类（按 window_size 选择论文对应层数）。

    输入：(B, 1, H, W)
    输出：(B, 2) logits
    """

    def __init__(self,
                 img_h: int = 64,
                 img_w: int = 60,
                 dropout: float = 0.5,
                 n_classes: int = 2,
                 window_size: int = 20,
                 channels: list[int] | tuple[int, ...] | None = None):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.window_size = window_size

        # 论文：第 1 层竖直 stride / dilation 随窗口变化
        if window_size <= 5:
            n_blocks = 2
            default_channels = [64, 128]
            s1, d1, p1 = 1, 1, (2, 1)   # stride=1, dilation=1
        elif window_size <= 20:
            n_blocks = 3
            default_channels = [64, 128, 256]
            # dilation=2 时 padding=7 使 64×60 输入 → FC=46080（对齐论文附录）
            s1, d1, p1 = 3, 2, (7, 1)
        else:
            n_blocks = 4
            default_channels = [64, 128, 256, 512]
            s1, d1, p1 = 3, 3, (6, 1)

        if channels is None:
            channels = default_channels
        else:
            if len(channels) != n_blocks or any(int(c) <= 0 for c in channels):
                raise ValueError(
                    f'channels must contain {n_blocks} positive widths for window_size={window_size}; '
                    f'got {channels}'
                )
            channels = [int(c) for c in channels]

        self.n_blocks = n_blocks
        self.channels = tuple(channels)
        layers = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(channels):
            if i == 0:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=(5, 3),
                                 stride=(s1, 1), dilation=(d1, 1), padding=p1)
            else:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=(5, 3),
                                 stride=(1, 1), dilation=(1, 1), padding=(2, 1))
            block = nn.ModuleDict({
                'conv': conv,
                'bn': nn.BatchNorm2d(out_ch),
                'mp': nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            })
            layers.append(block)
            in_ch = out_ch
        self.blocks = layers

        self._fc_in = self._calc_fc_in(img_h, img_w)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self._fc_in, n_classes)
        _xavier_init(self)

    def parameter_breakdown(self) -> dict:
        """Trainable parameters by architectural component for capacity controls."""
        conv = sum(p.numel() for block in self.blocks for p in block['conv'].parameters())
        batch_norm = sum(p.numel() for block in self.blocks for p in block['bn'].parameters())
        head = sum(p.numel() for p in self.fc.parameters())
        return {
            'conv': int(conv),
            'batch_norm': int(batch_norm),
            'head': int(head),
            'total': int(conv + batch_norm + head),
        }

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block['mp'](F.leaky_relu(block['bn'](block['conv'](x))))
        return x

    def _calc_fc_in(self, h: int, w: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            x = self._forward_blocks(dummy)
        return int(x.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


class CNN1DModel(nn.Module):
    """1D CNN 二分类（辅助，非论文主实验）"""

    def __init__(self, in_channels: int = 6, seq_len: int = 20,
                 dropout: float = 0.5, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.mp2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        fc_in = 64 * max(1, seq_len // 4)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(fc_in, n_classes)
        _xavier_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mp1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.mp2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


Model = CNNModel

# 论文附录模型名
MODEL_NAME_I5 = 'D5L2F53S11D11MP21F53S11D11MP21C128'
MODEL_NAME_I20 = 'D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C256'


def get_full_model_name(window_size: int = 20,
                        config_str: str = None) -> str:
    """按窗口返回论文对齐的模型名。"""
    if config_str is not None and config_str not in (
            BENCHMARK_MODEL_NAME, MODEL_NAME_I20, MODEL_NAME_I5, ''):
        return config_str
    if window_size <= 5:
        return MODEL_NAME_I5
    return MODEL_NAME_I20


def init_model(model_name: str = BENCHMARK_MODEL_NAME,
               img_h: int = 64,
               img_w: int = 60,
               dropout: float = DROPOUT_RATE,
               n_classes: int = 2,
               model_type: str = '2d',
               window_size: int = None,
               channels: list[int] | tuple[int, ...] | None = None) -> nn.Module:
    """初始化模型。window_size 决定层数/通道。"""
    if window_size is None:
        if isinstance(model_name, str) and model_name.startswith('D5L'):
            window_size = 5
        elif isinstance(model_name, str) and model_name.startswith('D60'):
            window_size = 60
        else:
            window_size = 20

    if model_type == '1d':
        return CNN1DModel(in_channels=6, seq_len=img_w,
                          dropout=dropout, n_classes=n_classes)
    return CNNModel(img_h=img_h, img_w=img_w, dropout=dropout,
                    n_classes=n_classes, window_size=window_size, channels=channels)
