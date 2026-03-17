# -*- coding: utf-8 -*-
"""
Model/cnn_model.py - CNN 模型定义

支持：
  - CNNModel (2D CNN)：输入 (B, 1, H, W)，输出 2 类概率
  - CNN1DModel (1D CNN)：输入 (B, C_in, T)，输出 2 类概率

模型名称约定（来自文档）：
  D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C64
  → 20日窗口，3层卷积，各层 filter/stride/dilation/maxpool 序列，64通道

公开接口：
  - get_full_model_name(window_size, config_str)
  - init_model(model_name, img_h, img_w, dropout, n_classes)
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from Misc.config import BENCHMARK_MODEL_NAME, DROPOUT_RATE


# ─────────────────────────────────────────────
# 2D CNN（图像输入）
# ─────────────────────────────────────────────
class CNNModel(nn.Module):
    """
    2D CNN 二分类模型。

    架构（对应文档模型名）：
      3 × [Conv2d + BatchNorm + LeakyReLU + MaxPool2d]
      → Flatten + Dropout + Linear → 2 类 logits

    输入：(B, 1, H, W)  例如 (B, 1, 64, 60) for 20 日 K 线图
    输出：(B, 2) logits
    """

    def __init__(self,
                 img_h: int = 64,
                 img_w: int = 60,
                 dropout: float = 0.5,
                 n_classes: int = 2):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w

        # ── 第 1 层卷积块 ──
        # filter=5x3, stride=3x1, dilation=2x1, maxpool=2x1
        self.conv1 = nn.Conv2d(1, 64,
                               kernel_size=(5, 3),
                               stride=(3, 1),
                               dilation=(2, 1),
                               padding=(4, 1))
        self.bn1   = nn.BatchNorm2d(64)
        self.mp1   = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # ── 第 2 层卷积块 ──
        # filter=5x3, stride=1x1, dilation=1x1, maxpool=2x1
        self.conv2 = nn.Conv2d(64, 128,
                               kernel_size=(5, 3),
                               stride=(1, 1),
                               dilation=(1, 1),
                               padding=(2, 1))
        self.bn2   = nn.BatchNorm2d(128)
        self.mp2   = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # ── 第 3 层卷积块 ──
        # filter=5x3, stride=1x1, dilation=1x1, maxpool=2x1
        self.conv3 = nn.Conv2d(128, 64,
                               kernel_size=(5, 3),
                               stride=(1, 1),
                               dilation=(1, 1),
                               padding=(2, 1))
        self.bn3   = nn.BatchNorm2d(64)
        self.mp3   = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # 计算 flatten 后的维度
        self._fc_in = self._calc_fc_in(img_h, img_w)

        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(self._fc_in, n_classes)

    def _calc_fc_in(self, h: int, w: int) -> int:
        """前向推导卷积输出的 flatten 维度"""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            x = self.mp1(F.leaky_relu(self.bn1(self.conv1(dummy))))
            x = self.mp2(F.leaky_relu(self.bn2(self.conv2(x))))
            x = self.mp3(F.leaky_relu(self.bn3(self.conv3(x))))
        return int(x.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mp1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.mp2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.mp3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回 softmax 概率 (B, 2)"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ─────────────────────────────────────────────
# 1D CNN（时间序列输入）
# ─────────────────────────────────────────────
class CNN1DModel(nn.Module):
    """
    1D CNN 二分类模型。

    输入：(B, C_in, T)  例如 (B, 6, 20)
    输出：(B, 2) logits
    """

    def __init__(self,
                 in_channels: int = 6,
                 seq_len: int = 20,
                 dropout: float = 0.5,
                 n_classes: int = 2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.mp1   = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.mp2   = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)

        fc_in = 64 * max(1, seq_len // 4)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(fc_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mp1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.mp2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────
# 通用 Model 别名（默认 2D CNN）
Model = CNNModel


def get_full_model_name(window_size: int = 20,
                        config_str: str = BENCHMARK_MODEL_NAME) -> str:
    """
    返回完整模型名称字符串（用于实验目录命名）。

    Args:
        window_size: 输入窗口天数
        config_str: 模型架构描述字符串

    Returns:
        如 'D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C64'
    """
    return config_str


def init_model(model_name: str = BENCHMARK_MODEL_NAME,
               img_h: int = 64,
               img_w: int = 60,
               dropout: float = DROPOUT_RATE,
               n_classes: int = 2,
               model_type: str = '2d') -> nn.Module:
    """
    根据名称和参数初始化模型。

    Args:
        model_name: 模型名称（目前仅用于记录，不解析架构）
        img_h, img_w: 图像高度和宽度（2D CNN）
        dropout: Dropout 率
        n_classes: 分类数量
        model_type: '2d' | '1d'

    Returns:
        初始化好的 nn.Module
    """
    if model_type == '1d':
        return CNN1DModel(in_channels=6, seq_len=img_w,
                          dropout=dropout, n_classes=n_classes)
    else:
        return CNNModel(img_h=img_h, img_w=img_w,
                        dropout=dropout, n_classes=n_classes)

