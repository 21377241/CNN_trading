# -*- coding: utf-8 -*-
"""
5_predict_with_model_res_0.py  ―  使用 model_res_0/ 预训练模型进行预测与评估

模型兼容性分析：
  ┌───────────────────────┬──────────────────────┬──────────────────────┐
  │ 项目                  │ model_res_0 (外部)   │ 本项目 CNNModel      │
  ├───────────────────────┼──────────────────────┼──────────────────────┤
  │ 图像尺寸              │ (64, 60) ← 推导得    │ (64, 60) ✓ 一致     │
  │ Conv1 通道            │ 64                   │ 64                   │
  │ Conv2 通道            │ 128                  │ 128                  │
  │ Conv3 通道            │ 256                  │ 64 (不同)            │
  │ Conv1 dilation        │ (1,1) 无膨胀         │ (2,1) (不同)         │
  │ Conv1 padding         │ (5,1)                │ (4,1) (不同)         │
  │ 状态字典命名          │ conv_layers.{i}.{j}  │ conv1/bn1/mp1...     │
  │ fc 输入维度           │ 46080                │ 7680 (不同)          │
  │ checkpoint keys       │ loss/accy/MCC/epoch  │ val_loss/img_h/img_w │
  └───────────────────────┴──────────────────────┴──────────────────────┘

  → 图像数据完全兼容（step2 生成的 .dat 文件可直接使用）
  → 需要专用 LegacyCNNModel 类加载权重（不能用 CNNModel）

用法：
  python 5_predict_with_model_res_0.py                # 预测所有可用 OOS 年份
  python 5_predict_with_model_res_0.py --years 2020 2021 2022
  python 5_predict_with_model_res_0.py --ckpt-dir WORK_SPACE/model_res_0
"""
import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_PKG = os.path.join(PROJECT_ROOT, 'trend_submit', 'trend_code_submit')
if CODE_PKG not in sys.path:
    sys.path.insert(0, CODE_PKG)

from Misc.config import (
    IS_YEARS, OOS_YEARS, N_ENSEMBLES, PORTFOLIO_DIR, COUNTRY, FREQ,
    BENCHMARK_MODEL_WINDOW, BENCHMARK_MODEL_PREDICT_WINDOW,
    MA_LAGS, VOLUME_BAR, EXP_DIR
)
from Portfolio.portfolio import run_portfolio_analysis

# 全体数据年份（IS + OOS，用作默认测试集）
ALL_YEARS = sorted(IS_YEARS + OOS_YEARS)
from Data.chart_dataset import EquityDataset
from Data.dgp_config import IMAGE_WIDTH, IMAGE_HEIGHT

# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── 路径 ──
CKPT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'model_res_0')
OUTPUT_DIR       = os.path.join(PROJECT_ROOT, 'WORK_SPACE', 'model_res_0')

# ── 图像参数（20日窗口标准尺寸）──
IMG_H = IMAGE_HEIGHT.get(BENCHMARK_MODEL_WINDOW, 64)   # = 64
IMG_W = IMAGE_WIDTH.get(BENCHMARK_MODEL_WINDOW, 60)    # = 60


# ══════════════════════════════════════════════
# 1. 兼容外部 checkpoint 的模型类
#    state_dict 键：conv_layers.{0,1,2}.{0=Conv,1=BN}
#    架构：conv3 输出 256 通道，conv1 无 dilation，padding=(5,1)
# ══════════════════════════════════════════════
class LegacyCNNModel(nn.Module):
    """
    与 model_res_0 checkpoint 兼容的 CNN 模型。

    架构：
      conv_layers[0]: Conv2d(1→64,   k=(5,3), s=(3,1), p=(5,1)) + BN + LeakyReLU + MaxPool(2,1)
      conv_layers[1]: Conv2d(64→128, k=(5,3), s=(1,1), p=(2,1)) + BN + LeakyReLU + MaxPool(2,1)
      conv_layers[2]: Conv2d(128→256,k=(5,3), s=(1,1), p=(2,1)) + BN + LeakyReLU + MaxPool(2,1)
      fc: Linear(46080, 2)

    输入：(B, 1, 64, 60)  ← 与本项目 step2 生成的图像一致
    输出：(B, 2) logits
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), padding=(5, 1)),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                nn.BatchNorm2d(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                nn.BatchNorm2d(256)
            ),
        ])
        self.pool    = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(p=dropout)

        # 推导 fc 输入维度（应为 46080）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMG_H, IMG_W)
            fc_in = self._forward_conv(dummy).numel()
        logger.debug(f"LegacyCNNModel fc_in={fc_in} (期望 46080)")
        assert fc_in == 46080, (
            f"fc_in={fc_in} 与 checkpoint 不符（期望 46080）。"
            f"请检查 IMG_H={IMG_H}, IMG_W={IMG_W} 是否正确。"
        )

        self.fc = nn.Linear(fc_in, 2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.conv_layers:
            x = self.pool(F.leaky_relu(blk(x)))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回 softmax 概率 (B, 2)"""
        return F.softmax(self.forward(x), dim=1)


# ══════════════════════════════════════════════
# 2. 加载 checkpoint
# ══════════════════════════════════════════════
def load_ensemble(ckpt_dir: str,
                  n_ensembles: int,
                  device: torch.device,
                  dropout: float = 0.0) -> list:
    """
    加载 n_ensembles 个 LegacyCNNModel checkpoint。
    推理时 dropout=0.0（eval 模式已禁用 dropout，但保留参数一致性）。

    Returns:
        list of nn.Module（已设为 eval 模式，移至 device）
    """
    models = []
    for i in range(n_ensembles):
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint{i}.pth.tar')
        if not os.path.exists(ckpt_path):
            logger.warning(f"  Checkpoint {i} 不存在: {ckpt_path}，跳过")
            continue

        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # 兼容 model_res_0 格式（keys: loss/accy/MCC/epoch/diff/model_state_dict）
        state_dict = ck.get('model_state_dict', ck)

        model = LegacyCNNModel(dropout=dropout).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        meta_keys = {k: v for k, v in ck.items() if k != 'model_state_dict'}
        logger.info(f"  ✓ Ensemble {i}: "
                    f"epoch={meta_keys.get('epoch','?')}, "
                    f"loss={meta_keys.get('loss', meta_keys.get('val_loss','?')):.4f}, "
                    f"accy={meta_keys.get('accy','?')}, "
                    f"MCC={meta_keys.get('MCC','?')}")
        models.append(model)

    return models


# ══════════════════════════════════════════════
# 3. 推理：集成预测
# ══════════════════════════════════════════════
@torch.no_grad()
def predict_year(models: list,
                 year: int,
                 batch_size: int,
                 device: torch.device) -> pd.DataFrame | None:
    """
    对指定年份的 OOS 数据集运行集成推理。

    Returns:
        DataFrame 含列：date, StockID, up_prob, down_prob, binary_pred, ret_val
        若该年数据不存在返回 None
    """
    try:
        dataset = EquityDataset(
            years=[year],
            window_size=BENCHMARK_MODEL_WINDOW,
            predict_window=BENCHMARK_MODEL_PREDICT_WINDOW,
            freq=FREQ,
            has_volume_bar=VOLUME_BAR,
            ma_lags=MA_LAGS,
            country=COUNTRY,
            image_scale='norm255',
        )
    except RuntimeError as e:
        logger.warning(f"  年份 {year} 数据加载失败: {e}")
        return None

    if len(dataset) == 0:
        logger.warning(f"  年份 {year}: 数据集为空")
        return None

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    all_up   = []
    all_down = []

    for imgs, labels in tqdm(loader, desc=f"  预测 {year}", leave=False):
        imgs = imgs.to(device)
        batch_up   = []
        batch_down = []
        for model in models:
            proba = model.predict_proba(imgs)   # (B, 2)
            batch_up.append(proba[:, 1].cpu().numpy())
            batch_down.append(proba[:, 0].cpu().numpy())

        # 集成平均
        all_up.append(np.mean(batch_up, axis=0))
        all_down.append(np.mean(batch_down, axis=0))

    up_probs   = np.concatenate(all_up)
    down_probs = np.concatenate(all_down)

    meta = dataset._meta.copy()
    meta['up_prob']    = up_probs
    meta['down_prob']  = down_probs
    meta['binary_pred'] = (up_probs > 0.5).astype(int)

    return meta


# ══════════════════════════════════════════════
# 4. 评估指标
# ══════════════════════════════════════════════
def evaluate(df: pd.DataFrame) -> dict:
    """
    计算预测性能指标：
      - 方向准确率 (Accuracy)
      - MCC (Matthews Correlation Coefficient)
      - 各十分位平均收益
      - H-L 收益差
    """
    from sklearn.metrics import matthews_corrcoef

    label_col = 'binary_label'
    ret_col   = 'ret_val'

    stats = {}

    if label_col in df.columns:
        valid = df[[label_col, 'binary_pred', ret_col, 'up_prob']].dropna()
        n     = len(valid)
        acc   = (valid[label_col] == valid['binary_pred']).mean()
        try:
            mcc = matthews_corrcoef(valid[label_col], valid['binary_pred'])
        except Exception:
            mcc = float('nan')

        stats['n_samples'] = n
        stats['accuracy']  = acc
        stats['MCC']       = mcc

        # 十分位分析（按 up_prob 分组）
        valid = valid.copy()
        valid['decile'] = pd.qcut(valid['up_prob'], q=10,
                                  labels=False, duplicates='drop')
        decile_ret = valid.groupby('decile')[ret_col].mean()
        stats['decile_returns'] = decile_ret

        # H-L 收益差（第10分位 - 第1分位）
        if len(decile_ret) >= 2:
            stats['hl_return'] = decile_ret.iloc[-1] - decile_ret.iloc[0]
        else:
            stats['hl_return'] = float('nan')

    return stats


def print_report(all_df: pd.DataFrame, output_dir: str):
    """打印并保存评估报告"""
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info("  评估报告（model_res_0 集成预测）")
    logger.info(f"{'='*60}")
    logger.info(f"  总样本数: {len(all_df)}")

    stats = evaluate(all_df)

    logger.info(f"  方向准确率 (Accuracy) : {stats.get('accuracy', float('nan')):.4f}")
    logger.info(f"  MCC                  : {stats.get('MCC', float('nan')):.4f}")
    logger.info(f"  H-L 收益差           : {stats.get('hl_return', float('nan')):.4f}")

    if 'decile_returns' in stats:
        logger.info("\n  十分位平均收益（1=最低 up_prob，10=最高 up_prob）：")
        for d, r in stats['decile_returns'].items():
            bar = '█' * int(abs(r) * 200) if not np.isnan(r) else ''
            sign = '+' if r >= 0 else '-'
            logger.info(f"    Decile {int(d)+1:2d}: {sign}{abs(r):.4f}  {bar}")

    # 按年份分解
    _date_col = 'date' if 'date' in all_df.columns else (
                'ending_date' if 'ending_date' in all_df.columns else None)
    if _date_col and 'year' not in all_df.columns:
        all_df['year'] = pd.to_datetime(all_df[_date_col]).dt.year
    if 'year' in all_df.columns:
        logger.info("\n  按年份统计：")
        logger.info(f"  {'Year':>6} {'N':>6} {'Acc':>7} {'MCC':>8} {'HL_Ret':>8}")
        for yr, grp in all_df.groupby('year'):  # type: ignore
            s = evaluate(grp)
            logger.info(f"  {yr:>6} {s.get('n_samples',0):>6} "
                        f"{s.get('accuracy',float('nan')):>7.4f} "
                        f"{s.get('MCC',float('nan')):>8.4f} "
                        f"{s.get('hl_return',float('nan')):>8.4f}")

    # 保存预测结果
    pred_path = os.path.join(output_dir, 'model_res_0_predictions.csv')
    all_df.to_csv(pred_path, index=False)
    logger.info(f"\n  预测结果已保存: {pred_path}")

    # 保存文字报告
    report_lines = [
        "model_res_0 集成预测评估报告",
        "=" * 50,
        f"总样本数  : {stats.get('n_samples', len(all_df))}",
        f"Accuracy  : {stats.get('accuracy', float('nan')):.4f}",
        f"MCC       : {stats.get('MCC', float('nan')):.4f}",
        f"H-L 收益  : {stats.get('hl_return', float('nan')):.4f}",
    ]
    report_path = os.path.join(output_dir, 'model_res_0_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"  报告已保存: {report_path}")


# ══════════════════════════════════════════════
# 5. Top-N 精选组合分析
# ══════════════════════════════════════════════
def run_topN_analysis(pred_csv: str,
                      output_dir: str,
                      top_ns: list = None) -> None:
    """
    按 up_prob 选取每期前 N / 后 N 支股票，计算：
      - 等权（EW）/ 市值加权（VW）月度收益
      - 年化收益、年化波动、Sharpe
      - Long-Short（Top N − Bottom N）

    Args:
        pred_csv  : 预测结果 CSV（model_res_0_predictions.csv）
        output_dir: 输出目录
        top_ns    : 要分析的 N 列表，默认 [5, 10, 20]
    """
    if top_ns is None:
        top_ns = [5, 10, 20]

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"  [TopN] 读取预测文件: {pred_csv}")
    df = pd.read_csv(pred_csv)
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    df = df.dropna(subset=['ret_val', 'up_prob'])

    n_periods = df['ending_date'].nunique()
    logger.info(f"  [TopN] 总样本: {len(df)} 条，期末日期: {n_periods} 个")

    ann_factor = 12   # 月度→年化

    # ── 辅助：计算统计量 ──────────────────
    def _stats(series: pd.Series) -> dict:
        s = series.dropna()
        if len(s) == 0:
            return dict(mean_monthly=np.nan, ann_return=np.nan,
                        ann_vol=np.nan, sharpe=np.nan, n=0)
        mean_m = float(s.mean())
        ann_r  = (1 + mean_m) ** ann_factor - 1
        ann_v  = float(s.std() * np.sqrt(ann_factor))
        sharpe = ann_r / ann_v if ann_v > 1e-10 else 0.0
        return dict(mean_monthly=mean_m, ann_return=ann_r,
                    ann_vol=ann_v, sharpe=sharpe, n=len(s))

    # ── 逐 N 计算 ────────────────────────
    all_series: dict = {}   # key → pd.Series(date→monthly_ret)
    grouped = list(df.groupby('ending_date'))

    for n in top_ns:
        ew_top, ew_bot, vw_top, vw_bot, dates = [], [], [], [], []

        for date, grp in grouped:
            grp = grp.dropna(subset=['ret_val', 'up_prob'])
            if len(grp) < n * 2:        # 至少有 2N 支股票才分析
                continue

            top = grp.nlargest(n, 'up_prob')
            bot = grp.nsmallest(n, 'up_prob')

            # ── 等权 ──
            ew_top.append(top['ret_val'].mean())
            ew_bot.append(bot['ret_val'].mean())

            # ── 市值加权 ──
            if 'MarketCap' in grp.columns:
                def _vw(sub):
                    w = sub['MarketCap'].clip(lower=0).values
                    return float(np.average(sub['ret_val'].values, weights=w)) \
                           if w.sum() > 0 else sub['ret_val'].mean()
                vw_top.append(_vw(top))
                vw_bot.append(_vw(bot))
            else:
                vw_top.append(ew_top[-1])
                vw_bot.append(ew_bot[-1])

            dates.append(date)

        idx = pd.DatetimeIndex(dates)
        s_ew_top = pd.Series(ew_top, index=idx, name=f'Top{n}_EW')
        s_ew_bot = pd.Series(ew_bot, index=idx, name=f'Bot{n}_EW')
        s_vw_top = pd.Series(vw_top, index=idx, name=f'Top{n}_VW')
        s_vw_bot = pd.Series(vw_bot, index=idx, name=f'Bot{n}_VW')

        all_series[f'Top{n}_EW']  = s_ew_top
        all_series[f'Bot{n}_EW']  = s_ew_bot
        all_series[f'Top{n}_VW']  = s_vw_top
        all_series[f'Bot{n}_VW']  = s_vw_bot
        all_series[f'LS{n}_EW']   = s_ew_top - s_ew_bot   # Long-Short EW
        all_series[f'LS{n}_VW']   = s_vw_top - s_vw_bot   # Long-Short VW

        logger.info(f"  [TopN] N={n:2d}: {len(dates)} 个期末日期有效")

    # ── 统计汇总 ──────────────────────────
    stats = {k: _stats(v) for k, v in all_series.items()}

    # ── 输出 CSV ──────────────────────────
    csv_path = os.path.join(output_dir, 'topN_monthly_returns.csv')
    pd.DataFrame(all_series).to_csv(csv_path)
    logger.info(f"  [TopN] 月度收益已保存: {csv_path}")

    # ── 写文字报告 ────────────────────────
    _write_topN_report(stats, top_ns, output_dir)

    # ── 绘图 ──────────────────────────────
    _plot_topN_cumulative(all_series, top_ns, output_dir)
    _plot_topN_bar(stats, top_ns, output_dir)
    _plot_topN_monthly(all_series, top_ns, output_dir)


def _write_topN_report(stats: dict, top_ns: list, output_dir: str) -> None:
    """写 Top-N 组合分析文字报告"""
    lines = []
    lines.append("=" * 66)
    lines.append("       model_res_0  Top-N 精选组合分析报告")
    lines.append("=" * 66)
    hdr = f"{'组合':>12} {'月均收益':>10} {'年化收益':>10} {'年化波动':>10} {'Sharpe':>8} {'月份数':>7}"
    sep = "-" * 66

    for n in top_ns:
        lines.append(f"\n── N={n} ──────────────────────────────────────────")
        lines.append(hdr)
        lines.append(sep)
        for key in [f'Top{n}_EW', f'Top{n}_VW',
                    f'Bot{n}_EW', f'Bot{n}_VW',
                    f'LS{n}_EW',  f'LS{n}_VW']:
            s = stats.get(key, {})
            if not s:
                continue
            lines.append(
                f"{key:>12} "
                f"{s['mean_monthly']:>10.4f} "
                f"{s['ann_return']:>10.4f} "
                f"{s['ann_vol']:>10.4f} "
                f"{s['sharpe']:>8.4f} "
                f"{s['n']:>7}"
            )

    report_path = os.path.join(output_dir, 'topN_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"  [TopN] 报告已保存: {report_path}")


def _plot_topN_cumulative(all_series: dict, top_ns: list, output_dir: str) -> None:
    """图1：Top-N EW/VW 累积净值对比"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Top-N 精选组合累积净值（model_res_0）', fontsize=13)

    colors_map = {5: '#e6194b', 10: '#3cb44b', 20: '#4363d8'}

    for ax, wt in zip(axes, ['EW', 'VW']):
        ax.set_title(f'等权 ({wt})' if wt == 'EW' else f'市值加权 ({wt})')
        for n in top_ns:
            key = f'Top{n}_{wt}'
            if key not in all_series:
                continue
            s = all_series[key].fillna(0)
            cum = (1 + s).cumprod()
            ax.plot(cum.index, cum.values,
                    label=f'Top {n}',
                    color=colors_map.get(n, 'gray'),
                    linewidth=1.8)
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel('日期')
        ax.set_ylabel('累积净值')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_1_cumulative.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [TopN] 图表已保存: {path}")


def _plot_topN_bar(stats: dict, top_ns: list, output_dir: str) -> None:
    """图2：Top-N / Long-Short 年化收益 & Sharpe 柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Top-N 组合年化收益与 Sharpe（model_res_0）', fontsize=13)

    labels, ann_rets, sharpes = [], [], []
    for n in top_ns:
        for key in [f'Top{n}_EW', f'Top{n}_VW', f'LS{n}_EW']:
            s = stats.get(key, {})
            if s and s.get('n', 0) > 0:
                labels.append(key)
                ann_rets.append(s['ann_return'] * 100)
                sharpes.append(s['sharpe'])

    x = range(len(labels))
    colors = ['steelblue' if v >= 0 else 'tomato' for v in ann_rets]

    axes[0].bar(x, ann_rets, color=colors, alpha=0.85)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    axes[0].set_ylabel('年化收益 (%)')
    axes[0].set_title('年化收益')
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    colors_s = ['steelblue' if v >= 0 else 'tomato' for v in sharpes]
    axes[1].bar(x, sharpes, color=colors_s, alpha=0.85)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Sharpe')
    axes[1].set_title('Sharpe 比率')

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_2_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [TopN] 图表已保存: {path}")


def _plot_topN_monthly(all_series: dict, top_ns: list, output_dir: str) -> None:
    """图3：各 N 的 Long-Short 月度收益时间序列"""
    n_plots = len(top_ns)
    fig, axes = plt.subplots(n_plots, 1, figsize=(13, 4 * n_plots), sharex=True)
    fig.suptitle('Long-Short (Top N − Bottom N) 月度收益（EW）', fontsize=12)
    if n_plots == 1:
        axes = [axes]

    colors_map = {5: '#e6194b', 10: '#3cb44b', 20: '#4363d8'}

    for ax, n in zip(axes, top_ns):
        key = f'LS{n}_EW'
        if key not in all_series:
            continue
        s = all_series[key].fillna(0)
        color = colors_map.get(n, 'gray')
        ax.bar(s.index, s.values * 100, color=color, alpha=0.7, width=20)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(f'Long-Short Top{n} − Bottom{n}（EW）', fontsize=10)
        ax.set_ylabel('月收益 (%)')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topN_3_ls_monthly.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [TopN] 图表已保存: {path}")


# ══════════════════════════════════════════════
# 6. 主函数
# ══════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description='model_res_0 预测与评估')
    parser.add_argument('--ckpt-dir', default=CKPT_DIR_DEFAULT,
                        help='checkpoint 目录（默认: WORK_SPACE/model_res_0）')
    parser.add_argument('--years', nargs='+', type=int, default=None,
                        help='要预测的年份列表（默认: 全体 IS+OOS 年份，即 2011-2025）')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='推理批大小（默认: 512）')
    parser.add_argument('--n-ensembles', type=int, default=5,
                        help='集成模型数量（默认: 5）')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help='输出目录')
    return parser.parse_args()


def main():
    args = parse_args()

    oos_years  = args.years or ALL_YEARS
    ckpt_dir   = args.ckpt_dir
    output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("  步骤 5：model_res_0 预训练模型预测")
    logger.info("=" * 60)
    logger.info(f"  Checkpoint 目录: {ckpt_dir}")
    logger.info(f"  预测年份       : {oos_years}")
    logger.info(f"  输出目录       : {output_dir}")
    logger.info(f"  图像尺寸       : H={IMG_H}, W={IMG_W}")

    # 验证 checkpoint 目录
    if not os.path.isdir(ckpt_dir):
        logger.error(f"Checkpoint 目录不存在: {ckpt_dir}")
        return 1

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
    logger.info(f"  发现 checkpoint 文件: {len(ckpt_files)} 个")
    if not ckpt_files:
        logger.error("未发现任何 .pth.tar 文件！")
        return 1

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  推理设备: {device}")

    # 加载模型
    logger.info("\n[1/3] 加载集成模型...")
    models = load_ensemble(ckpt_dir, args.n_ensembles, device)
    if not models:
        logger.error("没有成功加载任何模型！")
        return 1
    logger.info(f"  成功加载 {len(models)} 个集成模型")

    # 逐年预测
    logger.info(f"\n[2/3] 逐年推理（共 {len(oos_years)} 年）...")
    all_dfs = []
    for year in oos_years:
        logger.info(f"\n  → 年份 {year}")
        df_year = predict_year(models, year, args.batch_size, device)
        if df_year is not None and len(df_year) > 0:
            df_year['year'] = year
            all_dfs.append(df_year)
            logger.info(f"    样本数: {len(df_year)}")
        else:
            logger.warning(f"    年份 {year}: 无有效预测（数据可能尚未生成）")

    if not all_dfs:
        logger.error("所有年份预测均失败！请先运行步骤 2 生成图表数据。")
        return 1

    # 合并结果
    all_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\n  合并结果: {len(all_df)} 条预测记录")

    # 评估与报告
    logger.info("\n[3/4] 评估与保存预测报告...")
    print_report(all_df, output_dir)

    # 投资组合分析（十分位 / EW / VW / H-L）
    logger.info("\n[4/5] 投资组合分析（十分位分组、等权/市值加权/H-L）...")
    portfolio_dir = os.path.join(output_dir, 'portfolio')
    pred_csv = os.path.join(output_dir, 'model_res_0_predictions.csv')
    try:
        run_portfolio_analysis(
            result_files=[pred_csv],
            output_dir=portfolio_dir,
            n_ensembles=len(models),
            n_decile=10
        )
        logger.info(f"  投资组合分析完成，报告目录: {portfolio_dir}")
    except Exception as e:
        logger.error(f"  投资组合分析失败: {e}", exc_info=True)

    # Top-N 精选组合分析（前5/前10/前20）
    logger.info("\n[5/5] Top-N 精选组合分析（前5/前10/前20）...")
    topN_dir = os.path.join(output_dir, 'topN_portfolio')
    try:
        run_topN_analysis(
            pred_csv=pred_csv,
            output_dir=topN_dir,
            top_ns=[5, 10, 20]
        )
        logger.info(f"  Top-N 分析完成，报告目录: {topN_dir}")
    except Exception as e:
        logger.error(f"  Top-N 分析失败: {e}", exc_info=True)

    logger.info(f"\n[OK] 完成！输出目录: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

