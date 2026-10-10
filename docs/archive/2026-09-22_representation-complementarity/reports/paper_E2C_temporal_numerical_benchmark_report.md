# E2C 实验报告：Temporal Numerical Benchmark（1D-CNN-120）

## 1. 目的

E2C 检验图像 CNN 相对于 flattened numerical learner 保留的独立预测力，是否主要只是时间局部归纳偏置的结果。实验保持相同的 120D、20 日 OHLCV 信息集、样本、标签、chrono split 和 OOS 评价，仅将 flat input 改为 `20 x 6` 数值序列。

## 2. 训练设计

1D-CNN 使用两层 temporal convolution（32、64 channels，kernel size 3）、ReLU、global average pooling、dropout 和线性输出。训练期为 2000--2005，验证期为 2006--2007，OOS 为 2008--2025。五个随机种子的预测平均后形成最终 score；early stopping 由 validation Rank IC 决定。

## 3. 标准化结果

所有 Sharpe 均按周度 H--L 收益的 `mean / std * sqrt(52)` 计算。样本包含 922 个形成日和 2,687,822 个 stock-date。

| Model | Input | AUC | Rank IC | EW H--L Sharpe | VW H--L Sharpe |
|---|---|---:|---:|---:|---:|
| CNN | chart image | 0.5190 | 0.0308 | 2.3883 | 1.1098 |
| HistGB-120 | 120D flat | 0.5227 | 0.0256 | 2.5670 | 1.3773 |
| MLP-4-120 | 120D flat | -- | 0.0410 | 2.8071 | 1.3249 |
| 1D-CNN-120 | `20 x 6` sequence | 0.5261 | 0.0484 | 3.2065 | 1.6365 |

## 4. CNN--1D-CNN Spanning

| Statistic | Estimate |
|---|---:|
| Daily Pearson correlation | 0.3446 |
| Top-decile overlap | 0.2387 |
| CNN coefficient controlling 1D-CNN | 0.00048 (NW t = 3.90) |
| 1D-CNN coefficient controlling CNN | 0.00215 (NW t = 11.96) |
| CNN residual Rank IC | 0.0153 |
| CNN residual EW H--L Sharpe | 0.8111 |
| CNN + 1D-CNN EW H--L Sharpe | 3.4613 |

## 5. 结论

1D-CNN-120 是当前最强的 standalone learner，说明 temporal inductive bias 是价格历史预测力的重要来源。它吸收了 CNN 相对于 flattened MLP 的大量残差信息，但没有完全 span chart CNN：控制 1D-CNN 后，CNN 系数仍为正且显著，残差信号的 EW Sharpe 为 0.8111。两者的 ensemble Sharpe 为 3.4613，高于任一单模型。

因此，E2C 不支持将 CNN residual 强解释为纯粹的 image-representation effect；更稳妥的结论是，相同 OHLCV 历史在图像卷积、flattened MLP 和 temporal numerical convolution 下形成了部分重合但不完全相同的预测排序。

## 6. 可复现路径

| Artifact | Path |
|---|---|
| Training code | `experiments/E2C_temporal/train_1dcnn.py` |
| Evaluation code | `experiments/E2C_temporal/eval_e2c.py` |
| Config | `experiments/E2C_temporal/config_1dcnn_120.yaml` |
| OOS prediction | `WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv` |
| Machine-readable summary | `WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/key_six.json` |
