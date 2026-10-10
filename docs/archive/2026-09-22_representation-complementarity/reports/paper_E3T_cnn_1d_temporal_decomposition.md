# E3-T 实验报告：CNN 与 1D-CNN 的 Shared/Specific 分解

## 目的

E3-T 将 E2C 的 CNN--1D-CNN spanning 检验扩展为可直接进入后续机制研究的双向 residual decomposition。它只使用 E2C 的 120D common panel，因此不混入 180D MLP-4 Bag10 的指标特征。

## 定义

每天在横截面上进行带截距的 OLS：

- `CNN-specific given 1D`：CNN score 对 1D-CNN score 回归后的残差；
- `1D-specific given CNN`：1D-CNN score 对 CNN score 回归后的残差；
- `Shared`：两个日内标准化 score 的均值。

样本为 2008--2025 年的 922 个形成日、2,687,822 个 stock-date。Sharpe 使用周度 H--L 收益的 `mean / std * sqrt(52)`。

## 结果

| Score | Rank IC | NW t | EW H--L Sharpe | VW H--L Sharpe |
|---|---:|---:|---:|---:|
| Shared CNN + 1D | 0.0473 | 14.50 | 3.4613 | 1.7143 |
| CNN-specific given 1D | 0.0153 | 6.60 | 0.8111 | 0.4263 |
| 1D-specific given CNN | 0.0388 | 11.42 | 2.7674 | 1.4320 |
| Raw CNN | 0.0308 | 11.72 | 2.3883 | 1.1098 |
| Raw 1D-CNN | 0.0484 | 13.42 | 3.2065 | 1.6365 |

## 结论

1D-CNN 吸收了 chart CNN 相对 flattened learner 的大量预测力，但没有完全 span CNN：`CNN-specific given 1D` 仍有正的 Rank IC 和 0.8111 的 EW Sharpe。反过来，`1D-specific given CNN` 的 EW Sharpe 为 2.7674，说明 temporal numerical convolution 保留了更大部分不被图像 CNN 吸收的信号。

因此，后续 E4-T 和 E5-T 应以这两个双向 residual 为对象，检验它们的路径敏感性与经济实现是否不同。

## 可复现路径

- Code: `experiments/E3_shared_specific/run_e3t_temporal.py`
- Outputs: `WORK_SPACE/paper/E3T_cnn_1d_temporal/`
- Machine-readable summary: `e3t_score_summary.csv`
