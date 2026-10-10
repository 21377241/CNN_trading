# 论文实验结果统一更新报告

## 1. 更新内容

本轮统一了所有组合 Sharpe 的计算口径，并重新生成了主要 OOS 评价结果。

标准 Sharpe 定义为：

```text
Sharpe = mean(period return) / std(period return) * sqrt(52)
```

复合年化收益单独报告：

```text
Annual return = (1 + mean(period return))^52 - 1
```

旧版代码曾使用复合年化收益除以年化波动率作为 Sharpe，导致旧 Sharpe 偏高。当前报告和后续论文表格应使用标准口径。

## 2. E2 主 benchmark

E2 chronological common OOS panel：2,687,822 行、922 个周度观测日期。

| Model | Rank IC | EW H-L Sharpe | VW H-L Sharpe |
|---|---:|---:|---:|
| CNN | 0.0293 | 2.37 | 1.01 |
| Logit | 0.0048 | 1.10 | 0.25 |
| HistGB | 0.0256 | 2.57 | 1.38 |
| MLP | 0.0279 | 2.88 | 1.49 |

这些结果已写入：

```text
WORK_SPACE/paper/E2_matched_is_chrono/e2_summary.csv
WORK_SPACE/paper/E2_matched_is_chrono/e2_results_auto.md
```

## 3. E2C temporal numerical benchmark

| Model | Rank IC | EW H-L Sharpe | VW H-L Sharpe |
|---|---:|---:|---:|
| CNN | 0.0308 | 2.39 | 1.11 |
| 1D-CNN-120 | 0.0484 | 3.21 | 1.64 |
| CNN + 1D-CNN | 0.0473 | 3.46 | 1.71 |

CNN residual after controlling for 1D-CNN 的 EW Sharpe 为 `0.81`。

E2C 结果文件：

```text
WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/standalone_summary.csv
WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/key_six.json
```

## 4. E2D liquidity audit

E2D 使用已有 OOS 预测，不重新训练模型。流动性使用 `abs(Close) * abs(Vol)` 作为成交额代理。

1D-CNN 结果：

| Universe | Rank IC | EW H-L Sharpe |
|---|---:|---:|
| Full | 0.0484 | 3.21 |
| Largest 50% | 0.0420 | 2.67 |
| Largest 30% | 0.0397 | 2.26 |
| Large & Liquid | 0.0456 | 2.67 |

简单成本压力测试下，1D-CNN 的 EW H-L Sharpe 为：

| One-way cost | Sharpe |
|---:|---:|
| 0 bps | 3.21 |
| 25 bps | 2.48 |
| 50 bps | 1.75 |
| 100 bps | 0.30 |

完整报告：

[paper_E2D_liquidity_implementability_audit.md](../archive/superseded/paper_E2D_liquidity_implementability_audit.md)

## 5. E3 shared/specific

E3 已使用标准 Sharpe 重新评价：

| Score | Rank IC | EW H-L Sharpe |
|---|---:|---:|
| Shared | 0.0423 | 3.16 |
| CNN-specific | 0.0180 | 1.10 |
| MLP-specific | 0.0243 | 2.33 |
| HistGB-specific | 0.0026 | 0.50 |
| Raw CNN | 0.0308 | 2.39 |
| Raw MLP | 0.0406 | 3.13 |
| Raw HistGB | 0.0301 | 2.70 |

更新文件：

```text
WORK_SPACE/paper/E3_shared_specific_chrono_ric/e30_score_summary.csv
WORK_SPACE/paper/E3_shared_specific_chrono_ric/e30_portfolio/
```

## 6. E4 recent path

E4 的 39 个已有 portfolio 结果均已重评并重新绘图。更新内容包括：

- lag experiments；
- block ablations；
- portfolio summaries；
- decile return figures；
- Sharpe figures；
- H-L comparison figures；
- cumulative return figures。

更新目录：

```text
WORK_SPACE/paper/E4_recent_path_chrono_ric/
```

E4 baseline 标准 EW H-L Sharpe：

| Model | EW H-L Sharpe |
|---|---:|
| CNN | 1.23 |
| HistGB | 2.04 |
| MLP | 2.63 |

## 7. E5 mechanism

E5 Phase 1 和 Phase 2 已重新运行。E5 主体不是 Sharpe，而是：

- Rank IC；
- Newey-West t-statistics；
- Fama-MacBeth interaction coefficients；
- 条件 high-minus-low return spreads。

因此 E5 中的主体数字不应机械改写为 Sharpe。更新结果位于：

```text
WORK_SPACE/paper/E5_mechanism_chrono_ric/
```

代表性结果：

- CNN-specific `R_1_5` Rank IC：`0.0175`；
- MLP-specific `R_1_5` Rank IC：`0.0326`；
- CNN-specific × Pressure interaction NW t：`4.19`；
- MLP-specific × Pressure interaction NW t：`-0.46`；
- CNN-specific 在负近期路径上的 spread NW t：`5.58`；
- MLP-specific 在正近期路径上的 spread NW t：`10.46`。

## 8. 是否需要重新训练

本轮没有重新训练 CNN、MLP、HistGB 或 1D-CNN。原因是 Sharpe 属于 OOS prediction 之后的组合评价统计，模型预测文件没有改变。

只有在改变以下内容时才需要重新训练：

- 输入信息集；
- 标签定义；
- 训练/验证/OOS 时间划分；
- 模型结构；
- 训练目标；
- 选模标准。

本轮只是统一 Sharpe 公式、重新计算组合评价并更新图表，因此不属于需要重训的情形。

## 9. 当前论文主线

统一口径后，论文中的主要结论仍然成立，但数值应使用较低的标准 Sharpe：

> The temporal numerical model is the strongest standalone learner in the current matched-information benchmark. Its advantage over the chart CNN remains economically meaningful, while the chart CNN retains positive residual predictability and provides incremental ensemble value. These results survive large-stock and liquid-universe restrictions, although the signal is stronger among smaller stocks and remains subject to turnover, execution, and capacity constraints.

后续 Overleaf 正文、表格和摘要应以本报告中的数字为准。
