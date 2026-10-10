# E6A：CNN 与 MLP-4 的状态依赖相对预测优势

## 问题

在不重训模型、不组合预测、不估计 gate 的前提下，检验调仓日可观测的股票状态是否能系统性预测原始 CNN 与原始 MLP-4 的相对横截面预测能力。

本实验的对象是原始 `S_cnn`、`S_mlp`，而非 E3--E5 的 residual/specific component。后者只用于事前提出候选状态变量。

## 固定设计

- 固定预测：既有冻结 CNN 与 MLP-4 的 OOS 分数；不重新训练。
- 状态：过去五日累计收益 `CumRet5` 与既有 Pressure composite。
- Pressure：`[z(TurnoverProxy_5) + z(-MaxDD5) + z(VolEndOverMean5)] / 3`，只使用调仓日及以前信息。
- 每个周度调仓日，分别在全横截面上将两个状态各分为 tercile，形成 `3 x 3` 的九个状态格。
- 格内指标：CNN、MLP-4 对未来五日 `ret_val` 的 date-level Spearman Rank IC；相对优势定义为 `Delta IC = IC_CNN - IC_MLP4`。
- 推断：对周度 `Delta IC` 时间序列使用 Newey--West lag 4。
- Discovery/calibration：2008--2015；冻结验证：2016--2025。状态定义和分组规则在查看冻结期前已固定。

## 结果

| 最近收益 | Pressure | CNN IC（发现期） | MLP-4 IC（发现期） | Delta IC（发现期） | NW t | CNN IC（冻结期） | MLP-4 IC（冻结期） | Delta IC（冻结期） | NW t |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 负 | Low | 0.02270 | 0.02277 | -0.00007 | -0.01 | 0.01577 | 0.00836 | 0.00741 | 1.57 |
| 负 | Mid | 0.02964 | 0.03772 | -0.00808 | -1.50 | 0.01279 | 0.00916 | 0.00363 | 0.88 |
| 负 | High | 0.03461 | 0.03998 | -0.00537 | -0.90 | 0.01049 | 0.00242 | 0.00807 | 2.23 |
| 中性 | Low | 0.02785 | 0.03337 | -0.00552 | -1.01 | 0.02030 | 0.01502 | 0.00527 | 1.17 |
| 中性 | Mid | 0.03368 | 0.03841 | -0.00473 | -0.84 | 0.01874 | 0.01442 | 0.00432 | 1.00 |
| 中性 | High | 0.03538 | 0.04245 | -0.00707 | -1.08 | 0.01843 | 0.01616 | 0.00226 | 0.47 |
| 正 | Low | 0.01768 | 0.04139 | -0.02372 | -3.91 | 0.00452 | 0.01897 | -0.01445 | -3.19 |
| 正 | Mid | 0.03915 | 0.05600 | -0.01686 | -3.27 | 0.01633 | 0.02887 | -0.01254 | -2.59 |
| 正 | High | 0.03939 | 0.05741 | -0.01802 | -3.54 | 0.03632 | 0.05455 | -0.01823 | -4.02 |

对应 heatmap：

- [发现期 Delta IC heatmap](/E:/phd/LLM_trading/CNN_trading/docs/figures/E6A_delta_ic_discovery.png)
- [冻结期 Delta IC heatmap](/E:/phd/LLM_trading/CNN_trading/docs/figures/E6A_delta_ic_test.png)

## 结论

**MLP-4 的正近端收益状态优势得到清楚复现。** 在发现期与冻结期，近期正收益的三个 Pressure 组中，`Delta IC` 均为负；冻结期 NW t 分别为 `-3.19`、`-2.59` 和 `-4.02`。这与此前“MLP 更接近 continuation”的机制发现一致，而且现在直接发生在原始可交易预测上。

**CNN 的负近端/压力优势未通过发现期到冻结期的一致性检验。** 冻结期的负收益--高 Pressure 格中 CNN 较强（`Delta IC = 0.00807`，NW t `2.23`），但发现期该格反而为负且不显著。因此它目前只能作为待验证的 regime hypothesis，不能作为预先固定的 CNN overweight 规则。

因此，E6A 不支持直接进入一个双边状态 gate。它支持的最强、最稳健结论是：近期正收益状态系统性偏向 MLP-4；CNN 的比较优势尚未获得跨期复现。后续若探索组合，应先将这一结果作为单边 MLP overweight 的候选规则，并在新的、未参与状态发现的样本或外部市场中再次验证。

## 可复现性

- 代码：[run_e6a_state_map.py](/E:/phd/LLM_trading/CNN_trading/experiments/E6A_state_advantage/run_e6a_state_map.py)
- 原始逐周九格 Rank IC：`WORK_SPACE/paper/E6A_state_advantage/e6a_daily_cell_rank_ic.csv`
- 汇总表：`WORK_SPACE/paper/E6A_state_advantage/e6a_cell_summary.csv`
