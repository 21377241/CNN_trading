# E5 统一口径更新说明

E5 已重新运行 Phase 1 和 Phase 2：

```powershell
python experiments/E5_mechanism/run_phase1.py
python experiments/E5_mechanism/run_phase2.py
```

本轮没有重新训练 CNN、MLP 或任何机制模型。E5 使用已有的 `S_cnn_sp`、`S_mlp_sp` 预测分数和 forward-return panel。

## Sharpe 口径影响

E5 的主体结果不是 portfolio Sharpe，而是：

- Rank IC；
- Newey--West t-statistic；
- Fama--MacBeth interaction coefficient；
- 条件 high-minus-low return spread。

因此，Sharpe 统一规则不会改变 E5 的主体统计量。E5 中出现的括号数字主要是 NW t-stat，而不是 Sharpe，不应按 Sharpe 公式替换。

## 已更新输出

结果目录：`WORK_SPACE/paper/E5_mechanism_chrono_ric`

- `e51_horizon_ic.csv`
- `e52_session_ic.csv`
- `e52_on_minus_id.csv`
- `e53_state_delta.csv`
- `e54_pressure_ic.csv`
- `e55_interaction_fm.csv`
- `e56_cont_rev_spreads.csv`
- `e56_high_pressure_spreads.csv`

## 代表性结果

在 `R_1_5` 上：

- CNN-specific Rank IC：`0.0175`，H-L spread 的 NW t=`3.89`；
- MLP-specific Rank IC：`0.0326`，H-L spread 的 NW t=`12.06`；
- Pressure 分组的 CNN-specific IC 从 low=`0.0134` 到 high=`0.0180`；
- Pressure 分组的 MLP-specific IC 从 low=`0.0263` 到 high=`0.0448`；
- CNN-specific 与 pressure 的 interaction coefficient 为 `0.000354`，NW t=`4.19`；
- MLP-specific 与 pressure 的 interaction coefficient接近零，NW t=`-0.46`；
- Recent-path 分组中，CNN-specific 在负近期路径上的 spread 为 `0.00175`，NW t=`5.58`；
- MLP-specific 在正近期路径上的 spread 为 `0.00431`，NW t=`10.46`。

这些数值与旧 E5 报告的主体机制结论一致；本轮主要作用是确认所有 E5 输出已经基于当前 forward panel 重新生成，并明确 E5 不是 Sharpe 表，因此不应机械替换其中的 t-stat。
