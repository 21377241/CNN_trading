# E4T 实验报告：CNN 与 1D-CNN 特异信号的近期路径机制

## 目的

E2C 表明 1D-CNN-120 是最强的数值基准，但它没有完全张成 chart CNN。本实验检验两者在互相线性残差化后，是否仍依赖不同的 recent-5 价格路径成分。

设每个周度横截面上的原始分数为 `S_CNN` 与 `S_1D`。报告两个成分：

- `CNN | 1D`：将 `S_CNN` 对 `S_1D` 做截面线性投影后的残差；
- `1D | CNN`：将 `S_1D` 对 `S_CNN` 做截面线性投影后的残差。

样本为 2008--2025 年、922 个周度截面、2,687,822 个股票周观测。组合指标沿用全项目统一口径：每周形成等权 top-minus-bottom decile，Sharpe 为周度 H-L 平均收益除以周度标准差再乘以 `sqrt(52)`。

## 设计与可复现性

- 1D-CNN 保持 E2C 的 20x6 输入和 5 个已训练 seed（42--46），没有重训；每个干预只对冻结模型重新前向。
- 120D 每日输入依次为 adjusted `Open, High, Low, Close, MA20, normalized volume`。
- CNN 侧直接复用已验证的 chart redraw 干预预测缓存；所有结果均重新相对于未干预的对方模型残差化。
- 每个价格干预后重算 MA20，以保持 1D 输入的内部一致性。
- 干预是模型敏感性诊断，不能解释为对应市场变量的结构因果效应。

代码：`experiments/E4_recent_path/run_e4t_temporal.py`。
原始结果：`WORK_SPACE/paper/E4T_cnn_1d_recent_path/`。

## 1. 基线

| 特异成分 | Rank IC | EW H-L Sharpe | VW H-L Sharpe |
|---|---:|---:|---:|
| CNN \| 1D | 0.0153 | 0.8112 | 0.4264 |
| 1D \| CNN | 0.0388 | 2.7674 | 1.4320 |

基线与 E3T 的双向残差结果一致。1D-CNN 在剔除 CNN 可解释部分后仍保有较强的预测和组合表现；CNN 的残差较小但仍为正。

## 2. 路径形状超越累计收益

对每个周度横截面，先用 `CumRet5` 解释特异分数，再加入六个路径变量（LateEarly、PathVol、UpCount5、MaxDD5、Reversal5、Monotonicity）。下表是日度截面 R2 的时间均值。

| 特异成分 | CumRet5 R2 | 加入全部路径形状后 R2 | Delta R2 | Newey-West t |
|---|---:|---:|---:|---:|
| CNN \| 1D | 0.0159 | 0.0792 | 0.0633 | 55.08 |
| 1D \| CNN | 0.0673 | 0.1352 | 0.0680 | 43.66 |

两类特异分数都不能仅由五日累计收益概括。单个新增项中，CNN 侧的 PathVol（Delta R2=0.0258）、Reversal5（0.0249）与 LateEarly（0.0228）更突出；1D 侧的 PathVol（0.0275）、Reversal5（0.0210）与 MaxDD5（0.0190）更突出。

## 3. 逐日滞后扰动

把某个交易日的 OHLC 价格变动压平到前一日收盘价，并局部平滑该日成交量。表中为相对基线的 EW Sharpe 下降；负值表示干预后 Sharpe 上升。

| 被干预日 | CNN \| 1D Delta Sharpe | 1D \| CNN Delta Sharpe |
|---|---:|---:|
| t-4 | 0.1374 | -0.0029 |
| t-3 | 0.6003 | 0.1026 |
| t-2 | -0.0827 | -0.0516 |
| t-1 | 0.7112 | 0.2849 |
| t | 0.5135 | -0.5363 |

CNN 残差最依赖 `t-1`、`t-3` 和当日 `t` 的局部结构。1D-CNN 残差对 `t-1` 有温和依赖，但抹平当日会提高组合 Sharpe，表明其当日输入的使用方式并不等同于 CNN。

## 4. recent-5 组件干预

每次只改变一个 recent-5 成分：volume、wick、direction、return、gap、intraday。下表为 EW Sharpe 相对基线的变化。

| 组件 | CNN \| 1D Delta Sharpe | 1D \| CNN Delta Sharpe |
|---|---:|---:|
| volume | 0.2018 | -0.1746 |
| wick | 0.1617 | -0.0282 |
| direction | 0.1472 | 0.0114 |
| return | -0.1934 | 1.0056 |
| gap | -0.1088 | -0.5415 |
| intraday | 0.1441 | -0.1042 |

CNN 残差对 volume、wick、direction 和 intraday body 的扰动表现为明显下降，支持其保留了局部视觉形态与成交量排列相关的成分。1D-CNN 残差最清晰地依赖 recent-5 return path：去除该路径使 Sharpe 从 2.7674 降至 1.7618（下降 1.0056，36.3%）。相反，抹去 gap 会提高其 Sharpe，说明 gap 在这个数值时序表示中并非增益来源。

## 结论

E4T 支持“partial spanning 而非完全同质”的解释：两种卷积模型都从近期路径形状中获益，但残差机制并不相同。chart CNN 的剩余信息更集中在局部形态、成交量和近端日结构；1D-CNN 的剩余信息则更强地与连续价格路径有关。该结果应被表述为表征/归纳偏置差异下的模型敏感性证据，而不是图像变量的独立因果识别。

## 输出文件

- `e4t_pathshape/e4t_delta_r2_summary.csv`
- `e4t_lag/cnn/e4t_cnn_lag_summary.csv`
- `e4t_lag/1dcnn/e4t_1dcnn_lag_summary.csv`
- `e4t_blocks/cnn/e4t_cnn_blocks_summary.csv`
- `e4t_blocks/1dcnn/e4t_1dcnn_blocks_summary.csv`
