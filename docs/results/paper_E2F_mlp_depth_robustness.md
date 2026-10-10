# E2F：MLP 网络深度稳健性检验

## 1. 研究问题

当前扁平化数值 MLP 的结果是否依赖于任意选定的两层隐藏层结构？借鉴
Gu、Kelly 与 Xiu（2020）比较不同网络深度的思路，本实验仅改变网络深度，
其余信息集和训练协议保持不变。

## 2. 实验协议

- 输入：120D 匹配 OHLCV 特征。
- 训练集：2000--2005；验证集：2006--2007；样本外：2008--2025。
- 模型选择与早停：验证集 Rank IC。
- 随机种子：42--46；每个深度使用五个 seed 的预测概率平均作为 ensemble。
- 固定参数：`alpha=1e-4`、`learning rate=1e-3`、`max_iter=200`、`patience=15`。
- 组合 Sharpe：日度持有期收益的算术均值除以标准差，再乘以 `sqrt(52)` 年化。

## 3. 五 seed ensemble 结果

| 模型 | 隐藏层结构 | Rank IC | EW Sharpe | VW Sharpe |
|---|---|---:|---:|---:|
| MLP-2 | 128-64 | 0.0354 | 2.6496 | 1.3706 |
| MLP-3 | 128-64-32 | 0.0399 | 2.7966 | 1.1763 |
| MLP-4 | 128-64-32-16 | 0.0410 | 2.8071 | 1.3249 |
| MLP-5 | 128-64-32-16-8 | 0.0411 | 2.8961 | 1.1633 |

所有模型均在同一个 2,687,822 条观测、922 个调仓日的样本外面板上评价。

## 4. Seed 级别均值与标准差

| 模型 | Rank IC 均值（标准差） | EW Sharpe 均值（标准差） | VW Sharpe 均值（标准差） |
|---|---:|---:|---:|
| MLP-2 | 0.0276 (0.0074) | 2.3189 (0.6632) | 1.0331 (0.3028) |
| MLP-3 | 0.0318 (0.0053) | 2.3745 (0.3808) | 0.9432 (0.1771) |
| MLP-4 | 0.0320 (0.0028) | 2.4463 (0.4018) | 0.9915 (0.1825) |
| MLP-5 | 0.0329 (0.0045) | 2.4588 (0.2058) | 0.9494 (0.0811) |

ensemble 是先平均五个 seed 的预测概率、再进行截面排序和组合构造，因此其
Sharpe 不应等于五个单 seed Sharpe 的算术平均值。

## 5. 验证集 Rank IC

下表汇总每个 seed 在 2006--2007 验证集上、由 Rank IC early stopping 所选择的
最佳 checkpoint 分数。均值和标准差均跨五个 seed 计算。

| 模型 | 验证集 Rank IC 均值（标准差） | 各 seed 最佳 epoch |
|---|---:|---|
| MLP-2 | 0.0565 (0.0056) | 14, 2, 18, 2, 2 |
| MLP-3 | 0.0555 (0.0030) | 4, 14, 5, 3, 3 |
| MLP-4 | 0.0607 (0.0045) | 4, 15, 2, 14, 3 |
| MLP-5 | 0.0586 (0.0031) | 14, 4, 3, 4, 22 |

最终 five-seed ensemble 的验证集 Rank IC 如下：

| 模型 | ensemble 验证集 Rank IC |
|---|---:|
| MLP-2 | 0.0705 |
| MLP-3 | 0.0691 |
| MLP-4 | 0.0759 |
| MLP-5 | 0.0737 |

MLP-4 不仅在 seed 级验证集 Rank IC 均值上最高，也在与最终 OOS 组合完全一致的
five-seed ensemble 验证对象上最高，MLP-5 次之。该结果进一步说明，若需固定主
架构，应在预先规定的验证集标准和模型复杂度规则下选择，而不能事后依据样本外
EW Sharpe 的最高值选择 MLP-5。

## 6. 结果解释

结果表明，主结果并不依赖某一个偶然选定的两层 MLP。随着深度由两层增加至
五层，seed 级 EW Sharpe 均值温和提高，且 EW Sharpe 的跨 seed 波动总体下降。
样本外 Rank IC 的主要提升在三至四层已经实现：MLP-4 与 MLP-5 基本处于平台区。
验证集 Rank IC 在四层达到峰值，而 VW Sharpe 并未随深度单调改善。

因此，E2F 支持“扁平化 MLP 结论对网络深度具有一定稳健性”，但不足以仅基于
样本外 EW Sharpe 就将简约的 MLP-2 替换为 MLP-5。最终若需要固定主 MLP 架构，
应以预先规定的验证集 Rank IC 和模型复杂度规则为依据，而不是选择样本外
Sharpe 最高的一行。

## 7. 产物位置

- 脚本：`experiments/E2F_mlp_depth/run_depth.py`
- ensemble 汇总：`WORK_SPACE/paper/E2F_mlp_depth/e2f_depth_summary.csv`
- seed 汇总：`WORK_SPACE/paper/E2F_mlp_depth/e2f_depth_seed_summary.csv`
- seed 明细：`WORK_SPACE/paper/E2F_mlp_depth/mlp*_seed_metrics.csv`
- 样本外预测：`WORK_SPACE/paper/E2F_mlp_depth/preds/`
- 组合结果：`WORK_SPACE/paper/E2F_mlp_depth/portfolio/`
