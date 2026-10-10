# 论文主线整理：Same Information, Distinct Predictive Signals

> 本文档把 `docs/` 下 E1b-E5 的实验报告整理成一条可写成论文的主叙事。核心原则：不再把论文写成「CNN 是否优于表格模型」，而是写成「同一 OHLCV 信息集为什么会被不同非线性模型解析成不同且互补的预测信号」。

---

## 1. 一句话主线

在完全相同的 20 日 OHLCV 信息集下，图像 CNN 并不稳定优于强表格非线性模型；但 CNN、MLP 与 HistGB 产生的排序并非同义重复，而是包含 shared signal 与 model-specific signal。进一步拆解显示，这些 specific signals 对不同的近端价格路径、量能、几何形态和状态条件敏感，并以不同 horizon / market state 转化为未来收益。

更短版本：

> Same price history, different nonlinear representations, distinct and complementary trading signals.

---

## 2. 论文不应再讲什么

原始问题：

> CNN 是否因为图像表示而显著优于 tabular model？

现有证据不支持把它作为主故事：

- E2 主表中，CNN Sharpe 2.65，HistGB Sharpe 3.07，chrono bagged MLP Sharpe 3.70。
- MLP 稳定性实验显示，单 seed MLP 系统性低估浅层表格网络；bag k=10 后 MLP 成为当前最强单模。
- 因此「图像表示带来单模优势」应降级为被检验但不成立的起点，而不是结论。

应改写为：

> 图像表示不是收益预测的必要条件；关键问题是不同 representation / function class 如何从同一历史价格信息中抽取 shared 与 specific alpha。

---

## 3. 中心 Puzzle

如果 CNN、HistGB、MLP 都只使用相同的 20 日 OHLCV，那么：

1. 为什么强 tabular 模型可以追平或超过 CNN？
2. 为什么模型之间的日截面相关只有中等水平？
3. 为什么控制对方后仍有显著增量解释力和组合收益？
4. 这些 model-specific signals 分别对应什么价格路径结构？
5. 这些结构如何在未来收益中兑现，是否与中国市场状态或制度摩擦相容？

这组问题比「谁的 Sharpe 更高」更稳，也更有论文贡献。

---

## 4. 章节主线

### Part I. Matched-information benchmark：先把同信息集比较做公平

目的：建立同一信息集、同一 OOS 面板、chrono validation 的基准。

主结果：

| 模型 | Rank IC | EW H-L Sharpe | 叙事角色 |
|---|---:|---:|---|
| CNN E1b-ric | 0.0308 | 2.65 | 图像表示基准 |
| Logit | 0.0097 | 1.47 | 线性下界 |
| HistGB | 0.0301 | 3.07 | 强非线性 tabular 对照 |
| MLP chrono bag k=10 | 0.0406 | 3.70 | 稳定化 tabular 上界 |

应该写出的结论：

- CNN 明显强于线性 logit，说明非线性价格路径可预测性存在。
- HistGB / bagged MLP 不弱于 CNN，说明 image representation 不是必要条件。
- 论文问题自然转为：同信息、不同模型为何学出不同信号？

对应材料：

- `paper_results_E2.md`
- `paper_E2_experiment_report.md`
- `paper_results_E1b.md`
- `paper_results_mlp_stability.md`

---

### Part II. Feature engineering as accessibility：显式指标改变的是可达性，不是信息集

目的：解释为什么 120 -> 180 个显式 K 线/量能方向指标主要帮助线性模型。

主结果：

- Logit：+60 维后 Sharpe 增量约 +0.30。
- HistGB：增量较小，约 +0.12。
- MLP：全量 indicator 反而可能伤害，Sharpe 约 -0.29。
- 真正帮 Logit 的主要是 uP / uV 方向比特，delta 几乎无增量。

应该写出的结论：

- 显式图形指标不是带来新信息，而是降低某些模型访问阈值规则的难度。
- 灵活非线性模型已经能部分内生提取这些结构，因此 feature engineering 的边际收益较低。
- 这为后文铺垫：同一信息集下，差异来自 representation / function class 的 extraction bias。

对应材料：

- `paper_results_E2A.md`
- `paper_e2a_indicator_mechanism.md`

---

### Part III. Complementarity：不是谁替代谁，而是有可叠加增量

目的：证明模型分数不是同一信号的噪声版本。

CNN vs HistGB：

- 日截面相关约 0.37。
- Top10% 重叠约 26%。
- 双变量 FM 中两者仍显著。
- 条件双重排序、残差化后仍有正 Sharpe。
- ensemble Sharpe 3.43，高于任一单模。

CNN vs bagged MLP：

- 日截面相关约 0.33。
- Top10% 重叠约 23%。
- 控 MLP 后 CNN 仍显著；控 CNN 后 MLP 更强。
- ensemble Sharpe 3.91，高于最强单模 MLP 的 3.70。

应该写出的结论：

- 强 MLP 吸收了更多共变信息，但没有消灭 CNN 增量。
- Complementarity 是论文从 benchmark 走向 mechanism 的桥梁。
- 主叙事应避免「MLP 全面碾压 CNN」，改写为「MLP 是更强稳定 tabular learner，但 CNN-specific 仍可交易、可解释、可叠加」。

对应材料：

- `paper_results_E2B.md`
- `paper_results_E2B_mlp.md`
- `paper_mlp_single_vs_bagged.md`

---

### Part IV. Shared vs specific：把互补信号显式分解出来

目的：从 pairwise complementarity 进入多模型 signal decomposition。

主结果：

| 信号 | 核心发现 |
|---|---|
| Shared | 最强共识信号，Sharpe 约 3.72，接近或略高于 bagged MLP raw |
| MLP-specific | 显著且最强，Sharpe 约 2.57 |
| CNN-specific | 仍显著，Sharpe 约 1.14 |
| HistGB-specific | 在强 MLP 控制后大幅削弱，Sharpe 约 0.51 |

结构画像：

- Shared 主要对应近期收益、动量、量能共识。
- MLP-specific 是当前最强 specific，稳定化后更多贴近 recent / momentum / volume。
- CNN-specific 仍保留独立路径信息，尤其依赖近端 K 线。
- HistGB-specific 不宜作为主证据，可作为「被强 MLP 吸收」的削弱对照。

应该写出的结论：

- 三模共识是市场中最稳健的一层价格历史预测信号。
- 模型特异信号并非随机残差，而有显著预测力和可解释结构。
- 后文应聚焦 CNN-specific 与 MLP-specific，因为它们在强 MLP 口径下仍稳健。

对应材料：

- `paper_results_E3.md`
- `paper_e3_shared_specific_design.md`
- `paper_mlp_single_vs_bagged.md`

---

### Part V. Recent-path decomposition：specific signals 到底看什么路径

目的：把 specific alpha 从抽象残差翻译成价格路径结构。

主结果：

- Lag sensitivity：CNN 对 t-1 / t / t-3 等单日位置非常敏感；MLP 更平滑。
- 控制 CumRet5 后，路径形状仍有显著解释力。
- 成分敏感性剖面分离，CNN vs MLP normalized drop 相关约 0.29。
- CNN 更依赖 Volume、回撤、波动路径、近端日位置。
- MLP 更依赖 Intraday / Wick / Direction，并表现出更强 continuation 结构。

应该写出的结论：

- specific signal 不是简单 5 日动量。
- CNN 的 representation bias 更像「局部视觉路径/量能压力探测器」。
- MLP 的 function class 更像「表格化近端趋势、方向、几何规则组合器」。
- E4 是全文解释力最强的机制章节，应作为主图核心。

对应材料：

- `paper_results_E4.md`
- `paper_e4_recent_path_design.md`

---

### Part VI. Economic realization：这些路径信号如何兑现

目的：从「模型看什么」走向「为什么有收益」。

主结果：

| 维度 | CNN-specific | MLP-specific |
|---|---|---|
| Horizon | 短窗，t+1 较强，之后衰减 | 更持续，t+2 峰值，r1-r5 多日显著 |
| Session | 偏 intraday，但证据较弱 | ON/ID 无明显分工 |
| State | turnover / pressure 放大 | turnover + momentum / volatility 放大 |
| Pressure interaction | FM beta3 显著为正 | interaction 约 0 |
| Direction | rebound-leaning / mixed | continuation |
| T+1 / limit | T1_state interaction 与 FailedLimit 同向相容，但非因果 | T1_state 下反而更弱，LimitUp 上更强 |

应该写出的结论：

- CNN-specific 更像短窗、压力状态下的反弹/摩擦相关信号。
- MLP-specific 更像持续性 continuation 信号。
- T+1 / 涨跌停只能写成筛选性、相容性证据，不能写成已证明因果机制。

对应材料：

- `paper_results_E5.md`
- `paper_e5_mechanism_design.md`

---

## 5. 推荐论文结构

### Introduction

写作顺序：

1. 价格历史预测通常关心 representation：图像化 K 线是否优于表格 OHLCV？
2. 但在公平同信息集下，强 tabular nonlinear learners 可追平或超过 CNN。
3. 这提出更深问题：同一 OHLCV 为什么被不同模型解析成不同预测信号？
4. 本文贡献：benchmark、accessibility、complementarity、shared/specific decomposition、recent-path mechanism、economic realization。

### Data and Matched-information Design

放样本、A 股全市场、2000-2007 IS/val、2008-2025 OOS、20 日 OHLCV、120/180 维、Rank IC 与 long-short Sharpe。

### Benchmark Results

放 E1b + E2 主表 + MLP 稳定性。

主旨：CNN 有预测力，但不是最强；强非线性 tabular 证明问题不是 image superiority。

### Information Accessibility

放 E2A。

主旨：feature engineering 对线性模型帮助最大，说明差异是 extraction/accessibility。

### Complementarity

放 E2B 与 E2B-MLP。

主旨：相关中等、极端名单多数不重合、FM/残差/ensemble 支持可叠加信息。

### Signal Decomposition

放 E3。

主旨：Shared 最强；MLP-specific 与 CNN-specific 仍显著；HistGB-specific 被强 MLP 吸收。

### Recent-path Mechanism

放 E4。

主旨：CNN 与 MLP 的 specific 信号对应不同近端路径结构。

### Economic Realization

放 E5。

主旨：CNN 短窗、pressure/rebound-leaning；MLP persistent continuation；制度状态仅作相容性筛选。

### Conclusion

回到主题：同一信息不等于同一信号；模型比较不应只看谁赢，而应分解 representation-specific alpha。

---

## 6. 建议主表与主图

### 主表

1. Table 1：样本、split、模型、特征口径。
2. Table 2：E2 matched-information benchmark。
3. Table 3：E2A feature accessibility，120 -> 180 增量。
4. Table 4：E2B complementarity，相关、Top10 重叠、FM、ensemble。
5. Table 5：E3 shared/specific performance。
6. Table 6：E5 realization summary。

### 主图

1. Figure 1：论文证据链示意图：Same OHLCV -> Different representations -> Shared/specific -> Recent-path structure -> Economic realization。
2. Figure 2：MLP bagging curve，说明单 seed 不稳。
3. Figure 3：E2B score correlation / overlap 或 ensemble gain。
4. Figure 4：E3 shared/specific Sharpe bar。
5. Figure 5：E4 lag sensitivity。
6. Figure 6：E4 component sensitivity，CNN vs MLP normalized drop。
7. Figure 7：E5 horizon profile / pressure interaction。

---

## 7. 论文贡献写法

可以写成四点：

1. **Matched-information benchmark**：在同一 20 日 OHLCV 信息集下重新评估 chart-image CNN 与 tabular nonlinear learners，发现 image representation 并非稳定单模优势来源。
2. **Accessibility channel**：显式 K 线/量能方向指标主要帮助线性模型，说明 feature engineering 改变的是 information accessibility，而不是信息集本身。
3. **Complementary predictive signals**：CNN、MLP、HistGB 排序相关中等，控制彼此后仍有增量收益预测力，简单 ensemble 进一步提升 Sharpe。
4. **Economic interpretation of representation-specific alpha**：通过 shared/specific decomposition、路径干预和状态依赖分析，显示 CNN-specific 与 MLP-specific 对不同 recent-path structures 与收益兑现路径敏感。

---

## 8. 需要避免的强说法

不要写：

- CNN 显著优于所有 tabular models。
- MLP 全面证明 CNN 无意义。
- Feature engineering 增加了新信息。
- CNN-specific 已被证明来自 T+1 或涨跌停机制。
- E4 的每个干预单元格都是因果证明。

建议写：

- CNN 不提供稳定单模优势，但提供可叠加的 model-specific information。
- Bagged MLP 是强 tabular 上界；它吸收部分共变信号，但没有消灭 CNN-specific。
- 显式指标提高某些模型访问阈值/方向规则的能力。
- T+1 / price limit evidence is suggestive and state-consistent, not causal.
- 干预结果用于识别敏感性剖面，而非证明充分因果机制。

---

## 9. 摘要草稿

Using the same 20-day OHLCV information set in the Chinese stock market, we revisit whether chart-image CNNs outperform tabular models in return prediction. Under a fair chronological validation protocol, CNNs exhibit substantial predictability but do not consistently dominate strong nonlinear tabular learners: HistGB and especially bagged MLP achieve comparable or higher long-short Sharpe ratios. We show that explicit chart indicators mainly improve linear models, suggesting that feature engineering changes information accessibility rather than the information set itself. More importantly, CNN, MLP, and HistGB produce only moderately correlated rankings and retain incremental return predictability after controlling for one another; simple ensembles further improve performance. We decompose their predictions into shared and model-specific components and find that shared signals are strong, while CNN-specific and MLP-specific components remain economically meaningful. Path-level interventions reveal that CNN-specific signals are more sensitive to near-end visual path and volume-pressure structures, whereas MLP-specific signals align more with persistent continuation and tabular path geometry. These findings shift the role of representation from seeking a universally superior model to understanding how different nonlinear learners extract complementary trading signals from the same price history.

---

## 10. 中文摘要草稿

本文在中国 A 股全市场样本中，基于完全相同的 20 日 OHLCV 信息集，重新检验图像化 K 线 CNN 与表格非线性模型的收益预测能力。结果显示，在公平的时序验证协议下，CNN 具有显著预测力，但并不稳定优于强表格模型；HistGB 与稳定化 bagged MLP 可达到相当甚至更高的多空 Sharpe。进一步地，显式 K 线/量能方向指标主要提升线性模型，说明特征工程改变的是信息可达性，而非引入新的信息集。更重要的是，CNN、MLP 与 HistGB 的股票排序仅中等相关，在控制彼此后仍保留显著增量预测力，简单融合还能进一步提高组合表现。本文将模型预测分解为 shared 与 model-specific 信号，并通过路径干预和状态依赖分析表明：CNN-specific 更敏感于近端视觉路径、量能压力与短窗兑现，而 MLP-specific 更接近持续性动量和表格路径几何。由此，本文将模型表示的作用从寻找单一最优模型，转向解释不同非线性学习器如何从同一价格历史中提取互补的交易信号。

---

## 11. 最终标题候选

1. Same Information, Distinct Signals: Nonlinear Representations of Price History in China A-shares
2. Beyond Chart Images: Complementary Predictive Signals from the Same OHLCV History
3. What Do Chart CNNs Learn? Shared and Model-specific Signals in Stock Return Prediction
4. Representation-specific Alpha from Common Price Histories
5. From Image Superiority to Signal Complementarity: Evidence from China A-share OHLCV Data

