# E2A 结果：+60 维 indicators 机制（Chrono）

> 设计：[`paper_e2a_indicator_mechanism.md`](paper_e2a_indicator_mechanism.md)  
> 协议：Chrono Val（2000–2005 / 2006–2007）；120/180 复用 `val_auc` chrono 产物；不用随机 Val  
> 指标详解：[`paper_metrics_glossary.md`](paper_metrics_glossary.md)

---

## Track 总览

**背景**：E2 Chrono 设定下，120→180（+δ²⁰ / uᴾ / uⱽ）对 **Logit** 的 Rank IC 与 Sharpe 提升远大于 HistGB / MLP；需拆解「+60 维为何帮线性模型更多」。

**三个子问题**：

1. **A1**：+60 维带来的 Δ 是否对 Logit **统计上显著大于** HistGB / MLP？
2. **A2**：60 维 indicator 能否从 120 维 OHLC 结构**数值复现**（冗余性）？
3. **A3**：δ / uP / uV **各自**贡献多少（归因）？

**固定协议**：Train 2000–2005 / Val 2006–2007 / OOS 2008–2025；tabular 选参 `val_auc`；CNN 分数复用 I20R5（A1/A3 对照中不变）。

---

## 评测指标说明（本 Track）

| 指标 | 含义 | 本实验用法 |
|------|------|------------|
| **Rank IC** | 每个交易日截面上 Spearman(分数, 收益) 的跨日平均 | OOS 2008–2025；衡量排序质量 |
| **Δ Rank IC** | 180 维 − 120 维 的 Rank IC 差 | A1：+60 维带来的排序增益 |
| **EW H-L Sharpe** | 按分数十分位做多 D10、做空 D1 的周频多空 Sharpe（×52 年化，rf=0） | A1 Δ、A3 消融 |
| **Δ EW Sharpe** | 180 维与 120 维 Sharpe 之差 | A1 与组合表现挂钩 |
| **逐年 ΔIC > 0 比例** | OOS 各自然年中 Δ Rank IC > 0 的年份占比 | 增益是否跨年稳定 |
| **Wilcoxon p** | 对逐年 Δ Rank IC 做单侧检验（>0） | p 小 → 加维增益统计显著 |
| **OOS Acc** | 120 维回归预测 indicator，threshold@0.5 后的逐维分类准确率均值 | A2：二值块能否**判对 bit**；100% 表示完全复现 |
| **OOS R²** | 逐维 1−SS_res/SS_tot 再平均（连续回归拟合） | A2：辅助；对二值目标会低估「方向正确但幅度偏离 0/1」的情况 |

---

## A1 — 120 → 180 Δ（不重训）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 同一 tabular 模型在 120 维 vs 180 维下，OOS 表现差 Δ = 180 − 120 是否因模型类而异？ |
| **假设** | Δ_logit 在 Rank IC / Sharpe 上 **显著大于** Δ_histgb、Δ_mlp |
| **数据** | 复用 `E2_matched_is_chrono`（120）与 `E2_matched_is_ind_chrono`（180）的 OOS 预测；**不重训** |
| **模型** | logit / histgb / mlp（均已 chrono val + `val_auc` 网格选参） |
| **指标** | Δ Rank IC、Δ EW Sharpe；OOS 18 年逐年 ΔIC；Wilcoxon 单侧检验（ΔIC > 0）及 logit vs 他者的配对检验 |
| **脚本** | `experiments/E2_matched_tabular/run_e2a_delta_compare.py` |
| **产物** | `WORK_SPACE/paper/E2A_delta_chrono/` |

### 结果

| 模型 | Δ Rank IC | Δ EW Sharpe | 逐年 ΔIC>0 比例 | Wilcoxon p（ΔIC>0） |
|------|-----------|-------------|-----------------|---------------------|
| **logit** | **+0.0049** | **+0.301** | **94%** | **1.1e-5** |
| histgb | +0.0027 | +0.121 | 78% | 0.0024 |
| mlp | −0.0014 | −0.293 | 44% | 0.77 |

配对检验（逐年 ΔIC）：

- logit − histgb：mean +0.0022，72% 年 logit 更大，Wilcoxon p=**0.013**
- logit − mlp：mean +0.0062，78% 年，p=**0.009**

### 结论分析

- **主假设成立**：+60 维对 Logit 的 IC 与 Sharpe 增益在**幅度与跨年稳定性**上均显著强于 HistGB / MLP（94% 年份 ΔIC>0，Wilcoxon p≈0）。
- **HistGB 有增益但更小**：树模型已能从 OHLC 非线性结构部分「内生」方向信息，显式比特的边际价值较低（Δ Sharpe +0.12 vs Logit +0.30）。
- **MLP 全量 60 维略伤**：ΔIC 为负、44% 年份为正、Sharpe −0.29 → 对浅层 MLP，额外 indicator 更像噪声或冗余，与 E2 §4 180 维 MLP Sharpe 下降一致。
- **机制指向**：增益集中在「**线性模型装不下的非线性结构**」；A2/A3 进一步定位到具体 indicator 块。

---

## A2 — 120 → 60 可恢复性

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 60 维 indicator（δ²⁰ / uᴾ / uⱽ）能否从 120 维 base 特征**数值复现**？复现率高 → 加列冗余；复现率低 → 120 维缺显式信息 |
| **任务** | 用 X120 回归预测 Y60（整块或分块 δ / uP / uV）；目标均为 {0,1} 二值 |
| **回归器** | Ridge（线性）、MLPRegressor（非线性）、HistGradientBoostingRegressor（树）；Ridge 在 chrono val 上选 α |
| **指标** | 回归输出 **threshold@0.5 → Acc** 与 **连续 R²** 均报告（块内逐维平均）；Ridge 在 chrono val 上按 **Acc** 选 α；OOS 2008–2025 |
| **注意** | uP = 1{C≥O}、uV 为量柱方向，本为 X120 中 O/C 的阈值函数；本实验测的是「**专用映射能否复现 bit**」，非 end-to-end 收益预测 |
| **脚本** | `experiments/E2_matched_tabular/run_e2a_recover_60.py` |
| **产物** | `WORK_SPACE/paper/E2A_recover_60/recover_summary.csv` |

### 结果

**OOS Accuracy**（threshold@0.5，主指标）

| 目标 | Ridge Acc | MLP Acc | HGB Acc |
|------|-----------|---------|---------|
| 整块 60 | 96.7% | **98.8%** | 97.3% |
| δ²⁰（20） | **100%** | **100%** | **100%** |
| uᴾ（20） | 95.1% | **98.5%** | 96.0% |
| uⱽ（20） | 95.0% | **98.1%** | 95.8% |

**OOS R²**（连续回归，辅助）

| 目标 | Ridge R² | MLP R² | HGB R² |
|------|----------|--------|--------|
| 整块 60 | 0.62 | 0.61 | **0.84** |
| δ²⁰（20） | **1.00** | 0.00* | **1.00** |
| uᴾ（20） | 0.43 | **0.93** | 0.76 |
| uⱽ（20） | 0.43 | **0.90** | 0.76 |

\*MLP 对 δ 输出未饱和连续值，R²=0 但 Acc=100%（阶跃目标 + 回归器失配）；不影响「δ 可完全复现」结论。

### 结论分析

- **δ 完全冗余**：Acc 均 100%；Ridge/HGB R²=1 → 窗内确定性阶跃；与 A3「+δ only 零增益」一致。
- **uP/uV：Acc 与 R² 给出互补视角**——Ridge Acc ~95% 但 R² ~0.43：专用回归**方向大多判对**，连续预测离 0/1 仍远；MLP Acc ~98% 且 R² ~0.93：非线性映射两者皆优。
- **end-to-end Logit 仍需要显式列**：尽管 uP/uV 可数值复现（Acc），A3 显示 Logit 加 uP/uV 仍 +0.28~0.30 Sharpe → 瓶颈是 **SGD 线性分类器未自动提取 C≥O 规则**，而非 X120 缺信息。
- **非线性 > 线性**：HGB/MLP R² 与 Acc 均高于 Ridge → 与 A1 中 HistGB/MLP 的 Δ 小于 Logit 一致。

---

## A3 — 单指标消融（140 维补训）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | +60 维的总增益来自 δ、uP、uV 中的哪一块？ |
| **设定** | 从 180 维特征缓存抽列，仅补训缺失的 140 维设定；120 / 180 全量复用 chrono 产物 |
| **对照** | 120 基线 → +δ only（140）→ +uP only → +uV only → +all 180 |
| **模型** | logit / histgb / mlp；chrono val + `val_auc` 选参（与 A1 可比） |
| **指标** | OOS EW H-L Sharpe（主）；Rank IC 见各产物目录 |
| **脚本** | `experiments/E2_matched_tabular/run_e2a_ablation_train.py` |
| **产物** | `WORK_SPACE/paper/E2A_ind_{delta,up,uv}_chrono/` |

### 结果

相对 Chrono 120 基线（logit Sharpe 1.17 / histgb 2.90 / mlp 3.26）：

| 设定 | logit Sharpe | histgb Sharpe | mlp Sharpe |
|------|--------------|---------------|------------|
| 120 基线 | 1.17 | 2.90 | 3.26 |
| **+δ only** | 1.17（+0） | 2.90（+0） | 3.15 |
| **+uP only** | **1.45**（+0.28） | **3.13**（+0.23） | 2.79 |
| **+uV only** | **1.47**（+0.30） | 2.70 | **3.37** |
| +all 180 | **1.47**（+0.30） | 3.02 | 2.97 |

### 结论分析

- **Logit：增益几乎全部来自 uP / uV**；+δ 零增量；+uP 或 +uV 单独即可达到全量 180 水平（Sharpe 1.47）→ **方向比特是线性模型的关键增量**，δ 可忽略。
- **HistGB：uP 有帮助，uV 单独略负**；全量 180 居中（3.02）→ 树对两类方向的利用不对称，但总体仍能从 uP 受益。
- **MLP：单指标不稳定**（uV 升、uP 降；全量 180 低于 120 基线 3.26）→ 与 A1「MLP 全量 60 维略伤」一致；非线性模型不依赖显式比特，额外列可能引入噪声。
- **三实验合读**：A3 给**因果归因**（谁有用），A2 给**冗余性**（为何 δ 无用、uP/uV 为何仍值得显式给 Logit），A1 给**跨模型统计检验**（Logit Δ 显著更大）。

---

## Track A 综合结论

> +60 维中真正帮到线性**收益预测**的是 **K 线/量柱方向比特 uP/uV**（A3）；δ²⁰ 100% 可复现、零增量（A2+A3）。uP/uV 虽可由 X120 数值复现 ~95%（A2 Acc），但 end-to-end Logit 无法自动学到阈值规则，故 A1 中 Logit Δ 显著大于 HistGB / MLP。
>
> **论文叙事**：E2 180 维下 Logit 跃升并非「多了 60 维信息」，而是「多了线性模型读不出的显式方向特征」。
