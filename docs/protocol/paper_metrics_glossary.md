# 评测指标说明（E1/E2 实验通用）

> 实现参考：`experiments/common/metrics.py`、`experiments/E2_matched_tabular/run_eval.py`、`Portfolio/portfolio.py`

本文档解释论文实验结果表中各列的含义、计算方式与阅读要点。OOS 区间均为 **2008–2025**；收益频率为 **周频**（`ending_date` 为信号期末），年化因子 **`ann_factor = 52`**。

---

## 1. 预测分数与标签

| 符号 / 列 | 含义 |
|-----------|------|
| `up_prob` | 模型输出的「下一期上涨」概率或分数；**越大表示模型越看涨**。CNN / tabular 分类器直接输出；E2B 残差分数会先转为截面 rank 百分位再进组合模块。 |
| `binary_label` | 二分类标签：下一期收益 `ret_val > 0` 为 1，否则为 0。 |
| `ret_val` | 实现收益（周频），用于 IC、组合收益与 FM 回归因变量。 |

---

## 2. 分类与排序指标

### AUC（Area Under ROC Curve）

- **定义**：全样本上 `roc_auc_score(y, up_prob)`，衡量分数区分涨跌标签的能力。
- **范围**：0.5 = 随机；越高越好。
- **注意**：AUC 是**全局**指标，不直接反映截面排序能否赚钱；E2 主结论更看 Rank IC 与 H-L Sharpe。

### Rank IC（截面 Spearman 相关）

- **定义**：每个 `ending_date`，在当日股票截面上计算 Spearman(`up_prob`, `ret_val`)，再对全部交易日取**算术平均**。
- **含义**：分数能否把**当周**涨得多的股票排在前面；是资产定价里常用的「信息系数」。
- **选参**：`val_rank_ic` = 仅在 Val 期（如 2006–2007）上算的 Rank IC，用于网格选超参（§5 公平 chrono）。

### IC（截面 Pearson 相关）

- 与 Rank IC 类似，但用 Pearson 相关；对极端值更敏感。E2 主表以 **Rank IC** 为主。

### 逐年 Rank IC / Δ Rank IC

- **逐年 Rank IC**：按日 Rank IC 在**自然年**内再平均，得到每年一个数。
- **Δ Rank IC（E2A）**：同一模型在 180 维与 120 维设定下，逐年 Rank IC 之差（180 − 120）。正值表示加 60 维 indicators 后排序更好。

### 年份 Rank IC 胜率

- OOS 各年中，CNN 逐年 Rank IC **高于** HistGB 的年份比例（如 13/18 → 0.72）。用于预注册判定「CNN 是否在排序上稳定优于 HistGB」。

---

## 3. 组合指标（十分位 H-L）

### 十分位组合（Decile）

- 每个 `ending_date`，按 `up_prob` **升序**分成 10 组（`pd.qcut`；股票不足时自动降档）。
- **D10** = 分数最高组（最看涨），**D1** = 最低组。

### EW H-L（等权 High − Low）

- **定义**：每期收益 = D10 组内股票 `ret_val` 的**等权平均** − D1 组等权平均；得到一条周频 H-L 收益序列。
- **EW H-L 年化**：`mean(周频 H-L) × 52`（与 `portfolio.py` 中 compound 年化略有不同处，E2B 双重排序用简单 ×52）。
- **EW H-L Sharpe**：`年化收益 / 年化波动`，年化波动 = `std(周频 H-L) × √52`，**无风险利率 = 0**。

### VW H-L（市值加权 High − Low）

- 组内收益按 `MarketCap` 加权；其余同 EW。反映「大资金可交易性」下的排序效果，通常低于 EW。

### Corr(S_cnn, S_histgb)

- 每个交易日截面上 CNN 与 HistGB 分数的 Pearson 相关，再对日取平均。衡量两模型分数是否「说同样的话」。

---

## 4. 统计检验（E2A）

### Wilcoxon 符号秩检验

- **用途**：检验「180 维 − 120 维」的**逐年 Δ Rank IC** 是否**显著大于 0**（`alternative='greater'`）。
- **解读**：p 小 → 加维带来的 IC 提升在年份维度上稳健，而非个别年份驱动。
- **配对比较**：logit vs histgb 等，先算两模型逐年 Δ 的差值，再对差值做 Wilcoxon。

### 逐年 ΔIC > 0 比例

- OOS 年份中 Δ Rank IC > 0 的占比；与 Wilcoxon 互补（比例看频率，p 值看显著性）。

---

## 5. 可恢复性 Acc / R²（E2A A2）

- **任务**：用 120 维特征回归预测 180 维中的 60 维 indicator 块（整块或 δ / uP / uV 子块）；目标均为 **{0,1} 二值**。
- **Ridge / MLP / HGB**：在 IS train 上拟合，chrono val 选参（Ridge 按 **val Acc** 选 α）；**OOS Acc** 与 **OOS R²** 均在 2008–2025 上算。
- **Acc 定义**：回归预测值 ≥ 0.5 判为 1，否则 0；逐维分类准确率，块内平均。**主指标**。
- **R² 定义**：逐维 `1 − SS_res/SS_tot` 再平均。**辅助**；对二值目标，Ridge 可 Acc≈95% 而 R²≈0.43（方向对、连续值未贴近 0/1）。
- **解读**：Acc→100%（δ）→ 完全可复现；uP/uV 的 Acc 与 R² 互补；end-to-end Logit 仍从显式列受益见 A3。

---

## 6. 互补性指标（E2B）

### B1 日截面相关

- 每日 Pearson(`S_cnn`, `S_hgb`)；表内 mean/median/std 为**跨日**统计。
- 0.37 表示中等正相关：有重叠信号，但不是同一排序。

### B2 Decile 重合

| 指标 | 定义 |
|------|------|
| **重叠率** | 某日 CNN 的 Top q%（或 Bottom q%）中，同时落在 HistGB 同侧 q% 内的比例 = &#124;CNN ∩ HGB&#124; / &#124;CNN&#124;。 |
| **Jaccard** | &#124;CNN ∩ HGB&#124; / &#124;CNN ∪ HGB&#124;；对两集合相似度更对称。 |
| **随机基准** | 若两排序独立，Top 10% 期望重叠 ≈ 10%（0.1×0.1 / 0.1 = 0.1）。 |

### B3 Fama–MacBeth

- **截面回归**（每日）：`ret_val ~ z(S_cnn)` 或 `~ z(S_hgb)` 或 `~ z(S_cnn) + z(S_hgb)`，其中 z 为**当日**截面标准化。
- **报告量**：各回归系数在**全部交易日**上的时间序列均值 \(\bar\beta\)；**NW t** = Newey–West HAC t 统计量（lag=4），检验 \(\bar\beta ≠ 0\)。
- **解读**：M3 双变量中两系数仍显著 → 控制对方后仍有**增量**解释力。

### B4 条件双重排序（5×5）

- 先按**控制变量**（如 HistGB 分数）分成 5 个 quintile；在**每个** quintile 内再按**排序变量**（如 CNN）分 5 组，算组内 EW H-L。
- **报告**：各 quintile 的 H-L 再对 quintile **等权平均**，得到每日一条 H-L；Sharpe 计算同 §3。
- **含义**：「HistGB 分数相近的股票里，CNN 还能不能排出 spread？」→ 条件独立增量。

### B5 残差化

- **线性**：`u_cnn = S_cnn − E[S_cnn | S_hgb]`（OLS）；`u_hgb` 对称。
- **poly2**：用 HistGB 的二次项回归 CNN，再取残差。
- 残差经**截面 rank 百分位**映射后做十分位 H-L；若残差仍有 Sharpe / Rank IC → 正交化后仍有**独立**排序信息。

### B6 Ensemble

- `S_ens = w · S_cnn + (1−w) · S_hgb`；默认 **w = 0.5**。再按 §3 算 Rank IC 与 H-L Sharpe。

---

## 7. 预注册判定码（E2 主表）

| 代码 | 含义（相对 HistGB 网格最优） |
|------|------------------------------|
| `CNN_gt_HistGB` | Sharpe 差 ≥ 0.2 **或**（Sharpe 差 > 0 且年份 Rank IC 胜率 ≥ 2/3） |
| `CNN_approx_HistGB` | Sharpe 差 < 0.2 但 IC 胜率仍 ≥ 2/3 |
| `CNN_le_HistGB` | CNN Sharpe **低于** HistGB |

（具体阈值以 `run_eval.py` 中判定逻辑为准。）

---

## 8. 快速对照：该看哪个指标？

| 你想知道… | 优先看 |
|-----------|--------|
| 分数能否把股票排对 | **Rank IC**、逐年 Rank IC |
| 多空策略赚不赚钱 | **EW H-L Sharpe**、EW H-L 年化 |
| 标签分类准不准 | **AUC** |
| 加特征有没有用 | **Δ Rank IC**、**Δ Sharpe**、Wilcoxon p |
| 特征是否冗余 | **OOS Acc / R²**（A2）、单指标消融 Sharpe（A3） |
| 两模型是否互补 | **日相关**、**Top10% 重叠**、FM 双变量、**ensemble Sharpe** |
