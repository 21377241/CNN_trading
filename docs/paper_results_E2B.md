# E2B 结果：CNN vs HistGB 信息互补（Fair Chrono）

> 设计：[`paper_e2b_cnn_histgb_complement.md`](paper_e2b_cnn_histgb_complement.md)  
> 面板：`E2_matched_is_ind_chrono_ric`（E1b-ric CNN + RankIC HistGB）；n=2,687,822；922 日  
> 产物：`WORK_SPACE/paper/E2B_complement_chrono_ric/`  
> 姊妹：CNN vs MLP [`paper_results_E2B_mlp.md`](paper_results_E2B_mlp.md)  
> 指标详解：[`paper_metrics_glossary.md`](paper_metrics_glossary.md)

---

## Track 总览

**背景**：E2 fair chrono 下 HistGB（Sharpe 3.07）≥ CNN E1b-ric（2.65），但两模型日相关仅 ~0.37——既非独立，亦非同一信号。需系统检验：**是否捕捉不同、可叠加的可交易信息？**

**逻辑链**（由浅入深）：

| 步骤 | 问什么 |
|------|--------|
| B1 | 分数**像不像**？（相关） |
| B2 | 极端选股名单**重叠多少**？ |
| B3 | 对截面收益是否有**独立解释力**？（FM） |
| B4 | 控制对方后，还能否**排出 spread**？（双重排序） |
| B5 | 正交化残差是否仍有 **alpha**？ |
| B6 | **等权融合**是否优于单模？ |

**固定协议**：全程 Chrono + Rank IC 定模；复用 `E2_matched_is_ind_chrono_ric/preds/`，**不重训**；对齐键 `(StockID, ending_date)`；OOS 2008–2025。

---

## 评测指标说明（本 Track）

| 步骤 | 指标 | 含义 |
|------|------|------|
| **B1** | 日截面相关 mean | 每日 Pearson(S_cnn, S_hgb) 再跨日平均；≈0.37 表示中等重叠、非复制 |
| **B2** | 重叠率 | CNN Top q% 中同时落在 HistGB Top q% 的比例；随机基准 Top10%≈10% |
| **B2** | Jaccard | 交集/并集；两集合相似度的对称度量 |
| **B3** | beta_bar, NW t | Fama–MacBeth：每日 ret ~ 截面 z-score(分数)；beta_bar 为系数日均，NW t 为 HAC 显著性 |
| **B4** | 条件 EW H-L Sharpe | 先按控制模型分 5 档，档内再按另一模型分 5 档取 H-L，quintile 等权平均 |
| **B5** | 残差 Rank IC / Sharpe | u = 分数对另一模型回归残差；rank 百分位后做十分位 H-L |
| **B6** | Ensemble Sharpe | S = 0.5·S_cnn + 0.5·S_hgb 后的 Rank IC 与 H-L Sharpe |

通用定义（Rank IC、EW H-L Sharpe 等）见 [`paper_metrics_glossary.md`](paper_metrics_glossary.md) §2–3、§6。

---

## B1 — 日截面相关

### 设计

| 项 | 设定 |
|----|------|
| **问题** | CNN 与 HistGB 的 OOS 分数在每个交易日截面上有多相似？ |
| **做法** | 每日计算 Pearson(S_cnn, S_hgb)，报告跨日 mean / median / std / 分位数 |
| **分层** | 按年；牛/熊（市场等权收益 vs 中位数）；高/低波（截面收益 std vs 中位数） |
| **解读基准** | ρ→1：几乎同一排序；ρ→0：独立；E2 主表 random-val 约 0.41，fair chrono 约 0.37 |

### 结果

| 统计 | 值 |
|------|-----|
| mean | **0.375** |
| median | 0.384 |
| std | 0.091 |
| p25 / p75 | 0.325 / 0.438 |
| 牛 / 熊 mean | 见 `b1_corr_summary.json`（分层差异不大） |
| 高波 / 低波 | 同上 |

### 结论分析

- 相关 **中等偏弱**（~0.37）：两模型共享一部分信号（同面板、同标签），但远非复制关系（若完全复制应 >0.9）。
- 牛熊 / 高低波分层差异不大 → 互补性**非单一 regime artifact**。
- 为 B2–B6 提供起点：既然不是同一分数，极端组合与收益解释是否仍独立？

---

## B2 — Decile 重合（选股名单重叠）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 两模型各自最看涨/最看跌的股票集合，重叠程度如何？ |
| **做法** | 每日各取 Top/Bottom 10% 与 20%；算重叠率（CNN∩HGB 占 CNN 比例）与 Jaccard |
| **随机基准** | 若独立排序，Top q 期望重叠 ≈ q（如 Top10% → 10%） |
| **意义** | 直接对应「若只跟一套模型选股，另一套会漏掉多少名字」 |

### 结果

| 集合 | 重叠率 mean（CNN∩HGB / CNN） | Jaccard mean |
|------|------------------------------|--------------|
| Top 10% | **0.256** | 0.148 |
| Bottom 10% | **0.258** | 0.149 |
| Top 20% | 0.372 | 0.230 |
| Bottom 20% | 0.372 | 0.230 |

### 结论分析

- Top10% 重叠 **~26%**，约为随机基准（10%）的 **2.5×** → 有共同「共识股」，但 **~74% 的 CNN Top 名不在 HistGB Top**（反之亦然）。
- Top20% 重叠 ~37%，约 1.9× 随机 → 放宽带宽后共享增多，但仍有大量非重合。
- 与 B1 一致：**部分独立排序信息**，支持「两路信号可叠加」而非「选其一即可」。

---

## B3 — Fama–MacBeth（截面收益解释）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 控制对方后，各模型分数对**下一期截面收益**是否仍有显著边际解释力？ |
| **做法** | 每日截面 z-score 标准化分数后 OLS：M1 ret~z(CNN)；M2 ret~z(HGB)；M3 ret~z(CNN)+z(HGB) |
| **汇总** | 日系数序列取均值 beta_bar；Newey–West t（lag=4）检验显著性 |
| **关键判据** | M3 中**两系数仍显著** → 非「一个已包含另一个的全部信息」 |

### 结果

| 模型 | beta_bar | NW t | p |
|------|----------|------|---|
| M1 CNN | 0.00119 | 9.34 | ≈0 |
| M2 HistGB | 0.00144 | 11.76 | ≈0 |
| M3 CNN（控 HGB） | **0.00078** | **6.32** | ≈0 |
| M3 HistGB（控 CNN） | **0.00112** | **9.39** | ≈0 |

### 结论分析

- M1/M2 均高度显著 → 单模分数对收益有**定价意义**（预期内）。
- **M3 双变量仍同时显著** → 核心结论：HistGB 控住 CNN 后 CNN 仍显著（t=6.32）；CNN 控住 HistGB 后 HistGB 更显著（t=9.39）→ **增量解释力双向成立**。
- M3 系数小于 M1/M2 → 共享方差被剔除，符合 B1/B2「部分重叠」；但未被完全吸收。
- HistGB beta_bar 略大 → 与 E2 单模 Sharpe HistGB > CNN 一致；B3 在**收益回归**层面给出同样排序。

---

## B4 — Conditional Double Sort（5×5）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 在「对方分数相近」的股票子集内，本模型是否仍能排出显著 H−L spread？ |
| **做法** | 5×5 条件双重排序：先按控制变量分 5 quintile，组内再按排序变量分 5 档，各 quintile 内算 EW H−L，再对 quintile **等权平均** |
| **两个方向** | CNN 控 HistGB；HistGB 控 CNN |
| **意义** | 比 B3 更接近**组合可交易性**：在控制组内做多空是否赚钱 |

### 结果

| 设定 | EW H-L 年化 | EW H-L Sharpe |
|------|-------------|---------------|
| CNN（控 HistGB） | 0.107 | **1.67** |
| HistGB（控 CNN） | 0.151 | **2.16** |

### 结论分析

- 两方向条件 Sharpe 均 **>1.5** → 即使 HistGB 分数五分位内，CNN 仍能排出显著 spread（Sharpe 1.67）；反之 HistGB 条件 Sharpe 更高（2.16）。
- HistGB 条件增量更强 → 与单模 Sharpe 优势一致；但 **CNN 1.67 不可忽略**（远超噪声水平）。
- 与 B3 互补：FM 看平均边际系数；双重排序看**分组组合**是否可交易。

---

## B5 — Residualization（正交化残差）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 去掉与对方分数的（线性/二次）相关后，残差是否仍有排序与组合价值？ |
| **做法** | 每日截面：S_cnn = α + γ·S_hgb + u（及反向）；另做 degree-2 多项式残差 |
| **评价** | 残差经截面 rank 百分位映射后，算 Rank IC 与 EW H-L Sharpe |
| **判据** | 正交化后 Sharpe 仍显著 >0 → 独立 alpha，非简单线性相关可解释 |

### 结果

| 分数 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| raw CNN | 0.0308 | 2.65 |
| raw HistGB | 0.0301 | 3.07 |
| **u_CNN 线性** | 0.0218 | **1.52** |
| **u_HGB 线性** | 0.0185 | **2.24** |
| u_CNN poly2 | 0.0214 | 1.50 |
| u_HGB poly2 | 0.0188 | 2.27 |

### 结论分析

- 线性正交化后：**u_CNN Sharpe 1.52、u_HGB Sharpe 2.24** → 双方均保留可观独立组合表现（相对 raw 下降但未消失）。
- poly2 与 linear 几乎相同 → 互补**不是**简单二次相关可吸收；独立信息超出低阶多项式关系。
- u_HGB 残差 Sharpe 高于 u_CNN → HistGB 去掉 CNN 后保留更多 alpha，与 B3/B4 一致。
- 为 B6 铺垫：若两路残差各有 alpha，等权融合应优于单模。

---

## B6 — Ensemble（等权融合）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 若信息互补，简单融合是否**同时**提升 IC 与 Sharpe？ |
| **做法** | S_ens = 0.5·S_cnn + 0.5·S_hgb；与单模比 Rank IC、EW H-L Sharpe |
| **意义** | 最直接的「互补 → 可叠加」检验；未做 w 网格，仅报告 w=0.5 |

### 结果

| 模型 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| CNN | 0.0308 | 2.655 |
| HistGB | 0.0301 | 3.068 |
| **0.5/0.5 ens** | **0.0363** | **3.426** |

### 结论分析

- Ensemble **同时**抬高 Rank IC（0.0363 > 0.0308/0.0301）与 Sharpe（3.43 > 3.07 > 2.65）。
- 相对最强单模 HistGB，Sharpe **再 +0.36** → 强互补证据；若两模型只是噪声不同，融合不应稳定提升。
- 与 B1–B5 形成闭环：中等相关 + 低重叠 + FM/双重排序/残差独立增量 → **等权融合有效**。

---

## Track B 综合结论

> Fair chrono 下 CNN 与 HistGB **既非同义重复，也非完全独立**：日相关 ~0.37、Top10% 重叠 ~26%（约 2.5× 随机）；FM 双变量、条件双重排序、正交化残差均显示**双向独立增量**；等权 ensemble Sharpe **3.43** 高于任一单模。
>
> **论文叙事**：E2「CNN ≈ / ≤ HistGB」不否定 CNN 价值——在 fair 定模下，CNN 提供 **HistGB 未覆盖的可叠加信号**；实践上简单 0.5/0.5 融合即可获益。
>
> **局限**：仅 EW、仅 w=0.5；VW 与 w 网格可作附录扩展。
