# E4 结果：Decomposing Recent-Path Predictability

> 设计全文：[`paper_e4_recent_path_design.md`](paper_e4_recent_path_design.md)  
> 接续：[`paper_results_E3.md`](paper_results_E3.md)（两边都依赖近 5 日；E4 拆结构）  
> 面板：`E2_matched_is_ind_chrono_ric`；n=2,687,822；922 日；OOS 2008–2025  
> 产物：`WORK_SPACE/paper/E4_recent_path_chrono_ric/`  
> 脚本：`experiments/E4_recent_path/`  
> 指标详解：[`paper_metrics_glossary.md`](paper_metrics_glossary.md)

---

## Track 总览

**背景**：E3 已表明 CNN-spec 与 MLP-spec（bag k=10）都近乎必要地依赖近 5 日信息 \(W^{(5)}\)。E4 不再比“谁更强”，而专门回答：

\[
\boxed{
\text{CNN 与 MLP-Bag10 都依赖最近 5 天，但各自利用近 5 日内的哪些不同结构？}
}
\]

\[
W^{(5)}_{i,t}=\{O,H,L,C,V\}_{t-4:t}
\]

**逻辑链**：

| 步骤 | 问什么 | 证据类型 |
|------|--------|----------|
| E4-1 | 近 5 日中**哪一天**最重要？ | Lag flatten / drop → ΔSharpe\(^{sp}\) |
| E4-2 | 控住 CumRet5 后，**路径形状**还有增量吗？ | 嵌套截面 \(R^2\) + NW t |
| E4-3 | Volume / Wick / Direction / Return / Gap / Intraday 哪块伤 specific？ | Matched 语义干预 |
| E4-4 | 同 CumRet5 桶内，PathShape 是否仍分离 \(S^{sp}\)？ | 双重排序 spread |
| E4-5 | 极端分歧股票在 Recent-5 上差什么？ | 80/20 ΔX + NW t |

**叙事分支（预注册）**：

- 若 lag / component 剖面几乎相同 → *different functions, same short-horizon structure*  
- 若明显不同 → *representation-specific transformation of recent price paths*

---

## 实验方案（锁定协议）

### 模型与干预方式

| 项 | 设定 |
|----|------|
| **CNN** | E1b-ric 5-model ensemble；OHLCV 变换 → `DrawOHLC` → 前向；**不重训** |
| **MLP** | chrono Rank-IC early-stop + bag k=10；删列后 **同超参重训** |
| **HistGB** | 冻结超参；仅 E4-1/3 tabular 附录，不进主叙事 |
| **Panel** | OOS 2008–2025 common panel；对齐键 `(StockID, ending_date)` |
| **配对原则** | 同语义干预在 CNN（改 OHLCV）与 MLP（删列）各做一次，避免“一边 mask 一边删列”不可比 |

### Specific 构造（相对 E3 收窄）

E4 主角为 **两元**残差（不用三元含 HGB）：

\[
S_C^{sp}=S_C-\widehat{\mathbb{E}}[S_C\mid S_M],\qquad
S_M^{sp}=S_M-\widehat{\mathbb{E}}[S_M\mid S_C]
\]

- 每日截面 OLS（带截距）；当日 \(n\ge 50\)  
- 干预后评价时，**partner = 未扰动主线分数**（只扰动一侧）  
- 相对 E3：`mask_last5`（末 5 日整窗均值平板）→ E4 用单日 `flatten_day(k)` 钉到 \(C_{k-1}\)

### 评价指标与读表

| 指标 | 含义 | 怎么读 |
|------|------|--------|
| **Rank IC** | 每日 Spearman(分数, `ret_val`) 再跨日平均 | specific 排序力 |
| **EW / VW H-L Sharpe** | 十分位多空，`ann_factor=52` | 经济幅度；主看 **EW** |
| **ΔSharpe / ΔIC** | `baseline − intervened` | **正 = 变差**（干预伤了 specific） |
| **drop%** | \(\Delta Sharpe / |Sharpe_{base}| \times 100\) | 相对损伤；跨模型比剖面时用 **normalized drop** |
| **ΔR²** | \(R^2(CumRet5+Path)-R^2(CumRet5)\) | 控动量后的路径增量 |
| **NW t**（lag=4） | 日序列 HAC t | \|t\|>1.96 ≈ 5% 显著 |

### 预注册判定

| 假设 | 标准 | 本轮结果 |
|------|------|----------|
| **H1** 位置异质性 | 五 lag 中 max ΔSharpe ≥ 2× min（至少一侧） | **通过（两侧）**：CNN≈5.1×，MLP≈3.9× |
| **H2** PathShape 增量 | 控 CumRet5 后 ΔR²>0 且 NW 显著（至少一侧） | **通过（两侧）** |
| **H3** 成分剖面不同 | 六块 normalized drop：\(Corr(v_C,v_M)<0.7\) | **通过**：Corr=**0.29** |

---

## 0. 总判

两边都吃近 5 日，但 **不是同一短窗结构**：

- **CNN-spec**：日位置**尖锐**（t−1 / t / t−3）；成分上第一刀是 **Volume**，并吃回撤/波动路径形状。  
- **MLP-spec**：日位置更**平坦**、偏近端；成分上第一刀是 **Intraday**，Wick / Direction 次之；CumRet5 对分数解释力更强，但控动量后 Path 仍有显著增量。

→ 落入叙事分支二：**representation-specific transformation of recent price paths**。

---

## E4-1 — Lag sensitivity（图 1）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | \(W^{(5)}\) 内哪一日对 \(S^{sp}\) 最必要？ |
| **CNN** | `flatten_day(k)`，\(k\in\{15..19\}\)（lag \(t-4..t\)）：\(O=H=L=C=C_{k-1}\)，\(V=\)局部均值 |
| **MLP / HistGB** | `drop_day(k)`：删该日全部 9 列后重训 |
| **评价** | 干预后分数对未扰动 partner 残差化 → EW H-L Sharpe；报 Δ 与 drop% |

### 结果

Baseline EW Sharpe：CNN-sp **1.29**；MLP-sp **3.00**。

| Lag | CNN ΔSharpe | CNN drop% | MLP ΔSharpe | MLP drop% | HistGB drop% |
|-----|-------------|-----------|-------------|-----------|--------------|
| t−4 | 0.17 | 13% | −0.06 | −2% | 0.5% |
| t−3 | **0.71** | **55%** | −0.01 | −0% | −3% |
| t−2 | −0.20 | −16% | 0.05 | 2% | 5% |
| t−1 | **0.85** | **66%** | 0.10 | 3% | 12% |
| t   | **0.69** | **54%** | **0.21** | **7%** | **20%** |

### 解读

1. **H1 通过**：CNN 正损伤中 max/min≈5.1；MLP≈3.9——位置异质性成立。  
2. CNN 对 **单日** 极度敏感且**非单调**（t−2 flatten 反而升 Sharpe），说明图像表示把“某一根 K 线”编码成高杠杆特征，而不是均匀摊在 5 日上。  
3. MLP 删单日最多掉 ~7%：specific 更依赖**跨日联合**，单日可替代性强；但仍显示 **越近越重要** 的弱梯度。  
4. HistGB 附录落在两者之间（近端更重），支持“tabular 近端梯度”稳健，但幅度远小于 CNN。

→ **RQ1**：同一近 5 日窗内，CNN 的“日位置”远比 MLP 尖锐。

---

## E4-2 — CumRet5 + PathShape（图 3）

### 设计

| 项 | 设定 |
|----|------|
| **问题** | \(S^{sp}\) 是否只是 5 日动量水平？控住 CumRet5 后路径形状还有解释力吗？ |
| **Y** | 每日 cs_zscore(\(S_C^{sp}\)) / cs_zscore(\(S_M^{sp}\)) |
| **嵌套** | \(R^2(CumRet5)\) vs \(R^2(CumRet5 + Path)\)；Path = LateEarly, PathVol, UpCount5, MaxDD5, Reversal5, Monotonicity |
| **推断** | 日度 ΔR² 序列 + NW lag=4 |

### 结果

| Score | \(R^2(CumRet5)\) | \(R^2(+Path)\) | ΔR² | NW t |
|-------|------------------|----------------|-----|------|
| \(S_C^{sp}\) | 0.021 | 0.072 | **0.052** | 45.6 |
| \(S_M^{sp}\) | 0.048 | 0.089 | **0.041** | 36.6 |

单特征增量（控 CumRet5 后 ΔR²，按大到小）：

| 排序 | CNN | MLP |
|------|-----|-----|
| 1 | MaxDD5 (0.021) | Monotonicity (0.011) |
| 2 | PathVol (0.021) | MaxDD5 (0.011) |
| 3 | LateEarly (0.016) | PathVol (0.010) |
| 4 | Reversal5 (0.015) | LateEarly (0.008) |

### 解读

1. **H2 通过**：两侧 ΔR²>0 且 NW 极显著——specific **不是**纯 CumRet5。  
2. MLP 的 CumRet5 基准 R²（0.048）明显高于 CNN（0.021）：tabular specific 更“像动量水平”；但 Path 增量仍接近 CNN 量级。  
3. CNN 更吃 **MaxDD / PathVol / LateEarly**（回撤与波动路径）；MLP 相对更吃 **Monotonicity**（走势是否单向）。  
4. 这为 E4-3/4 预告：即使控住收益水平，形状通道仍可能分开两边。

---

## E4-3 — Matched blocks（图 2）

### 设计

同语义六块；CNN 改近 5 日 OHLCV 后重画；MLP/HistGB 只删 recent5 对应列后重训。

| Block | CNN 干预 | MLP 删列（recent5） |
|-------|----------|---------------------|
| Volume | V → 更早窗均值 | Vn |
| Wick | flatten wick（H/L→body） | H, L |
| Direction | 中和 C−O 符号，保 range | uP, uV |
| Return | cumret 归零，保 H−L | C, delta |
| Gap | \(O_k=C_{k-1}\)，平移 H/L/C | O |
| Intraday | 压缩 body（O=C=mid），保 H/L | O, C, uP |

评价同 E4-1；H3 用六维 **normalized** \(\Delta Sharpe\) 向量相关。

### 结果

| Block | CNN ΔSharpe | CNN drop% | MLP ΔSharpe | MLP drop% | HistGB drop% |
|-------|-------------|-----------|-------------|-----------|--------------|
| Volume | **0.44** | **34%** | 0.16 | 5% | 2% |
| Wick | 0.05 | 4% | 0.34 | 11% | 3% |
| Direction | 0.21 | 16% | 0.30 | 10% | 9% |
| Return | 0.04 | 3% | 0.07 | 2% | 7% |
| Gap | −0.17 | −13% | 0.08 | 3% | 9% |
| Intraday | 0.19 | 15% | **0.49** | **16%** | **12%** |

Normalized drop 相关：**Corr(\(v_C,v_M\))=0.29**。

### 解读

1. **H3 通过**：相关远低于 0.7，拒绝 “same structure”。  
2. **CNN 第一刀 Volume**：图像里量柱是显式通道；抹掉近端量能路径会大幅伤害 CNN-spec。MLP 对 Volume 块仅弱损伤 → 两边不是在用同一套“放量/缩量”规则。  
3. **MLP 第一刀 Intraday**，其次 Wick / Direction：表格特征里 body / 影线 / 方向指示可直接列示，删掉后 specific 掉得最明显。  
4. **纯 Return 块两边都几乎不伤**：与 E4-2 一致——specific 剩余部分更像形状/几何，而不是再砍一层 cumret。  
5. CNN 的 Gap 干预 **升** Sharpe（Δ 为负）：隔夜缝合可能破坏了 CNN 实际依赖的视觉线索，或引入与 partner 更正交的噪声；不解读为“Gap 对 CNN 有害必要”，只说明 Gap 不是 CNN-spec 的主轴（E4-5 中 Gap 亦不显著）。

→ **RQ2**：成分敏感性剖面分离——CNN≈量能路径，MLP≈日内几何。

---

## E4-4 — Matched-Return path

### 设计

| 项 | 设定 |
|----|------|
| **问题** | 在 **相同 CumRet5 水平** 内，LateEarly 是否仍拉开 \(S^{sp}\)？ |
| **排序** | 先 CumRet5 十分位；桶内再 LateEarly 五分位；比较 Q5−Q1 的 \(S^{sp}\) 均值差 |
| **推断** | 日度 spread + NW；并报各动量十分位与 pooled |

### 结果

| | Pooled \(S^{sp}\) spread | NW t | 备注 |
|--|--------------------------|------|------|
| \(S_C^{sp}\) | **0.019** | 30.1 | 几乎每个 CumRet 十分位内显著；高动量端更大 |
| \(S_M^{sp}\) | 0.003 | 2.65 | 弱；最高 CumRet 十分位 **反号**（−0.010） |

（未来收益 spread 在多数桶不显著——本实验主目标是 **分数对路径的条件敏感**，不是再刷一层可交易 alpha。）

### 解读

1. 控住 5 日收益水平后，**路径形状主要仍是 CNN-spec 的鉴别器**。  
2. 与 E4-2 一致：MLP-spec 更已被 CumRet5 吸收；CNN-spec 在“同涨同跌”股票里仍区分 late vs early 路径。  
3. 这强化论文说法：互补不只来自“一个看动量、一个不看”，而是 **同动量下的路径编码不同**。

---

## E4-5 — Disagreement × Recent-5

### 设计

| 项 | 设定 |
|----|------|
| **问题** | CNN 强烈看多、MLP 强烈看空的股票，在 **仅近 5 日** 特征上差什么？ |
| **样本** | 每日 80/20：\(C^+M^-\) vs \(C^-M^+\)；两侧各 \(n\ge5\) |
| **特征** | CumRet5, LateEarly, PathVol, UpCount5, MaxDD5, Reversal5, Monotonicity, WickMean5, GapMean5, VolEndOverMean5 |
| **统计** | \(\Delta X_t=\bar X_{C^+M^-}-\bar X_{C^-M^+}\)；跨日 NW |

### 结果

| 特征 | Δ | NW t |
|------|---|------|
| MaxDD5 | +0.016 | 28.1 |
| CumRet5 | +0.024 | 19.3 |
| Monotonicity | +0.198 | 19.0 |
| LateEarly | +0.015 | 16.4 |
| UpCount5 | +0.192 | 12.6 |
| VolEndOverMean5 | +0.130 | 12.4 |
| Reversal5 | +0.005 | 11.0 |
| PathVol | −0.003 | −10.6 |
| WickMean5 | +0.011 | 8.8 |
| GapMean5 | ≈0 | n.s. |

### 解读

1. \(C^+M^-\) 侧：更高 CumRet / 更单调 / 更深回撤、相对更高量能末端、**更低 PathVol**——像“近端顺势但路径不那么毛刺”的股票。  
2. 与 E4-2/3 交叉：动量+回撤+量能出现在分歧轴上，符合 CNN 对 Volume / MaxDD 敏感、MLP 更吃水平动量与几何的分工。  
3. **Gap 不显著**：与 E4-3 中 Gap 非 CNN 主损伤轴一致——隔夜缺口不是这对模型吵架的主因。

---

## 结论分析

### 回答两个 RQ

| RQ | 答案 |
|----|------|
| **RQ1（哪一天）** | CNN：t−1 / t / t−3 极关键，损伤非均匀；MLP：平坦近端梯度，单日不可替代性弱。 |
| **RQ2（什么结构）** | 控 CumRet5 后两边都有 Path 增量；成分上 CNN→**Volume（+路径波动/回撤）**，MLP→**Intraday / Wick / Direction**。 |

### 对论文叙事的含义

1. **拒绝** “CNN 与 bagged MLP 只是同一 recent-5 动量的两种函数近似”。E4-2 的 ΔR²、E4-3 的低相关 drop 向量、E4-4 的 matched-return 分离，构成三角互证。  
2. **接受** “同一 \(W^{(5)}\) 经不同表示 inductive bias 抽出不同可交易残差”：  
   - 图像 CNN 显式画出 **量柱与 K 线序列位置** → 对单日与 Volume 杠杆高；  
   - 表格 MLP 直接吃 **标准化 OHLC 几何列** → 对日内 body/影线/方向更敏感，对单日更可替代。  
3. 这解释 E2B/E3 的残差与 ensemble 增益：**Shared** 吸收了重叠的近端动量；**Specific** 分别挂在量能路径 vs 日内几何上，组合仍有增量。  
4. **不等价声明**：干预是同语义近似配对，不是证明因果充分；CNN Gap 升 Sharpe、MLP 删 t−4 微升等现象说明局部替代与重训噪声存在，主结论应读 **剖面相对差异**，而非每个单元格的点估计神话。

### 局限与下一步

| 局限 | 含义 |
|------|------|
| CNN 不重训 / MLP 重训 | 协议不对称是设计选择（匹配部署方式）；数字不宜逐点横比绝对值 |
| Bagged MLP vs 单 HistGB | 附录 HistGB 仅 robustness；主叙事保持 CNN↔MLP |
| 经济机制（涨跌停、T+1 等） | E4 停在**价格路径结构**；制度/行为链接仍属 Part V |

**建议写作顺序**：先 E3 Shared/Specific 必要性（近 5 日）→ 再 E4 拆成 lag + PathShape + blocks → 用 H1–H3 收束为 representation-specific recent-path → 最后才进机制章节。

---

## 文件索引

| 实验 | 主表 |
|------|------|
| Baseline scores | `baseline_scores.parquet` / `pathshape.parquet` |
| E4-1 | `e41_lag/{cnn,mlp,histgb}/e41_*_lag_summary.csv` |
| E4-2 | `e42_pathshape/e42_delta_r2_summary.csv` |
| E4-3 | `e43_blocks/{cnn,mlp,histgb}/e43_*_block_summary.csv` |
| E4-4 | `e44_matched_return/e44_summary.csv` |
| E4-5 | `e45_disagreement/e45_delta_summary.csv` |
