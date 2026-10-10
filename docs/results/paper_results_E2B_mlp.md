# E2B-MLP 结果：CNN vs MLP 信息互补（Fair Chrono）

> 设计：[`paper_e2b_cnn_mlp_complement.md`](../design/paper_e2b_cnn_mlp_complement.md)  
> 面板：`E2_matched_is_ind_chrono_ric`（E1b-ric CNN + **chrono bag k=10 MLP**）；n=2,687,822；922 日  
> 产物：`WORK_SPACE/paper/E2B_cnn_mlp_chrono_ric/`  
> 姊妹：CNN vs HistGB [`paper_results_E2B.md`](paper_results_E2B.md)  
> MLP 稳定性：[`paper_results_mlp_stability.md`](paper_results_mlp_stability.md)  
> 指标详解：[`paper_metrics_glossary.md`](../protocol/paper_metrics_glossary.md)

---

## Track 总览

**背景**：稳定化后 MLP（Rank IC 0.041 / Sharpe **3.70**）强于 CNN（2.65）与 HistGB（3.07）。问题：强 MLP 是否吞掉 CNN 的独立信息，还是仍可叠加？

**协议**：与 CNN–HistGB 的 E2B **完全同构**（B1–B6）；第二模型为 bagged MLP。不重训 CNN。

| 对照 | CNN vs HistGB（E2B） | CNN vs MLP bag（本页） |
|------|----------------------|------------------------|
| 日相关 mean | 0.375 | **0.326** |
| Top10% 重叠 | 0.256 | **0.232** |
| Ensemble Sharpe | 3.43 | **3.91** |

相对旧单 seed MLP：相关与重叠上升（更“同向”），但 ensemble 仍明显高于最强单模。

---

## B1 — 日截面相关

| 统计 | 值 |
|------|-----|
| mean | **0.326** |
| median | 0.332 |
| std | 0.099 |
| p25 / p75 | 0.274 / 0.388 |

- 仍显著低于 1，且低于 CNN–HistGB（0.37）不多 → 排序函数仍有可观分歧。
- 高于旧单 seed 的 0.22：bagging 抬升了与 CNN 的共变。

---

## B2 — Decile 重合

| 集合 | 重叠率 mean | Jaccard mean |
|------|-------------|--------------|
| Top 10% | **0.232** | 0.132 |
| Bottom 10% | **0.238** | 0.136 |
| Top 20% | 0.345 | 0.210 |
| Bottom 20% | 0.350 | 0.213 |

Top10% ≈ **2.3× 随机**；多数极端选股仍不重合。

---

## B3 — Fama–MacBeth

每日截面 z-score；Newey–West t（lag=4）。

| 模型 | beta_bar | NW t | p |
|------|----------|------|---|
| M1 CNN | 0.00119 | 9.34 | ≈0 |
| M2 MLP | 0.00183 | 13.87 | ≈0 |
| M3 CNN（控 MLP） | **0.00067** | **5.39** | ≈0 |
| M3 MLP（控 CNN） | **0.00158** | **12.14** | ≈0 |

- 双变量下两系数仍显著 → **双向增量**仍在；MLP 边际更大（与 raw 强弱一致）。
- CNN 控 MLP 后 β 下降但仍显著 → 图像分数未被 bagged MLP 完全吸收。

---

## B4 — Conditional Double Sort（5×5）

| 设定 | EW H-L 年化 | EW H-L Sharpe |
|------|-------------|---------------|
| CNN（控 MLP） | 0.095 | **1.44** |
| MLP（控 CNN） | 0.217 | **2.70** |

两方向条件 spread 均 >0；MLP\|CNN 更强，符合 raw 排序。

---

## B5 — Residualization

| 分数 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| raw CNN | 0.0308 | 2.65 |
| raw MLP | 0.0406 | 3.70 |
| **u_CNN 线性** | 0.0189 | **1.29** |
| **u_MLP 线性** | 0.0309 | **3.00** |
| u_CNN poly2 | 0.0184 | 1.24 |
| u_MLP poly2 | 0.0311 | 3.05 |

- 正交化后双方仍可交易；MLP 残差接近 raw 量级。
- poly2 ≈ linear。

---

## B6 — Ensemble（w=0.5）

| 模型 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| CNN | 0.0308 | 2.655 |
| MLP bag | 0.0406 | 3.695 |
| **0.5/0.5 ens** | **0.0444** | **3.914** |

相对最强单模 MLP 再 **+0.22 Sharpe**；相对 CNN–HistGB ensemble（3.43）明显更高。

---

## 总判定

| 问题 | 判定 |
|------|------|
| bagged MLP 是否消灭 CNN 增量？ | **否**（FM M3、残差、条件排序、ensemble 均支持） |
| 互补强度 vs 旧单 seed | 相关上升，但 ensemble 收益更大 |
| 叙事 | 稳定化 MLP 是更强 tabular 非线性；CNN 仍提供可叠加的特定信息 |

复现：

```bash
python experiments/E2_matched_tabular/run_e2b_complement.py --partner mlp --step all
```
