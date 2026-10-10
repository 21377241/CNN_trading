# 论文研究框架（更新版）：Same Information → Distinct Predictive Signals

> **状态**：主线已切换（2026-09）  
> **E2 主表**：公平 Chrono 180 + Rank IC — [`paper_results_E2.md`](../results/paper_results_E2.md) / [`paper_E2_experiment_report.md`](../results/paper_E2_experiment_report.md)  
> **机制**：E2A [`paper_results_E2A.md`](../results/paper_results_E2A.md)；互补 E2B [`paper_results_E2B.md`](../results/paper_results_E2B.md) / [`paper_results_E2B_mlp.md`](../results/paper_results_E2B_mlp.md)  
> **Part IV（E3）**：[`paper_results_E3.md`](../results/paper_results_E3.md) / [`paper_e3_shared_specific_design.md`](../design/paper_e3_shared_specific_design.md)（**已完成**）  
> **Part IV-b（E4）**：[`paper_e4_recent_path_design.md`](../design/paper_e4_recent_path_design.md) / [`paper_results_E4.md`](../results/paper_results_E4.md) — Recent-5 path decomposition  
> **Part V（E5）**：[`paper_e5_mechanism_design.md`](../design/paper_e5_mechanism_design.md) / [`paper_results_E5.md`](../results/paper_results_E5.md) — 收益兑现路径与状态依赖（Phase 1）

---

## 1. 核心问题（已切换）

**旧问题（证据已不支持作为主故事）**：

> CNN 是否因为图像表示而优于 tabular model？

**新问题**：

> 在完全相同的 OHLCV 信息集下，为什么不同的非线性模型会提取出不同、但都具有收益预测力的信号？这些信号分别对应什么价格路径结构（以及最终何种经济机制）？

形式化：

```
Same Information (W_{i,t} = 20-day OHLCV)
  → Different Representations / Function Classes (CNN, HistGB, MLP)
  → Distinct Predictive Signals (shared + model-specific)
  → Economic Structure (price geometry → China institutions, later)
```

---

## 2. 论文结构（五部分）

| Part | 标题 | 内容 | 状态 |
|------|------|------|------|
| **I** | Predictability | 同信息集下 CNN / Logit / HistGB / MLP 均可预测（或线性不可） | **完成**（E2 主表） |
| **II** | Representation vs Nonlinearity | CNN 不稳优于强 tabular；120→180 显示 FE 改 accessibility 而非信息集 | **完成**（E2 + E2A） |
| **III** | Complementarity | 相关中等偏低；FM / 双重排序 / 残差 / ensemble 显示增量 | **完成**（E2B + MLP） |
| **IV** | What Do They Learn? | Shared vs Specific + Recent-5 结构分解 | **E3+E4 完成** |
| **V** | Economic Mechanism | Horizon/state/Pressure/方向 + 制度筛选 | **E5 完成（7=筛选非因果）** |

---

## 3. 已完成实验与主结论

### 3.1 E2 主表（公平 Chrono 180 + val_rank_ic）

| 模型 | Rank IC | EW H-L Sharpe | 角色 |
|------|---------|---------------|------|
| CNN E1b-ric | 0.0308 | **2.655** | Image |
| HistGB | 0.0301 | **3.068** | 主对照 |
| MLP（chrono bag k=10） | **0.0406** | **3.695** | 深度表格（稳定化） |
| Logit | 0.0097 | 1.470 | 线性下界 |

判定：`CNN_le_HistGB`（Sharpe 差 −0.41）；bagged MLP 为当前最强单模。  
**含义**：Image representation **不是**收益预测的必要条件；**nonlinear extraction** 更关键；单 seed MLP 会系统性低估表格浅网。

随机 Val 仅作协议敏感性附录（旧口径下曾接近打平）。

### 3.2 E2A：Feature engineering = accessibility

- Logit Δ Sharpe +0.30 ≫ HistGB +0.12 / MLP −0.29  
- δ 100% 可复现、零增量；uP/uV 驱动 Logit 增益  
- Acc 高但 Ridge R² 低 → **Information ≠ Accessibility**

### 3.3 E2B：Complementarity

| 对照 | ρ | Top10% 重叠 | Ensemble Sharpe |
|------|---|-------------|-----------------|
| CNN–HistGB | 0.375 | 26% | **3.43** |
| CNN–MLP（bag k=10） | **0.326** | **23%** | **3.91** |

FM / 双重排序 / 残差均显示**双向独立增量**。  
**含义**：Similar performance ≠ same learned information；model-specific signals 有经济价值。

---

## 4. 真正的 Puzzle（RQ）

| RQ | 问题 |
|----|------|
| RQ1 | Shared signal 是什么？ |
| RQ2 | CNN-specific 是什么？ |
| RQ3 | MLP- / HistGB-specific 是什么？ |
| RQ4 | Specific signals 对应何种价格路径结构？ |
| RQ5 | 结构能否链接到中国市场制度 / 行为？（后置） |

---

## 5. Part IV（E3）与 E4

E3 Phase 1+2 **已完成**（结果 [`paper_results_E3.md`](../results/paper_results_E3.md)）：

| 期 | 内容 | 状态 |
|----|------|------|
| **Phase 1** | E3-0 Shared/Specific；E3-1 Disagreement；E3-2 Spec×Features；E3-5 汇总 | **完成** |
| **Phase 2** | E3-3 CNN perturbation；E3-4 tabular ablation | **完成** |

**E4 — Decomposing Recent-Path Predictability**（结果 [`paper_results_E4.md`](../results/paper_results_E4.md)）：H1/H2/H3 **均通过**。

| 实验 | 目的 | 状态 |
|------|------|------|
| E4-1 Lag sensitivity | 近 5 日哪一天最重要 | **完成** |
| E4-2 CumRet5 + PathShape | 控动量后路径形状是否有增量 | **完成** |
| E4-3 Matched blocks | Volume/Wick/Direction/Return/Gap/Intraday | **完成** |
| E4-4 Matched-Return path | 同 CumRet5 下 path 敏感性 | **完成** |
| E4-5 Disagreement × Recent-5 | 机制交叉验证 | **完成** |

主结论：CNN-spec 对 **日位置 + 量能路径** 尖锐敏感；MLP-spec 更依赖 **日内/影线几何**，成分剖面相关仅 0.29 → representation-specific recent-path。

| 优先级 | 实验 | 期 | 目的 |
|--------|------|----|------|
| ★★★★★ | Shared / Specific 构造（E3-0） | E3 | **完成** |
| ★★★★★ | Disagreement sample（E3-1） | E3 | **完成** |
| ★★★★★ | Spec × Features（E3-2） | E3 | **完成** |
| ★★★★ | CNN/tabular perturbation（E3-3/4） | E3 | **完成** |
| ★★★★★ | Recent-5 lag / PathShape / blocks（E4） | E4 | **完成** |
| ★★★★★ | Horizon / ON-ID / State（E5-1/2/3） | E5 | **完成** |
| ★★★ | Pressure / Interaction / Cont-Rev（E5-4/5/6） | E5 | **完成** |
| ★★ | T+1 / Price Limit（E5-7） | E5 | **完成（筛选，非因果）** |

### Part V（E5）摘要

设计 [`paper_e5_mechanism_design.md`](../design/paper_e5_mechanism_design.md)；结果 [`paper_results_E5.md`](../results/paper_results_E5.md)（Phase 1+2）。  
CNN-spec：短窗、Pressure 放大、rebound-leaning；T1_state 上 FM \(\beta_3>0\)，FailedLimit 抬 IC、LimitUp 伤隔日 IC（**非因果证明**）。MLP-spec：continuation；T1_state 上反而更弱。

---

## 6. 工作摘要（可进 abstract 草稿）

> Using the same 20-day OHLCV information set, image CNNs do not consistently outperform strong nonlinear tabular models in the Chinese stock market. Explicit chart indicators substantially improve linear models but add little to flexible nonlinear learners, suggesting that feature engineering primarily changes information accessibility rather than the underlying information set. More importantly, CNN, MLP, and HistGB generate only moderately correlated stock rankings and retain substantial incremental return predictability after controlling for one another. Simple model combinations materially improve both rank IC and portfolio Sharpe ratios, indicating that different nonlinear representations extract distinct and economically complementary signals from the same price history.

---

## 7. 一句话主线

> **论文不再是「图像模型是否更强」，而是「同一价格历史为什么会被不同非线性模型解析成不同且互补的预测信号」。**
