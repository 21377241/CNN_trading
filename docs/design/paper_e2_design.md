# E2 实验设计说明（Matched-Information Benchmark）

> 状态：**已完成**；**主表 = 公平 Chrono 180 + Rank IC**（2026-09 口径切换）  
> 统一结果：[`paper_results_E2.md`](../results/paper_results_E2.md)；完整报告：[`paper_E2_experiment_report.md`](../results/paper_E2_experiment_report.md)  
> 新论文主线：[`paper_research_framework.md`](../framework/paper_research_framework.md)  
> 切分：IS 2000–2007 / OOS 2008–2025  
> 主对照：HistGB；Image：E1b-ric CNN（与 tabular chrono 对称）

---

## 1. 研究问题

**E2 原问题**（预注册）：在完全相同的 20 日 OHLCV 信息集下，图像 CNN 相对表格模型是否仍有预测优势？

**主表答案**：否（`CNN_le_HistGB`）。E2 在全文中转为 predictability + 「nonlinear > image representation」证据。

**全文新问题**（见研究框架）：同一信息集下，不同非线性模型为何提取不同且互补的预测信号？

---

## 2. 样本与切分

| 集合 | 年份 | 用法 |
|------|------|------|
| IS | 2000–2007 | 拟合 tabular |
| **主 Val** | **2006–2007**（chrono；train 2000–2005） | 网格选参 / 监控 |
| OOS | 2008–2025 | 只评价 |
| **附录 Val** | IS 内随机 70/30 | 协议敏感性 |

- Common panel 与 CNN `ensem_res` 对齐  
- 主表 CNN：E1b-ric（chrono + Rank IC）；**不重训**于 E2 脚本内

---

## 3. 信息集匹配（核心约束）

同一窗口 \(W_{i,t}\)：Image = K 线+MA20+量图；Tabular = 仅由该窗构造的 120/180 维特征。  
**不放入**：MA5、MACD/RSI、市值/行业、\(t\) 之后信息。

| 设定 | 维度 |
|------|------|
| 120 | \(20\times6\) OHLC+V+MA20 |
| **180（主表）** | + δ²⁰ / uᴾ / uⱽ |

---

## 4. 模型

| 模型 | 角色 |
|------|------|
| CNN E1b-ric | Image（主表） |
| SGD Logit | 线性下界 |
| **HistGB** | **主对照** |
| MLP | 深度表格稳健性 |

主表选参：tabular **`val_rank_ic`**；CNN 已在 E1b-ric 固定。

---

## 5. 评价指标

Rank IC、EW/VW H-L Sharpe、AUC、Corr(S_cnn, S_tab)、年份 Rank IC 胜率。  
主表列：**CNN | Logit | HistGB | MLP**（180 维公平 chrono）。

---

## 6. 预注册判定（相对 HistGB）

仍按原规则报告判定码；主表实现为 `CNN_le_HistGB`。  
叙事上**不再**把「CNN 明显优于 HistGB」当作目标结论，而转向互补与 specific signals（E2B / E3）。

---

## 7. 交付

- 脚本：`experiments/E2_matched_tabular/`  
- 主表产物：`WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/`  
- 附录：`E2_matched_is*`（随机）、`E2_matched_is_*_chrono/`（val_auc）
