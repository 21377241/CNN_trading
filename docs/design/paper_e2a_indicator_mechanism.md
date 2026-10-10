# E2A 设计：+60 维 chart indicators 机制拆解（Chrono）

> 状态：已完成  
> 结果：[`paper_results_E2A.md`](../results/paper_results_E2A.md)  
> 父实验：[`paper_E2_experiment_report.md`](../results/paper_E2_experiment_report.md)（**主表 = 公平 Chrono**）  
> 全文主线：[`paper_research_framework.md`](../framework/paper_research_framework.md)

## 1. 研究问题

E2 显示：在 Chrono Val 下，120→180（+δ²⁰ / uᴾ / uⱽ）对 **Logit** 的提升远大于 HistGB / MLP。本 track 拆解：

1. Δ 是否在统计上显著大于后两者？
2. 60 维是否可由 120 维线性/非线性「恢复」？
3. 三个 indicator 各自贡献多少？

## 2. 协议（固定）

| 项 | 设定 |
|----|------|
| Val | **Chrono**：train 2000–2005 / val 2006–2007 |
| OOS | 2008–2025 |
| 选参 | 与已有 chrono 一致：`val_auc`（保证 120 vs 180 Δ 可比） |
| 不用 | 随机 70/30 Val |

## 3. 复用 vs 补训

| 产物 | 路径 | 动作 |
|------|------|------|
| Chrono 120 | `WORK_SPACE/paper/E2_matched_is_chrono/` | **复用** |
| Chrono 180 | `WORK_SPACE/paper/E2_matched_is_ind_chrono/` | **复用** |
| 180 特征缓存 | `WORK_SPACE/paper/E2_matched_is_ind/features/` | **复用**（列抽取做消融/恢复） |
| 140 维 δ / uP / uV | `WORK_SPACE/paper/E2A_ind_{delta,up,uv}_chrono/` | **补训** |
| 120→60 回归器 | `WORK_SPACE/paper/E2A_recover_60/` | **新训** |

特征列序（180，每天 9 维）：`[O,H,L,C,MA20,δ,V_n,uP,uV]`。  
120 等价抽取：每天 `[0,1,2,3,4,6]`；60 维：`[5,7,8]`。

## 4. 子实验

### A1 — Δ 对比（不重训）

\(\Delta_m(X)=X_m^{180}-X_m^{120}\)，\(m\in\{\mathrm{logit},\mathrm{histgb},\mathrm{mlp}\}\)。

指标：AUC、Rank IC、Pearson IC、EW/VW H-L Sharpe；**逐年 Rank IC 差**。  
主假设：\(\Delta_{\mathrm{logit}}\) 在 Rank IC / Sharpe 上大于 \(\Delta_{\mathrm{histgb}}\)、\(\Delta_{\mathrm{mlp}}\)。  
检验：18 年 \(\Delta\mathrm{IC}\) 的 Wilcoxon 符号秩 / 配对符号检验。

脚本：`experiments/E2_matched_tabular/run_e2a_delta_compare.py`

### A2 — 120→60 可恢复性

从 \(Z^{120}\) 预测整块 60 维及 \(\delta^{20}\)、\(u^P\)、\(u^V\) 分块。  
回归器：Ridge / MLPRegressor / HistGradientBoostingRegressor；**主指标** Acc（threshold@0.5），**辅助** 逐维 R²；Ridge 在 chrono val 上按 Acc 选 \(\alpha\)。  
报告：OOS 块平均 Acc、R² 及逐维明细。

脚本：`experiments/E2_matched_tabular/run_e2a_recover_60.py`

### A3 — 单指标消融

| 设定 | 维数 | 每天列 |
|------|------|--------|
| baseline 120 | 120 | 复用 chrono |
| +δ only | 140 | OHLCV+MA+δ |
| +uP only | 140 | OHLCV+MA+uP |
| +uV only | 140 | OHLCV+MA+uV |
| +all 180 | 180 | 复用 ind_chrono |

脚本：`experiments/E2_matched_tabular/run_e2a_ablation_train.py`（从 180 缓存抽列 + chrono 训评）

## 5. 交付

- 设计：本文档  
- 结果：`docs/results/paper_results_E2A.md`  
- 产物：`WORK_SPACE/paper/E2A_*/`
