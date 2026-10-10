# 论文 E2 结果：Matched-Information Benchmark

> **主表口径（2026-09）**：**公平 Chrono 180 + Rank IC**（§1）  
> 完整报告：[`paper_E2_experiment_report.md`](paper_E2_experiment_report.md)  
> 新主线：[`paper_research_framework.md`](paper_research_framework.md)  
> 指标详解：[`paper_metrics_glossary.md`](paper_metrics_glossary.md)  
> 随机 Val / 旧 chrono 见 §3–4 **附录**

---

## 评测指标说明

| 列名 | 含义 | 阅读要点 |
|------|------|----------|
| **AUC** | 全样本 ROC-AUC | 全局分类力 |
| **Rank IC** | 日截面 Spearman 均值 | **核心排序指标**；主表选参 val_rank_ic |
| **EW H-L Sharpe** | 十分位多空 Sharpe（×52，rf=0） | **核心组合指标** |
| **VW H-L Sharpe** | 市值加权 H-L Sharpe | 可交易性 |
| **判定码** | `CNN_le_HistGB` 等 | 见 glossary §7 |

- OOS：2008–2025；common panel n=2,687,822；`ann_factor=52`
- 主表 CNN：E1b-ric；tabular：chrono + val_rank_ic 网格
- **MLP**：`chrono_es` + **bag k=10**（seeds 42–51）；单 seed 旧值见 [`paper_results_mlp_stability.md`](paper_results_mlp_stability.md)

---

## 1. 主表：公平 Chrono 180 + Rank IC

- Train 2000–2005 / Val 2006–2007 / OOS 2008–2025  
- 配置：`config_ind_chrono_ric.yaml`  
- 产物：`WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/`  
- Corr(S_cnn, S_histgb)≈0.37；Sharpe 差 (CNN−HistGB)=**−0.41**；年份 Rank IC 胜率 **0.50** → **`CNN_le_HistGB`**

| 模型 | AUC | Rank IC | EW H-L 年化 | EW H-L Sharpe | VW H-L Sharpe |
|------|-----|---------|-------------|---------------|---------------|
| cnn（E1b-ric） | 0.5190 | 0.0308 | 0.2359 | **2.6548** | 1.2049 |
| logit | 0.5123 | 0.0097 | 0.1747 | 1.4703 | 0.5052 |
| logit_paper | 0.5121 | 0.0067 | 0.1445 | 1.2016 | 0.2931 |
| **histgb**（网格） | 0.5219 | 0.0301 | 0.2915 | **3.0683** | 1.2372 |
| histgb_paper | 0.5218 | 0.0235 | 0.2532 | 2.4213 | 0.9736 |
| **mlp**（chrono bag k=10） | 0.5211 | **0.0406** | 0.3938 | **3.6951** | 1.6350 |

### 网格最优（chrono val_rank_ic）

| 模型 | 超参 | val_rank_ic | OOS EW Sharpe |
|------|------|-------------|---------------|
| Logit | α=1e-5, η₀=0.05, passes=10 | 0.0377 | 1.47 |
| HistGB | lr=0.08, leaves=63, min_leaf=**200**, l2=1 | 0.0593 | **3.07** |

### 主表解读

- \(CNN \gg Logit\)；\(HistGB > CNN\)；稳定化后 **\(MLP > HistGB > CNN\)**（单 seed 曾低估 MLP）  
- **Image representation 不是必要条件**；nonlinear extraction 更关键  
- 与 E2B 衔接：单模不碾压 ≠ 无独立信息（CNN–MLP ensemble Sharpe **3.91**）

---

## 2. 主表对照：Chrono 120 / 180（val_auc；CNN=现网）

产物：`E2_matched_is_chrono/`、`E2_matched_is_ind_chrono/`。CNN 仍为现网 I20R5（2.65），tabular 用 chrono + val_auc。

### 2.1 120 维 chrono

| 模型 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| cnn | 0.0293 | 2.65 |
| logit | 0.0048 | 1.17 |
| **histgb** | 0.0256 | **2.90** |
| mlp | 0.0279 | 3.26 |

判定：`CNN_le_HistGB`（差 −0.25）。

### 2.2 180 维 chrono

| 模型 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| cnn | 0.0293 | 2.65 |
| logit | 0.0097 | 1.47 |
| **histgb** | 0.0283 | **3.02** |
| mlp | 0.0266 | 2.97 |

判定：`CNN_le_HistGB`（差 −0.37）。方向与主表一致。

---

## 3. 附录：随机 Val 70/30（旧主表；协议敏感性）

> 不再作为主文数字。用于说明：随机切分下相对排序可接近打平，且线性模型易过拟合选参。

### 3.1 120 维随机

产物：`E2_matched_is/`。Corr≈0.41；Sharpe 差 +0.16；胜率 0.72 → 旧判定 `CNN_gt_HistGB`（窄）。

| 模型 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| cnn | 0.0293 | 2.65 |
| logit | −0.0067 | 0.47 |
| **histgb** | 0.0197 | **2.49** |
| mlp | 0.0263 | **3.35** |

### 3.2 180 维随机

产物：`E2_matched_is_ind/`。Sharpe 差 +0.09；胜率 0.56 → `CNN_approx_HistGB`。

| 模型 | Rank IC | EW H-L Sharpe |
|------|---------|---------------|
| cnn | 0.0293 | 2.65 |
| logit | 0.0036 | **1.17** |
| **histgb** | 0.0220 | **2.56** |
| mlp | 0.0244 | 2.91 |

120→180：Logit 受益最大；HistGB 略升；MLP 可能略降（详见 E2A）。

---

## 4. 三方协议对照（180 维）

| 设定 | Val | 选参 | CNN | HistGB | MLP | 角色 |
|------|-----|------|-----|--------|-----|------|
| **§1 主表** | chrono | val_rank_ic | E1b-ric 2.65 | **3.07** | 2.97 | **主文** |
| §2.2 | chrono | val_auc | I20R5 2.65 | 3.02 | 2.97 | 对照 |
| §3.2 附录 | 随机 | val_auc | I20R5 2.65 | 2.56 | 2.91 | 敏感性 |

---

## 5. 实现说明

- 曾误把 `binary_label=-99` 当第三类；已剔除。  
- 配置：主表 `config_ind_chrono_ric.yaml`；附录 `config.yaml` / `config_ind.yaml`。  
- 物理存储：`F:\LLM_trading\CNN_trading\WORK_SPACE\`（E 盘 junction）。

---

*主文以 §1 公平 Chrono 为准。机制与互补见 E2A / E2B；结构解释见 E3。*
