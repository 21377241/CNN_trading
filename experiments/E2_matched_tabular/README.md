# E2 — Matched-Information Benchmark

对照设计见 [`docs/paper_e2_design.md`](../../docs/paper_e2_design.md)。

## 设定摘要

- 同一 20 日窗；特征 = 归一化 OHLC + Vol + **仅 MA20**（无 MA5）
- 模型：SGD Logit（原文规格 + IS-val 网格） | **HistGB**（主） | MLP；CNN = 已有 I20R5 `up_prob`
- **主协议切分**：IS 2000–2007（70/30, seed=42）；OOS 2008–2025
- 产物：`WORK_SPACE/paper/E2_matched_is/`（120 维）与 `E2_matched_is_ind/`（180 维）
- 统一结果：[`docs/paper_results_E2.md`](../../docs/paper_results_E2.md)

### 额外稳健性：时序 Val（不覆盖主产物）

- Train 2000–2005 / Val 2006–2007 / OOS 2008–2025；复用既有特征缓存
- 配置：`config_chrono.yaml`、`config_ind_chrono.yaml`
- 产物：`WORK_SPACE/paper/E2_matched_is_chrono/`、`E2_matched_is_ind_chrono/`（见结果文档 §4）

### 公平 chrono + Rank IC（180 维；§5）

- 与上相同 chrono Val；tabular `search_metric: val_rank_ic`；CNN = E1b-ric 最优（不重训）
- 配置：`config_ind_chrono_ric.yaml`
- 产物：`WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/`（见结果文档 §5）

## 命令

```powershell
python experiments/E2_matched_tabular/build_dataset.py
python experiments/E2_matched_tabular/run_train.py
# 仅重训 Logit（不覆盖 HistGB / MLP）：
python experiments/E2_matched_tabular/run_train.py --only logit
python experiments/E2_matched_tabular/run_eval.py

# 时序 Val 稳健性（独立 out_root）
python experiments/E2_matched_tabular/run_train.py --config experiments/E2_matched_tabular/config_chrono.yaml
python experiments/E2_matched_tabular/run_eval.py --config experiments/E2_matched_tabular/config_chrono.yaml
python experiments/E2_matched_tabular/run_train.py --config experiments/E2_matched_tabular/config_ind_chrono.yaml
python experiments/E2_matched_tabular/run_eval.py --config experiments/E2_matched_tabular/config_ind_chrono.yaml

# 公平 chrono + Rank IC（180 维；CNN = E1b-ric，见 §5）
python experiments/E2_matched_tabular/run_train.py --config experiments/E2_matched_tabular/config_ind_chrono_ric.yaml
python experiments/E2_matched_tabular/run_eval.py --config experiments/E2_matched_tabular/config_ind_chrono_ric.yaml
```
