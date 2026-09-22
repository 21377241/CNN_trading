# 论文实验索引（A 股图像化价格模式）

编排层：只调用 `ashare_train/` 与 `trend_submit/` 已有能力，**不覆盖** `WORK_SPACE/ashare_model_res_*`。

| ID | 名称 | 状态 | 依赖 | 产物 |
|----|------|------|------|------|
| E1 | Image baseline I20/R5 (IS/OOS) | **done** | — | `ashare_model_res_I20R5/`；E1b 搜参见下 |
| E1b | CNN A-share HP localization (chrono val) | **done** | E1 图数据 | val 最优 ≠ Jiang；OOS 2.37 < 现网 2.65；[`docs/paper_results_E1b.md`](../docs/paper_results_E1b.md) |
| E2 | Matched tabular (Logit / HistGB / MLP) | **done** | E1 OOS preds + I20R5 labels | `E2_matched_is/` + `E2_matched_is_ind/`；chrono 稳健性：`E2_matched_is_chrono/` + `E2_matched_is_ind_chrono/`；[`docs/paper_results_E2.md`](../docs/paper_results_E2.md) |
| E3+ | Traditional / perturbation / … | planned | E2 | — |

设计文档：[`docs/paper_e2_design.md`](../docs/paper_e2_design.md)  
执行总计划：[`docs/ashare_paper_execution_plan.md`](../docs/ashare_paper_execution_plan.md)

## 复现 E2

```powershell
cd e:\phd\LLM_trading\CNN_trading
python experiments/E2_matched_tabular/build_dataset.py
python experiments/E2_matched_tabular/run_train.py
python experiments/E2_matched_tabular/run_eval.py
```
