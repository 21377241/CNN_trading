# 实验索引（Same Information, Distinct Signals；目标期刊 JEF）

约定：
- 每个实验一个目录 `experiments/E*_*/`，产物写入 `WORK_SPACE/paper/` 下的对应目录，不覆盖 `WORK_SPACE/ashare_model_res_*`。
- 规范模型与预测文件以 [`docs/protocol/model_registry.md`](../docs/protocol/model_registry.md) 为准，统计口径以 [`docs/protocol/E0_reconciliation.md`](../docs/protocol/E0_reconciliation.md) 第 5 节为准。
- 新代码的公共函数放在 `common/`：面板读取 `panel.py`，样本切分 `splits.py`，IC、Newey–West t 和 Sharpe 在 `metrics.py`。旧实验目录里各自的 `utils.py` 保持不变，以免影响已有结果的复现。
- 全部文档的索引见 [`docs/README.md`](../docs/README.md)。

## 已完成

| ID | 内容 | 目录 | 产物（`WORK_SPACE/paper/`） | 结果文档 |
|---|---|---|---|---|
| E1 / E1b | 图像基线 I20R5；A 股 CNN 超参本地化（按验证期 Rank IC 选参） | `E1_image_baseline/`，训练用 `ashare_train/` | `E1b_cnn_hp_I20R5/`、`E1b_cnn_hp_I20R5_ric/` | [results/paper_results_E1b.md](../docs/results/paper_results_E1b.md) |
| E2 | 同信息集基准：Logit、HistGB、MLP | `E2_matched_tabular/` | `E2_matched_is*`（规范版为 `E2_matched_is_ind_chrono_ric/`） | [results/paper_results_E2.md](../docs/results/paper_results_E2.md) |
| E2A | +60 维图形指标的机制拆解 | `E2_matched_tabular/run_e2a_*.py` | `E2A_*` | [results/paper_results_E2A.md](../docs/results/paper_results_E2A.md) |
| E2B | CNN 与 HistGB/MLP 的信息互补 | `E2_matched_tabular/run_e2b_complement.py` | `E2B_*` | [results/paper_results_E2B.md](../docs/results/paper_results_E2B.md) |
| E2C | 时序数值基准 1D-CNN-120 | `E2C_temporal/` | `E2C_temporal_1dcnn_120/` | [results/paper_E2C_temporal_numerical_benchmark_report.md](../docs/results/paper_E2C_temporal_numerical_benchmark_report.md) |
| E2D | 流动性与可实施性审计（已被取代，由 E9 重做） | `E2D_liquidity/` | `E2D_liquidity_audit*` | [archive/superseded/](../docs/archive/superseded/) |
| E2F | MLP 深度稳健性（2/3/4/5 层，120D） | `E2F_mlp_depth/` | `E2F_mlp_depth/` | [results/paper_E2F_mlp_depth_robustness.md](../docs/results/paper_E2F_mlp_depth_robustness.md) |
| E2G | 参数量匹配的 chart-CNN 容量控制 | `E2G_chart_capacity/` | `E2G_chart_capacity/`（旧的无效结果在 `_invalid_fullwidth/`） | [results/paper_results_E2G.md](../docs/results/paper_results_E2G.md) |
| E3 / E3T | 共享信号与模型特有信号的分解 | `E3_shared_specific/` | `E3_shared_specific_*`、`E3T_cnn_1d_temporal/` | [results/paper_results_E3.md](../docs/results/paper_results_E3.md) |
| E4 / E4T | 近期价格路径的可预测性分解 | `E4_recent_path/` | `E4_recent_path_*`、`E4T_cnn_1d_recent_path/` | [results/paper_results_E4.md](../docs/results/paper_results_E4.md) |
| E5 / E5T | 特有信号的经济机制 | `E5_mechanism/` | `E5_mechanism_*`、`E5T_cnn_1d_mechanism/` | [results/paper_results_E5.md](../docs/results/paper_results_E5.md) |
| E6A | CNN 与 MLP-4 的状态依赖相对优势 | `E6A_state_advantage/` | `E6A_state_advantage/` | [results/paper_E6A_state_dependent_comparative_advantage.md](../docs/results/paper_E6A_state_dependent_comparative_advantage.md) |
| E0 | 模型登记、口径对账、数据与执行时点审计 | `E0_audit/` | `E0_audit/` | [protocol/](../docs/protocol/) |
| E7 | 传统控制变量下的增量预测信息（按冻结协议） | `E7_incremental/` | `E7_incremental/` | [results/paper_results_E7.md](../docs/results/paper_results_E7.md) |

## 待开展（JEF 投稿计划）

| ID | 内容 | 目录 | 协议或设计 |
|---|---|---|---|
| E9 | 可交易性与执行：次日开盘成交、涨跌停与停牌约束、交易成本、换手 | `E9_tradable/` | 待写 |
| E8 | 风险调整后的价值：自建的规模、价值、动量、反转、流动性因子 | `E8_risk_adjusted/` | 待写 |
| E10 | 组合增量：固定组合规则 + block bootstrap 置信区间 | `E10_combination/` | 待写 |

执行顺序为 E9、E8、E10，之后重写全文。E7 的结论是 chart-CNN 的增量信息被 1D-CNN 包含，所以后续实验以 1D-CNN 为主要对象，chart-CNN 作为对照。E7 新建的扩展日面板、执行一致收益和控制变量面板（`E7_incremental/build_*.py`）供 E8–E10 复用。

## 复现示例

```powershell
cd e:\phd\LLM_trading\CNN_trading
python experiments/E0_audit/build_registry.py
python experiments/E0_audit/reconcile_metrics.py
python experiments/E0_audit/audit_timing.py
```
