# 文档索引

状态说明：
- **现行**：内容与当前规范口径一致，可以直接引用；
- **待重算**：结论有效，但数字使用旧 Sharpe 口径或非规范模型版本，定稿前需要按 [protocol/E0_reconciliation.md](protocol/E0_reconciliation.md) 重算；
- **历史**：记录当时的设计或计划，不再更新。

模型与预测文件的对应关系，以 [protocol/model_registry.md](protocol/model_registry.md) 为准。

## protocol/：口径、协议与审计（全部现行）

| 文档 | 内容 |
|---|---|
| [model_registry.md](protocol/model_registry.md) | 规范模型、预测文件与 sha1；各报告使用了哪个文件 |
| [E0_reconciliation.md](protocol/E0_reconciliation.md) | 统一口径下的主表；各报告数字差异的来源 |
| [E0_data_timing.md](protocol/E0_data_timing.md) | 数据字典、信息时点、标签与执行摩擦审计 |
| [E7_protocol.md](protocol/E7_protocol.md) | E7 冻结分析协议（tag `protocol-e7-v1`） |
| [sharpe_convention.md](protocol/sharpe_convention.md) | Sharpe 计算口径 |
| [paper_sample_and_split.md](protocol/paper_sample_and_split.md) | 样本与训练、验证、OOS 切分规则 |
| [paper_metrics_glossary.md](protocol/paper_metrics_glossary.md) | 评测指标定义 |

## framework/：论文主线与计划

| 文档 | 状态 |
|---|---|
| [paper_main_storyline.md](framework/paper_main_storyline.md) | 待重算（旧 Sharpe 口径） |
| [paper_research_framework.md](framework/paper_research_framework.md) | 待重算（旧 Sharpe 口径） |
| [ashare_paper_execution_plan.md](framework/ashare_paper_execution_plan.md) | 历史（早期执行计划，已被 E0/E7–E10 计划取代） |

## design/：实验设计（历史，E2G 为现行）

| 实验 | 文档 |
|---|---|
| E1b | [paper_cnn_hp_design.md](design/paper_cnn_hp_design.md) |
| E2 | [paper_e2_design.md](design/paper_e2_design.md) |
| E2A | [paper_e2a_indicator_mechanism.md](design/paper_e2a_indicator_mechanism.md) |
| E2B | [paper_e2b_cnn_histgb_complement.md](design/paper_e2b_cnn_histgb_complement.md)、[paper_e2b_cnn_mlp_complement.md](design/paper_e2b_cnn_mlp_complement.md) |
| E2G | [paper_E2G_chart_capacity_design.md](design/paper_E2G_chart_capacity_design.md) |
| E3 | [paper_e3_shared_specific_design.md](design/paper_e3_shared_specific_design.md) |
| E4 | [paper_e4_recent_path_design.md](design/paper_e4_recent_path_design.md) |
| E5 | [paper_e5_mechanism_design.md](design/paper_e5_mechanism_design.md) |

## results/：实验结果

| 实验 | 文档 | 状态 |
|---|---|---|
| E1b | [paper_results_E1b.md](results/paper_results_E1b.md) | 待重算（旧 Sharpe 口径） |
| E2 | [paper_results_E2.md](results/paper_results_E2.md)、[paper_E2_experiment_report.md](results/paper_E2_experiment_report.md) | 待重算（旧 Sharpe 口径） |
| E2A | [paper_results_E2A.md](results/paper_results_E2A.md) | 现行 |
| E2B | [paper_results_E2B.md](results/paper_results_E2B.md)、[paper_results_E2B_mlp.md](results/paper_results_E2B_mlp.md) | 待重算（旧口径；ensemble 结果在 E10 重做） |
| E2C | [paper_E2C_temporal_numerical_benchmark_report.md](results/paper_E2C_temporal_numerical_benchmark_report.md) | 现行 |
| E2C | [paper_results_E2C_temporal_benchmark.md](results/paper_results_E2C_temporal_benchmark.md) | 待重算（正文 Sharpe 为旧口径） |
| E2F | [paper_E2F_mlp_depth_robustness.md](results/paper_E2F_mlp_depth_robustness.md) | 现行 |
| E2G | [paper_results_E2G.md](results/paper_results_E2G.md) | 现行（MLP-4-120 基准已于 E0 更正） |
| E3 | [paper_results_E3.md](results/paper_results_E3.md) | 分解结果现行；单模型表的 Sharpe 为旧口径，待重算 |
| E3T | [paper_E3T_cnn_1d_temporal_decomposition.md](results/paper_E3T_cnn_1d_temporal_decomposition.md) | 现行 |
| E4 / E4T | [paper_results_E4.md](results/paper_results_E4.md)、[paper_E4T_cnn_1d_recent_path.md](results/paper_E4T_cnn_1d_recent_path.md) | 现行 |
| E5 / E5T | [paper_results_E5.md](results/paper_results_E5.md)、[paper_E5T_cnn_1d_economic_realization.md](results/paper_E5T_cnn_1d_economic_realization.md)、[paper_E5_standard_update.md](results/paper_E5_standard_update.md) | 现行 |
| E6A | [paper_E6A_state_dependent_comparative_advantage.md](results/paper_E6A_state_dependent_comparative_advantage.md) | 现行 |
| MLP 稳定性 | [paper_results_mlp_stability.md](results/paper_results_mlp_stability.md)、[paper_mlp_single_vs_bagged.md](results/paper_mlp_single_vs_bagged.md) | 待重算（旧 Sharpe 口径） |
| MLP-4 替换 | [mlp4_propagation_audit.md](results/mlp4_propagation_audit.md) | 现行 |
| 口径统一 | [paper_standardized_results_update.md](results/paper_standardized_results_update.md) | 现行 |
| E7 | [paper_results_E7.md](results/paper_results_E7.md) | 现行（按冻结协议运行；H1、H3 成立，H2 不成立；追加的 ChartCNN-S 检验表明包含结论对容量稳健） |

## repro/：Jiang et al. (2023) 复现阶段

[ashare_paper_repro_report.md](repro/ashare_paper_repro_report.md)、[论文主实验复现报告_A股全市场.md](repro/论文主实验复现报告_A股全市场.md)、[ashare_topk_robust.md](repro/ashare_topk_robust.md)（汇总表 `ashare_topk_robust_summary.csv`）、[项目实现方案_CodeAgent复现指南.md](repro/项目实现方案_CodeAgent复现指南.md)、[SPX_volume_price_数据结构说明.md](repro/SPX_volume_price_数据结构说明.md)。均为历史。

## 其他

- `figures/`：文档引用的图片。
- `state_guided_combination/`：状态引导组合的研究设想（与 E6A 相关）。
- `archive/2026-09-22_representation-complementarity/`：上一版论文（representation complementarity）的完整快照，只读。
- `archive/superseded/`：已被取代的报告。E2D 流动性审计建立在旧版模型文件上，由 E9 重做。
