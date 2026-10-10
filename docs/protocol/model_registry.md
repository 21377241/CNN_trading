# E0-T1 统一模型结果登记表

- 生成方式：`python experiments/E0_audit/build_registry.py`
- 人工映射：`experiments/E0_audit/registry_manual.yaml`
- 机器可读结果：`WORK_SPACE/paper/E0_audit/registry.csv`（共 603 个文件：64 个主文件、538 个干预实验中间文件、1 个空文件）
- 登记日期：2026-10-11

## 1. 全局检查结果

- **样本一致**：64 个主预测文件和 score panel 都是 2,687,822 行、922 周（2008-01-04 至 2025-12-31），4,923 只股票，没有重复键。与参照文件（`E2_matched_is_ind_chrono_ric/preds/pred_cnn_oos.csv`）的键集合完全相同。
- **标签一致**：所有文件的 `ret_val` 与参照文件的最大绝对差为 0，`binary_label` 没有不一致。所有历史结果共用同一套标签：裁剪后的 close-to-close 周收益，见 `E0_data_timing.md`。
- **空文件**：`WORK_SPACE/paper/E1b_cnn_hp_I20R5_ric/e2_bridge/pred_cnn_oos_tuned.csv` 为 0 字节。规范 CNN 预测并非由 `run_e2_cnn_bridge.py` 写出，而是直接放在 `E2_matched_is_ind_chrono_ric/preds/` 下。要从 checkpoint 复现规范 CNN 预测，还需补一个从 E1b-ric 的 `ensem_res` 拼接的脚本（列入 E7 前置任务）。
- **干预实验中间文件**：E3-3 扰动和 E4-1/E4-3 的 538 个文件只登记大小和时间，不计算哈希。它们只服务于 E3/E4 的敏感性分析，不进入主表。

## 2. 规范模型与候选版本

| 登记名 | 角色 | 输入 | 结构 | ensemble | 选模 | 文件 | sha1（前 16 位） |
|---|---|---|---|---|---|---|---|
| `chartcnn_e1b_ric` | 规范 | I20 图像 | Jiang CNN 64-128-256，708,866 参数 | 5 checkpoint | val Rank IC | `E2_matched_is_ind_chrono_ric/preds/pred_cnn_oos.csv` | be0a89e6b16f522c |
| `onedcnn_120` | 规范 | 120D | Conv1D [32, 64]，7,073 参数 | seed 42–46 | val Rank IC | `E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv` | 6f1982cad36565e9 |
| `mlp4_120_s5` | 规范 | 120D | MLP (128,64,32,16)，26,369 参数 | seed 42–46 | val Rank IC 早停 | `E2F_mlp_depth/preds/pred_mlp4_oos.csv` | 4bf2a2214272f312 |
| `mlp2_180_bag10` | 稳健性 | 180D | MLP (128,64) | seed 42–51 | val Rank IC 早停 | `E2_matched_is_ind_chrono_ric/preds/pred_mlp_oos.csv` | c7a831bc49d865fb |
| `mlp4_180_bag10` | 稳健性 | 180D | MLP (128,64,32,16) | seed 42–51 | val Rank IC 早停 | `E2_matched_is_ind_chrono_ric_mlp4/preds/pred_mlp_oos.csv` | f06502c16f624699 |
| `mlp2/3/5_120_s5` | 稳健性 | 120D | 不同深度 | seed 42–46 | val Rank IC 早停 | `E2F_mlp_depth/preds/pred_mlp{2,3,5}_oos.csv` | 见 csv |
| `histgb_180` | 附录 | 180D | HistGB，按 val Rank IC 网格搜索 | 单模型 | val Rank IC | `E2_matched_is_ind_chrono_ric/preds/pred_histgb_oos.csv` | becf1e4bb546aa72 |
| `logit_180` | 附录 | 180D | SGD logit | 单模型 | val Rank IC | `E2_matched_is_ind_chrono_ric/preds/pred_logit_oos.csv` | 27587f90ef115887 |
| `chartcnn_pre_ric` | 旧版 | I20 图像 | 同上，Rank IC 选参之前 | 5 checkpoint | random val 或 MCC | `E2_matched_is*/preds/pred_cnn_oos.csv`、`E2A_*/preds/pred_cnn_oos.csv`（7 个目录相同） | f28090d17dbbd7c2 |
| `mlp2_180_seed42` | 旧版 | 180D | MLP (128,64) | 单 seed | val Rank IC | `..._ric/preds/pred_mlp_oos_seed42_legacy.csv`（与 `E2_matched_is_ind_chrono/preds/pred_mlp_oos.csv` 相同） | f5b9d71d0a1f41ab |
| `e2_chrono_120_legacy` | 旧版 | 120D | 旧 CNN 与单 seed MLP | — | — | `E2_matched_is_chrono/preds/pred_{cnn,mlp}_oos.csv` | f28090…、b70ddfc1… |

各模型的训练配置和生成脚本见 `registry_manual.yaml`。

**哈希分组揭示的复用关系**（不影响结果，但写作时需注意）：
- `..._ric/preds/pred_logit_oos.csv` 与 `pred_histgb_paper_oos.csv` 分别和 `E2_matched_is_ind_chrono/preds/` 下的同名文件逐字节相同。所谓 “ric” 版 Logit，是沿用 `ind_chrono` 那一轮的结果。
- `E2A_ind_delta_chrono/preds/pred_histgb_oos.csv` 与 `E2_matched_is_chrono/preds/pred_histgb_oos.csv` 相同。

## 3. 论文表格与报告所用文件

| 报告 | 使用的模型 | 产出脚本 | 状态 |
|---|---|---|---|
| E2 主表（`paper_results_E2.md`、框架文档） | `chartcnn_e1b_ric`、`histgb_180`、`mlp2_180_bag10`、`logit_180` | `E2_matched_tabular/run_eval.py` | Sharpe 为旧口径，需重算 |
| MLP-4 替换版（`mlp4_propagation_audit.md`，E3–E5 的 `_mlp4`） | `chartcnn_e1b_ric`、`mlp4_180_bag10`、`histgb_180` | `E2_matched_tabular/run_train.py`、E3/E4/E5 脚本 | 新口径 |
| E2C（`paper_E2C_temporal_numerical_benchmark_report.md`） | `chartcnn_e1b_ric`、`onedcnn_120` | `E2C_temporal/eval_e2c.py` | 新口径 |
| E2F（`paper_E2F_mlp_depth_robustness.md`） | `mlp{2..5}_120_s5` | `E2F_mlp_depth/run_depth.py` | 新口径 |
| E2D（`paper_E2D_liquidity_implementability_audit.md`） | `e2_chrono_120_legacy`、`onedcnn_120` | `E2D_liquidity/run_e2d_audit.py` | 用了旧版文件，由 E9 取代 |
| E2G（`paper_results_E2G.md`） | `chartcnn_e1b_ric`、`onedcnn_120`、`mlp4_120_s5` | `E2G_chart_capacity/eval_e2g.py` | 原先误用 `mlp4_180_bag10`；已改为 `mlp4_120_s5` 并重跑 |
| E3/E4/E5 主线（`paper_results_E3/E4/E5.md`） | `chartcnn_e1b_ric`、`mlp2_180_bag10`、`histgb_180` | `E3_shared_specific_chrono_ric/scores.parquet` | 解释性章节 |
| E3T/E4T/E5T | `chartcnn_e1b_ric`、`onedcnn_120` | `E3T_cnn_1d_temporal/scores.parquet` | 解释性章节 |

## 4. 使用规则（自 E7 起）

1. 新实验只能通过本登记表的登记名引用预测文件，不能直接写路径。新增模型要先登记，再使用。
2. 主表只使用三个规范模型：`chartcnn_e1b_ric`、`onedcnn_120`、`mlp4_120_s5`。其余版本只用于稳健性检验或附录，并在表注中写明登记名。
3. 对已登记的文件重新训练或覆盖时，必须更新 sha1，并重跑 `build_registry.py`。
