# E1 — Image baseline (I20/R5)

## E1 主结果（已完成）

| 项 | 值 |
|----|-----|
| 设定 | window_size=20，freq=week，标签 Ret_week |
| 切分 | IS 2000–2007（70/30 随机 val）；OOS 2008–2025 |
| 模型 | CNN `D20…C256`，5 ensembles |
| 预测 | `WORK_SPACE/ashare_model_res_I20R5/**/ensem_res/ensem5_res_YYYY.csv` |
| 回测 | EW H-L Sharpe ≈ **2.65** |
| 报告 | `docs/repro/ashare_paper_repro_report.md` |

E2 只**读取**上述 OOS `up_prob`，不修改该目录。

## E1b：A 股 CNN 超参本地化（chrono val）

- 设计：[`docs/design/paper_cnn_hp_design.md`](../../docs/design/paper_cnn_hp_design.md)
- 结果：[`docs/results/paper_results_E1b.md`](../../docs/results/paper_results_E1b.md)
- 配置：[`config_cnn_hp.yaml`](config_cnn_hp.yaml)
- 产物：`WORK_SPACE/paper/E1b_cnn_hp_I20R5/`（不覆盖 `ashare_model_res_I20R5/`）

## E1b-ric：用截面 Rank IC 重选（修正 MCC 选参）

昨天实验用 **val_mcc** 选参，与组合目标不对齐。重跑：

- 配置：[`config_cnn_hp_ric.yaml`](config_cnn_hp_ric.yaml)
- 产物：`WORK_SPACE/paper/E1b_cnn_hp_I20R5_ric/`（不覆盖 MCC 版）
- 选参：`search_metric: val_rank_ic`；早停仍用 val_loss

```powershell
python ashare_train/3b_cnn_hp_search.py --config experiments/E1_image_baseline/config_cnn_hp_ric.yaml --phase all
```
