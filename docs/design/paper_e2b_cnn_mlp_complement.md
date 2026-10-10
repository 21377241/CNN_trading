# E2B-MLP 设计：CNN vs MLP 信息互补性（Fair Chrono）

> 状态：已完成  
> 结果：[`paper_results_E2B_mlp.md`](../results/paper_results_E2B_mlp.md)  
> 姊妹实验：CNN vs HistGB [`paper_e2b_cnn_histgb_complement.md`](paper_e2b_cnn_histgb_complement.md)  
> 父实验：[`paper_E2_experiment_report.md`](../results/paper_E2_experiment_report.md)

## 1. 研究问题

E2 fair chrono 下 MLP EW Sharpe（2.97）略高于 CNN（2.65），且 MLP 与 CNN 同属「浅层非线性」族。问题：

1. CNN 与 MLP 是否捕捉**不同**可交易信息（类似 CNN–HistGB）？
2. 图像表示相对表格 MLP 是否仍有**独立增量**？
3. 等权融合能否同时抬高 IC / Sharpe？

协议与指标定义与 E2B（CNN–HistGB）**完全相同**，仅将第二模型换成 MLP。

## 2. 主面板（不重训）

**Fair Chrono 180**：`WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/preds/`

| 分数 | 文件 | 来源 |
|------|------|------|
| S_cnn | `pred_cnn_oos.csv` | E1b-ric（chrono + Rank IC） |
| S_mlp | `pred_mlp_oos.csv` | chrono 180；固定 128→64 |
| 收益 | `ret_val` | 周收益 |

对齐键：`(StockID, ending_date)`。全程 Chrono；**不用随机 Val 面板**。

## 3. 子实验（与 E2B 同构）

| 步骤 | 内容 |
|------|------|
| B1 | 日截面 corr(S_cnn, S_mlp) |
| B2 | Top/Bottom 10%/20% 重叠率与 Jaccard |
| B3 | Fama–MacBeth：M1/M2/M3（双变量） |
| B4 | 5×5 条件双重排序（双向） |
| B5 | 线性 / poly2 残差 → 十分位 H-L |
| B6 | 0.5/0.5 ensemble vs 单模 |

## 4. 脚本与产物

```powershell
python experiments/E2_matched_tabular/run_e2b_complement.py --partner mlp --step all
```

- 产物：`WORK_SPACE/paper/E2B_cnn_mlp_chrono_ric/`
- 结果：`docs/results/paper_results_E2B_mlp.md`
