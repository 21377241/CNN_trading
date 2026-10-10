# E2B 设计：CNN vs HistGB 信息互补性（Fair Chrono）

> 状态：已完成  
> 结果：[`paper_results_E2B.md`](paper_results_E2B.md)  
> 姊妹：CNN vs MLP [`paper_e2b_cnn_mlp_complement.md`](paper_e2b_cnn_mlp_complement.md) / [`paper_results_E2B_mlp.md`](paper_results_E2B_mlp.md)  
> 父实验：[`paper_E2_experiment_report.md`](paper_E2_experiment_report.md)

## 1. 研究问题

E2 下 CNN 无法碾压 HistGB，但日均相关仅 ~0.35–0.41。二者是否捕捉**不同**可交易信息？

## 2. 主面板（不重训）

**Fair Chrono 180**：`WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/preds/`

| 分数 | 文件 | 来源 |
|------|------|------|
| \(S^{CNN}\) | `pred_cnn_oos.csv` | E1b-ric（chrono + Rank IC） |
| \(S^{HistGB}\) | `pred_histgb_oos.csv` | chrono + val_rank_ic 网格 |
| 收益 | `ret_val` | 周收益 |

对齐键：`(StockID, ending_date)`。全程 Chrono 定模；**不用随机 Val 面板**。

## 3. 子实验

### B1 — 日截面相关

每日 \(\rho_t=\mathrm{corr}(S^{CNN},S^{HistGB})\)：mean / median / std / p25 / p75；按年；牛/熊（市场等权收益中位数上下）；高/低波动（截面收益 std 中位数上下）。

### B2 — Decile 重合

Top/Bottom 10% 与 20% 的重叠率与 Jaccard；时间均值 + 牛熊分层。

### B3 — Fama–MacBeth

截面标准化分数后：

- M1: \(R\sim S^{CNN}\)
- M2: \(R\sim S^{HistGB}\)
- M3: \(R\sim S^{CNN}+S^{HistGB}\)

报告 \(\bar\beta\) 与 Newey–West t（lag≈4）。

### B4 — Conditional Double Sort（5×5）

先 HistGB 五分位，组内 CNN 五分位 → 条件 CNN H−L；反过来再做；EW 主报，VW 附录。

### B5 — Residualization

- 线性：每日 \(S^{CNN}=\alpha+\gamma S^{HistGB}+u\)（及反向）
- 非线性：degree-2 多项式同日拟合残差

残差做十分位 H−L；仍显著 → 独立信息。

### B6 — Ensemble

\(S^{ens}=0.5\,S^{CNN}+0.5\,S^{HistGB}\) vs 单模型 Rank IC / EW H-L Sharpe。

## 4. 脚本与产物

```powershell
# CNN vs HistGB（默认）
python experiments/E2_matched_tabular/run_e2b_complement.py --partner histgb --step all
# CNN vs MLP
python experiments/E2_matched_tabular/run_e2b_complement.py --partner mlp --step all
```

- 入口：`run_e2b_complement.py`（`--partner histgb|mlp`；`--step all|corr|overlap|fm|doublesort|resid|ens`）
- HistGB 产物：`WORK_SPACE/paper/E2B_complement_chrono_ric/`
- MLP 产物：`WORK_SPACE/paper/E2B_cnn_mlp_chrono_ric/`
- 结果：`docs/paper_results_E2B.md`、`docs/paper_results_E2B_mlp.md`
