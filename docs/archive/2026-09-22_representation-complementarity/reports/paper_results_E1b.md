# 论文 E1b 结果：A 股 CNN 超参本地化

> 设计：[`paper_cnn_hp_design.md`](paper_cnn_hp_design.md)  
> 指标详解：[`paper_metrics_glossary.md`](paper_metrics_glossary.md)（Rank IC、EW H-L Sharpe、val_mcc）

---

- 产物：`WORK_SPACE/paper/E1b_cnn_hp_I20R5/`
- 问题：MCC 与截面排序 / H-L 目标不对齐；val 最优未转化为 OOS Sharpe。

| 配置 | val_mcc | EW H-L Sharpe | vs 现网 2.65 |
|------|---------|---------------|--------------|
| 现网 I20R5 | — | **2.65** | — |
| MCC 搜参最优 3e-5/0.6/pat5 | 0.054 | 2.37 | −0.28 |
| Jiang 锚点 chrono 全训 | 0.036 | 2.38 | −0.27 |

## B. 第二轮（val Rank IC 选参）— 已完成

- 配置：`experiments/E1_image_baseline/config_cnn_hp_ric.yaml`
- 产物：`WORK_SPACE/paper/E1b_cnn_hp_I20R5_ric/`
- 选参：每日截面 Spearman(分数, ret_val) 再平均；早停仍用 val_loss

### Val 粗搜 / 精筛

| lr | dropout | patience | val Rank IC |
|----|---------|----------|-------------|
| **1e-4** | **0.5** | **5** | **0.0509**（最优） |
| 1e-4 | 0.5 | 2 | 0.0506 |
| 1e-5 | 0.5 | 2（Jiang） | 0.0312 |

### OOS（Phase 3，5-ensemble）

| 配置 | val Rank IC | EW H-L Sharpe | vs 现网 2.65 |
|------|-------------|---------------|--------------|
| 现网 I20R5（随机 val + Jiang） | — | **2.65** | — |
| **Rank IC 最优** 1e-4 / 0.5 / pat5 | 0.0509 | **2.655** | **≈0** |
| Rank IC 次优 1e-4 / 0.5 / pat2 | 0.0506 | 2.45 | −0.20 |
| Jiang 锚点 chrono 全训 | 0.031 | 2.37 | −0.28 |

## 解读

- [x] 用 **Rank IC** 选参明显优于 MCC：OOS Sharpe 从 2.37 拉回到 **≈现网 2.65**。
- [x] 最优超参变为 **lr=1e-4**（高于 Jiang 的 1e-5）、dp=0.5、patience=5；chrono 全训下与现网打平，**未显著超过**（Δ≪0.15）。
- [x] 主文仍可沿用 `ashare_model_res_I20R5`；E2 bridge **不触发**。
- [x] 说明：昨天 MCC 实验的漏洞主要在 **选参指标**，而非「A 股完全不能调参」。

```powershell
python ashare_train/3b_cnn_hp_search.py --config experiments/E1_image_baseline/config_cnn_hp_ric.yaml --phase all
```
