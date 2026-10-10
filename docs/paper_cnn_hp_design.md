# E1b：A 股 CNN 超参本地化

> 状态：进行中。结果：[`paper_results_E1b.md`](paper_results_E1b.md)  
> 编排：[`ashare_train/3b_cnn_hp_search.py`](../../ashare_train/3b_cnn_hp_search.py)  
> 配置：[`experiments/E1_image_baseline/config_cnn_hp.yaml`](../../experiments/E1_image_baseline/config_cnn_hp.yaml)

## 归属：E1b（不是 E2）

| 维度 | E1 / E1b | E2 |
|------|----------|-----|
| 问题 | 图像 CNN **如何训**、A 股最优训练超参 | **同一信息集**下 CNN vs 表格 |
| 改动 | lr / dropout / patience | 仅 tabular；CNN 分数只读 |
| 产物 | checkpoint、`ensem5_res_*.csv` | `pred_*_oos.csv` |

E2 主实验不重训 CNN。仅当 E1b 全训 OOS 明显优于现网时，可在 E2 **附录**换 `up_prob` 重评。

## 研究问题

Jiang 在美国市场定的 lr/dp/早停，在 A 股全市场 I20R5 上是否仍为最优？chrono val 下本地化超参能带来多少 OOS 增益？

## 对照

- **OOS 历史基线**（不重训）：`WORK_SPACE/ashare_model_res_I20R5/`，EW H-L Sharpe ≈ **2.65**（随机 val + Jiang 默认）。
- **锚点**：`(lr=1e-5, dp=0.5, patience=2)` 纳入搜索网格；**不**单独 chrono 重训锚点对照线。
- Phase 3：top 1–2 全训；锚点未进 top 2 时额外全训一次锚点。

## 固定设定

- I20 / R5 / 周频；架构 `D20…C256`
- 图数据：复用 `ashare_stocks_dataset/I20R5`
- `batch_size=128`，`max_epochs=50`

## Val 协议（chrono）

| 集合 | 年份 | 用途 |
|------|------|------|
| Train | 2000–2005 | 拟合 |
| Val | 2006–2007 | 选参 |
| OOS | 2008–2025 | 锁定后评一次 |

选参指标：**val_mcc**（越高越好）。禁止用 OOS Sharpe 选参。

## 搜索空间

| 参数 | 取值 | 锚点 |
|------|------|------|
| lr | 3e-6, 1e-5, 3e-5, 1e-4 | 1e-5 |
| dropout | 0.40, 0.50, 0.60 | 0.50 |
| patience | 2, 5 | 2 |

## 三阶段

1. **Phase 1**（~12 轻训）：1 ensemble，patience=1，分轴搜 + 锚点 → `search_phase1.csv`
2. **Phase 2**（3 中训）：top 3 val_mcc，1 ensemble，patience=2 → `search_phase2.csv`
3. **Phase 3**（1–3 全训）：5 ensemble + OOS + portfolio

产物根目录：`WORK_SPACE/paper/E1b_cnn_hp_I20R5/`（不覆盖 `ashare_model_res_I20R5/`）。

## 命令

```powershell
cd e:\phd\LLM_trading\CNN_trading
python ashare_train/3b_cnn_hp_search.py --phase 1
python ashare_train/3b_cnn_hp_search.py --phase 2
python ashare_train/3b_cnn_hp_search.py --phase 3
python ashare_train/3b_cnn_hp_search.py --phase all
```
