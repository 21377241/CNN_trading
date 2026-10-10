# MLP 稳定性探测：完整结果与解读

> 脚本：`experiments/E2_matched_tabular/run_mlp_stability_probe.py`  
> 产物：`WORK_SPACE/paper/E2_mlp_stability_probe/`  
> **主线状态（已切换）**：`config_ind_chrono_ric.yaml` 使用 **chrono_es + bag k=10**；正式 pred 为 `E2_matched_is_ind_chrono_ric/preds/pred_mlp_oos.csv`（旧单 seed 备份：`pred_mlp_oos_seed42_legacy.csv`）  
> **k=20 校验（2026-09）**：chrono bag 曲线已延伸至 k=20；**k=10→20 无实质增益**，主表维持 k=10。

**动机**：主表 / E3 中的 MLP 为 **单 seed=42 + sklearn 随机 validation early-stop**。探测要回答：(1) 单 seed 方差有多大；(2) 换 chrono val Rank-IC early-stop 能否压稳；(3) bagging 规模 k 取多少够用。

---

## 1. 协议定义

| 协议 | 做法 |
|------|------|
| **legacy_es** | `MLPClassifier(early_stopping=True, validation_fraction=0.15)`，与现有 `run_train` 主路径一致 |
| **chrono_es** | 固定 2000–05 train / 2006–07 val；按 **val Rank IC** early-stop（`max_iter=200`, patience=15），恢复最佳 epoch 权重 |
| **bag k** | 同协议下前 k 个 seed 的 OOS `up_prob` **等权平均**，再评 IC / Sharpe |

网络固定：`StandardScaler` + `(128, 64)`，`α=1e-4`，`lr=1e-3`。

---

## 2. 单 seed 汇总（n=10）

| protocol | Rank IC mean±std | min–max | EW Sharpe mean±std | min–max | VW Sharpe mean±std | seed 间 OOS Spearman | 平均 best_epoch |
|----------|-----------------:|--------:|-------------------:|--------:|-------------------:|---------------------:|----------------:|
| legacy_es | 0.0297±0.0037 | 0.0239–0.0342 | **3.15±0.30** | 2.67–3.62 | 1.21±0.28 | **0.467** | 30.0 |
| chrono_es | 0.0314±0.0047 | 0.0229–0.0386 | **3.07±0.40** | 2.46–3.73 | 1.20±0.20 | **0.561** | 5.7 |

相对论文主表 MLP（seed=42 legacy：Rank IC **0.0266**，EW Sharpe **2.97**）：落在 legacy 分布**偏下端**，不是“中位代表值”。

### 解读

- **Sharpe 波动远大于 Rank IC**：只报 IC 会低估不稳定；单 seed Sharpe 可相差约 **1 个 Sharpe 单位**。
- **chrono 并未压低单 seed 方差**：RIC std、Sharpe std 反而略高于 legacy；但 seed 间 OOS 排序相关更高（0.56 vs 0.47）→ 换 early-stop 让预测更“同向”，**不能**替代 bagging。
- chrono 常在 **epoch 2–10** 就取最优（mean best_epoch≈5.7），偏早停；单模型仍抖，符合“初始化敏感 + 欠拟合波动”。

---

## 3. 逐 seed 明细

### 3.1 legacy_es

| seed | n_iter | OOS Rank IC | EW H-L ann | EW Sharpe | VW Sharpe |
|-----:|-------:|------------:|-----------:|----------:|----------:|
| 42 | 24 | **0.0266** | 0.2483 | **2.967** | 1.321 |
| 43 | 37 | 0.0312 | 0.2686 | 3.229 | 1.136 |
| 44 | 30 | 0.0342 | 0.3111 | 3.248 | 1.270 |
| 45 | 22 | 0.0263 | 0.2333 | 2.743 | 1.223 |
| 46 | 30 | 0.0315 | 0.2816 | 3.290 | 1.230 |
| 47 | 25 | 0.0307 | 0.2610 | 2.970 | 0.815 |
| 48 | 31 | 0.0261 | 0.2734 | 3.445 | 1.697 |
| 49 | 38 | 0.0329 | 0.2686 | 3.283 | 1.535 |
| 50 | 37 | 0.0239 | 0.2247 | 2.674 | 0.840 |
| 51 | 26 | 0.0337 | 0.3093 | 3.620 | 1.013 |

### 3.2 chrono_es

| seed | stop@ | best_ep | val Rank IC | OOS Rank IC | EW Sharpe | VW Sharpe |
|-----:|------:|--------:|------------:|------------:|----------:|----------:|
| 42 | 18 | 3 | 0.0586 | 0.0323 | 2.906 | 1.244 |
| 43 | 19 | 4 | 0.0652 | 0.0300 | 2.712 | 1.065 |
| 44 | 18 | 3 | 0.0561 | 0.0317 | 2.976 | 1.229 |
| 45 | 19 | 4 | 0.0602 | 0.0289 | 2.991 | 1.179 |
| 46 | 22 | 7 | 0.0602 | 0.0332 | 3.347 | 1.230 |
| 47 | 25 | 10 | 0.0576 | 0.0378 | 3.644 | 1.486 |
| 48 | 22 | 7 | 0.0582 | 0.0322 | 3.048 | 0.999 |
| 49 | 17 | 2 | 0.0524 | **0.0229** | **2.459** | 0.874 |
| 50 | 23 | 8 | 0.0486 | 0.0268 | 2.876 | 1.175 |
| 51 | 24 | 9 | 0.0616 | **0.0386** | **3.730** | 1.544 |

训练墙钟约 **66–89 s / seed**（全量 IS fit）；瓶颈在组合评测而非拟合。

---

## 4. Bagging 曲线（k=1…20，chrono）

对前 k 个 seed 的概率做等权平均（种子顺序固定为 42,43,…）。主线协议为 **chrono**。

| k | Rank IC | EW ann | EW Sharpe | VW Sharpe |
|--:|--------:|-------:|----------:|----------:|
| 1 | 0.0323 | 0.281 | 2.906 | 1.244 |
| 5 | 0.0378 | 0.360 | 3.385 | 1.657 |
| **10** | **0.0406** | **0.394** | **3.695** | **1.635** |
| 15 | 0.0399 | 0.402 | 3.695 | 1.648 |
| **20** | **0.0403** | **0.405** | **3.642** | **1.593** |

### 4.1 够不够：5 / 10 / 20

| 对比 | ΔRank IC | ΔEW Sharpe | 门槛判定（\|ΔIC\|<0.001 且 \|ΔSharpe\|<0.05） |
|------|---------:|-----------:|:---|
| k=5 → 10 | +0.0028 | +0.311 | **未收敛** |
| k=10 → 20 | **−0.0003** | **−0.054** | IC 已收敛；Sharpe 贴门槛（略超 0.05，且为小幅回落） |

**判定**：增益主要在 **k≤10**；k=10→20 **几乎不再抬升**（甚至略回落）。主表继续用 **k=10** 足够；不必升到 20。

曲线在 k=10–18 附近在 Sharpe≈3.58–3.72 小幅震荡，属 bag 噪声，不是系统性再提升。

---

## 5. 与现有论文数字对照

| 对象 | Rank IC | EW Sharpe | 说明 |
|------|--------:|----------:|------|
| E2 主表 MLP（seed=42 legacy） | 0.0266 | 2.97 | 现行报告值 |
| legacy 10-seed 均值 | 0.0297 | 3.15 | 单 seed 无偏中心（相对 seed=42） |
| chrono 10-seed 均值 | 0.0314 | 3.07 | |
| legacy bag k=5 | 0.0397 | 3.73 | |
| legacy bag k=10 | **0.0415** | **4.03** | 本探测建议定稿候选 |
| chrono bag k=5 | 0.0378 | 3.38 | |
| chrono bag k=10 | 0.0406 | 3.70 | |
| chrono bag k=20 | 0.0403 | 3.64 | 相对 k=10：ΔIC≈0，ΔSharpe≈−0.05 → **k=10 已够** |
| E2/E3 中 CNN raw（参考） | ~0.0308 | ~2.65 | bagged MLP 超过 CNN raw |
| E3 Shared | ~0.0423 | ~3.72 | 与 bag@10 接近 |

---

## 6. 总判定

| 问题 | 判定 |
|------|------|
| 单 seed 是否够稳？ | **否**。Spearman~0.55，Sharpe std（20 seeds）~0.51。 |
| 只换 chrono ES 够不够？ | **不够**。略增一致性，不降单模型 Sharpe 方差。 |
| k=5 够不够？ | **定方向够；定稿偏紧**（→10 仍有 +0.3 Sharpe）。 |
| k=10 够不够？ | **够**。k=10→20：ΔIC≈0、ΔSharpe≈−0.05（无再抬升）。 |
| 推荐报告协议 | **chrono + bag k=10**（主表已采用）；不必升到 20。 |

---

## 7. 产物清单

| 文件 | 内容 |
|------|------|
| `mlp_stability_detail.csv` | 逐 seed IC / Sharpe / epoch |
| `mlp_stability_summary.csv` | 协议级 mean±std |
| `mlp_bagging_curve.csv` | k=1…20 曲线（chrono） |
| `mlp_enough_seeds_check.csv` | 5 / 10 / 20 差分 |
| `mlp_stability_verdict.txt` / `mlp_stability_meta.json` | 自动判定与元数据 |
| `prob_chrono_seed*.npy` | OOS 概率（可 resume） |
| `portfolio/*` | 各 seed / bag 组合明细 |

复现：

```bash
python experiments/E2_matched_tabular/run_mlp_stability_probe.py --seeds 20 --protocols chrono
```
