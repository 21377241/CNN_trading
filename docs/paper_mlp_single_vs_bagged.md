# 单 seed MLP vs Bagged MLP：对比报告

> 口径：Fair Chrono 180 + Rank IC；OOS 2008–2025；n=2,687,822；922 日；`ann_factor=52`  
> **单 seed**：legacy sklearn ES，seed=42（旧主表 / `pred_mlp_oos_seed42_legacy.csv`）  
> **Bagged**：chrono Rank-IC ES + **bag k=10**（seeds 42–51；现主表）  
> CNN / HistGB **未改**；凡依赖 MLP 的下游随 bag 重算

---

## 1. 一句话对照

| | 单 seed | Bag k=10 |
|--|---------|----------|
| MLP 在主表中的角色 | 与 CNN 接近、略弱于 HistGB | **最强单模**，明显超过 CNN 与 HistGB |
| 论文叙事风险 | 低估表格浅网 | 易读成“MLP 全面碾压”，需强调 **互补仍在** 与 **协议不对等** |
| 科学含义 | 初始化敏感，结论脆弱 | 更接近该架构在同信息集下的稳定表现 |

---

## 2. E2 主表：单模预测力

| 模型 | Rank IC（单 / bag） | EW Sharpe（单 / bag） | ΔSharpe |
|------|---------------------|----------------------|--------:|
| CNN（不变） | 0.0308 | 2.65 | — |
| HistGB（不变） | 0.0301 | 3.07 | — |
| **MLP** | **0.0266 → 0.0406** | **2.97 → 3.70** | **+0.73** |

**排序变化**

| | 单 seed | Bagged |
|--|---------|--------|
| Rank IC | HistGB ≈ CNN > MLP | **MLP ≫ HistGB ≈ CNN** |
| EW Sharpe | HistGB > MLP ≥ CNN | **MLP > HistGB > CNN** |

单 seed=42 落在 10-seed legacy 分布偏下端（探测：mean Sharpe≈3.15，seed42≈2.97），故旧主表**系统性低估** MLP。

---

## 3. E2B（CNN–MLP）：互补是否还在？

| 指标 | 单 seed MLP | Bagged MLP | 变化 |
|------|-------------|------------|------|
| 日相关 mean | **0.218** | **0.326** | 更同向 |
| Top10% 重叠 | **0.184** | **0.232** | 重叠↑ |
| Ensemble Sharpe | **3.47** | **3.91** | +0.44 |
| 相对最强单模增量 | +0.50（vs MLP 2.97） | **+0.22**（vs MLP 3.70） | 边际变薄但仍为正 |
| CNN 线性残差 Sharpe | ~2.04 | **1.29** | CNN 独有变弱但仍可交易 |
| MLP 线性残差 Sharpe | ~2.31 | **3.00** | MLP 独有变强 |

**判定**：不是“互补消失”，而是 **MLP 变强后吸收更多共变**；FM / 残差 / ensemble 仍支持 CNN 有增量，只是相对空间缩小。

---

## 4. E3-0：Shared / Specific

| 分数 | Rank IC 单→bag | EW Sharpe 单→bag | 解读 |
|------|----------------|------------------|------|
| S_sh | 0.0386 → **0.0423** | 3.71 → **3.72** | Shared 仍最强共识组合，幅度几乎不动 |
| S_mlp raw | 0.0266 → **0.0406** | 2.97 → **3.70** | 单模抬升主因 |
| S_cnn raw | 0.0308（不变） | 2.65 | — |
| S_hgb raw | 0.0301（不变） | 3.07 | — |
| S_mlp_sp | 0.0136 → **0.0243** | 1.74 → **2.57** | 独有 alpha 变强 |
| S_cnn_sp | 0.0204 → **0.0180** | 1.39 → **1.14** | 仍显著，变弱 |
| S_hgb_sp | 0.0129 → **0.0026** | 1.72 → **0.51** | **IC NW 不再显著**（p≈0.13） |

**核心结构变化**：三模“各有显著 specific” → **Shared + MLP-spec + CNN-spec**；HGB-spec 被强 MLP 线性吸收，不宜再作主证据。

---

## 5. E3-1 / E3-2：结构画像变了什么

| 块 | 单 seed 叙事 | Bagged 叙事 |
|----|--------------|-------------|
| CNN–MLP 分歧 | 相对量 + 下影 + 隔夜/贴板 | **近端动量 + 量能 + 市值**（更像 CNN–HGB） |
| MLP–HGB 分歧 | 剧烈、覆盖日多 | 仍显著，但极端日变少（≈793） |
| block R² | MLP-spec Recent 较弱 | MLP-spec Recent 升至与 CNN-spec 同级 |

含义：旧“图 vs 表靠影线/相对量吵架”部分依赖弱 MLP；稳定化后分歧轴更贴近**动量–量能**。

---

## 6. E3-3 / E3-4：命门是否变？

| 检验 | 单 seed | Bagged | 结论 |
|------|---------|--------|------|
| E3-3 CNN-spec baseline Sharpe | 1.39 | 1.14 | 基准下降 |
| mask_last5 | 打穿（→负） | 打穿（→负） | **命门不变** |
| E3-4 MLP-spec baseline | 1.74 | **2.57** | 独有更强 |
| drop_recent5 | 打穿 | 打穿 | **命门不变** |

近端路径仍是两侧 specific 的共同脆弱点。

---

## 7. 这是不是“MLP 全方位碾压”？

### 看起来像碾压的地方

- 单模 IC / Sharpe 全面最高  
- E3 MLP-spec 最强  
- Ensemble 以 MLP 为主导抬升  

### 并不等于“CNN / 图像无意义”

1. **Shared 仍 ≥ MLP raw**（3.72 ≈ 3.70）：共识不是被 MLP 单独替代。  
2. **CNN 残差 / FM M3 / ensemble 增量仍显著** → 图像通道未被完全吸收。  
3. **协议不对等**：MLP 用了 10-seed bag；CNN 是固定 E1b 五模型平均（另一种 ensemble）；HistGB 仍是单模网格。直接比“谁碾压谁”不公平。  
4. **叙事主线本就不是“谁最强”**，而是：同信息集下不同非线性提取 → shared + specific；bag 后 puzzle 变成“强 tabular 下 CNN-spec 还剩什么、对应何结构”。

### 建议写法（论文）

- 主表：**报告 bagged MLP 为稳定化 tabular 上界**；附录保留单 seed + 稳定性曲线。  
- 避免写“CNN 失败”；写“单 seed MLP 低估了表格浅网；稳定后 MLP 领先，但与 CNN 仍互补”。  
- E3 主证据转向：**Shared + MLP-spec + CNN-spec**；HGB-spec 作削弱对照。

---

## 8. 数字速查卡

| 对象 | 单 seed | Bag k=10 |
|------|--------:|---------:|
| MLP Rank IC | 0.0266 | 0.0406 |
| MLP EW Sharpe | 2.97 | 3.70 |
| CNN–MLP ρ | 0.218 | 0.326 |
| CNN–MLP ens Sharpe | 3.47 | 3.91 |
| Shared Sharpe | 3.71 | 3.72 |
| MLP-spec Sharpe | 1.74 | 2.57 |
| CNN-spec Sharpe | 1.39 | 1.14 |
| HGB-spec Sharpe | 1.72 | 0.51 |

产物路径：

- 旧 pred：`WORK_SPACE/paper/E2_matched_is_ind_chrono_ric/preds/pred_mlp_oos_seed42_legacy.csv`  
- 新 pred：`.../pred_mlp_oos.csv`  
- 探测：`docs/paper_results_mlp_stability.md`  
- 现结论：`docs/paper_results_E2.md` / `E2B_mlp.md` / `E3.md`
