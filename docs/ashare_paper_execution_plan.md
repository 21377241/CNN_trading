# A 股图像化价格模式论文：可执行计划

> 依据：`中国A股图像化价格模式研究方案_V1.docx`  
> 主线：Prediction → Interpretation → Economic Mechanism  
> 主设定：**I20/R5**（20 日图像 → 未来约 5 日收益）  
> 原则：机制由前置结果驱动，不预绑死 T+1 / 涨跌停

---

## 0. 现状盘点（已完成）

| 项 | 状态 | 路径 / 备注 |
|----|------|-------------|
| A 股预处理 + ST 过滤 | ✅ | `ashare_ret.feather` |
| I20/R20、I5/R5、I20/R5 图表与训练 | ✅ | `ashare_model_res_*` |
| 十分位 / H-L / TopN | ✅ | I20R5 EW H-L Sharpe ≈ 2.65 |
| 剔涨停 + 市值宇宙 TopK | ✅ | I20R5 几乎不变；I5 失效 |
| Jiang 式切分 IS2000–2007 / OOS2008–2025 | ✅ | 与方案 chronological 不同 |
| E2 Matched tabular | ✅ | 主协议 + chrono Val §4 |
| E3 Traditional + FM | ❌ | |
| E4–E9 解释与机制 | ❌ | |

---

## Phase 1 — 现象确认（论文 Table 1–5 基础）

**门槛**：E1（IS 切分下 I20/R5）已完成；E2 明确相对 Logit / GBDT 的位置。

### P1.1 样本规则文档
- ✅ `docs/paper_sample_and_split.md` + `PAPER_*`（供日后 chronological 可选）

### P1.2 E1 基线（IS 切分）
- ✅ 已完成：I20/R5 等，见复现报告（EW H-L Sharpe ≈ 2.65）
- **E1b（已完成）**：A 股 CNN 超参本地化；chrono val 搜参 val_mcc 最优 lr=3e-5/dp=0.6/pat=5，但 OOS EW Sharpe 2.37 **低于** 现网 2.65 → 主文仍用原 I20R5；见 [`docs/paper_results_E1b.md`](paper_results_E1b.md)

### P1.3 E2 Matched-Information Benchmark（**已完成；IS 切分**）
> 设计：[`docs/paper_e2_design.md`](paper_e2_design.md)。结果：[`docs/paper_results_E2.md`](paper_results_E2.md)。

- **切分**：与 E1 相同 —— IS 2000–2007（70/30）/ OOS 2008–2025  
- **Image**：复用 I20R5 `up_prob`（不重训）  
- **Tabular**：同一 20 日窗归一化 OHLC + Vol + **仅 MA20**（**不加 MA5**）；180 维 + indicators 见结果 §2  
- **模型**：Logit | **HistGB（主对照）** | MLP  
- **产物**：`experiments/E2_matched_tabular/`；`WORK_SPACE/paper/E2_matched_is/`、`E2_matched_is_ind/`  
- **判定（主协议）**：120 维 CNN EW Sharpe 2.65 vs HistGB 2.49 → **CNN_gt_HistGB**（窄）；180 维 ≈ HistGB  
- **额外稳健性（时序 Val）**：train 2000–2005 / val 2006–2007；产物 `E2_matched_is_chrono/`、`E2_matched_is_ind_chrono/`；结果文档 §4（不覆盖主表）  

### P1.4 E3 Traditional 基准 + 增量回归初版
- **做什么**
  - 特征：MOM、STR、MACD、RSI、MA cross、52WH、5/20 日收益等
  - Traditional-Logit / Traditional-GBDT vs Raw-OHLCV-GBDT vs Image
  - 横截面 Corr(S_image, 指标)；Fama–MacBeth：`R ~ S_image + traditional`
- **交付物**
  - `ashare_train/paper_e3_traditional_fm.py`
  - `docs/paper_results_E3.md`
- **验收**
  - 报告 Image-only / Traditional-only / Joint 系数与 Newey–West t

**Phase 1 完成标准**：可写出论文「实证设定 + 主预测结果 + 公平比较」三节草稿。

---

## Phase 2 — 解释模型（Table 6–9）

**门槛**：指出模型最依赖的结构，并量化翻译变量对 image score 的解释度。

### P2.1 E4 基础 Perturbation（固定模型，只改测试图）
| ID | 操作 | 检验 |
|----|------|------|
| P-rec5 | 遮最近 5 根 | Recency |
| P-old5 | 遮最早 5 根（placebo） | Recent vs Old |
| P-flip | 时间左右翻转 | Temporal order |
| P-nowick | 去掉上下影 | Intraday extremes |
| P-novol | 去掉成交量面板 | Volume |
| P-gray | 去涨跌色 | Color |

每项输出：`E|ΔS|`、Corr(S,S')、IC、H-L Sharpe。

### P2.2 E5 中国特定 Perturbation（按 E4 结果择优）
- Upper vs Lower wick
- Limit-related wick vs 普通 wick
- （可选）NHT：Negative × High Turnover + 三类 placebo
- （可选）Overnight vs Intraday（难度高，可后置）

### P2.3 E6 Translation Ladder
- Blocks：Traditional → Recent → Wick → Sequence → Price-Volume → China-Specific
- 指标：IC / Sharpe / `R²(S_image, Ŝ_tabular)`

### P2.4 E7 Score Decomposition
- `S = explained + residual`；分别做收益预测与 FM
- 仅当 residual 仍显著时，强化 Phase 3

**Phase 2 完成标准**：Figure「遮挡敏感性」+ Figure「翻译阶梯闭合 gap」可出。

---

## Phase 3 — 机制（Table 10–11，择一做深）

| 前置发现 | 优先机制 |
|----------|----------|
| Recent + Neg×HighTurnover | **E8 T+1**（overnight vs intraday） |
| Limit-related wick | **E9 涨跌停** |
| Traditional+translated 已解释大部分 | 机制降为辅助，主打可翻译性 |
| Image ≈ matched GBDT | 主打提取效率，机制弱化 |

- E8：`R_overnight` / `R_intraday` ~ S_image；按 turnover/size 异质性  
- E9：LimitHit / FailedLimit 与 NormalWick 重要性对比（可复用现有涨停代码）

**Phase 3 完成标准**：一条与中国制度一致的机制叙述 + 至少一张机制表。

---

## Phase 4 — 稳健性与成文（Table 12 + 全文）

- Window 5/20/60；Horizon 1/5/20（已有部分）
- EW vs VW；大/小盘；交易成本与可交易性（已有涨停 TopK）
- 复权方案；牛熊/高低波动
- 整理 Table 1–12、Figure 1–7；中英文摘要与贡献段落

---

## 执行顺序（严格依赖）

```
P1.1 样本与切分
  → P1.2 E1 chronological I20/R5
    → P1.3 E2 matched tabular
      → P1.4 E3 traditional + FM
        → P2.1 E4 perturbation
          → P2.2 E5（择优）
            → P2.3–P2.4 E6/E7
              → P3 E8 或 E9
                → P4 稳健性 + 成文
```

**模型策略**：全文主模型 = 现有 CNN；ViT/Hiera 仅 Phase 4 稳健性（可选）。

---

## 当前进度

| 步骤 | 状态 |
|------|------|
| 执行计划文档 | ✅ |
| P1.1 样本与切分文档 | ✅ |
| P1.2 E1（IS 切分 I20R5） | ✅ 已完成（复现实验） |
| P1.3 E2 Matched tabular | ✅ `docs/paper_results_E2.md`；主协议 CNN≈/略>HistGB；chrono Val 见 §4 |
| P1.4–P4 | ⬜ |

---

## 每次迭代产出约定

1. 改代码 / 跑实验  
2. 更新对应 `docs/paper_results_*.md`  
3. 更新本文件「当前进度」  
4. 简短告知：完成了什么、门槛是否过、下一步是什么  
