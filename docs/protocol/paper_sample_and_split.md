# 论文样本与时序切分规则（P1.1）

> 对应执行计划：`docs/framework/ashare_paper_execution_plan.md`  
> 配置源：`trend_submit/trend_code_submit/Misc/config.py`

---

## 1. 宇宙与主样本

| 项 | 规则 | 当前实现 |
|----|------|----------|
| 宇宙 | 沪深 A 股（含主板 / 创业板 / 科创板等） | `ashare_ret.feather`，约 1530 万行 |
| ST / *ST | 主样本剔除 | 预处理按「是否ST」行级过滤（`1_preprocess_ashare_data.py`） |
| 日收益 | clip ±11%（涨跌停缓冲） | `ASHARE_RET_CLIP = 0.11` |
| 价格 | 复权 OHLC | 原始「复权」目录 |
| 市值 | 流通市值（VW 权重） | `MarketCap` 列 |
| IPO 初期 | 上市后前 **60** 个交易日剔除（`PAPER_IPO_EXCLUDE_DAYS`） | **待补**：feather 无上市日字段；P1.2 前用「该股首次出现日后 60 日」近似，或从原始 CSV 补字段 |
| 长期停牌 | 窗口内有效交易日不足则样本不生成 | 图表生成阶段已有缺行情跳过 |
| 涨跌停幅度 | 按板块与历史制度逐日判定 | 稳健性脚本已实现；feather **未存** LimitUp/Down 价 |

---

## 2. 时间切分（主文 vs 附录）

### 2.1 主文：Chronological（方案 V1）

```
Train: 2005–2014   (PAPER_TRAIN_YEARS)
Val:   2015–2017   (PAPER_VAL_YEARS)      ← 按年整段验证，禁止与 Train 样本随机混洗
Test:  2018–2025   (PAPER_TEST_YEARS)     ← 仅一次 OOS，不重训
```

- 约束：`max(Train) < min(Val) < min(Test)`  
- 验证集用途：早停 / 超参；**不得**并入 Test  
- 集成：仍可 5 ensembles，但 Val 固定为 2015–2017（不再对 Train 年内随机 70/30）

### 2.2 附录：Jiang 式固定 IS（已完成）

```
IS:  2000–2007  随机 70% train / 30% val
OOS: 2008–2025
```

路径：`WORK_SPACE/ashare_model_res_I20R5/` 等。主文引用时标明切分差异。

---

## 3. 主实验设定（E1）

| 项 | 取值 |
|----|------|
| 图像窗口 | 20 交易日（I20） |
| 标签 | `Ret_week`（约 5 日），二分类 `1(R>0)` |
| 调仓频率 | 周频（与现有 I20R5 一致） |
| 模型 | CNN（现有 `D20…C256`）；ViT 仅稳健性 |
| 实验目录 | `PAPER_EXP_DIR_E1` = `WORK_SPACE/ashare_paper_E1_I20R5/` |

图表数据可复用：`WORK_SPACE/data/ashare_stocks_dataset/I20R5/`（与切分无关，按年存盘）。

---

## 4. Feather 字段清单与缺口

**已有**：`Date, Open, High, Low, Close, StockID, Vol, MarketCap, sec_name, Ret, EWMA_vol, Ret_5d/20d/…, Ret_week/month/…`

**缺口（后续补）**

| 字段 | 用途 | 优先级 |
|------|------|--------|
| 上市日 / IPO | 严格剔除上市 60 日 | P1.2 前用近似；E3 前建议补齐 |
| 涨跌停价或标记 | E5/E9 | 可继续按代码规则现算 |
| 换手率 | E5 NHT、E8 异质性 | **高**：tabular/机制需要；需从原始或 CSMAR 并入 |
| 停牌标记 | 清洗审计 | 中 |

---

## 5. 验收（P1.1）

- [x] `PAPER_TRAIN/VAL/TEST_YEARS` 写入 config  
- [x] 独立论文实验目录常量  
- [x] 本规则文档可复现  
- [ ] 训练脚本支持「按年 Val、禁止 Train 内随机切分」（→ **P1.2**）  
- [ ] 换手率字段入库（→ E2/E5 前）  

---

## 6. 下一步

**P1.2**：扩展训练入口，使 I20R5 在 `PAPER_*_YEARS` 下训练并输出 Test 期组合结果到 `ashare_paper_E1_I20R5/`。
