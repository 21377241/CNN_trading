# E5 结果：Economic Mechanism of Recent-Path Signals

> 设计：[`paper_e5_mechanism_design.md`](paper_e5_mechanism_design.md)  
> 接续：[`paper_results_E4.md`](paper_results_E4.md)  
> 产物：`WORK_SPACE/paper/E5_mechanism_chrono_ric/`  
> Specific：E4 两元 \(S_C^{sp},S_M^{sp}\)；**不重训**  
> 范围：**Phase 1（E5-1/2/3）+ Phase 2（E5-4/5/6）+ E5-7（制度筛选）**

---

## Track 总览

\[
\text{Different recent-path representations}
\rightarrow
\text{different specific signals}
\rightarrow
\text{different return-realization dynamics}
\]

Phase 1 缩小机制空间；Phase 2 判定 **continuation vs reversal** 与 Pressure 放大。

| Phase | 步骤 | 输出 |
|-------|------|------|
| 1 | E5-1 Horizon | `e51_horizon_ic.csv` |
| 1 | E5-2 ON/ID | `e52_session_ic.csv` |
| 1 | E5-3 State | `e53_state_delta.csv` |
| 2 | E5-4 Pressure | `e54_pressure_ic.csv` |
| 2 | E5-5 Interaction | `e55_interaction_fm.csv` |
| 2 | E5-6 Cont/Rev | `e56_cont_rev_spreads.csv` |
| 3 | E5-7 T+1 / Limit | `e57_*.csv` |

**数据**：日频 \(r_{t+h}\)/\(R_{1:5}\)；`TurnoverProxy_5`；Pressure 预注册见设计文。

---

## 预注册判定

| 假设 | 判定 | 摘要 |
|------|------|------|
| **H1** Horizon | **通过** | CNN：t+1 峰值后快衰；MLP：t+2 峰，多日显著 |
| **H2** ON/ID | **弱** | CNN 偏 ID（边缘）；MLP 无分工 |
| **H3** State | **通过** | 高 TurnoverProxy 双侧更强；MLP 另有高 CumRet5/波动 |
| **H4** Pressure × S | **通过（CNN）** | CNN \(\beta_3>0\) 显著；MLP interaction≈0 |
| **H5** Cont vs Rev | **可区分** | MLP：正近端后 spread 最强 → **continuation**；CNN：负近端后 spread 仍强 → 更偏 **rebound / mixed** |

---

## Phase 1 摘要（E5-1/2/3）

### E5-1 Horizon

| | CNN-sp | MLP-sp |
|--|--------|--------|
| 形态 | **短窗**：IC(r1)=0.018，r4 后不显著；\(R_{3:5}\) 弱 | **持续**：峰值 r2=0.028；r1–r5 均显著；\(R_{1:10}\) 仍强 |
| 情形 | A（快速兑现） | B/C 混合 |

### E5-2 Session

CNN：Intraday IC > Overnight（Δ NW t≈−1.6～−1.9）→ *tendency toward intraday realization*，未证成流动性机制。MLP：无显著 ON/ID 差。

### E5-3 State（\(R_{1:5}\)）

高 TurnoverProxy：CNN ΔIC=+0.010（t=3.6）；MLP +0.017（t=6.4）。MLP 另在高 CumRet5（Δ=+0.027）与高波动显著。

---

## E5-4 — Pressure tercile IC

### 设计

\[
Pressure=\tfrac{1}{3}\big[z(TurnoverProxy_5)+z(-MaxDD_5)+z(VolEndOverMean_5)\big]
\]

每日 tercile；报 Low/Mid/High 上 \(S^{sp}\) 对 \(R_{1:5}\) / \(r_1\) 的 Rank IC。

### 结果

| ret | score | IC Low | Mid | High | High−Low | NW t |
|-----|-------|-------:|----:|-----:|---------:|-----:|
| R_1_5 | CNN | 0.0134 | 0.0144 | 0.0180 | +0.0046 | 1.61 |
| R_1_5 | MLP | 0.0263 | 0.0320 | **0.0448** | **+0.0185** | **6.68** |
| r1 | CNN | 0.0161 | 0.0148 | **0.0217** | **+0.0056** | **2.00** |
| r1 | MLP | 0.0109 | 0.0170 | 0.0205 | +0.0097 | 3.50 |

### 解读

- MLP：Pressure 上 IC **单调增强**（主窗极显著）。  
- CNN：主窗边缘；**隔日 r1 显著** → 压力放大更贴短 horizon，与 E5-1 一致。

---

## E5-5 — \(S^{sp}\times Pressure\) FM

### 设计

日截面（变量均 cs_zscore）：

\[
R=\alpha+\beta_1 S^{sp}+\beta_2 Pressure+\beta_3(S^{sp}\times Pressure)+\gamma_1 Size+\gamma_2 CumRet5+\varepsilon
\]

Fama–MacBeth + NW lag=4。Controls 不含 Turnover（已进 Pressure）。

### 结果（关键 \(\beta_3\)）

| ret | score | \(\bar\beta_1\) (t) | \(\bar\beta_3\) interact (t) |
|-----|-------|---------------------:|-----------------------------:|
| R_1_5 | CNN | 0.00039 (3.85) | **0.00035 (4.19)** |
| R_1_5 | MLP | 0.00135 (11.85) | −0.00003 (−0.46) |
| r1 | CNN | 0.00032 (6.97) | **0.00024 (7.21)** |
| r1 | MLP | 0.00034 (7.86) | −0.00001 (−0.30) |

### 解读

**H4：CNN 通过，MLP 不通过。**  
→ Pressure 是 **CNN-spec 预测力的状态放大器**；MLP 虽在高 Pressure 桶 IC 更高，但线性 interaction 不显著（水平效应 / 非线性分层，而非稳健乘积项）。

---

## E5-6 — Continuation vs Reversal

### 设计

CumRet5 × \(S^{sp}\) 各 tercile；桶内 High\(S^{sp}\)−Low\(S^{sp}\) 的未来收益 spread。正近端桶 spread>0 → continuation；负近端桶 spread>0 → 高分股在跌后相对更强（rebound / 信号排序，非 CumRet 动量延续）。

### 结果（High−Low \(S^{sp}\) spread）

| ret | score | 负近端 | t | 正近端 | t |
|-----|-------|-------:|--:|-------:|--:|
| R_1_5 | CNN | **0.00175** | 5.58 | 0.00068 | 1.75 |
| R_1_5 | MLP | 0.00204 | 6.86 | **0.00431** | **10.46** |
| r1 | CNN | 0.00084 | 6.41 | 0.00066 | 3.73 |
| r1 | MLP | 0.00077 | 5.89 | 0.00043 | 2.63 |

高 Pressure 子集（同表 `e56_high_pressure_spreads.csv`）：CNN/MLP 在负、正近端 spread 仍为正；MLP 正近端 \(R_{1:5}\) 仍最强。

### 解读（H5）

1. **MLP-spec → continuation**：正 CumRet5 后 High−Low spread **最大且最显著**。  
2. **CNN-spec → mixed / rebound-leaning**：负近端后 spread **更强**；正近端 \(R_{1:5}\) 仅边缘 → 不像干净的动量延续，更像短窗下「路径压力后的相对反弹排序」与弱 continuation 并存。  
3. 高 Pressure 不改变方向定性，但抬高幅度（与 E5-4/5 一致）。

**不要过度解读**：负近端正 spread ≠ 已证明 temporary price pressure 制度；只说明方向性上 CNN 与「跌后相对修复」更合拍，MLP 与「涨后继续」更合拍。

---

## 综合叙事

| | CNN-spec | MLP-spec |
|--|----------|----------|
| Horizon | short-lived (t+1) | persistent (peak t+2) |
| Session | intraday-leaning (弱) | 无分工 |
| State | turnover / Pressure（短窗） | turnover + momentum/vol |
| Pressure × S | **\(\beta_3>0\)** | ≈0 |
| Direction | rebound-leaning / mixed | **continuation** |

\[
\boxed{
\text{CNN-spec: short-lived, pressure-amplified, rebound-leaning}
}
\]

\[
\boxed{
\text{MLP-spec: persistent, momentum-state continuation}
}
\]

**E5-7 已跑（筛选级，非因果）**：T+1-adjacent 状态对 CNN 有显著 \(S\times State\) 放大，但对简单 IC 差分不稳；涨停日伤 CNN 隔日 IC，FailedLimit 抬升 CNN IC。**仍不能写成“已证明 T+1/涨跌停机制”。**

---

## E5-7 — T+1-adjacent & Price Limit（筛选，非因果）

### 设计

| 块 | 定义 | 评价 |
|----|------|------|
| **T1_state** | CumRet5 最低 tercile × TurnoverProxy 最高 tercile（约 11.2% 样本） | 组内 Rank IC vs 组外；FM \(S^{sp}\times T1\_state\) |
| **LimitHit** | 收盘触及当日涨跌停价（板规则：主板 10%、科创/创业板改革后 20%、北交所 30%；**未单独识别 ST 5%**） | 组内 vs 组外 IC |
| **FailedLimit** | 日内触及但收盘未封板 | 同上 |
| **LimitUp / LimitDn** | 收盘涨停 / 跌停 | 同上 |

标签主看 **r1**（隔日，贴近 T+1 叙事）与 \(R_{1:5}\)。

### A. T+1-adjacent

| ret | score | IC in | IC out | Δ | NW t(Δ) | FM \(\beta_3\) (t) |
|-----|-------|------:|-------:|--:|--------:|-------------------:|
| r1 | CNN | 0.020 | 0.018 | +0.002 | 0.66 | **0.00016 (2.20)** |
| r1 | MLP | 0.017 | 0.013 | +0.004 | 1.32 | 0.00007 (1.05) |
| R_1_5 | CNN | 0.019 | 0.017 | +0.001 | 0.38 | **0.00071 (3.93)** |
| R_1_5 | MLP | 0.011 | 0.035 | **−0.024** | **−7.84** | **−0.00077 (−4.54)** |

解读：

1. 简单 IC 差分：**不能**说 T1_state 提高了 CNN 的 IC（Δ 不显著）。  
2. FM interaction：**CNN \(\beta_3>0\)**（r1 与 \(R_{1:5}\)）→ 在控住主效应后，该状态放大 CNN-spec 的收益载荷。  
3. MLP 在 T1_state（负近端×高换手）下 \(R_{1:5}\) IC **显著更弱**，interaction 为负 → 与 continuation / 高 CumRet5 放大器一致，**反对**把 MLP 主叙事绑到 T+1-adjacent 卖方压力。  
4. **禁止**把 \(\beta_3>0\) 写成“证明了 T+1”；只能写 *consistent with delayed adjustment under high-turnover loser states*。

### B. Price limits

覆盖：LimitHit≈1.3%，FailedLimit≈0.7%，LimitUp≈1.0%，LimitDn≈0.3%（跌停日有效截面天数仅约 74，噪声大）。

| 状态 | ret | CNN ΔIC (t) | MLP ΔIC (t) |
|------|-----|------------:|------------:|
| LimitHit | r1 | −0.007 (−0.74) | −0.005 (−0.48) |
| LimitHit | R_1_5 | +0.012 (1.29) | +0.014 (1.60) |
| **FailedLimit** | **r1** | **+0.026 (2.45)** | −0.007 (−0.64) |
| FailedLimit | R_1_5 | +0.026 (2.34) | +0.002 (0.21) |
| **LimitUp** | **r1** | **−0.041 (−4.20)** | **+0.029 (2.93)** |
| LimitUp | R_1_5 | −0.006 (−0.51) | +0.033 (3.26) |

解读：

1. **封板 LimitHit 整体**未稳定抬升 CNN 隔日 IC。  
2. **FailedLimit** 下 CNN IC 显著高于非该状态 → 与“未完全成交的涨跌停摩擦 / 次日继续博弈”方向相容，仍非因果。  
3. **涨停 LimitUp**：CNN 隔日 IC **显著变差**（甚至为负）；MLP 反而更强 → 两边在涨停日上的兑现相反，提醒制度状态会**分化**而非统一放大 specific。  
4. 跌停样本太少，不作主结论。

### E5-7 总判

| 问题 | 答案 |
|------|------|
| 能否进正文当“T+1 机制已证实”？ | **否** |
| 有没有筛选价值？ | **有**：CNN 与 T1_state / FailedLimit 的放大或分化模式，与短窗+pressure+rebound 叙事**同向相容**；MLP 不支持 T+1-adjacent 主故事 |
| 涨跌停？ | 不是简单“涨停→更强”；**涨停日伤 CNN 隔日、FailedLimit 抬 CNN** 更值得写进稳健讨论 |

---

## 综合叙事（含 E5-7）

| | CNN-spec | MLP-spec |
|--|----------|----------|
| Horizon | short-lived (t+1) | persistent (peak t+2) |
| Session | intraday-leaning (弱) | 无分工 |
| Pressure × S | **\(\beta_3>0\)** | ≈0 |
| Direction | rebound-leaning / mixed | **continuation** |
| T1_state | IC Δ 弱；**FM \(\beta_3>0\)** | \(R_{1:5}\) **更弱** |
| Limits | FailedLimit↑；LimitUp 伤 r1 | LimitUp 上更强 |

\[
\boxed{
\text{CNN-spec: short-lived, pressure-amplified, rebound-leaning;}
}
\]
制度侧仅达 *screening consistency*，未达因果识别。

\[
\boxed{
\text{MLP-spec: persistent momentum-state continuation}
}
\]
不宜用 T+1 / 跌停叙事主推。

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `forward_labels.parquet` | Phase 1 面板 |
| `e51_*` / `e52_*` / `e53_*` | Phase 1 |
| `e54_pressure_ic.csv` | Pressure tercile IC |
| `e55_interaction_fm.csv` | FM \(\beta_3\) |
| `e56_cont_rev_spreads.csv` | Cont/Rev spreads |
| `e56_high_pressure_spreads.csv` | 高压子集 |
| `e56_33_summary.csv` | 3×3 单元格 |
| `e57_t1_state_ic.csv` | T1_state IC |
| `e57_t1_interaction_fm.csv` | T1 / NegCumRet FM |
| `e57_limit_state_ic.csv` | 涨跌停状态 IC |
| `e57_limit_vs_normal_ic.csv` | vs Normal |
| `e57_coverage.csv` | 状态覆盖率 |
