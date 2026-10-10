# E5 设计：Economic Mechanism of Recent-Path Signals

> 接续：E4 [`paper_results_E4.md`](../results/paper_results_E4.md) / [`paper_e4_recent_path_design.md`](paper_e4_recent_path_design.md)  
> 主模型：CNN-Ens5 vs MLP-Bag10  
> 面板：`E2_matched_is_ind_chrono_ric`；Specific 复用 E4 两元 \(S_C^{sp},S_M^{sp}\)  
> OOS：2008–2025；产物：`WORK_SPACE/paper/E5_mechanism_chrono_ric/`  
> 脚本：`experiments/E5_mechanism/`

---

## 1. 研究背景

E4 已表明：两边都依赖近 5 日，但 lag / PathShape / block 剖面不同（H1–H3 通过）。E5 不再问「是否不同」，而问：

\[
\boxed{\text{这些不同的 recent-path signals 为什么会转化为未来收益？}}
\]

先不把 T+1 / 涨跌停当作既定机制，而用 **收益兑现路径 + 状态依赖** 筛选方向。

---

## 2. Research Questions

| RQ | 问题 |
|----|------|
| RQ1 | Alpha 在什么 horizon 兑现？（立即 / 延迟 / 持续） |
| RQ2 | 兑现发生在 overnight 还是 intraday？ |
| RQ3 | 是否在特定市场状态（换手代理 / 波动 / 规模 / 近端收益）更强？ |

---

## 3. 固定协议

| 项 | 设定 |
|----|------|
| CNN / MLP | E1b-ric Ens5；chrono bag k=10；**不重训** |
| Specific | \(S_C^{sp}=S_C-\widehat E[S_C\mid S_M]\) 等（E4 `baseline_scores.parquet`） |
| 面板 | 同 E4 common panel；状态变量仅用 \(t\) 及以前信息 |
| 推断 | 日序列 + NW lag=4（重叠收益） |
| HistGB | 附录 only |

### 本仓库数据 realization（重要）

| 点 | 说明 |
|----|------|
| 训练标签 | E2 主标签是 **`Ret_week`**（周频期末→下期末），不是严格日历 \(t{+}1{:}t{+}5\) |
| E5-1 主分析 | 用日频 `ashare_ret.Ret` 构造 \(r_{t+h}\)，并附 `Ret_5d` / `Ret_week` 对照行 |
| ON/ID | 从 `Open/Close` 现算 log return；无现成列 |
| Turnover | 面板无换手率 → 用 **`TurnoverProxy_5`**=近 5 日 \(\mathrm{mean}(Vol/MarketCap)\)，结果文不得写作真实 Turnover |

---

## 4. 分期

| Phase | 内容 | 本轮 |
|-------|------|------|
| **Phase 1** | E5-1 Horizon；E5-2 ON/ID；E5-3 State | **执行** |
| **Phase 2** | E5-4 Pressure；E5-5 Interaction；E5-6 Cont/Rev | 条件触发（设计保留） |
| **Phase 3** | E5-7 T+1 / Price Limit | 仅当 Phase1–2 指向 trading pressure |

执行顺序：\(E5\text{-}1\to 2\to 3\)；再决定是否进入 4–6 / 7。

---

## 5. E5-1 Future-Horizon Decomposition

- 构造 \(r_{t+1},\ldots,r_{t+10}\) 与 \(R_{1:2},R_{3:5},R_{1:5},R_{1:10}\)  
- 对 \(S_C^{sp},S_M^{sp}\) 报 \(\overline{\mathrm{RankIC}}_h\) + NW t；可选 H-L  
- 解释：A 快速衰减；B 峰值后移；C 持续至 t+5/10  

## 6. E5-2 Overnight vs Intraday

\[
r^{ON}_{t+h}=\ln\frac{O_{t+h}}{C_{t+h-1}},\quad
r^{ID}_{t+h}=\ln\frac{C_{t+h}}{O_{t+h}}
\]

Next-day 与 \(\sum_{h=1}^5\)；比较 ON vs ID 的 IC / H-L。

## 7. E5-3 Market-State Dependence

Tercile（或 CumRet5：负/中/正）：TurnoverProxy_5、PathVol（RV 代理）、Size、CumRet5。  
\(\Delta IC = IC^{High}-IC^{Low}\)；日序列 NW。标签用主累计窗 \(R_{1:5}\)（与训练 horizon 量级对齐的交易日累计）。

## 8. E5-4～7

| 步 | 内容 | 状态 |
|----|------|------|
| **E5-4** | Pressure（3 变量预注册） | Phase 2 **执行** |
| **E5-5** | \(S^{sp}\times Pressure\) FM | Phase 2 **执行** |
| **E5-6** | Continuation vs reversal | Phase 2 **执行** |
| **E5-7** | T+1 / 涨跌停 | **已跑（筛选级）** |

### Pressure 预注册（跑收益前锁定）

\[
Pressure=\frac{z(TurnoverProxy_5)+z(-MaxDD_5)+z(VolEndOverMean_5)}{3}
\]

`MaxDD5` 越负回撤越深 → 取 \(-MaxDD5\) 使指标越大表示路径压力越大。禁止用未来收益调号。

## 9. 预注册成功标准

| H | 判据 | Phase |
|---|------|-------|
| H1 Horizon heterogeneity | 不同 h 的 IC/H-L 有衰减或峰值结构 | 1 |
| H2 ON/ID heterogeneity | 至少一侧 ON vs ID 差异显著 | 1 |
| H3 State dependence | 至少一个预设 state 的 High−Low ΔIC NW 显著 | 1 |
| H4 Pressure interaction | \(\beta_3\) 显著且方向稳定 | 2 |
| H5 Economic direction | continuation / reversal 可区分 | 2 |

不要求全部成立；H3/H4 失败则停止 trading-pressure story。

## 10. 证据链（目标）

Recent-path structure → specific signal → 特定状态更强 → 特定 horizon/session 兑现 → 再提经济机制。

## 11. 避免

- 高换手 ≠ 直接「投资者行为」  
- next-day ≠ 直接 T+1  
- 勿把全部 E4 特征塞进 Pressure  
- 勿追求每个 subgroup 显著；找统一兑现机制  
