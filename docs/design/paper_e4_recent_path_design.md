# E4 设计：Decomposing Recent-Path Predictability

> 接续 E3：两边都依赖近 5 日路径；本阶段问**近 5 日里什么不同结构**在驱动 CNN-spec / MLP-spec。  
> 产物：`WORK_SPACE/paper/E4_recent_path_chrono_ric/`  
> 脚本：`experiments/E4_recent_path/`

---

## 1. 核心问题

\[
\boxed{
\text{CNN 与 MLP-Bag10 都依赖 } W^{(5)}\text{，但各自利用近 5 日内的哪些不同结构？}
}

W^{(5)}_{i,t}=\{O,H,L,C,V\}_{t-4:t}
\]

| RQ | 问题 |
|----|------|
| RQ1 | 近 5 天中**哪一天**最重要？（lag） |
| RQ2 | 近 5 天中**什么结构**最重要？（PathShape / blocks） |

**主角**：\(S_C^{sp}\)、\(S_M^{sp}\)（两元残差，不用三元含 HGB）。  
**HistGB**：附录 robustness。

---

## 2. 固定协议

| 项 | 设定 |
|----|------|
| CNN | E1b-ric 5-model；OHLCV 变换 → `DrawOHLC` → 前向；**不重训** |
| MLP | chrono Rank-IC ES + bag k=10；删列后 **同超参重训** |
| Panel | OOS 2008–2025 common panel |
| Specific | \(S_C^{sp}=S_C-\widehat E[S_C\mid S_M]\)；\(S_M^{sp}=S_M-\widehat E[S_M\mid S_C]\)；partner = **未扰动**主线分数 |
| 指标 | Rank IC、EW/VW H-L Sharpe；\(\Delta IC=IC_{base}-IC_k\)，\(\Delta Sharpe\) 同理 |
| 推断 | 日序列 + NW lag=4 |

**配对原则**：同语义干预在 CNN（改 OHLCV）与 MLP（删列重训）两侧各做一次。

### 相对 E3

| E3 | E4 |
|----|-----|
| `mask_last5` = 末 5 日整窗均值平板 | 单日 flatten 到 \(C_{k-1}\) |
| 三元残差含 HGB | 两元 CNN↔MLP |
| 块级删列粗粒度 | lag + PathShape + matched blocks |

---

## 3. 预注册判定

| 假设 | 标准 |
|------|------|
| **H1** 位置异质性 | 五 lag 中 max \(\Delta Sharpe\) ≥ 2× min（至少一侧） |
| **H2** PathShape 增量 | 控 \(CumRet_5\) 后 \(\Delta R^2>0\) 且 NW 显著（至少一侧） |
| **H3** 成分剖面不同 | 六块 normalized drop 向量 \(Corr(v_C,v_M)<0.7\)；若 ≈1 则写 same structure |

---

## 4. 实验清单

### E4-1 Lag sensitivity

- CNN：`flatten_day(k)`，\(k\in\{15..19\}\)（lag \(t-4..t\)）：\(O=H=L=C=C_{k-1}\)，\(V=\)局部均值  
- MLP：`drop_day(k)` 删当日全部 9 列，bag10 重训  
- 图 1：Lag → \(\Delta Sharpe^{sp}\)

### E4-2 CumRet5 + PathShape \(\Delta R^2\)

特征：`CumRet_5`、`LateEarly`、`PathVol`、`UpCount_5`、`MaxDD_5`、`Reversal_5`、`Monotonicity`  
\[
\Delta R^2 = R^2(CumRet_5+Path)-R^2(CumRet_5)
\]
图 3：CNN vs MLP \(\Delta R^2\)。

### E4-3 Matched blocks

| Block | CNN | MLP |
|-------|-----|-----|
| Volume | 近5日 V→更早窗均 | drop recent5 Vn |
| Range/Wick | 近5日 flatten wick | drop recent5 H,L |
| Direction | 近5日中和 C−O 符号保 range | drop recent5 uP,uV |
| Return | 近5日 cumret 归零保日内形状 | drop 近5日可直接表达 return 的列（C 相对） |
| Gap | \(O_t=C_{t-1}\)，按比例调 H/L/C | drop overnight 相关（O 相对前日 C 的表达） |
| Intraday | 压缩日内 body 保 H/L | drop 近5日 body 相关列 |

每块附 sanity：CumRet5 / PathVol / wick 等是否按预期变。

### E4-4 Matched-Return path

按 \(CumRet_5\) 十分位内再按 PathShape（主用 `LateEarly`）双重排序，比较 \(S^{sp}\) spread。

### E4-5 Disagreement × Recent-5

80/20 \(C^+M^-\) vs \(C^-M^+\)，只比 Recent-5 特征；交叉验证 E4-2/3。

### 附录 HistGB

E4-1 / E4-3 tabular 侧同步跑 HistGB 冻结超参（不进主叙事）。

---

## 5. 核心三图

1. Lag sensitivity（ΔSharpe）  
2. Component sensitivity（六块 ΔSharpe）  
3. Momentum-controlled PathShape \(\Delta R^2\)

---

## 6. 叙事分支

- 若 lag/component 几乎同 → *different functions, same short-horizon structure*  
- 若明显不同 → *representation-specific transformation of recent price paths*
