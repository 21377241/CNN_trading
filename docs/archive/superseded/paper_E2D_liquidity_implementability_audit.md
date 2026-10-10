# E2D 实验报告：流动性、小市值暴露与可实施性审计

## 1. 实验目的

EW H-L Sharpe 明显高于 VW H-L Sharpe，提示模型收益可能集中在小市值、低容量或难交易股票。本实验不重新训练模型，而是复用已有 chronological OOS prediction，回答三个问题：

1. 去掉小市值股票后，预测关系是否仍然存在；
2. 去掉低成交额股票后，预测关系是否仍然存在；
3. 简单交易成本压力下，组合收益是否仍然成立。

## 2. 数据与口径

- 模型：CNN、MLP、1D-CNN-120；
- OOS：2008--2025；
- common panel：2,687,822 行，922 个观测日期，4,923 只股票；
- 预测频率：以周度 period-end 为主；
- 市值变量：标签文件中的 `MarketCap`；
- 流动性代理：`abs(Close) * abs(Vol)`；
- 投资组合：预测分数最高 10% 做 long，最低 10% 做 short；
- 年化：52 个周度周期。

这里的流动性变量是成交额代理，不是经过交易所口径确认的 ADV。结果应解释为流动性审计，而不是完整的市场冲击成本模型。

## 3. 一个重要的 Sharpe 口径说明

旧版组合代码使用：

```text
Sharpe_old = compounded annual return / annualized volatility
```

E2D 主表使用更标准的 Sharpe：

```text
Sharpe = mean weekly H-L return / weekly volatility * sqrt(52)
```

年化复合收益仍然单独报告。由于复合年化收益和算术平均收益不是同一个量，E2D 的 Sharpe 不应与旧报告中的 Sharpe 直接逐项比较。

## 4. Full、规模和流动性筛选

以下为 E2D 使用标准 Sharpe 口径得到的结果：

| Model | Universe | Rank IC | EW Sharpe | Top Sharpe | Bottom Sharpe | Top-Market Sharpe |
|---|---|---:|---:|---:|---:|---:|
| CNN | Full | 0.0293 | 2.37 | 0.64 | -0.08 | 1.48 |
| CNN | Largest 50% | 0.0259 | 1.87 | 0.46 | -0.22 | 1.27 |
| CNN | Largest 30% | 0.0230 | 1.48 | 0.44 | -0.17 | 1.13 |
| CNN | Large & Liquid | 0.0264 | 1.73 | 0.36 | -0.34 | 1.21 |
| MLP | Full | 0.0279 | 2.87 | 0.73 | -0.07 | 2.03 |
| MLP | Largest 50% | 0.0256 | 2.48 | 0.56 | -0.24 | 1.90 |
| MLP | Largest 30% | 0.0239 | 2.09 | 0.50 | -0.29 | 1.49 |
| MLP | Large & Liquid | 0.0279 | 2.15 | 0.38 | -0.36 | 1.48 |
| 1D-CNN | Full | 0.0484 | 3.21 | 0.85 | -0.48 | 2.10 |
| 1D-CNN | Largest 50% | 0.0420 | 2.67 | 0.70 | -0.52 | 2.01 |
| 1D-CNN | Largest 30% | 0.0397 | 2.26 | 0.74 | -0.47 | 1.98 |
| 1D-CNN | Large & Liquid | 0.0456 | 2.67 | 0.62 | -0.72 | 2.02 |

结论：1D-CNN 的 Full EW Sharpe 从 3.21 降到 Large & Liquid 的 2.67，但没有消失。Rank IC 也从 0.0484 降至 0.0456，说明主要预测关系并不只是小盘股暴露。

CNN 和 MLP 也呈现类似结果：限制大市值股票后收益下降，但仍保留明显正的预测关系。

## 5. 市值和流动性分组

| Model | Dimension | Bucket | Rank IC | EW Sharpe | Top-Market Sharpe |
|---|---|---|---:|---:|---:|
| 1D-CNN | MarketCap | SmallCap | 0.0523 | 3.39 | 1.78 |
| 1D-CNN | MarketCap | MidCap | 0.0482 | 3.32 | 2.10 |
| 1D-CNN | MarketCap | LargeCap | 0.0403 | 2.32 | 1.95 |
| 1D-CNN | Liquidity proxy | Illiquid | 0.0414 | 2.23 | 2.02 |
| 1D-CNN | Liquidity proxy | MidLiquid | 0.0472 | 3.18 | 2.30 |
| 1D-CNN | Liquidity proxy | Liquid | 0.0546 | 2.97 | 1.98 |

市值分组显示出较清晰的规模梯度：小市值股票的 Rank IC 和 Sharpe 高于大市值股票。但流动性分组并没有呈现“越不流动越强”的单调关系，Liquid 组的 Rank IC 反而最高。因此当前证据更支持“小市值暴露”，而不是简单的“低流动性异象”。

## 6. Long、short 和 long-only 分解

1D-CNN Full universe：

| Portfolio | Sharpe |
|---|---:|
| Top decile long | 0.85 |
| Bottom decile short leg return | -0.48 |
| Top minus Bottom | 3.21 |
| Top minus Market | 2.10 |

多空收益不是完全由 short leg 产生。Top-minus-Market 的 Sharpe 仍为 2.10，说明 long leg 有经济价值；但 short leg 仍然贡献了相当一部分 H-L 表现，而 A 股空头可实施性需要单独谨慎讨论。

## 7. 交易成本压力测试

组合换手定义为相邻周度目标权重变化的半绝对差之和。1D-CNN 的平均换手约为 0.737；CNN 和 MLP 分别约为 0.855 和 0.888。

| Model | Cost | Net annual return | Net Sharpe |
|---|---:|---:|---:|
| CNN | 0 bps | 24.8% | 2.37 |
| CNN | 25 bps | 11.7% | 1.18 |
| CNN | 50 bps | -0.1% | -0.01 |
| MLP | 0 bps | 28.3% | 2.87 |
| MLP | 25 bps | 14.4% | 1.55 |
| MLP | 50 bps | 1.9% | 0.22 |
| 1D-CNN | 0 bps | 52.3% | 3.21 |
| 1D-CNN | 25 bps | 38.5% | 2.48 |
| 1D-CNN | 50 bps | 25.9% | 1.75 |
| 1D-CNN | 100 bps | 4.0% | 0.30 |

按当前简化成本模型，1D-CNN 的毛收益 break-even 单边成本约为 110 bps；CNN 和 MLP 分别约为 50 bps 和 54 bps。这个结果说明 1D-CNN 的毛收益对成本更有韧性，但 100 bps 下 Sharpe 已明显下降，且该成本模型尚未包含冲击成本、涨跌停和容量限制。

## 8. 结论

E2D 不支持“高 Sharpe 完全由低流动性垃圾股造成”的强结论，但支持存在明显的小市值暴露：

- 大市值筛选会降低 Rank IC 和 EW Sharpe；
- 1D-CNN 在 Large & Liquid universe 中仍有 Rank IC 0.0456 和 EW Sharpe 2.67；
- 流动性 tercile 没有呈现低流动性单调优势；
- long-only Top-minus-Market 仍有正的经济表现；
- 1D-CNN 在简单成本压力测试下比 CNN 和 MLP 更稳健。

因此论文应避免把 Full-universe gross Sharpe 直接解释为可无限扩容的套利收益。更稳妥的表述是：

> Predictability is stronger among smaller stocks, but it remains economically meaningful after restricting the universe to larger and more liquid stocks. The remaining implementability claim is limited by the use of a turnover-proportional cost proxy and the absence of explicit price-impact, suspension, and limit-hit execution rules.

## 9. 文件与复现

代码：`experiments/E2D_liquidity/run_e2d_audit.py`

运行：

```powershell
python experiments/E2D_liquidity/run_e2d_audit.py
```

结果目录：`WORK_SPACE/paper/E2D_liquidity_audit`

主要输出：

- `universe_summary.csv`
- `strata_summary.csv`
- `cost_sensitivity.csv`
- `audit_panel.parquet`
- `audit_metadata.json`

本轮没有重新训练模型；流动性变量使用 `abs(Close) * abs(Vol)` 作为成交额代理，后续如有可靠的成交金额或 ADV 数据，应再做一次更严格的可实施性审计。
