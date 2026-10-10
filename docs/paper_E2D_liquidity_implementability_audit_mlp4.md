# E2D MLP-4 流动性与可实施性审计

本报告将 E2D 的 MLP 输入替换为 E2F 的 120D 五种子 MLP-4 ensemble OOS 预测；CNN、1D-CNN、样本、流动性代理和成本计算保持不变。

## 样本与口径

- OOS: 2008--2025，922 个周度形成日，2,687,822 个 stock-date。
- 流动性代理：`abs(Close) * abs(Vol)`。
- Sharpe：周度 H--L 收益的 `mean / std * sqrt(52)`。
- 成本：按单边换手率扣除 0、10、25、50、100 bps。

## MLP-4 结果

| Universe | Rank IC | EW H--L Sharpe | Top-minus-market Sharpe |
|---|---:|---:|---:|
| Full | 0.0410 | 2.8100 | 1.8931 |
| Largest 50\% | 0.0353 | 2.3835 | 1.5214 |
| Largest 30\% | 0.0321 | 1.8760 | 1.2705 |
| Liquid 50\% | 0.0453 | 2.6710 | 1.7317 |
| Liquid 30\% | 0.0475 | 2.3748 | 1.6255 |
| Large and liquid | 0.0381 | 2.3264 | 1.5287 |

| One-way cost | EW H--L Sharpe |
|---:|---:|
| 0 bps | 2.8100 |
| 10 bps | 2.3929 |
| 25 bps | 1.7671 |
| 50 bps | 0.7238 |
| 100 bps | -1.3633 |

Mean one-way turnover is 0.8898. Thus, MLP-4 predictability remains positive in the large-and-liquid universe, but the gross return should not be interpreted as frictionless net alpha.

## Reproducibility

```powershell
python experiments/E2D_liquidity/run_e2d_audit.py `
  --mlp-pred WORK_SPACE/paper/E2F_mlp_depth/preds/pred_mlp4_oos.csv `
  --mlp-label MLP-4 `
  --out WORK_SPACE/paper/E2D_liquidity_audit_mlp4
```
