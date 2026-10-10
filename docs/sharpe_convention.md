# Sharpe Ratio Convention

All portfolio Sharpe ratios in the paper should use one convention:

```text
period mean return = mean(r_t)
period volatility = std(r_t, ddof=1)
annualized return = (1 + period mean return)^ann_factor - 1
annualized volatility = period volatility * sqrt(ann_factor)
Sharpe = period mean return / period volatility * sqrt(ann_factor)
```

For the weekly OOS panel, `ann_factor = 52`.

The compounded annualized return is reported separately. It must not be used as the numerator of the Sharpe ratio. This convention is implemented in:

- `trend_submit/trend_code_submit/Portfolio/portfolio.py`
- `experiments/E2C_temporal/eval_e2c.py`
- `experiments/E2D_liquidity/run_e2d_audit.py`

Existing CSV summaries generated before this change should be regenerated before being used in the manuscript.
