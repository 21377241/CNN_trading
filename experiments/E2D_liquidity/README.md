# E2D Liquidity and Implementability Audit

This audit reuses existing chronological OOS predictions. It does not retrain models.

```powershell
python experiments/E2D_liquidity/run_e2d_audit.py
```

The liquidity proxy is `abs(Close) * abs(Vol)` from the same-day daily panel. Results are written to `WORK_SPACE/paper/E2D_liquidity_audit`.
