# E6A: State-Dependent Comparative Advantage

Runs the frozen-score test of whether raw CNN and MLP-4 Rank IC differs
systematically across the `CumRet5 x Pressure` state matrix. It does not train models,
build a portfolio, select weights, or estimate a gating network.

```powershell
python experiments/E6A_state_advantage/run_e6a_state_map.py
```

Inputs are the existing MLP-4 baseline-score panel and pressure-state panel. The output
uses 2008--2015 as discovery and 2016--2025 as a fixed validation period.
