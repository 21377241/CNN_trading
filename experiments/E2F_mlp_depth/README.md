# E2F: MLP depth robustness

This experiment compares MLPs with two to five hidden layers while keeping the
120D chronological matched-information protocol fixed. It is a robustness
experiment, not a new model-selection track for the main benchmark.

- input: 120D OHLCV features
- train: 2000--2005; validation: 2006--2007; OOS: 2008--2025
- hidden layers: `(128,64)`, `(128,64,32)`, `(128,64,32,16)`, `(128,64,32,16,8)`
- seeds: 42--46
- early stopping: validation Rank IC
- fixed: alpha=1e-4, learning rate=1e-3, max_iter=200, patience=15

Run:

```powershell
python experiments/E2F_mlp_depth/run_depth.py
```
