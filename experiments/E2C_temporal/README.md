# E2C — Temporal Numerical Benchmark

Question: does the CNN residual over MLP reflect image representation, or does it mainly reflect the lack of explicit temporal local structure in flattened tabular MLP?

Minimum run:

```powershell
python experiments/E2C_temporal/train_1dcnn.py --config experiments/E2C_temporal/config_1dcnn_120.yaml
python experiments/E2C_temporal/eval_e2c.py --config experiments/E2C_temporal/config_1dcnn_120.yaml
```

Main outputs:

- `WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv`
- `WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/key_six.json`
- `WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/e2c_1dcnn120_report.md`

Key six numbers:

1. Rank IC of 1D-CNN-120
2. EW H-L Sharpe of 1D-CNN-120
3. Corr(CNN, 1D-CNN-120)
4. Top10 overlap
5. CNN residual Sharpe after controlling for 1D-CNN-120
6. CNN + 1D-CNN-120 ensemble Sharpe

