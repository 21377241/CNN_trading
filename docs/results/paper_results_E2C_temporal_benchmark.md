# E2C — Temporal Numerical Benchmark

## Purpose

This experiment addresses the main remaining identification concern in the current draft: whether the CNN residual relative to flattened tabular learners reflects image representation, or whether it mainly reflects the absence of explicit temporal inductive bias in the MLP benchmark.

The first-stage benchmark trains a 1D-CNN on the same 120D OHLCV information set, reshaped as a 20-by-6 sequence. The label, common panel, chronological split, validation criterion, early stopping logic, and OOS evaluation period are kept aligned with the existing matched-information experiments.

## Implementation

- Config: `experiments/E2C_temporal/config_1dcnn_120.yaml`
- Training script: `experiments/E2C_temporal/train_1dcnn.py`
- Evaluation script: `experiments/E2C_temporal/eval_e2c.py`
- Output root: `WORK_SPACE/paper/E2C_temporal_1dcnn_120`
- OOS prediction: `WORK_SPACE/paper/E2C_temporal_1dcnn_120/preds/pred_1dcnn120_oos.csv`
- Main report: `WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/e2c_1dcnn120_report.md`

The 1D-CNN architecture is deliberately simple:

`20x6 -> Conv1D(6,32,k=3) -> ReLU -> Conv1D(32,64,k=3) -> ReLU -> GlobalAveragePooling -> Linear(64,1)`

Five seeds are trained and averaged.

## Key Results

| Metric | Value |
|---|---:|
| 1D-CNN-120 Rank IC | 0.0484 |
| 1D-CNN-120 EW H-L Sharpe | 3.9693 |
| Corr(CNN, 1D-CNN-120) | 0.3446 |
| Top-10 overlap | 0.2387 |
| CNN residual EW H-L Sharpe after controlling 1D-CNN | 0.8373 |
| CNN + 1D-CNN ensemble EW H-L Sharpe | 4.2296 |

## Interpretation

The result materially changes the paper's emphasis.

First, the temporal numerical model is the strongest standalone learner in the current matched-information set. Its Rank IC is 0.0484 and equal-weighted high-minus-low Sharpe is 3.97, exceeding the chart CNN, HistGB, and the 10-seed MLP benchmark.

Second, the temporal model does not fully span the chart CNN. The daily Pearson correlation between CNN and 1D-CNN scores is only 0.345, top-decile overlap is 23.9%, and the CNN coefficient remains positive in the joint Fama--MacBeth regression after controlling for the temporal model. The CNN residual score has a positive Rank IC of 0.0153 and EW Sharpe of 0.84.

Third, the residual is economically thinner than in the CNN-vs-MLP comparison. This implies that a substantial part of the earlier CNN-specific alpha relative to flattened MLPs was temporal inductive bias rather than image encoding per se.

Fourth, the CNN plus 1D-CNN ensemble reaches an EW Sharpe of 4.23, above either standalone model. This supports the revised claim that image and temporal numerical representations remain complementary, although the strongest benchmark now absorbs a large share of the original CNN-specific signal.

## Drafting Implication

The safest paper claim is no longer:

> Image representation creates a large distinct signal relative to tabular learning.

The safer claim is:

> Strong temporal numerical learners absorb a large share of the apparent CNN-specific alpha relative to flattened tabular models, but they do not fully span the chart CNN. Even under the same OHLCV history, temporal numerical and image representations retain moderate disagreement, positive residual predictability, and ensemble gains.

## Next Robustness, Not Required for Current Draft

- 1D-CNN-180: add engineered chart indicators as a stronger numerical temporal benchmark.
- TCN-120: replace ordinary local convolution with dilated residual temporal convolution.
- E3/E4 extension: decompose CNN, MLP, HistGB, and 1D-CNN jointly only if the paper decides to make E2C central enough to require mechanism reruns.
