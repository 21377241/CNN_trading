# E2 Experiment Report: Matched-Information Benchmark

> Status: current standardized version. This report and
> `paper_E2C_temporal_numerical_benchmark_report.md` use the same 120D OOS
> common panel, prediction files, and Sharpe convention.

## 1. Question

With the same 20-day OHLCV history, does a chart-image CNN have a stable
single-model advantage? E2 holds the information set fixed and compares chart
images, flattened numerical inputs, and temporal numerical sequences.

## 2. Current Main Specification

| Item | Setting |
|---|---|
| Information set | 20-day, 120D matched OHLCV: adjusted Open, High, Low, Close, MA20, normalized volume |
| Split | 2000--2005 training; 2006--2007 chronological validation; 2008--2025 OOS |
| OOS panel | 922 weekly formation dates; 2,687,822 stock-date observations |
| Selection | Validation Rank IC; MLP-4 and 1D-CNN use five-seed ensembles |
| Sharpe | Weekly decile H--L: `mean / std * sqrt(52)` |

## 3. Current 120D Matched Benchmark

| Model | Input | AUC | Rank IC | EW H--L Sharpe | VW H--L Sharpe |
|---|---|---:|---:|---:|---:|
| CNN (E1b-ric) | chart image | 0.5190 | 0.0308 | 2.3883 | 1.1098 |
| Logit | 120D flat | 0.5110 | 0.0048 | 1.0954 | 0.2466 |
| HistGB-120 | 120D flat | 0.5227 | 0.0256 | 2.5670 | 1.3773 |
| MLP-4-120 | 120D flat | -- | 0.0410 | 2.8071 | 1.3249 |
| 1D-CNN-120 | 20 x 6 sequence | 0.5261 | 0.0484 | 3.2065 | 1.6365 |

The first four rows are the E2 matched numerical benchmark. The last row is
E2C on the same 120D information set and OOS panel. Therefore, the overlapping
rows in this report and the E2C report must match exactly, and now do.

## 4. Relation Between E2 and E2C

E2C is not a different market sample or Sharpe definition. It reshapes E2's
120D flattened input as a `20 x 6` sequence and applies local temporal
convolution. E2C additionally reports the 1D-CNN, CNN--1D spanning, pairwise
residuals, and ensemble performance.

The valid comparison is `CNN / HistGB-120 / MLP-4-120 / 1D-CNN-120`. Do not
compare E2C's 120D table with historical 180D or pre-standardization tables.

## 5. Separate 180D MLP-4 Supplement

| Model | Input | Rank IC | EW H--L Sharpe | VW H--L Sharpe |
|---|---|---:|---:|---:|
| MLP-4-180 | 20 x 9 flattened features | 0.0395 | 2.7571 | 1.3306 |

This model is in `WORK_SPACE/paper/E2_matched_is_ind_chrono_ric_mlp4/`. It
uses engineered indicators and is independently trained. It shares dates and
labels with the 120D design but is not the same information specification, so
it cannot be presented as the E2C main benchmark.

## 6. Historical Numbers Removed

The prior version of this report listed CNN 2.6548, HistGB 3.0683, MLP 3.6951,
and MLP Rank IC 0.0406. Those were historical summaries mixing earlier
predictions/specifications and pre-standardization portfolio statistics. They
are not valid current-paper main-table values and must not be used in the
manuscript, figures, or presentations.

## 7. Reproducible Artifacts

| Content | Location |
|---|---|
| Current E2 paper table | `overleaf/tables/benchmark.tex` |
| E2C report | `docs/results/paper_E2C_temporal_numerical_benchmark_report.md` |
| 1D-CNN results | `WORK_SPACE/paper/E2C_temporal_1dcnn_120/results/` |
| 120D MLP depth results | `WORK_SPACE/paper/E2F_mlp_depth/` |
| 180D MLP-4 supplement | `WORK_SPACE/paper/E2_matched_is_ind_chrono_ric_mlp4/e2_summary.csv` |
