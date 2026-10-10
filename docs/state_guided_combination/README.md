# State-Guided Cross-Sectional Forecast Combination

## Research Question

Can observable, stock-specific price-path states determine when a chart-CNN or a
flattened MLP has a cross-sectional forecasting advantage, and can that information
improve an out-of-sample ranking-based portfolio combination?

The object is cross-sectional return prediction, not point-price forecasting:

`score(i,t) -> cross-sectional rank at t -> forward 5-trading-day return`.

## Claim Boundary

Do not claim that the chart and the MLP use different information. Both use the same
20-day OHLCV history. The proposed claim is that their representations create
systematic, state-dependent comparative forecasting advantages.

## Required Temporal Separation

The existing E3--E5 results discovered candidate states. They must be treated as
development evidence, not as a final test of a state-guided combination.

1. Split the current OOS interval into a development period and a final untouched holdout.
2. Use only the development period to choose state variables, transformations, gating
   functional form, and any hyperparameters.
3. Freeze the specification.
4. Report all combination performance only once on the final holdout.

If possible, use 2008--2016 for development and 2017--2025 for the final holdout;
the exact boundary must be fixed before inspecting combination results.

## Candidate State Set

Use only observable functions of information available at the rebalance date. Begin
with a short, pre-specified set motivated by the archived E4/E5 evidence:

- recent cumulative return and sign;
- local path-shape or reversal statistic;
- price-volume pressure;
- intraday range/wick structure.

Standardize each state cross-sectionally at each rebalance date. Do not use realized
future returns, model residual returns, or a state selected after inspecting the final
holdout.

## Models and Benchmarks

The focal comparison is CNN versus MLP-4. Scores must be cross-sectionally transformed
to percentile ranks or z-scores before combination.

Report, in this order:

1. CNN and MLP-4 standalone scores.
2. Equal-weight combination: `0.5 z(CNN) + 0.5 z(MLP)`.
3. Global constant weight chosen on the development period.
4. A simple performance-based or stacking benchmark fitted only on the development
   period.
5. State-guided combination with the same information and tuning budget.

The 1D-CNN belongs in robustness. It tests whether the CNN--MLP contrast is merely a
temporal-inductive-bias effect; it should not silently disappear from the analysis.

## Preferred Gating Specification

Start with an interpretable constrained gate rather than a flexible black-box meta-model:

`w(i,t) = logistic(a0 + a' z(state(i,t)))`

`combined_score(i,t) = w(i,t) z(CNN(i,t)) + [1-w(i,t)] z(MLP(i,t))`.

Estimate `a` on the development period only, regularize it, and cap weights away from
zero and one. Compare it with a small number of pre-specified rule-based gates. A more
flexible meta-learner is a robustness exercise, not the headline model.

## Evidence Required

For every standalone and combination score, report:

- date-level Rank IC and Newey--West t-statistic;
- non-overlapping 5-day, equal- and value-weighted H-L Sharpe ratios;
- annualized arithmetic H-L return, turnover, and a transaction-cost grid;
- yearly/subperiod results;
- score correlation, overlap, and incremental Fama--MacBeth tests.

The central test is incremental performance over both equal weighting and an
equally-tuned generic combination benchmark. Be explicit that a better combination is
not novel by itself; the contribution is whether ex-ante economic states provide useful
stock-specific weights.
