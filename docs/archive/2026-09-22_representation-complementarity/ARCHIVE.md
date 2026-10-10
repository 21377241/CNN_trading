# Archived Research Line

**Archive date:** 2026-09-22  
**Status:** frozen as a completed empirical foundation; do not overwrite.

## Question Addressed

How do chart-CNN, flattened numerical learners, and temporal numerical learners
transform the same 20-day OHLCV history into partially overlapping cross-sectional
return-prediction signals in China A-shares?

## Contents

- `overleaf/`: the compilable manuscript source, tables, figures, bibliography, and
  the corresponding `main.pdf`.
- `reports/`: the standardized experimental reports used by the manuscript, including
  the Jiang-style transfer benchmark, matched numerical comparison, 1D-CNN temporal
  benchmark, liquidity audit, MLP-depth robustness, and temporal E3--E5 analyses.

## Frozen Main Claim

Different nonlinear representations of the same price history generate partially
spanning cross-sectional rankings. Explicit temporal structure materially attenuates,
but does not eliminate, the chart-CNN's distinct predictive content.

## Boundary

This archive is a historical snapshot. Ongoing work should retain it as evidence for
the new state-guided forecast-combination project, rather than revise it in place.
All new code, reports, manuscript drafts, and results should be created under a
separate combination-focused workstream.
