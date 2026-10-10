# MLP-4 Propagation Audit

Date: 2026-09-22

## Scope

The validation-selected MLP-4 architecture is `(128, 64, 32, 16)`. This audit verifies the result chain used by the standardized manuscript.

| Stage | Source | Status |
|---|---|---|
| 120D depth selection | `WORK_SPACE/paper/E2F_mlp_depth/e2f_depth_summary.csv` | Complete; ensemble validation Rank IC = 0.075925 |
| 180D Bag10 prediction | `E2_matched_is_ind_chrono_ric_mlp4` | Complete; Rank IC = 0.039480, EW Sharpe = 2.7571 |
| CNN--MLP-4 spanning | `E2B_cnn_mlp_chrono_ric_mlp4` | Complete; CNN conditional FM t = 5.75 |
| Shared/specific decomposition | `E3_shared_specific_chrono_ric_mlp4` | Complete; CNN-specific Sharpe = 1.0806, MLP-4-specific Sharpe = 1.8781 |
| Recent-path interventions | `E4_recent_path_chrono_ric_mlp4` | Complete |
| Mechanism tests E5-1 to E5-7 | `E5_mechanism_mlp4` | Complete |
| E2D liquidity and cost audit | `E2D_liquidity_audit_mlp4` | Complete; LargeLiquid EW Sharpe = 2.3264 |
| Standardized manuscript | `overleaf/main.pdf` | Updated and compiled |

## Explicit exclusions

- The original E2D liquidity report contains pre-MLP-4 MLP rows and is retained only as a historical record. Use `paper_E2D_liquidity_implementability_audit_mlp4.md` for the current MLP-4 audit.
- Earlier E2/E3 reports, bagging reports, and default two-layer YAML files are preserved as reproducible historical records, not as the current manuscript specification.
- The original E2C report contains pre-standardization Sharpe values; the current-paper values are those in `overleaf/tables/temporal_benchmark.tex` and `overleaf/tables/temporal_spanning.tex`.

## Reproducibility controls

- `experiments/E2_matched_tabular/config_ind_chrono_ric_mlp4.yaml` is the 180D MLP-4 Bag10 training configuration.
- E4 defaults to MLP-4. `MLP_HIDDEN_LAYERS` can override the architecture only for an explicitly requested historical reproduction.
- E3--E5 output roots can be versioned with `E3_OUT_ROOT`, `E4_OUT_ROOT`, and `E5_OUT_ROOT`.
