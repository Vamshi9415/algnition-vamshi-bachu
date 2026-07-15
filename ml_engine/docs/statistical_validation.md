# Statistical Validation & Scientific Evaluation

> AIgnition Forecast Studio follows a rigorous, reproducible statistical evaluation methodology.
> Every forecast claim is backed by significance tests, calibration metrics, and residual diagnostics.

---

## Philosophy

A model should not be considered "better" just because it has a lower MAE on one test split.
We demonstrate that:

- Improvements are **statistically significant** (Wilcoxon / Friedman)
- Forecasts are **well-calibrated** (PICP ≈ target coverage)
- Residuals show **no autocorrelation** (Ljung-Box pass)
- Models **generalize across time** (Walk-Forward CV)
- Improvements are **practically meaningful** (Cohen’s d effect size)

---

## Module Map

| Module | Tests Implemented | Location |
|--------|------------------|----------|
| Descriptive Stats | Mean, Std, Skewness, Kurtosis, CV, IQR | `src/evaluation/descriptive_stats.py` |
| Stationarity | ADF, KPSS, STL Decomposition | `src/evaluation/stationarity.py` |
| Residual Diagnostics | Ljung-Box, Durbin-Watson, Shapiro-Wilk | `src/evaluation/residual_diagnostics.py` |
| Forecast Metrics | MAE, RMSE, WMAPE, SMAPE, PICP, MPIW, Pinball | `src/evaluation/forecast_metrics.py` |
| Significance Tests | Wilcoxon, Friedman, Bootstrap CI, Cohen’s d | `src/evaluation/significance_tests.py` |
| Backtesting | Walk-Forward (3 folds, 30-day horizon) | `src/evaluation/backtester.py` |
| Model Comparison | Ranked table + pairwise Wilcoxon vs best | `src/evaluation/model_comparison.py` |
| Explainability | SHAP TreeExplainer (top-15 features) | `src/evaluation/explainability.py` |
| Report Generator | Markdown reports per analysis | `src/evaluation/report_generator.py` |

---

## Acceptance Criteria

A model is considered **production-ready** only if ALL of the following hold:

| Criterion | Threshold | Test |
|-----------|-----------|------|
| WMAPE | < 20% | Point metrics on 30-day holdout |
| Residual autocorrelation | Not significant | Ljung-Box lag-10 p > 0.05 |
| Interval coverage | Within ±10% of target | PICP ≈ 0.80 |
| Calibration | Well-calibrated or Over-covering | Calibration status |
| Backtesting stability | All folds complete | Walk-forward CV |

---

## Reports Generated

All reports are written to `reports/` after each pipeline run:

```
reports/
  descriptive_statistics.md
  stationarity.md
  residual_analysis.md
  forecast_metrics.md
  model_comparison.md
  significance_tests.md
```

---

## Key Test Decisions

### Why Wilcoxon over t-test?
Marketing revenue residuals are rarely normally distributed (skewed, fat-tailed).
The Wilcoxon Signed-Rank test makes no distributional assumptions.

### Why PICP for calibration?
A 90% prediction interval should contain 90% of actual observations.
PICP directly measures this — unlike point metrics, it evaluates uncertainty quality.

### Why Bootstrap CI for WMAPE?
Single-split metrics are noisy. Bootstrap over 1,000 resamples gives a 95% CI,
making the improvement claim defensible and reproducible.

### Why STL Decomposition?
STL separates trend, seasonality, and remainder without assumptions about frequency.
Seasonality strength > 0.6 indicates that seasonal features (Fourier terms) are critical.
Trend strength > 0.6 indicates that lag and rolling features should capture it.

### Why SHAP over permutation importance?
SHAP values are consistent, locally accurate, and additive.
They explain each prediction individually, not just global feature rankings.
