# Known Limitations & Mitigations

## Model Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| LightGBM cannot extrapolate beyond training range | Medium | Ensemble with Prophet which handles trends |
| Prophet is slow for many campaigns (>100) | Medium | Fit in parallel; fall back to LightGBM-only |
| Budget elasticity assumes log-linear relationship | Medium | Documented; real elasticity may be S-shaped |
| Quantile crossing possible (P90 < P50) | Low | `clip(lower=0)` and sorted quantile post-processing |
| Cold-start: campaigns < 14 days of data | High | Falls back to channel-level aggregate model |

## Data Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| Currency not normalised across platforms | Medium | Flag in validation report; assume single currency |
| Platform attribution differences (click vs view) | High | Documented in canonical schema; not reconciled |
| Missing conversion tracking (some Meta campaigns) | Medium | ROAS imputed from channel median |

## Statistical Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| Walk-forward CV uses 3 folds (small) | Medium | Report fold std; flag if > 5% |
| Wilcoxon assumes continuous distribution | Low | Acceptable for revenue data |
| PICP computed on 30-day holdout only | Medium | Backtest across 3 folds for robustness |
