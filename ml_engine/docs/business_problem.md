# Business Problem Statement

## Executive Summary

Digital marketing agencies manage hundreds of campaigns across Google, Meta, and Microsoft Ads simultaneously. Today, revenue forecasting is done manually in spreadsheets or via platform-native tools that are siloed, non-probabilistic, and unable to model cross-channel interactions.

**AIgnition Forecast Studio** replaces this fragmented workflow with a single AI-powered forecasting engine that produces statistically rigorous, probabilistic revenue forecasts with automated explanation and budget optimisation.

---

## Current Agency Workflow (Problem)

```
Account Manager exports CSVs from Google/Meta/Bing
    ↓
 Manually copies data into Excel
    ↓
 Applies a linear trend or seasonal adjustment by feel
    ↓
 Sends a single-point forecast to the client (e.g. "$45,000 next month")
    ↓
 Client asks: "How confident are you?"
    ↓
 Account Manager says: "Very confident" (with no statistical basis)
```

### Limitations of Existing Forecasting

| Limitation | Impact |
|-----------|--------|
| Siloed per-platform exports | Cannot model cross-channel effects |
| Single-point forecasts only | No uncertainty, no risk management |
| Manual Excel-based process | 4–6 hours per client per month |
| No holiday / seasonality modelling | Systematic forecast errors around peaks |
| No spend–revenue elasticity | Budget decisions are based on intuition |
| No statistical validation | Impossible to know if the model is trustworthy |
| No explainability | Cannot justify recommendations to clients |

---

## Target Users

| User | Need |
|------|------|
| **Account Managers** | Automate monthly forecast reports (save 4h/client) |
| **Media Planners** | Simulate budget reallocation before committing spend |
| **Agency Directors** | Portfolio-level revenue confidence intervals |
| **Marketing Clients** | Understand forecast risk and expected ROAS range |

---

## Business KPIs

| KPI | Target |
|-----|--------|
| Forecast accuracy (WMAPE) | < 15% on 30-day horizon |
| Time to forecast (per client) | < 2 minutes (vs 4–6 hours manually) |
| Prediction interval coverage | 80% interval captures actual ≥ 78% of the time |
| Budget simulation accuracy | Revenue delta error < 10% vs actual |
| Client retention lift | +20% via proactive budget recommendations |

## Technical KPIs

| KPI | Target |
|-----|--------|
| Pipeline runtime (90-day data) | < 60 seconds |
| Test coverage | > 80% |
| Ljung-Box residual autocorrelation | Not significant (p > 0.05) |
| PICP (80% interval) | 0.75 – 0.85 |
| Bootstrap WMAPE 95% CI upper bound | < 25% |
| CI: run.sh exit code on clean clone | 0 |
