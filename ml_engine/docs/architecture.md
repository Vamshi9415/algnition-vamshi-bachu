# System Architecture

## Data Flow

```
┌────────────────────┐
│  Raw CSVs (data/raw/)  │  ← Google Ads, Meta Ads, Microsoft Ads
└───────────┬────────┘
                 │
         ┌─────┴─────┐
         │ Ingestion   │  ← Auto-detect platform (filename + column signatures)
         └─────┬─────┘
                 │
         ┌─────┴─────┐
         │  Canonical   │  ← 15-column unified schema
         │  Schema       │
         └─────┬─────┘
                 │
         ┌─────┴─────┐
         │ Validation  │  ← 12 checks → validation_report.json + .html
         └─────┬─────┘
                 │
         ┌─────┴─────┐
         │  Cleaning    │  ← 8-step deterministic pipeline
         └─────┬─────┘
                 │
         ┌─────┴─────┐
         │  Features    │  ← Calendar + KPI + Lag + Rolling + Fourier (60+ cols)
         └─────┬─────┘
                 │
    ┌────┴──────────┐
    │  Forecasting Ensemble  │
    │  LightGBM (P10/50/90)  │
    │  Prophet (per campaign)│
    │  Weighted Ensemble     │
    └────┬───────────┘
                 │
    ┌────┴──────────┐
    │ Statistical Validation │
    │ ADF+KPSS, STL, LB      │
    │ Wilcoxon, PICP, SHAP   │
    │ Walk-Forward CV        │
    └────┬───────────┘
                 │
    ┌────┴──────────┐
    │  Output Layer           │
    │  predictions.csv        │
    │  reports/*.md           │
    │  AI insights (Gemini)   │
    └──────────────────┘
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend API | FastAPI 0.110, Python 3.11 |
| Forecasting | LightGBM 4.3, Prophet 1.1.5 |
| Statistics | statsmodels 0.14, scipy 1.13 |
| Explainability | SHAP 0.45 |
| Feature Store | pandas 2.2, numpy 1.26 |
| LLM | Google Gemini 2.5 Flash |
| Frontend | React 18, Recharts, Tailwind CSS |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |

## Module Dependency Graph

```
cli/predict.py
  └─ pipeline/orchestrator.py
        ├─ ingestion/ (loader, detector)
        ├─ canonical/schema.py
        ├─ validation/validator.py
        ├─ preprocessing/cleaner.py
        ├─ features/feature_store.py
        │     ├─ calendar_features.py
        │     ├─ kpi_features.py
        │     └─ lag_features.py
        ├─ forecasting/
        │     ├─ lgbm_model.py
        │     ├─ prophet_model.py
        │     └─ ensemble.py
        ├─ uncertainty/intervals.py
        ├─ budget/simulator.py
        ├─ evaluation/
        │     ├─ descriptive_stats.py
        │     ├─ stationarity.py
        │     ├─ residual_diagnostics.py
        │     ├─ forecast_metrics.py
        │     ├─ significance_tests.py
        │     ├─ backtester.py
        │     ├─ model_comparison.py
        │     ├─ explainability.py
        │     └─ report_generator.py
        ├─ model_store/serializer.py
        └─ llm/insights.py
```
