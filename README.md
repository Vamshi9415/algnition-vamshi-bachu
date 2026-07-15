# ⚡ AIgnition Forecast Studio

> AI-Assisted Probabilistic Revenue Forecasting for Digital Marketing Agencies

[![CI](https://github.com/Vamshi9415/algnition-vamshi-bachu/actions/workflows/ci.yml/badge.svg)](https://github.com/Vamshi9415/algnition-vamshi-bachu/actions/workflows/ci.yml)

---

## 🚀 Judge Quick Start (Single Command)

```bash
git clone https://github.com/Vamshi9415/algnition-vamshi-bachu
cd algnition-vamshi-bachu
pip install -r requirements.txt

# Place your ad-platform CSVs in data/raw/
cp /path/to/your/*.csv data/raw/

# Run the full pipeline
DATA_DIR=data/raw MODEL_PATH=pickle/model.pkl OUTPUT_PATH=output/predictions.csv bash run.sh
```

**Output:** `output/predictions.csv` with columns:
`date, channel, campaign_name, revenue_p10, revenue_p50, revenue_p90, confidence, pipeline_wmape, pipeline_picp, production_ready`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/raw` | Directory with raw ad-platform CSVs |
| `MODEL_PATH` | `pickle/model.pkl` | Save/load path for trained model bundle |
| `OUTPUT_PATH` | `output/predictions.csv` | Predictions output path |
| `HORIZON_DAYS` | `60` | Forecast horizon in days |
| `SKIP_TRAIN` | `0` | Set to `1` to skip retraining |

### Skip-Training Mode (re-run on same data)

```bash
SKIP_TRAIN=1 DATA_DIR=data/raw OUTPUT_PATH=output/predictions.csv bash run.sh
```

---

## 📁 Supported Input Formats

| Platform | Auto-detected by |
|----------|------------------|
| Google Ads | Filename (`google`) or column (`Cost`, `Conv. value`) |
| Meta Ads | Filename (`meta`, `facebook`) or column (`Amount spent (USD)`) |
| Microsoft/Bing Ads | Filename (`bing`, `microsoft`) or column (`TimePeriod`, `Spend`) |

Multiple CSVs from different platforms can be placed in `DATA_DIR` simultaneously.

---

## 🏗️ Architecture

```
raw CSVs
  └─ Ingestion (auto-detect platform)
      └─ Canonical Schema (13 unified columns)
          └─ Validation (nulls, negatives, duplicates)
              └─ Cleaning (8-step deterministic)
                  └─ Feature Engineering
                  │     Calendar (20+ cols) + KPIs + Lags/Rolling + Fourier
                  └─ Forecasting Ensemble
                  │     LightGBM P10/P50/P90 + Prophet + Weighted Ensemble
                  └─ Statistical Validation
                  │     ADF+KPSS + STL + Ljung-Box + Wilcoxon + PICP + SHAP
                  └─ Output
                        predictions.csv + reports/ + AI insights
```

---

## 🧪 Statistical Validation (Hackathon Differentiator)

| Test | Purpose | Module |
|------|---------|--------|
| ADF + KPSS | Stationarity with combined verdict | `stationarity.py` |
| STL Decomposition | Trend/seasonality strength | `stationarity.py` |
| Ljung-Box | Residual autocorrelation | `residual_diagnostics.py` |
| Shapiro-Wilk | Residual normality | `residual_diagnostics.py` |
| PICP | Prediction interval calibration | `forecast_metrics.py` |
| Pinball Loss | Quantile forecast quality | `forecast_metrics.py` |
| Wilcoxon Signed-Rank | Two-model significance | `significance_tests.py` |
| Bootstrap CI | 95% CI for WMAPE | `significance_tests.py` |
| Cohen’s d | Effect size | `significance_tests.py` |
| SHAP | Feature explainability | `explainability.py` |
| Walk-Forward CV | Temporal generalization | `backtester.py` |

### Acceptance Criteria

A model is **production-ready** only if ALL hold:
- WMAPE < 20% on 30-day holdout
- Ljung-Box residual autocorrelation: not significant (p > 0.05)
- PICP within ±10% of 80% target
- Calibration status: Well-calibrated or Over-covering

---

## 📂 Project Structure

```
algnition-vamshi-bachu/
├── run.sh                    ← Judge entry point
├── requirements.txt          ← Pinned dependencies
├── pickle/
│   └── model.pkl             ← Committed stub model (retrain via run.sh)
├── data/raw/                 ← Place CSVs here
├── output/                   ← predictions.csv written here
├── reports/                  ← Markdown stat reports (generated at runtime)
├── config/                   ← YAML configuration files
├── src/
│   ├── cli/predict.py        ← CLI called by run.sh
│   ├── pipeline/             ← End-to-end orchestrator
│   ├── ingestion/            ← Platform detection + CSV loading
│   ├── canonical/            ← Unified 13-column schema
│   ├── validation/           ← Data quality checks
│   ├── preprocessing/        ← 8-step deterministic cleaner
│   ├── features/             ← Calendar + KPI + Lag + Fourier
│   ├── forecasting/          ← LightGBM + Prophet + Ensemble
│   ├── uncertainty/          ← P10/P50/P90 interval enrichment
│   ├── budget/               ← Elasticity-based budget simulator
│   ├── evaluation/           ← Stats tests + metrics + SHAP
│   ├── model_store/          ← Pickle serializer
│   ├── llm/                  ← GPT-4o insights
│   └── api/                  ← FastAPI backend
├── frontend/               ← React 18 + Recharts dashboard
├── tests/                  ← pytest (unit + integration)
└── docs/                   ← Statistical validation methodology
```

---

## 🔧 Development Setup

```bash
# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload

# Frontend
cd frontend && npm install && npm run dev

# Tests
pytest tests/ -v
```

---

## 🧭 How It Works

1. **Upload** Google/Meta/Microsoft CSVs via the web UI or `DATA_DIR`
2. **Platform auto-detected** from filename and column signatures
3. **Canonical schema** unifies all platforms into 13 columns
4. **Feature engineering** adds 60+ calendar, KPI, lag, rolling, and Fourier features
5. **LightGBM** trains separate P10/P50/P90 quantile regressors
6. **Prophet** fits per-campaign seasonal time series models
7. **Ensemble** combines both (LightGBM 60% + Prophet 40%)
8. **Statistical validation** runs ADF/KPSS, Ljung-Box, PICP, SHAP, and walk-forward CV
9. **Acceptance criteria** gate ensures forecast is production-ready
10. **AI insights** (GPT-4o) generate executive summary and risk analysis
11. **Output CSV** written to `OUTPUT_PATH` with P10/P50/P90 per campaign per day
