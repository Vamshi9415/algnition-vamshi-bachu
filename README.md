# AIgnition Forecast Studio

AI-assisted probabilistic revenue forecasting for digital marketing agencies. The dashboard loads a pre-trained model at startup — pick a date range and get calibrated P10/P50/P90 revenue forecasts in an interactive Plotly chart, plus a Gemini-generated executive summary and risk analysis. No upload step in the live app; training happens offline (see [Quick Start](#quick-start)) against the Google/Meta/Microsoft Ads data checked into the repo.

[![CI](https://github.com/Vamshi9415/algnition-vamshi-bachu/actions/workflows/ci.yml/badge.svg)](https://github.com/Vamshi9415/algnition-vamshi-bachu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

---

## Contents

- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Supported Input Formats](#supported-input-formats)
- [Architecture](#architecture)
- [Statistical Validation](#statistical-validation)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [How It Works](#how-it-works)

---

## Quick Start

Training and serving are separate steps. `run.sh` trains the model and writes `ml_engine/pickle/model.pkl`; the backend then loads that file at startup and serves forecasts over the checked-in historical data (`google_ads_campaign_stats.csv`, `meta_ads_campaign_stats.csv`, `bing_campaign_stats.csv`) — no upload needed in the running app.

```bash
git clone https://github.com/Vamshi9415/algnition-vamshi-bachu
cd algnition-vamshi-bachu
pip install -r requirements.txt

cp .env.example .env   # then set GEMINI_API_KEY (required for AI insights)

# Train and save the model bundle to ml_engine/pickle/model.pkl
DATA_DIR=ml_engine/data/raw MODEL_PATH=ml_engine/pickle/model.pkl OUTPUT_PATH=ml_engine/output/predictions.csv bash run.sh
```

**Output:** `ml_engine/output/predictions.csv` with columns:
`date, channel, campaign_name, revenue_p10, revenue_p50, revenue_p90, confidence, pipeline_wmape, pipeline_picp, production_ready`, plus `ml_engine/pickle/model.pkl` — the bundle the backend loads on boot (see [Development Setup](#development-setup)).

To point `run.sh` at your own ad-platform CSVs instead of `ml_engine/data/raw/`, set `DATA_DIR` to that directory. Note: if you retrain on different data, also update `backend/api/state.py`'s `DATA_FILES` to match, since the live dashboard reads historical data straight from those files.

### Skip-Training Mode (re-run on same data)

```bash
SKIP_TRAIN=1 DATA_DIR=ml_engine/data/raw OUTPUT_PATH=ml_engine/output/predictions.csv bash run.sh
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required for the AI Insights step (executive summary, risk analysis). See `.env.example`; also accepts `GEMINI_API_KEYS` (comma-separated) or `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, … for key rotation. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for insight generation |
| `DATA_DIR` | `ml_engine/data/raw` | Directory with raw ad-platform CSVs |
| `MODEL_PATH` | `ml_engine/pickle/model.pkl` | Save/load path for the trained model bundle |
| `OUTPUT_PATH` | `ml_engine/output/predictions.csv` | Predictions output path |
| `HORIZON_DAYS` | `60` | Forecast horizon in days |
| `SKIP_TRAIN` | `0` | Set to `1` to skip retraining and reuse `MODEL_PATH` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Backend bind address |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed origin for the frontend dev server |

---

## Supported Input Formats

| Platform | Auto-detected by |
|----------|------------------|
| Google Ads | Filename (`google`) or column (`Cost`, `Conv. value`) |
| Meta Ads | Filename (`meta`, `facebook`) or column (`Amount spent (USD)`) |
| Microsoft/Bing Ads | Filename (`bing`, `microsoft`) or column (`TimePeriod`, `Spend`) |

Multiple CSVs from different platforms can be placed in `DATA_DIR` simultaneously. All platforms are normalized into a single 15-column canonical schema:
`date, channel, campaign_id, campaign_name, campaign_type, spend, clicks, impressions, conversions, revenue, roas, currency, device, country, status`.

---

## Architecture

```
raw CSVs
  └─ Platform Detection
      └─ Canonical Schema (15 columns)
          └─ Validation
              └─ Cleaning
                  └─ Feature Store (60+ features)
                      └─ Anomaly Detection
                          └─ Forecast Models (LightGBM + Prophet)
                              └─ Weighted Ensemble
                                  └─ Calibration (conformal intervals)
                                      └─ Statistical Tests
                                          └─ Decision Engine
                                              └─ Budget Simulator
                                                  └─ Gemini Insights
                                                      └─ Dashboard / Reports / CSV Output
```

---

## Statistical Validation

| Test | Purpose | Module |
|------|---------|--------|
| ADF + KPSS | Stationarity with combined verdict | `ml_engine/evaluation/stationarity.py` |
| STL Decomposition | Trend/seasonality strength | `ml_engine/evaluation/stationarity.py` |
| Ljung-Box | Residual autocorrelation | `ml_engine/evaluation/residual_diagnostics.py` |
| Shapiro-Wilk | Residual normality | `ml_engine/evaluation/residual_diagnostics.py` |
| PICP | Prediction interval calibration | `ml_engine/evaluation/forecast_metrics.py` |
| Pinball Loss | Quantile forecast quality | `ml_engine/evaluation/forecast_metrics.py` |
| Wilcoxon Signed-Rank | Two-model significance | `ml_engine/evaluation/significance_tests.py` |
| Bootstrap CI | 95% CI for WMAPE | `ml_engine/evaluation/significance_tests.py` |
| Cohen's d | Effect size | `ml_engine/evaluation/significance_tests.py` |
| SHAP | Feature explainability | `ml_engine/evaluation/explainability.py` |
| Walk-Forward CV | Temporal generalization | `ml_engine/evaluation/backtester.py` |
| Per-channel conformal intervals | Calibrated P10/P90 bounds per channel | `ml_engine/uncertainty/intervals.py` |

### Acceptance Criteria

A model is **production-ready** only if all of the following hold:
- WMAPE < 20% on 30-day holdout
- Ljung-Box residual autocorrelation: not significant (p > 0.05)
- PICP within ±10% of the 80% target
- Calibration status: Well-calibrated or Over-covering

---

## API Reference

FastAPI serves interactive docs at `/docs` (Swagger) and `/redoc` once the backend is running.

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Service liveness check |
| `GET`  | `/api/v1/model` | Info about the currently loaded model: algorithm, training date, WMAPE, campaigns, channels, data cutoff, and the earliest date it can forecast |
| `POST` | `/api/v1/forecast?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` | Predicts per-campaign P10/P50/P90 for the requested date window using the pre-trained model (loaded once at startup — no retraining per request). `start_date` must be after the model's data cutoff; both default to the next 60 days if omitted. |
| `POST` | `/api/v1/insights` | Generate Gemini executive summary / risk analysis for a given forecast (call after `/forecast`) |
| `POST` | `/api/v1/simulate` | Simulate a single budget reallocation scenario |
| `POST` | `/api/v1/simulate/all` | Simulate all budget scenarios for comparison |
| `GET`  | `/api/v1/report` | Download the generated HTML forecast report |
| `POST` | `/api/v1/upload` | Upload raw platform CSVs and validate/canonicalize them. Not used by the current dashboard (which forecasts from the pre-trained model instead) — kept for programmatic/future use. |

The frontend calls two endpoints in sequence: **forecast → insights**. `/api/v1/model` is fetched once on load to populate the date pickers.

---

## Project Structure

```
algnition-vamshi-bachu/
├── backend/
│   ├── api/
│   │   ├── main.py            FastAPI app, router registration, model warm-up on startup
│   │   ├── state.py           Loads the pre-trained model + historical CSVs once, cached in memory
│   │   ├── routes/            health, model, forecast, insights, simulate, report, upload
│   │   └── services/          upload/validation service layer
│   └── tests/
├── frontend/                  React 18 + Tailwind + Plotly dashboard (black/white theme)
│   └── src/
│       ├── App.jsx
│       └── components/        ForecastControls, ForecastDashboard, SummaryCards, RevenueChart, CampaignTable, AIInsights
├── ml_engine/
│   ├── ingestion/              platform detection + raw CSV loading
│   ├── canonical/               canonical schema mapping
│   ├── preprocessing/          validation + cleaning
│   ├── features/               feature store (lag, rolling, Fourier, calendar)
│   ├── anomaly/                 anomaly detection
│   ├── forecasting/            LightGBM quantile models, Prophet, ensemble
│   ├── uncertainty/            conformal prediction intervals
│   ├── evaluation/             stationarity, residual diagnostics, significance tests, SHAP, backtesting
│   ├── budget/                 budget reallocation simulator
│   ├── model_store/            model serialization + registry
│   ├── llm/                    Gemini insights client
│   ├── pipeline/               orchestrator tying every stage together
│   ├── cli/                    `predict.py` command-line entry point
│   ├── config/                 forecasting.yaml pipeline config
│   ├── docs/                   architecture, business problem, feature catalog, audit notes
│   └── tests/
├── run.sh                     Judge entry point
├── requirements.txt           Pinned dependencies
└── LICENSE                    MIT
```

---

## Development Setup

The backend and frontend run as two independent services, each in its own terminal.

**Terminal 1 — Backend (FastAPI, port 8000)**

```bash
pip install -r requirements.txt
cp .env.example .env   # set GEMINI_API_KEY, or /api/v1/insights will fail
uvicorn backend.app:app --reload --port 8000
```

On startup the backend loads `ml_engine/pickle/model.pkl` and the three root CSVs into memory (see `backend/api/state.py`). If `model.pkl` doesn't exist yet, run the training step in [Quick Start](#quick-start) first — `/api/v1/forecast` and `/api/v1/model` will 503 until it's there.

**Terminal 2 — Frontend (Vite dev server, port 5173)**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend proxies `/api/*` to `http://localhost:8000` (see `frontend/vite.config.js`), so **start the backend first** — it fetches model info from `/api/v1/model` on load. Verify the backend is up with `curl http://localhost:8000/health`.

**ML engine (standalone CLI run, no servers needed)**

```bash
python -m ml_engine.cli.predict --data-dir ml_engine/data/raw --model-path ml_engine/pickle/model.pkl --output-path ml_engine/output/predictions.csv
```

**Tests**

```bash
pytest backend/tests ml_engine/tests -v
```

---

## How It Works

### Offline: training (`run.sh` / `ml_engine.cli.predict`)

1. **Load** Google/Meta/Microsoft CSVs from `DATA_DIR`
2. **Platform auto-detected** from filename and column signatures
3. **Canonical schema** unifies all platforms into 15 columns
4. **Validation & cleaning** flag/fix missing values, outliers, and schema violations
5. **Feature engineering** adds 60+ calendar, KPI, lag, rolling, and Fourier features
6. **Anomaly detection** flags abnormal spend/revenue days before training
7. **LightGBM** trains separate P10/P50/P90 quantile regressors
8. **Prophet** fits per-campaign seasonal time series models
9. **Statistical validation** runs ADF/KPSS, Ljung-Box, PICP, SHAP, and walk-forward CV
10. **Acceptance criteria** gate ensures the forecast is production-ready
11. **Model bundle saved** to `ml_engine/pickle/model.pkl` (LightGBM + Prophet + budget simulator + metadata)
12. **Output CSV** written to `OUTPUT_PATH` with P10/P50/P90 per campaign per day

### Live: serving (backend + dashboard)

1. **Startup** loads `model.pkl` and the historical CSVs once (`backend/api/state.py`) — no training happens in the request path
2. **Dashboard** fetches `/api/v1/model` and shows model info (WMAPE, campaigns, data cutoff) plus a date-range picker defaulted to the day after the data cutoff
3. **User picks a start/end date** and clicks Generate Forecast
4. **`/api/v1/forecast`** builds one future feature row per campaign per requested day (carrying the last known feature values forward), runs the cached **LightGBM 60% + Prophet 40% ensemble**, applies **conformal calibration**, and returns per-campaign P10/P50/P90 for exactly that window
5. **Plotly chart** renders the aggregated P10/P50/P90 band; the campaign table shows the per-campaign breakdown
6. **`/api/v1/insights`** (Gemini) generates an executive summary, risk analysis, and budget recommendations from that forecast
