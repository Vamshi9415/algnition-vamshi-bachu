# AIgnition 2026 — AI-Assisted Probabilistic Revenue Forecasting Platform

> *A production-grade, end-to-end marketing intelligence system for digital ad agencies.*

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Is

This platform answers questions like:

> *"If I increase Google Ads budget by 20%, how much revenue should I expect over the next 60 days?"*

Instead of `CSV → XGBoost → Number`, we built:

```
CSV Upload → Validation → Canonical Schema → Feature Engineering
          → Ensemble Forecasting → Probabilistic Intervals
          → Budget Simulator → LLM Insights → Interactive Dashboard
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Vamshi9415/algnition-vamshi-bachu.git
cd algnition-vamshi-bachu

# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload

# Frontend
cd frontend
npm install && npm run dev
```

---

## Architecture

```
project/
├── config/          # YAML configs (schema, channels, holidays, forecasting, LLM)
├── data/            # raw/ → processed/ → canonical/ → features/
├── src/
│   ├── ingestion/   # Source detection + CSV loading
│   ├── validation/  # Schema contracts + ValidationReport
│   ├── preprocessing/ # Cleaning pipeline
│   ├── canonical/   # Unified schema builder
│   ├── features/    # Feature store (calendar, KPIs, lags, rolling)
│   ├── forecasting/ # Ensemble: LightGBM + Prophet + AutoGluon
│   ├── uncertainty/ # Quantile / conformal prediction intervals
│   ├── budget/      # Budget simulator engine
│   ├── llm/         # AI insight generator
│   ├── api/         # FastAPI REST backend
│   └── visualization/ # Chart builders
├── frontend/        # React dashboard
├── tests/           # pytest unit + integration tests
├── reports/         # Auto-generated HTML/JSON validation reports
└── notebooks/       # EDA notebooks
```

---

## Supported Channels

| Channel | Date Column | Revenue Column | Spend Column |
|---------|-------------|----------------|--------------|
| Google Ads | Date | Conversion Value | Cost |
| Meta Ads | Reporting Starts | Purchase Value | Amount Spent |
| Microsoft (Bing) | TimePeriod | Revenue | Spend |

---

## Key Features

- **Automatic source detection** — upload any CSV, system identifies the channel
- **Canonical schema** — all platforms mapped to one unified format
- **Probabilistic forecasts** — always shows `[P10, P50, P90]` intervals, never a single number
- **Budget simulator** — "What if I increase Google spend by 20%?"
- **LLM explanations** — AI-generated executive summaries for every forecast
- **Interactive dashboard** — React UI with Recharts visualizations
- **Export reports** — PDF/HTML downloadable reports

---

## Tests

```bash
pytest tests/ -v --tb=short
```

---

## License

MIT — Built for AIgnition 2026 Hackathon
