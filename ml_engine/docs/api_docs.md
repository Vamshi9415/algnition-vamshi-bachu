# API Documentation

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Endpoints

### `GET /health`
Returns service status.
```json
{"status": "ok", "service": "AIgnition Forecast Studio"}
```

---

### `POST /api/v1/forecast`
Upload ad-platform CSVs and receive a probabilistic forecast.

**Parameters:**
- `files` (multipart/form-data): One or more CSV files
- `horizon_days` (query, int, default 60): Forecast horizon

**Response:**
```json
{
  "status": "success",
  "horizon_days": 60,
  "summary": {
    "total_revenue_p50": 284750.00,
    "total_revenue_p10": 201300.00,
    "total_revenue_p90": 368200.00,
    "campaigns_forecasted": 12,
    "channels": ["google", "meta", "microsoft"]
  },
  "forecast": [
    {
      "date": "2026-08-01",
      "channel": "google",
      "campaign_name": "Search_US",
      "revenue_p10": 1200.50,
      "revenue_p50": 1850.00,
      "revenue_p90": 2400.75,
      "confidence": "High"
    }
  ],
  "evaluation": {
    "holdout_metrics": {"wmape": 8.3, "picp_80pct_interval": 0.82},
    "acceptance_criteria": {"production_ready": true}
  },
  "ai_insights": {
    "executive_summary": "...",
    "risk_analysis": "..."
  }
}
```

---

### `POST /api/v1/simulate`
Simulate revenue impact of budget changes.

**Body:**
```json
{
  "forecast_id": "abc123",
  "spend_changes": {
    "Search_US": 0.20,
    "Meta_Retargeting": -0.10
  }
}
```

**Response:**
```json
{
  "forecast_id": "abc123",
  "spend_changes": {"Search_US": 0.20},
  "delta_revenue": 18500.00,
  "delta_pct": 6.5
}
```

---

### `POST /api/v1/insights`
Generate AI insights for a forecast.

**Body:** Same as `/forecast` response.

**Response:**
```json
{
  "executive_summary": "...",
  "risk_analysis": "...",
  "budget_recommendations": "..."
}
```

---

### `GET /api/v1/report`
Returns an HTML forecast report.
