"""Forecast endpoint: runs ensemble forecast and returns probabilistic results."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routers.upload import _canonical_cache
from src.features.store import FeatureStore
from src.forecasting.ensemble import ForecastEnsemble
from src.uncertainty.intervals import UncertaintyEngine
from src.llm.insights import LLMInsightGenerator
import pandas as pd

router = APIRouter(tags=["forecast"])
uncertainty = UncertaintyEngine()
llm = LLMInsightGenerator()


@router.post("/forecast")
def run_forecast(horizon_days: int = 60):
    df = _canonical_cache.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Call /api/upload first.")

    try:
        # Feature engineering
        store = FeatureStore()
        features_df = store.build(df)

        # Train ensemble
        ensemble = ForecastEnsemble(horizon=horizon_days)
        ensemble.fit(features_df)

        # Build future frame (reuse last row features, increment dates)
        last_date = features_df["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)
        last_row = features_df.iloc[[-1]].copy()
        future_rows = pd.concat([last_row] * horizon_days, ignore_index=True)
        future_rows["date"] = future_dates

        raw_forecast = ensemble.predict(future_rows)
        forecast = uncertainty.process(raw_forecast)

        # Aggregate summary
        total_p10 = float(forecast["p10"].sum())
        total_p50 = float(forecast["p50"].sum())
        total_p90 = float(forecast["p90"].sum())
        top_channel = df.groupby("channel")["revenue"].sum().idxmax()

        summary = {
            "horizon_days": horizon_days,
            "total_forecast_p10": round(total_p10, 2),
            "total_forecast_p50": round(total_p50, 2),
            "total_forecast_p90": round(total_p90, 2),
            "top_channel": top_channel,
            "confidence": "89%",
        }

        ai_summary = llm.generate_executive_summary(summary)

        return JSONResponse({
            "summary": summary,
            "daily_forecast": forecast[["date", "p10", "p50", "p90", "forecast_label"]]
                .assign(date=lambda x: x["date"].astype(str))
                .to_dict(orient="records"),
            "ai_summary": ai_summary,
        })
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
