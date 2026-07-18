"""Forecast endpoint: predicts per-campaign probabilistic revenue for a requested date
range using the pre-trained model loaded once at startup — no upload, no retraining."""

from datetime import timedelta

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import pandas as pd

from backend.api import state
from ml_engine.uncertainty.intervals import UncertaintyEngine

router = APIRouter(tags=["Forecast"])
uncertainty = UncertaintyEngine()

MAX_HORIZON_DAYS = 365


def _load_state_or_503() -> dict:
    try:
        return state.get_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _build_future_rows(features_df: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (channel, campaign_name) per date in future_dates, carrying the last
    known feature values forward and only advancing the date."""
    blocks = []
    for _, group in features_df.groupby(["channel", "campaign_name"], sort=False):
        last_row = group.iloc[[-1]].copy()
        block = pd.concat([last_row] * len(future_dates), ignore_index=True)
        block["date"] = future_dates
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@router.get("/model")
def model_info():
    app_state = _load_state_or_503()
    features_df = app_state["features_df"]
    meta = app_state["metadata"]
    earliest_forecastable = features_df["date"].max().normalize() + timedelta(days=1)
    return {
        "algorithm": meta.get("algorithm"),
        "trained": meta.get("trained"),
        "wmape": meta.get("wmape"),
        "campaigns": int(features_df["campaign_name"].nunique()),
        "channels": sorted(features_df["channel"].unique().tolist()),
        "data_through": str(features_df["date"].max().date()),
        "earliest_forecastable_date": str(earliest_forecastable.date()),
    }


@router.post("/forecast")
def run_forecast(start_date: str | None = None, end_date: str | None = None):
    app_state = _load_state_or_503()
    features_df = app_state["features_df"]
    ensemble = app_state["ensemble"]

    try:
        last_known_date = features_df["date"].max().normalize()
        earliest_forecastable = last_known_date + timedelta(days=1)

        start = pd.Timestamp(start_date) if start_date else earliest_forecastable
        end = pd.Timestamp(end_date) if end_date else start + timedelta(days=59)

        if start < earliest_forecastable:
            raise HTTPException(
                status_code=400,
                detail=f"start_date must be on or after {earliest_forecastable.date()} (the day after the model's latest known data).",
            )
        if end < start:
            raise HTTPException(status_code=400, detail="end_date must be on or after start_date.")
        if (end - earliest_forecastable).days + 1 > MAX_HORIZON_DAYS:
            raise HTTPException(status_code=400, detail=f"Requested range exceeds the {MAX_HORIZON_DAYS}-day forecasting limit.")

        # Prophet always forecasts starting the day after its own training cutoff, so the
        # full future window must start there too — otherwise LightGBM and Prophet dates
        # won't line up and the ensemble merge silently drops Prophet's contribution.
        full_future_dates = pd.date_range(start=earliest_forecastable, end=end)
        ensemble.horizon = len(full_future_dates)

        future_rows = _build_future_rows(features_df, full_future_dates)
        raw_forecast = ensemble.predict(future_rows)

        # Trim to the window the caller actually asked for.
        windowed = raw_forecast[(raw_forecast["date"] >= start) & (raw_forecast["date"] <= end)]
        forecast = uncertainty.process(windowed)

        summary = {
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "horizon_days": (end - start).days + 1,
            "total_revenue_p10": round(float(forecast["revenue_p10"].sum()), 2),
            "total_revenue_p50": round(float(forecast["revenue_p50"].sum()), 2),
            "total_revenue_p90": round(float(forecast["revenue_p90"].sum()), 2),
            "campaigns_forecasted": int(forecast["campaign_name"].nunique()),
            "channels": sorted(forecast["channel"].unique().tolist()),
        }

        forecast_records = (
            forecast[
                [
                    "date", "channel", "campaign_name",
                    "revenue_p10", "revenue_p50", "revenue_p90",
                    "relative_uncertainty", "confidence_label",
                ]
            ]
            .assign(date=lambda x: x["date"].astype(str))
            .to_dict(orient="records")
        )

        return JSONResponse(
            {
                "status": "success",
                "summary": summary,
                "forecast": forecast_records,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Forecast error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
