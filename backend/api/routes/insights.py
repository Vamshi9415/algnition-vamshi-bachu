"""AI Insights endpoint."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from ml_engine.llm.insights import InsightGenerator

router = APIRouter()


class InsightsRequest(BaseModel):
    forecast: list[dict[str, Any]]
    summary: dict[str, Any] = {}


@router.post("/insights")
def generate_insights(req: InsightsRequest):
    """Generate AI-powered executive summary and risk analysis from a forecast payload."""
    import pandas as pd
    forecast_df = pd.DataFrame(req.forecast)
    llm = InsightGenerator()
    exec_summary = llm.executive_summary(forecast_df)
    risk_text    = llm.risk_analysis(pd.DataFrame(), forecast_df)
    budget_rec   = llm.budget_recommendation(req.summary)
    return {
        "executive_summary": exec_summary,
        "risk_analysis": risk_text,
        "budget_recommendations": budget_rec,
    }

