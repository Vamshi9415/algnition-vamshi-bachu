"""LLM insight generator: produces AI explanations for forecasts."""
import os
from typing import Optional

import pandas as pd
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class InsightGenerator:
    """Generates AI-powered executive summaries and risk analyses."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if OPENAI_AVAILABLE else None

    def _call(self, system_prompt: str, user_content: str) -> str:
        if not self.client:
            return "[LLM unavailable — set OPENAI_API_KEY to enable AI insights]"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=800,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM error: {e}]"

    def executive_summary(self, forecast: pd.DataFrame, top_campaigns: int = 5) -> str:
        total_p50 = forecast["revenue_p50"].sum()
        total_p10 = forecast["revenue_p10"].sum()
        total_p90 = forecast["revenue_p90"].sum()
        horizon = forecast["date"].nunique()
        top = (
            forecast.groupby("campaign_name")["revenue_p50"]
            .sum()
            .sort_values(ascending=False)
            .head(top_campaigns)
            .to_dict()
        )
        channels = forecast.groupby("channel")["revenue_p50"].sum().to_dict()

        data_str = (
            f"Forecast horizon: {horizon} days\n"
            f"Expected revenue (P50): ${total_p50:,.0f}\n"
            f"Conservative estimate (P10): ${total_p10:,.0f}\n"
            f"Optimistic estimate (P90): ${total_p90:,.0f}\n"
            f"Top campaigns: {top}\n"
            f"Revenue by channel: {channels}"
        )
        system = (
            "You are a senior digital marketing analyst. "
            "Write a concise 3-paragraph executive summary that a marketing director can act on. "
            "Highlight expected revenue range, top campaigns, and key opportunities."
        )
        return self._call(system, data_str)

    def risk_analysis(self, df: pd.DataFrame, forecast: pd.DataFrame) -> str:
        low_roas = df[df["roas"] < 1.5][["campaign_name", "channel", "roas"]].head(5).to_string(index=False)
        high_unc = forecast[forecast["relative_uncertainty"] > 0.5][["campaign_name", "revenue_p50"]].head(5).to_string(index=False) if "relative_uncertainty" in forecast.columns else "N/A"
        system = (
            "You are a risk analyst for a digital marketing agency. "
            "Identify the top 3 risks from the data. For each: state the risk, why it matters, and what to do."
        )
        user = f"Low ROAS campaigns:\n{low_roas}\n\nHigh uncertainty forecasts:\n{high_unc}"
        return self._call(system, user)

    def budget_recommendation(self, simulator_summary: dict) -> str:
        system = (
            "You are an expert media planner. "
            "Based on the budget simulation results, write specific, numbered budget recommendations. "
            "Be precise (e.g., 'Increase Google Search by 20%'). Explain expected revenue impact."
        )
        user = str(simulator_summary)
        return self._call(system, user)
