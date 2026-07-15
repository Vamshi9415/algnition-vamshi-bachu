"""LLM insight generator: produces AI-written executive summaries and recommendations."""
import os
import json
from typing import Optional

from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMInsightGenerator:
    """
    Uses GPT-4o to generate:
    - Executive summary
    - Budget recommendations
    - Risk analysis
    Falls back to rule-based insights if OpenAI unavailable.
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 800):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if OPENAI_AVAILABLE else None

    def generate_executive_summary(self, forecast_summary: dict) -> str:
        prompt = self._build_summary_prompt(forecast_summary)
        return self._call_llm(prompt) or self._fallback_summary(forecast_summary)

    def generate_budget_recommendation(self, simulation_results: list[dict]) -> str:
        prompt = self._build_budget_prompt(simulation_results)
        return self._call_llm(prompt) or self._fallback_budget(simulation_results)

    def generate_risk_analysis(self, metrics: dict) -> str:
        prompt = self._build_risk_prompt(metrics)
        return self._call_llm(prompt) or self._fallback_risk(metrics)

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not OPENAI_AVAILABLE or not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _build_summary_prompt(self, data: dict) -> str:
        return (
            "You are a senior digital marketing analyst. "
            "Write a concise 3-paragraph executive summary based on this forecast data. "
            "Use plain language a CMO would understand. Highlight: expected revenue, confidence range, top campaigns, risks.\n\n"
            f"Data: {json.dumps(data, default=str)}"
        )

    def _build_budget_prompt(self, data: list) -> str:
        return (
            "You are an expert media planner. "
            "Based on these budget simulation results, give specific budget reallocation recommendations. "
            "Be precise (e.g., 'Increase Google Search by $500/day'). Explain expected revenue impact.\n\n"
            f"Simulation data: {json.dumps(data, default=str)}"
        )

    def _build_risk_prompt(self, data: dict) -> str:
        return (
            "You are a risk analyst for a digital marketing agency. "
            "Identify the top 3 risks from this campaign performance data. "
            "For each: state what it is, why it matters, and what to do.\n\n"
            f"Data: {json.dumps(data, default=str)}"
        )

    # --- Fallback rule-based insights ---
    def _fallback_summary(self, data: dict) -> str:
        p50 = data.get("total_forecast_p50", 0)
        p10 = data.get("total_forecast_p10", 0)
        p90 = data.get("total_forecast_p90", 0)
        return (
            f"Over the next {data.get('horizon_days', 60)} days, total revenue is forecast at "
            f"${p50:,.0f} (range: ${p10:,.0f}\u2013${p90:,.0f}). "
            f"Top channel by revenue: {data.get('top_channel', 'N/A')}. "
            f"Forecast confidence: {data.get('confidence', 'N/A')}."
        )

    def _fallback_budget(self, data: list) -> str:
        if not data:
            return "No simulation data available."
        best = max(data, key=lambda x: x.get("delta_revenue_estimated", 0))
        return best.get("explanation", "No recommendation available.")

    def _fallback_risk(self, data: dict) -> str:
        risks = []
        if data.get("roas_trend", 1) < 1:
            risks.append("Declining ROAS trend detected — review campaign targeting and creatives.")
        if data.get("zero_revenue_days_pct", 0) > 0.3:
            risks.append("High proportion of zero-revenue days — check conversion tracking.")
        if data.get("spend_concentration", 0) > 0.8:
            risks.append("Spend heavily concentrated in one channel — consider diversification.")
        return " | ".join(risks) if risks else "No critical risks identified."
