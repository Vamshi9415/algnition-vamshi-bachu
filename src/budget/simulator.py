"""Budget simulation engine: models revenue impact of spend changes."""
import pandas as pd
import numpy as np
from loguru import logger


class BudgetSimulator:
    """
    Simulates "what-if" budget scenarios.
    Uses historical ROAS to estimate incremental revenue from spend changes.
    """

    def __init__(self):
        self._channel_roas: dict = {}
        self._channel_spend: dict = {}

    def fit(self, canonical_df: pd.DataFrame) -> None:
        """Learn baseline ROAS and spend patterns per channel."""
        recent = canonical_df[canonical_df["date"] >= canonical_df["date"].max() - pd.Timedelta(days=30)]
        for channel, grp in recent.groupby("channel"):
            total_spend = grp["spend"].sum()
            total_revenue = grp["revenue"].sum()
            roas = total_revenue / total_spend if total_spend > 0 else 0
            self._channel_roas[channel] = round(roas, 4)
            self._channel_spend[channel] = round(total_spend, 2)
        logger.info(f"Budget simulator fitted. Channels: {list(self._channel_roas.keys())}")

    def simulate(
        self,
        channel: str,
        spend_change_pct: float,
        base_forecast_p50: float,
        base_spend: float = None,
    ) -> dict:
        """
        Args:
            channel: e.g. 'google'
            spend_change_pct: e.g. 20.0 for +20%
            base_forecast_p50: current P50 forecast revenue
            base_spend: override base spend (optional)
        Returns dict with simulated revenue and delta.
        """
        roas = self._channel_roas.get(channel, 2.0)
        base = base_spend or self._channel_spend.get(channel, 1000.0)

        delta_spend = base * (spend_change_pct / 100)
        delta_revenue = delta_spend * roas

        # Apply diminishing returns for large increases (>30%)
        if spend_change_pct > 30:
            diminishing_factor = 0.75
            delta_revenue *= diminishing_factor

        new_revenue_p50 = base_forecast_p50 + delta_revenue
        new_revenue_p10 = new_revenue_p50 * 0.88
        new_revenue_p90 = new_revenue_p50 * 1.12

        return {
            "channel": channel,
            "spend_change_pct": spend_change_pct,
            "delta_spend": round(delta_spend, 2),
            "delta_revenue_estimated": round(delta_revenue, 2),
            "new_revenue_p10": round(new_revenue_p10, 2),
            "new_revenue_p50": round(new_revenue_p50, 2),
            "new_revenue_p90": round(new_revenue_p90, 2),
            "baseline_roas": roas,
            "explanation": (
                f"Increasing {channel.title()} spend by {spend_change_pct:.0f}% (${delta_spend:,.0f}) "
                f"is estimated to generate an additional ${delta_revenue:,.0f} in revenue "
                f"based on a 30-day baseline ROAS of {roas:.2f}x."
            ),
        }

    def simulate_all_channels(self, spend_change_pct: float, base_forecast_p50: float) -> list[dict]:
        return [
            self.simulate(ch, spend_change_pct, base_forecast_p50)
            for ch in self._channel_roas
        ]
