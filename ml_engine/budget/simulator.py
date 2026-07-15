"""Budget simulation engine: estimates revenue impact of spend changes."""
import pandas as pd
import numpy as np
from loguru import logger


class BudgetSimulator:
    """
    Simulates revenue impact of budget changes using marginal ROAS elasticity.
    Uses a log-linear diminishing returns model: Revenue ~ a * Spend^b
    """

    def __init__(self):
        self.elasticity_map: dict = {}  # campaign -> spend elasticity

    def fit(self, df: pd.DataFrame):
        """Estimate spend elasticity per campaign from historical data."""
        for (channel, campaign), grp in df.groupby(["channel", "campaign_name"]):
            grp = grp[(grp["spend"] > 0) & (grp["revenue"] > 0)].copy()
            if len(grp) < 10:
                continue
            log_spend = np.log(grp["spend"])
            log_rev = np.log(grp["revenue"])
            # OLS: log(revenue) = a + b * log(spend)
            b = np.polyfit(log_spend, log_rev, 1)[0]
            b = float(np.clip(b, 0.1, 2.0))  # sensible bounds
            self.elasticity_map[(channel, campaign)] = b
        logger.info(f"Budget simulator fitted for {len(self.elasticity_map)} campaigns")

    def simulate(
        self,
        forecast: pd.DataFrame,
        spend_changes: dict,  # {campaign_name: pct_change}  e.g. {"Search_US": 0.20}
    ) -> pd.DataFrame:
        """
        Returns a modified forecast with adjusted revenue estimates.
        spend_changes: {campaign_name: fractional_change}  e.g. 0.20 = +20%
        """
        out = forecast.copy()
        for campaign, pct in spend_changes.items():
            mask = out["campaign_name"] == campaign
            if not mask.any():
                logger.warning(f"Campaign '{campaign}' not found in forecast")
                continue
            rows = out.loc[mask]
            channel = rows["channel"].iloc[0] if "channel" in rows.columns else None
            b = self.elasticity_map.get((channel, campaign), 0.8)  # default elasticity

            multiplier = (1 + pct) ** b
            for q in ["revenue_p10", "revenue_p50", "revenue_p90"]:
                if q in out.columns:
                    out.loc[mask, q] = (out.loc[mask, q] * multiplier).clip(lower=0)

            logger.info(f"Simulated {campaign}: spend +{pct*100:.0f}% → revenue x{multiplier:.3f}")
        return out

    def what_if_summary(self, baseline: pd.DataFrame, simulated: pd.DataFrame) -> dict:
        """Return a summary comparing baseline vs simulated totals."""
        return {
            "baseline_revenue_p50": round(float(baseline["revenue_p50"].sum()), 2),
            "simulated_revenue_p50": round(float(simulated["revenue_p50"].sum()), 2),
            "baseline_revenue_p10": round(float(baseline["revenue_p10"].sum()), 2),
            "simulated_revenue_p10": round(float(simulated["revenue_p10"].sum()), 2),
            "baseline_revenue_p90": round(float(baseline["revenue_p90"].sum()), 2),
            "simulated_revenue_p90": round(float(simulated["revenue_p90"].sum()), 2),
            "delta_revenue": round(
                float(simulated["revenue_p50"].sum()) - float(baseline["revenue_p50"].sum()), 2
            ),
            "delta_pct": round(
                (float(simulated["revenue_p50"].sum()) / (float(baseline["revenue_p50"].sum()) + 1e-9) - 1) * 100, 2
            ),
        }

