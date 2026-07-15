"""Marketing KPI feature generator."""
import numpy as np
import pandas as pd


class MarketingKPIGenerator:
    """Computes CTR, CPC, CPM, CPA, CVR, AOV, and derived ROAS features."""

    EPS = 1e-9  # avoid division by zero

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        spend = df["spend"]
        revenue = df["revenue"]
        clicks = df.get("clicks", pd.Series(0, index=df.index))
        impressions = df.get("impressions", pd.Series(0, index=df.index))
        conversions = df.get("conversions", pd.Series(0, index=df.index))

        df["ctr"] = (clicks / (impressions + self.EPS)).round(6)
        df["cpc"] = (spend / (clicks + self.EPS)).round(4)
        df["cpm"] = (spend / ((impressions + self.EPS) / 1000)).round(4)
        df["cpa"] = (spend / (conversions + self.EPS)).round(4)
        df["cvr"] = (conversions / (clicks + self.EPS)).round(6)
        df["revenue_per_click"] = (revenue / (clicks + self.EPS)).round(4)
        df["revenue_per_impression"] = (revenue / (impressions + self.EPS)).round(6)
        df["aov"] = (revenue / (conversions + self.EPS)).round(4)  # Avg Order Value
        df["roas"] = (revenue / (spend + self.EPS)).round(4)

        # Efficiency score (composite)
        df["efficiency_score"] = (
            df["roas"] * df["cvr"] * 100
        ).round(4)

        return df

