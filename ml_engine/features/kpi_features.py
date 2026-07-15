"""Marketing KPI feature generator."""
import numpy as np
import pandas as pd
from loguru import logger


class KPIFeatureGenerator:
    """Computes standard marketing KPIs from canonical data."""

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        eps = 1e-9  # avoid divide by zero

        df["ctr"] = (df["clicks"] / (df["impressions"] + eps)).round(6)
        df["cpc"] = (df["spend"] / (df["clicks"] + eps)).round(4)
        df["cpm"] = (df["spend"] / ((df["impressions"] + eps) / 1000)).round(4)
        df["cpa"] = (df["spend"] / (df["conversions"] + eps)).round(4)
        df["cvr"] = (df["conversions"] / (df["clicks"] + eps)).round(6)
        df["rpc"] = (df["revenue"] / (df["clicks"] + eps)).round(4)   # Revenue per click
        df["roas"] = (df["revenue"] / (df["spend"] + eps)).round(4)
        df["aov"] = (df["revenue"] / (df["conversions"] + eps)).round(4)  # Avg order value

        # Cap extreme values (e.g. ROAS > 500 is data error)
        df["roas"] = df["roas"].clip(0, 500)
        df["cpa"] = df["cpa"].clip(0, 10_000)

        logger.info("KPI features generated: ctr, cpc, cpm, cpa, cvr, rpc, roas, aov")
        return df

