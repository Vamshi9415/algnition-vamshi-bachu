"""Uncertainty engine: computes and enriches probabilistic forecast intervals."""
import numpy as np
import pandas as pd
from loguru import logger


class UncertaintyEngine:
    """Enriches a forecast DataFrame with confidence metadata."""

    def enrich(self, forecast: pd.DataFrame) -> pd.DataFrame:
        df = forecast.copy()

        # Enforce monotonic quantile ordering (P10 <= P50 <= P90); ensemble averaging
        # of two independent models can otherwise produce crossed quantiles.
        q_cols = ["revenue_p10", "revenue_p50", "revenue_p90"]
        df[q_cols] = np.sort(df[q_cols].to_numpy(), axis=1)
        df[q_cols] = df[q_cols].clip(lower=0)

        # Interval width
        df["interval_width"] = (df["revenue_p90"] - df["revenue_p10"]).clip(lower=0)

        # Relative uncertainty (coefficient of variation of interval)
        df["relative_uncertainty"] = (
            df["interval_width"] / (df["revenue_p50"] + 1e-9)
        ).round(4).clip(0, 10)

        # Confidence label
        df["confidence_label"] = df["relative_uncertainty"].apply(self._label_confidence)

        logger.info("Uncertainty intervals enriched")
        return df

    def process(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Alias for enrich(), kept for callers expecting the older method name."""
        return self.enrich(forecast)

    @staticmethod
    def _label_confidence(rel_unc: float) -> str:
        if rel_unc < 0.15:
            return "High"
        elif rel_unc < 0.35:
            return "Medium"
        else:
            return "Low"

