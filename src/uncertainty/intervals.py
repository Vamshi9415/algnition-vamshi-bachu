"""Uncertainty engine: computes and enriches probabilistic forecast intervals."""
import pandas as pd
from loguru import logger


class UncertaintyEngine:
    """Enriches a forecast DataFrame with confidence metadata."""

    def enrich(self, forecast: pd.DataFrame) -> pd.DataFrame:
        df = forecast.copy()

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

    @staticmethod
    def _label_confidence(rel_unc: float) -> str:
        if rel_unc < 0.15:
            return "High"
        elif rel_unc < 0.35:
            return "Medium"
        else:
            return "Low"
