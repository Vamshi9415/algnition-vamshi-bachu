"""Uncertainty engine: computes and validates probabilistic forecast intervals."""
import pandas as pd
import numpy as np
from loguru import logger


class UncertaintyEngine:
    """
    Post-processes forecast intervals:
    - Ensures P10 <= P50 <= P90
    - Computes interval width and confidence score
    - Adds human-readable range string
    """

    def process(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        df = forecast_df.copy()
        df = self._enforce_ordering(df)
        df = self._compute_metrics(df)
        df = self._add_labels(df)
        return df

    def _enforce_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        df["p10"] = df[["p10", "p50", "p90"]].min(axis=1).clip(lower=0)
        df["p90"] = df[["p10", "p50", "p90"]].max(axis=1).clip(lower=0)
        df["p50"] = df["p50"].clip(lower=df["p10"], upper=df["p90"])
        return df

    def _compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["interval_width"] = (df["p90"] - df["p10"]).round(2)
        df["confidence_score"] = (
            1 - (df["interval_width"] / (df["p50"] + 1e-9))
        ).clip(0, 1).round(3)
        return df

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df["forecast_range"] = df.apply(
            lambda r: f"${r['p10']:,.0f} – ${r['p90']:,.0f}", axis=1
        )
        df["forecast_label"] = df.apply(
            lambda r: f"Expected ${r['p50']:,.0f} (89% range: {r['forecast_range']})",
            axis=1,
        )
        return df
