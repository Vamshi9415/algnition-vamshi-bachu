"""Baseline forecast models: Naive and Seasonal Naive."""
import numpy as np
import pandas as pd
from loguru import logger


class NaiveForecaster:
    """Forecast = last observed value (random walk baseline)."""

    def __init__(self):
        self.last_values: dict = {}

    def fit(self, df: pd.DataFrame):
        for (ch, camp), grp in df.groupby(["channel", "campaign_name"]):
            last = grp.sort_values("date").iloc[-1]["revenue"]
            self.last_values[(ch, camp)] = float(last)
        logger.info(f"NaiveForecaster fitted for {len(self.last_values)} campaigns")

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        out = future_df[["date", "channel", "campaign_name"]].copy()
        out["revenue_p50"] = out.apply(
            lambda r: self.last_values.get((r["channel"], r["campaign_name"]), 0.0), axis=1
        )
        out["revenue_p10"] = out["revenue_p50"] * 0.7
        out["revenue_p90"] = out["revenue_p50"] * 1.3
        return out


class SeasonalNaiveForecaster:
    """Forecast = value from same day of week, 1 week ago."""

    def __init__(self, period: int = 7):
        self.period = period
        self.history: dict = {}

    def fit(self, df: pd.DataFrame):
        for (ch, camp), grp in df.groupby(["channel", "campaign_name"]):
            self.history[(ch, camp)] = grp.sort_values("date")[["date", "revenue"]].copy()
        logger.info(f"SeasonalNaiveForecaster fitted for {len(self.history)} campaigns")

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in future_df.iterrows():
            key = (row["channel"], row["campaign_name"])
            hist = self.history.get(key)
            if hist is None or len(hist) < self.period:
                rev = 0.0
            else:
                # Find same weekday from period days ago
                target_date = pd.Timestamp(row["date"]) - pd.Timedelta(days=self.period)
                closest = hist.iloc[(hist["date"] - target_date).abs().argsort()[:1]]
                rev = float(closest["revenue"].values[0])
            rows.append({
                "date": row["date"], "channel": row["channel"],
                "campaign_name": row["campaign_name"],
                "revenue_p50": rev, "revenue_p10": rev * 0.75, "revenue_p90": rev * 1.25,
            })
        return pd.DataFrame(rows)
