"""Lag and rolling window feature generator."""
import pandas as pd
from loguru import logger


LAG_DAYS = [1, 3, 7, 14, 30]
ROLL_WINDOWS = [3, 7, 14, 30]
LAG_COLS = ["revenue", "spend", "roas", "conversions", "clicks", "ctr"]


class LagFeatureGenerator:
    """Adds time-shifted and rolling statistics for each campaign."""

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["channel", "campaign_name", "date"]).reset_index(drop=True)
        group_keys = ["channel", "campaign_name"]

        for col in LAG_COLS:
            if col not in df.columns:
                continue
            for lag in LAG_DAYS:
                df[f"{col}_lag{lag}"] = (
                    df.groupby(group_keys)[col]
                    .shift(lag)
                )

        for col in ["revenue", "spend", "roas"]:
            if col not in df.columns:
                continue
            for w in ROLL_WINDOWS:
                grp = df.groupby(group_keys)[col]
                df[f"{col}_roll_mean_{w}d"] = grp.transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                df[f"{col}_roll_std_{w}d"] = grp.transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
                )
                df[f"{col}_roll_max_{w}d"] = grp.transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).max()
                )

        # EWM features
        for col in ["revenue", "spend", "roas"]:
            if col not in df.columns:
                continue
            for alpha in [0.3, 0.7]:
                label = str(int(alpha * 10))
                df[f"{col}_ewm{label}"] = df.groupby(group_keys)[col].transform(
                    lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean()
                )

        # Growth features
        for col in ["revenue", "spend"]:
            if col not in df.columns:
                continue
            prev = df.groupby(group_keys)[col].shift(7)
            df[f"{col}_wow_growth"] = ((df[col] - prev) / (prev.abs() + 1e-9)).round(4)
            prev30 = df.groupby(group_keys)[col].shift(30)
            df[f"{col}_mom_growth"] = ((df[col] - prev30) / (prev30.abs() + 1e-9)).round(4)

        logger.info(f"Lag/rolling features added. Shape: {df.shape}")
        return df
