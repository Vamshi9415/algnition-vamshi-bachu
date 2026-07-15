"""Lag and rolling window feature generator."""
import pandas as pd


LAG_WINDOWS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [3, 7, 14, 30, 60]
EWM_SPANS = [7, 14, 30]

TARGET_COLS = ["revenue", "spend", "roas", "conversions", "clicks", "ctr", "cpa"]


class LagFeatureGenerator:
    """Generates lag, rolling, EWM, and growth features per campaign."""

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["channel", "campaign_name", "date"])
        group_cols = ["channel", "campaign_name"]

        for col in TARGET_COLS:
            if col not in df.columns:
                continue

            grp = df.groupby(group_cols)[col]

            # Lag features
            for lag in LAG_WINDOWS:
                df[f"{col}_lag_{lag}"] = grp.shift(lag)

            # Rolling statistics
            for w in ROLLING_WINDOWS:
                shifted = grp.shift(1)  # avoid leakage
                roll = df.groupby(group_cols)[col].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                df[f"{col}_rolling_mean_{w}d"] = roll
                df[f"{col}_rolling_std_{w}d"] = df.groupby(group_cols)[col].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).std()
                )

            # EWM
            for span in EWM_SPANS:
                df[f"{col}_ewm_{span}d"] = df.groupby(group_cols)[col].transform(
                    lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
                )

            # Growth features
            df[f"{col}_wow_growth"] = grp.pct_change(7).round(4)
            df[f"{col}_mom_growth"] = grp.pct_change(30).round(4)

        return df
