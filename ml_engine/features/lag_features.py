"""Lag, rolling, EWM, and growth feature generator."""
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

        # Lag features
        for col in LAG_COLS:
            if col not in df.columns:
                continue
            for lag in LAG_DAYS:
                df[f"{col}_lag{lag}"] = df.groupby(group_keys)[col].shift(lag)

        # Rolling features
        for col in ["revenue", "spend", "roas"]:
            if col not in df.columns:
                continue
            for w in ROLL_WINDOWS:
                grp = df.groupby(group_keys)[col]
                df[f"{col}_roll_mean_{w}d"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f"{col}_roll_std_{w}d"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
                df[f"{col}_roll_max_{w}d"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                df[f"{col}_roll_min_{w}d"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())
                df[f"{col}_roll_median_{w}d"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())

        # EWM features
        for col in ["revenue", "spend", "roas"]:
            if col not in df.columns:
                continue
            for alpha in [0.3, 0.7]:
                label = str(int(alpha * 10))
                df[f"{col}_ewm{label}"] = df.groupby(group_keys)[col].transform(
                    lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean()
                )

        # Growth features: WoW, MoM, QoQ.
        # BUGFIX: previously compared df[col] (today's raw, unshifted value -- i.e. the
        # target itself for col="revenue") against a lagged reference. Combined with the
        # *_lagN features already in the set, that made revenue exactly reconstructible
        # algebraically -- a target leak, not a real signal (see reports/recipe_verification.md
        # errata). Anchor "current" on the last KNOWN value (lag1, yesterday) instead, and
        # push the reference point back by one extra day so the window length (7/30/90d) is
        # unchanged.
        for col in ["revenue", "spend"]:
            if col not in df.columns:
                continue
            asof = df.groupby(group_keys)[col].shift(1)
            prev7  = df.groupby(group_keys)[col].shift(8)
            prev30 = df.groupby(group_keys)[col].shift(31)
            prev90 = df.groupby(group_keys)[col].shift(91)
            df[f"{col}_wow_growth"] = ((asof - prev7)  / (prev7.abs()  + 1e-9)).round(4)
            df[f"{col}_mom_growth"] = ((asof - prev30) / (prev30.abs() + 1e-9)).round(4)
            df[f"{col}_qoq_growth"] = ((asof - prev90) / (prev90.abs() + 1e-9)).round(4)

        # Lifecycle & momentum features (leak-free: derived from date and already-shifted lags).
        # Both ranked in the top 15 of the leak-free feature set in the ablation
        # (see reports/feature_engineering.md); days_since_series_start captures the
        # campaign ramp/decay curve, rev_lag1_over_lag7 captures week-over-week momentum.
        df["days_since_series_start"] = (
            df.groupby(group_keys)["date"].transform(lambda s: (s - s.min()).dt.days)
        )
        if "revenue_lag1" in df.columns and "revenue_lag7" in df.columns:
            df["rev_lag1_over_lag7"] = (
                (df["revenue_lag1"] / (df["revenue_lag7"] + 1e-9))
                .replace([float("inf"), float("-inf")], 0.0)
                .clip(-100, 100)
                .round(4)
            )

        # Interaction features
        if "spend" in df.columns and "month" in df.columns:
            df["spend_x_month"] = df["spend"] * df["month"]
        if "spend" in df.columns and "is_holiday" in df.columns:
            df["spend_x_holiday"] = df["spend"] * df["is_holiday"]
        if "is_weekend" in df.columns:
            df["channel_weekend"] = df["channel"].astype("category").cat.codes * df["is_weekend"]

        logger.info(f"Lag/rolling/growth/interaction features added. Shape: {df.shape}")
        return df

