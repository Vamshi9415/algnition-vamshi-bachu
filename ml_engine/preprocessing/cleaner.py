"""8-step deterministic data cleaning pipeline."""
import numpy as np
import pandas as pd
from loguru import logger


class DataCleaner:
    """Cleans a canonical DataFrame in 8 deterministic steps."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n0 = len(df)

        # Step 1: Parse dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Step 2: Lowercase / strip strings
        for col in ["channel", "campaign_name", "campaign_type", "currency", "device", "country", "status"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        # Step 3: Fill missing numeric cols with 0
        for col in ["spend", "clicks", "impressions", "conversions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

        # Step 4: Fill missing revenue with 0 (cannot impute with forward-fill)
        if "revenue" in df.columns:
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0).clip(lower=0)

        # Step 5: Remove duplicate (date, channel, campaign) rows
        df = df.drop_duplicates(subset=["date", "channel", "campaign_name"], keep="last")

        # Step 6: Sort by channel, campaign, date
        df = df.sort_values(["channel", "campaign_name", "date"]).reset_index(drop=True)

        # Step 7: Recompute ROAS
        if "revenue" in df.columns and "spend" in df.columns:
            df["roas"] = (df["revenue"] / (df["spend"] + 1e-9)).round(4)

        # Step 8: Outlier detection (IQR-based flag, do not remove)
        if "revenue" in df.columns:
            q1, q3 = df["revenue"].quantile(0.25), df["revenue"].quantile(0.75)
            iqr = q3 - q1
            df["revenue_outlier_flag"] = ((df["revenue"] < q1 - 3 * iqr) | (df["revenue"] > q3 + 3 * iqr)).astype(int)
            n_outliers = df["revenue_outlier_flag"].sum()
            if n_outliers:
                logger.warning(f"Outlier detection: {n_outliers} revenue outliers flagged (IQR x3)")

        # Currency normalization placeholder (flag non-USD)
        if "currency" in df.columns:
            non_usd = df["currency"].ne("usd").sum()
            if non_usd > 0:
                logger.warning(f"{non_usd} rows with non-USD currency — conversion not applied")

        logger.info(f"Cleaning complete: {n0} → {len(df)} rows")
        return df

