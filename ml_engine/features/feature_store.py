"""Feature store: orchestrates all feature generators and saves parquet artifacts."""
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_engine.features.calendar_features import CalendarFeatureGenerator
from ml_engine.features.kpi_features import KPIFeatureGenerator
from ml_engine.features.lag_features import LagFeatureGenerator


class FeatureStore:
    """Runs the full feature pipeline and persists output to disk."""

    def __init__(self, output_dir: str = "data/features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calendar = CalendarFeatureGenerator()
        self.kpi = KPIFeatureGenerator()
        self.lag = LagFeatureGenerator()

    def build(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """Run all feature generators in order and return enriched DataFrame."""
        logger.info("Building feature store")
        df = self.kpi.generate(df)
        df = self.calendar.generate(df)
        df = self.lag.generate(df)

        # Leakage check: ensure no future values
        self._assert_no_leakage(df)

        if save:
            out_path = self.output_dir / "daily_features.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Feature store saved: {out_path} ({len(df)} rows, {df.shape[1]} cols)")

        return df

    def _assert_no_leakage(self, df: pd.DataFrame):
        """Basic leakage guard: lag/rolling cols must have NaN at start of each series."""
        # _roll_std_ is deliberately fillna(0)'d (std of a single point is undefined,
        # not a leak) so it's excluded from the strict first-row-must-be-NaN check.
        lag_cols = [
            c for c in df.columns
            if ("_lag" in c or "_roll" in c) and "_roll_std_" not in c
        ]
        sorted_df = df.sort_values(["channel", "campaign_name", "date"])
        first_rows = sorted_df.groupby(["channel", "campaign_name"], as_index=False).nth(0)
        for col in lag_cols:
            leaked = first_rows[col].notna()
            if leaked.any():
                bad_campaigns = first_rows.loc[leaked, "campaign_name"].unique().tolist()
                raise ValueError(
                    f"Leakage detected in '{col}': first-row value is not NaN for campaigns {bad_campaigns}"
                )
        logger.info(f"Leakage check passed ({len(lag_cols)} lag/roll cols confirmed shifted)")

