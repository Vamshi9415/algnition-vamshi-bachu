"""Feature store: orchestrates all feature generators and persists outputs."""
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_engine.features.calendar_features import CalendarFeatureGenerator
from ml_engine.features.marketing_kpis import MarketingKPIGenerator
from ml_engine.features.lag_features import LagFeatureGenerator


class FeatureStore:
    """Runs the full feature pipeline and saves outputs to disk."""

    def __init__(self, output_dir: str = "data/features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calendar_gen = CalendarFeatureGenerator()
        self.kpi_gen = MarketingKPIGenerator()
        self.lag_gen = LagFeatureGenerator()

    def build(self, canonical_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building feature store...")
        df = canonical_df.copy()
        df = self.kpi_gen.generate(df)
        df = self.calendar_gen.generate(df)
        df = self.lag_gen.generate(df)
        df = df.sort_values(["channel", "campaign_name", "date"]).reset_index(drop=True)
        self._save(df)
        logger.info(f"Feature store complete: {df.shape[1]} columns, {len(df)} rows")
        return df

    def _save(self, df: pd.DataFrame):
        path = self.output_dir / "features.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Feature store saved to {path}")

    def load(self) -> pd.DataFrame:
        path = self.output_dir / "features.parquet"
        if not path.exists():
            raise FileNotFoundError("Feature store not found. Run build() first.")
        return pd.read_parquet(path)

