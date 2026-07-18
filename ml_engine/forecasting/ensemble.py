"""Ensemble: weighted average of LightGBM and Prophet forecasts."""

import pandas as pd
from loguru import logger

from ml_engine.forecasting.lgbm_model import LGBMForecaster
from ml_engine.forecasting.prophet_model import ProphetForecaster


class EnsembleForecaster:
    """Combines predictions from multiple models into a single probabilistic output."""

    def __init__(self, weights: dict = None):
        # weights: {"lgbm": 0.6, "prophet": 0.4}
        self.weights = weights or {"lgbm": 0.6, "prophet": 0.4}

    def combine(
        self,
        lgbm_preds: pd.DataFrame,
        prophet_preds: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        def _suffix_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
            renamed = frame.copy()
            for q in ["p10", "p50", "p90"]:
                col = f"revenue_{q}"
                if col in renamed.columns:
                    renamed = renamed.rename(columns={col: f"{col}_{source}"})
            return renamed

        sources: list[tuple[str, pd.DataFrame]] = [("lgbm", lgbm_preds)]
        if prophet_preds is not None and not prophet_preds.empty:
            sources.append(("prophet", prophet_preds))

        if len(sources) == 1:
            logger.info("Ensemble: using LightGBM only (other forecasters unavailable)")
            return lgbm_preds

        merge_keys = ["date", "channel", "campaign_name"]
        merged = _suffix_frame(lgbm_preds, "lgbm")
        for source, frame in sources[1:]:
            merged = merged.merge(_suffix_frame(frame, source), on=merge_keys, how="left")

        available_weights = {
            source: float(self.weights.get(source, 0.0))
            for source, frame in sources
            if any(f"revenue_p{q}_{source}" in merged.columns for q in [10, 50, 90])
        }
        if not available_weights:
            available_weights = {source: 1.0 for source, _ in sources}

        total_weight = sum(available_weights.values()) or 1.0
        normalized_weights = {source: weight / total_weight for source, weight in available_weights.items()}

        for q in ["p10", "p50", "p90"]:
            combined = None
            for source, weight in normalized_weights.items():
                col = f"revenue_{q}_{source}"
                if col not in merged.columns:
                    continue
                series = merged[col].fillna(0)
                combined = series * weight if combined is None else combined + series * weight
            merged[f"revenue_{q}"] = (combined if combined is not None else merged[f"revenue_{q}_lgbm"].fillna(0)).clip(lower=0)

        out_cols = merge_keys + ["revenue_p10", "revenue_p50", "revenue_p90"]
        logger.info(f"Ensemble complete: {len(merged)} rows using {list(normalized_weights.keys())}")
        return merged[out_cols]


class ForecastEnsemble:
    """Higher-level forecast wrapper used by the API and legacy pipeline."""

    def __init__(self, horizon: int = 60, weights: dict | None = None):
        self.horizon = horizon
        self.lgbm = LGBMForecaster()
        self.prophet = ProphetForecaster()
        self.ensemble = EnsembleForecaster(weights)

    def fit(self, df: pd.DataFrame):
        self.lgbm.fit(df)
        self.prophet.fit(df)

    def predict(self, future_rows: pd.DataFrame) -> pd.DataFrame:
        lgbm_preds = self.lgbm.predict(future_rows)
        prophet_preds = self.prophet.predict(self.horizon)
        return self.ensemble.combine(lgbm_preds, prophet_preds)

