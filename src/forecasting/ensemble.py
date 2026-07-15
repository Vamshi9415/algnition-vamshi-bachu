"""Forecast ensemble: combines LightGBM and Prophet predictions."""
import pandas as pd
import numpy as np
from loguru import logger

from src.forecasting.lightgbm_model import LGBMForecaster
from src.forecasting.prophet_model import ProphetForecaster


class ForecastEnsemble:
    """
    Combines LightGBM (quantile) and Prophet forecasts via weighted average.
    Falls back to LightGBM only if Prophet unavailable.
    """

    def __init__(self, horizon: int = 60, lgbm_weight: float = 0.5):
        self.horizon = horizon
        self.lgbm_weight = lgbm_weight
        self.prophet_weight = 1.0 - lgbm_weight
        self.lgbm = LGBMForecaster()
        self.prophet = ProphetForecaster(horizon=horizon)
        self._prophet_fitted = False
        self._lgbm_fitted = False

    def fit(self, features_df: pd.DataFrame) -> None:
        # Fit LightGBM on full feature set
        train = features_df.dropna(subset=["revenue"])
        self.lgbm.fit(train)
        self._lgbm_fitted = True

        # Fit Prophet on aggregated daily
        try:
            self.prophet.fit(features_df)
            self._prophet_fitted = True
        except Exception as e:
            logger.warning(f"Prophet fit failed: {e}")

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        forecast_df: future-dated rows with feature columns filled.
        Returns DataFrame with [date, p10, p50, p90, model].
        """
        lgbm_preds = self.lgbm.predict(forecast_df)
        lgbm_preds["date"] = forecast_df["date"].values

        if self._prophet_fitted:
            prophet_preds = self.prophet.predict()
            # Align on date
            merged = lgbm_preds.merge(prophet_preds, on="date", suffixes=("_lgbm", "_prophet"), how="left")
            for q in ["p10", "p50", "p90"]:
                lgbm_col = f"{q}_lgbm"
                prophet_col = f"{q}_prophet"
                if prophet_col in merged.columns:
                    merged[q] = (
                        self.lgbm_weight * merged[lgbm_col].fillna(0) +
                        self.prophet_weight * merged[prophet_col].fillna(0)
                    ).clip(lower=0)
                else:
                    merged[q] = merged[lgbm_col]
            result = merged[["date", "p10", "p50", "p90"]]
            result["model"] = "ensemble"
        else:
            result = lgbm_preds.rename(columns={"p10": "p10", "p50": "p50", "p90": "p90"})
            result["model"] = "lightgbm"

        return result.reset_index(drop=True)
