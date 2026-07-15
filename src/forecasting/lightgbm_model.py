"""LightGBM quantile forecasting model."""
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger

QUANTILES = [0.10, 0.50, 0.90]

FEATURE_COLS = [
    "spend", "clicks", "impressions", "conversions",
    "ctr", "cpc", "cpa", "roas", "aov",
    "year", "quarter", "month", "week_of_year",
    "day_of_week", "day_of_month", "is_weekend",
    "is_holiday", "near_black_friday", "near_christmas",
    "sin_doy", "cos_doy", "sin_month", "cos_month",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
    "revenue_rolling_mean_7d", "revenue_rolling_mean_30d",
    "revenue_rolling_std_7d", "revenue_ewm_14d",
    "spend_lag_1", "spend_lag_7", "spend_rolling_mean_7d",
    "roas_lag_1", "roas_lag_7", "roas_rolling_mean_7d",
]


class LGBMForecaster:
    """Trains three LightGBM models for P10, P50, P90 quantile forecasts."""

    def __init__(self, params: Optional[dict] = None):
        self.models: dict[float, lgb.LGBMRegressor] = {}
        base_params = params or {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "verbose": -1,
        }
        for q in QUANTILES:
            self.models[q] = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **base_params,
            )
        self.feature_cols: list[str] = []

    def _get_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_COLS if c in df.columns]

    def fit(self, train_df: pd.DataFrame, target: str = "revenue") -> None:
        self.feature_cols = self._get_features(train_df)
        X = train_df[self.feature_cols].fillna(0)
        y = train_df[target]
        for q, model in self.models.items():
            model.fit(X, y)
            logger.info(f"LightGBM Q{int(q*100)} trained on {len(X)} samples")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].fillna(0)
        preds = {}
        for q, model in self.models.items():
            preds[f"p{int(q*100)}"] = np.maximum(model.predict(X), 0)
        return pd.DataFrame(preds, index=df.index)

    def feature_importance(self) -> pd.DataFrame:
        model = self.models[0.50]
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
