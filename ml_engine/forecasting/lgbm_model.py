"""LightGBM quantile regression forecaster."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger


FEATURE_COLS_EXCLUDE = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "currency", "daily_budget", "holiday_name", "revenue",
]

QUANTILES = [0.1, 0.5, 0.9]


class LGBMForecaster:
    """Trains one LightGBM model per quantile and returns P10/P50/P90 forecasts."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.models: dict[float, lgb.Booster] = {}
        self.feature_cols: list[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        exclude = set(FEATURE_COLS_EXCLUDE)
        return [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    def fit(self, df: pd.DataFrame):
        self.feature_cols = self._get_feature_cols(df)
        X = df[self.feature_cols].fillna(0)
        y = df["revenue"].fillna(0)

        base_params = {
            "n_estimators": self.config.get("n_estimators", 400),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "num_leaves": self.config.get("num_leaves", 63),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "verbose": -1,
        }

        for q in QUANTILES:
            params = {**base_params, "objective": "quantile", "alpha": q}
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self.models[q] = model
            logger.info(f"LightGBM trained for quantile={q}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].fillna(0)
        out = df[["date", "channel", "campaign_name"]].copy()
        for q in QUANTILES:
            preds = self.models[q].predict(X)
            preds = np.maximum(preds, 0)  # revenue cannot be negative
            out[f"revenue_p{int(q*100)}"] = preds
        return out

    @property
    def feature_importance(self) -> pd.DataFrame:
        model = self.models.get(0.5)
        if model is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp

