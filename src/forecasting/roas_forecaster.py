"""ROAS forecasting: probabilistic P10/P50/P90 ROAS using LightGBM quantile regression."""
import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False


class ROASForecaster:
    """Trains separate LightGBM quantile models for ROAS P10/P50/P90."""

    QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

    def __init__(self, params: dict = None):
        self.params = params or {
            "n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
            "min_child_samples": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        }
        self.models: dict = {}
        self.feature_cols: list = []

    def fit(self, df: pd.DataFrame):
        if not LGBM_OK:
            logger.warning("LightGBM not available — ROASForecaster skipped")
            return self

        target = "roas"
        if target not in df.columns:
            logger.warning("No 'roas' column found — skipping ROAS forecaster")
            return self

        exclude = {"date", "channel", "campaign_name", "campaign_id",
                   "currency", "device", "country", "status", "holiday_name",
                   "revenue", "roas", "conversions", "spend"}
        self.feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]

        X = df[self.feature_cols].fillna(0)
        y = df[target].fillna(0).clip(lower=0, upper=500)

        for label, q in self.QUANTILES.items():
            p = {**self.params, "objective": "quantile", "alpha": q, "metric": "quantile"}
            model = lgb.LGBMRegressor(**p)
            model.fit(X, y)
            self.models[label] = model
            logger.info(f"ROAS {label.upper()} model trained (quantile={q})")

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            return pd.DataFrame()

        X = df[self.feature_cols].fillna(0)
        out = df[["date", "channel", "campaign_name"]].copy().reset_index(drop=True)
        for label, model in self.models.items():
            out[f"roas_{label}"] = np.clip(model.predict(X), 0.01, 500)

        # Enforce monotonicity
        if {"roas_p10", "roas_p50", "roas_p90"}.issubset(out.columns):
            out["roas_p10"] = out[["roas_p10", "roas_p50"]].min(axis=1)
            out["roas_p90"] = out[["roas_p90", "roas_p50"]].max(axis=1)

        return out
