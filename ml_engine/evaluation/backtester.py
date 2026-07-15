"""Walk-forward backtesting with fold-level metrics."""
import pandas as pd
import numpy as np
from loguru import logger

from ml_engine.evaluation.forecast_metrics import ForecastMetrics


class WalkForwardBacktester:
    """
    Walk-forward validation: trains on expanding window, tests on fixed horizon.
    Records metrics per fold and computes aggregate stats.
    """

    def __init__(self, n_splits: int = 3, test_size_days: int = 30):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.metrics_engine = ForecastMetrics()

    def run(self, df: pd.DataFrame, model_cls, model_kwargs: dict = None) -> dict:
        """
        df: features DataFrame with 'date' and 'revenue' columns.
        model_cls: a class with .fit(df) and .predict(df) returning DataFrame with revenue_p50.
        Returns: per-fold metrics and aggregate summary.
        """
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].unique()
        n_days = len(dates)
        fold_results = []

        for fold in range(self.n_splits):
            test_end_idx = n_days - fold * self.test_size_days
            test_start_idx = test_end_idx - self.test_size_days
            if test_start_idx <= self.test_size_days:
                logger.warning(f"Fold {fold}: not enough training data, skipping")
                continue

            train_dates = dates[:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]

            try:
                model = model_cls(model_kwargs or {})
                model.fit(train_df)
                preds = model.predict(test_df)
            except Exception as e:
                logger.error(f"Fold {fold} model error: {e}")
                continue

            # Merge on (date, channel, campaign_name)
            merge_keys = ["date", "channel", "campaign_name"]
            merged = test_df[[*merge_keys, "revenue"]].merge(
                preds, on=merge_keys, how="inner"
            )
            if merged.empty:
                continue

            y_true = merged["revenue"].values
            y_pred = merged["revenue_p50"].values

            metrics = self.metrics_engine.point_metrics(y_true, y_pred)
            metrics["fold"] = fold
            metrics["train_days"] = len(train_dates)
            metrics["test_days"] = len(test_dates)
            metrics["test_start"] = str(test_dates[0])[:10]
            metrics["test_end"] = str(test_dates[-1])[:10]
            fold_results.append(metrics)
            logger.info(f"Fold {fold}: MAE={metrics['mae']:.2f}, WMAPE={metrics['wmape']:.2f}%")

        if not fold_results:
            return {"error": "No folds completed"}

        fold_df = pd.DataFrame(fold_results)
        summary = {
            "n_folds": len(fold_results),
            "mae_mean": round(float(fold_df["mae"].mean()), 4),
            "mae_std": round(float(fold_df["mae"].std()), 4),
            "wmape_mean": round(float(fold_df["wmape"].mean()), 4),
            "wmape_std": round(float(fold_df["wmape"].std()), 4),
            "rmse_mean": round(float(fold_df["rmse"].mean()), 4),
            "folds": fold_results,
        }
        return summary

