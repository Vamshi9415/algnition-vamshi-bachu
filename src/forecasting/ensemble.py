"""Ensemble: weighted average of LightGBM + Prophet forecasts."""
import pandas as pd
from loguru import logger


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
        if prophet_preds is None or prophet_preds.empty:
            logger.info("Ensemble: using LightGBM only (Prophet unavailable)")
            return lgbm_preds

        merge_keys = ["date", "channel", "campaign_name"]
        merged = lgbm_preds.merge(prophet_preds, on=merge_keys, suffixes=("_lgbm", "_prophet"), how="left")

        w_l = self.weights["lgbm"]
        w_p = self.weights["prophet"]

        for q in ["p10", "p50", "p90"]:
            l_col = f"revenue_{q}_lgbm"
            p_col = f"revenue_{q}_prophet"
            if p_col in merged.columns:
                merged[f"revenue_{q}"] = (
                    w_l * merged[l_col].fillna(0) + w_p * merged[p_col].fillna(0)
                ).clip(lower=0)
            else:
                merged[f"revenue_{q}"] = merged[l_col].fillna(0)

        out_cols = merge_keys + ["revenue_p10", "revenue_p50", "revenue_p90"]
        logger.info(f"Ensemble complete: {len(merged)} rows")
        return merged[out_cols]
