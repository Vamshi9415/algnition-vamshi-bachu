"""SHAP-based explainability for LightGBM forecaster."""
import pandas as pd
import numpy as np
from loguru import logger

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    logger.warning("SHAP not installed. Run: pip install shap")


class SHAPExplainer:
    """Computes SHAP values for a fitted LGBMForecaster."""

    def explain(self, model, X: pd.DataFrame, max_samples: int = 500) -> dict:
        if not SHAP_OK:
            return {"error": "shap not installed"}
        # Use median quantile model (P50)
        lgbm_model = model.models.get(0.5)
        if lgbm_model is None:
            return {"error": "P50 model not fitted"}

        X_sample = X[model.feature_cols].fillna(0).head(max_samples)
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": model.feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        top_features = shap_df.head(15).to_dict(orient="records")
        logger.info(f"SHAP computed. Top feature: {shap_df.iloc[0]['feature']}")
        return {
            "top_features": top_features,
            "explanation": (
                f"The most influential feature is '{shap_df.iloc[0]['feature']}' "
                f"with mean |SHAP| = {shap_df.iloc[0]['mean_abs_shap']:.4f}."
            ),
        }

