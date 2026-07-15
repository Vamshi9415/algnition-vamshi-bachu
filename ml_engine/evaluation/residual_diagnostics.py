"""Residual diagnostics: autocorrelation, normality, Ljung-Box."""
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False


class ResidualDiagnostics:
    """Diagnoses model residuals for autocorrelation, normality, and bias."""

    def diagnose(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        residuals = (y_true - y_pred).dropna()
        result = {
            "n": int(len(residuals)),
            "mean": round(float(residuals.mean()), 4),
            "std": round(float(residuals.std()), 4),
            "skewness": round(float(stats.skew(residuals)), 4),
            "kurtosis": round(float(stats.kurtosis(residuals)), 4),
        }

        # Bias
        result["mean_forecast_error"] = result["mean"]
        result["bias_interpretation"] = (
            "Unbiased" if abs(result["mean"]) < max(result["std"] * 0.1, 1e-9)
            else ("Over-forecast" if result["mean"] < 0 else "Under-forecast")
        )

        # Normality: Shapiro-Wilk
        if len(residuals) <= 5000:
            sw_stat, sw_p = stats.shapiro(residuals[:5000])
            result["shapiro_wilk"] = {
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 4),
                "is_normal": bool(sw_p > 0.05),
            }

        # Ljung-Box test for residual autocorrelation (H0: no autocorrelation)
        if STATSMODELS_OK and len(residuals) >= 20:
            lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
            result["ljung_box"] = {
                "lag_10": {
                    "statistic": round(float(lb["lb_stat"].iloc[0]), 4),
                    "p_value": round(float(lb["lb_pvalue"].iloc[0]), 4),
                    "has_autocorrelation": bool(lb["lb_pvalue"].iloc[0] < 0.05),
                },
                "lag_20": {
                    "statistic": round(float(lb["lb_stat"].iloc[1]), 4),
                    "p_value": round(float(lb["lb_pvalue"].iloc[1]), 4),
                    "has_autocorrelation": bool(lb["lb_pvalue"].iloc[1] < 0.05),
                },
                "interpretation": (
                    "Residuals have significant autocorrelation — model is missing temporal patterns."
                    if lb["lb_pvalue"].iloc[0] < 0.05
                    else "No significant residual autocorrelation — model captures temporal structure."
                ),
            }

        # Durbin-Watson
        if STATSMODELS_OK:
            dw = durbin_watson(residuals)
            result["durbin_watson"] = {
                "statistic": round(float(dw), 4),
                "interpretation": (
                    "Positive autocorrelation" if dw < 1.5
                    else "No autocorrelation" if dw <= 2.5
                    else "Negative autocorrelation"
                ),
            }

        logger.info(f"Residual diagnostics: bias={result['bias_interpretation']}, "
                    f"normal={result.get('shapiro_wilk', {}).get('is_normal', 'N/A')}")
        return result

