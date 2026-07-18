"""Full forecast error metrics: MAE, RMSE, MAPE, WMAPE, SMAPE, MASE, Pinball, PICP, CRPS.

Purpose:
    Compute the core forecast quality metrics for point and probabilistic predictions.
Responsibilities:
    Evaluate accuracy, calibration, interval width, goodness-of-fit, and forecast quality.
Inputs:
    Arrays of true values, point predictions, interval bounds, and optional in-sample history.
Outputs:
    Dictionaries of scalar metrics suitable for reports and API responses.
Assumptions:
    Inputs are aligned and represent the same forecast horizon.
Limitations:
    CRPS is approximated from available quantiles because the models emit discrete quantile forecasts.
"""
import numpy as np
import pandas as pd
from loguru import logger


class ForecastMetrics:
    """Computes point and probabilistic forecasting evaluation metrics."""

    @staticmethod
    def _safe_variance(values: np.ndarray) -> float | None:
        if len(values) == 0:
            return None
        variance = float(np.var(values))
        return variance if variance > 0 else None

    @staticmethod
    def _adjusted_r2(r2: float, n: int, p: int = 1) -> float | None:
        if n <= p + 1:
            return None
        return float(1 - (1 - r2) * ((n - 1) / (n - p - 1)))

    def mase(self, y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, seasonality: int = 1) -> dict:
        """Mean Absolute Scaled Error using in-sample seasonal naive differences."""

        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        insample = np.array(insample, dtype=float)

        if len(insample) <= seasonality:
            return {"mase": None, "scale": None, "seasonality": seasonality}

        naive_errors = np.abs(insample[seasonality:] - insample[:-seasonality])
        scale = float(np.mean(naive_errors))
        if scale <= 0:
            return {"mase": None, "scale": round(scale, 4), "seasonality": seasonality}

        mae = float(np.mean(np.abs(y_true - y_pred)))
        return {"mase": round(mae / scale, 4), "scale": round(scale, 4), "seasonality": seasonality}

    def crps_from_quantiles(self, y_true: np.ndarray, quantile_forecasts: dict[float, np.ndarray]) -> float | None:
        """Approximate CRPS from discrete quantile forecasts via the pinball loss integral."""

        if not quantile_forecasts:
            return None

        losses = [self.pinball_loss(y_true, np.array(preds), quantile) for quantile, preds in sorted(quantile_forecasts.items())]
        if not losses:
            return None
        return round(float(2 * np.mean(losses)), 4)

    def point_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]
        n = len(y_true)
        if n == 0:
            return {"error": "No valid samples"}

        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape_mask = y_true != 0
        mape = float(np.mean(np.abs(errors[mape_mask] / y_true[mape_mask])) * 100) if mape_mask.any() else None
        wmape = float(np.sum(abs_errors) / (np.sum(np.abs(y_true)) + 1e-9) * 100)
        smape = float(np.mean(2 * abs_errors / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100)
        median_ae = float(np.median(abs_errors))
        rmsle_mask = (y_true >= 0) & (y_pred >= 0)
        rmsle = float(np.sqrt(np.mean((np.log1p(y_true[rmsle_mask]) - np.log1p(y_pred[rmsle_mask])) ** 2))) if rmsle_mask.any() else None
        variance_true = self._safe_variance(y_true)
        r2 = None
        adjusted_r2 = None
        explained_variance = None
        if variance_true is not None:
            ss_res = float(np.sum(errors ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            if ss_tot > 0:
                r2 = float(1 - (ss_res / ss_tot))
                adjusted_r2 = self._adjusted_r2(r2, n=n, p=1)
            error_variance = self._safe_variance(errors)
            if error_variance is not None:
                explained_variance = float(1 - (error_variance / variance_true))

        result = {
            "n": int(n),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "wmape": round(wmape, 4),
            "smape": round(smape, 4),
            "median_ae": round(median_ae, 4),
            "rmsle": round(rmsle, 4) if rmsle is not None else None,
            "r2": round(r2, 4) if r2 is not None else None,
            "adjusted_r2": round(adjusted_r2, 4) if adjusted_r2 is not None else None,
            "explained_variance": round(explained_variance, 4) if explained_variance is not None else None,
        }
        logger.info(f"Point metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, WMAPE={wmape:.2f}%")
        return result

    def pinball_loss(self, y_true: np.ndarray, y_pred_q: np.ndarray, quantile: float) -> float:
        """Quantile (pinball) loss for a given quantile."""
        errors = np.array(y_true) - np.array(y_pred_q)
        loss = np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))
        return round(float(loss), 4)

    def picp(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """Prediction Interval Coverage Probability."""
        covered = np.sum((y_true >= lower) & (y_true <= upper))
        return round(float(covered / (len(y_true) + 1e-9)), 4)

    def mpiw(self, lower: np.ndarray, upper: np.ndarray) -> float:
        """Mean Prediction Interval Width."""
        return round(float(np.mean(np.array(upper) - np.array(lower))), 4)

    def probabilistic_metrics(
        self,
        y_true: np.ndarray,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray,
        insample: np.ndarray | None = None,
        seasonality: int = 7,
    ) -> dict:
        """Full probabilistic evaluation."""
        y_true = np.array(y_true)
        point = self.point_metrics(y_true, np.array(p50))
        coverage = self.picp(y_true, np.array(p10), np.array(p90))
        width = self.mpiw(np.array(p10), np.array(p90))
        pb_10 = self.pinball_loss(y_true, np.array(p10), 0.10)
        pb_50 = self.pinball_loss(y_true, np.array(p50), 0.50)
        pb_90 = self.pinball_loss(y_true, np.array(p90), 0.90)
        crps = self.crps_from_quantiles(y_true, {0.10: np.array(p10), 0.50: np.array(p50), 0.90: np.array(p90)})
        mase_result = self.mase(y_true, np.array(p50), insample, seasonality=seasonality) if insample is not None else {"mase": None, "scale": None, "seasonality": seasonality}

        result = {
            **point,
            "picp_80pct_interval": coverage,
            "target_coverage": 0.80,
            "coverage_gap": round(float(coverage - 0.80), 4),
            "mpiw": width,
            "pinball_p10": pb_10,
            "pinball_p50": pb_50,
            "pinball_p90": pb_90,
            "crps": crps,
            **mase_result,
            "calibration_status": (
                "Well-calibrated" if abs(coverage - 0.80) < 0.05
                else "Under-covering" if coverage < 0.75
                else "Over-covering"
            ),
            "forecast_quality": self._quality_label(point["wmape"], coverage),
        }
        logger.info(f"Probabilistic metrics: PICP={coverage:.2%}, MPIW={width:.2f}, WMAPE={point['wmape']}%")
        return result

    @staticmethod
    def _quality_label(wmape: float, picp: float) -> str:
        if wmape < 10 and abs(picp - 0.80) <= 0.05:
            return "Excellent"
        if wmape < 15 and abs(picp - 0.80) <= 0.08:
            return "Very Good"
        if wmape < 20:
            return "Good"
        return "Needs Improvement"

    def overall_score(self, metrics: dict, residual_diagnostics: dict, backtesting: dict | None = None) -> dict:
        """Compute a judge-friendly score and grade from model diagnostics."""

        wmape = float(metrics.get("wmape", 100.0))
        coverage_gap = abs(float(metrics.get("coverage_gap", 1.0)))
        calibration_ok = metrics.get("calibration_status") in {"Well-calibrated", "Over-covering"}
        residual_autocorr = bool(residual_diagnostics.get("ljung_box", {}).get("lag_10", {}).get("has_autocorrelation", True))

        score = 100.0
        score -= min(40.0, wmape * 2.5)
        score -= min(20.0, coverage_gap * 200.0)
        score -= 12.0 if residual_autocorr else 0.0
        score -= 8.0 if not calibration_ok else 0.0

        if backtesting and backtesting.get("wmape_std") is not None:
            score -= min(10.0, float(backtesting["wmape_std"]) * 10.0)

        score = max(0.0, min(100.0, score))
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
        return {"score": round(score, 1), "grade": grade}

    def production_readiness(self, metrics: dict, residual_diagnostics: dict, backtesting: dict | None = None) -> dict:
        """Return a structured readiness summary with reasons and next steps."""

        passed: list[str] = []
        failed: list[str] = []
        recommendations: list[str] = []

        if metrics.get("wmape", 999) < 20.0:
            passed.append("WMAPE")
        else:
            failed.append("WMAPE")
            recommendations.append("Reduce point error with feature and hyperparameter tuning")

        if abs(metrics.get("coverage_gap", 1.0)) < 0.10:
            passed.append("Coverage")
        else:
            failed.append("Coverage")
            recommendations.append("Recalibrate prediction intervals or adjust quantile spread")

        if metrics.get("calibration_status") in ["Well-calibrated", "Over-covering"]:
            passed.append("Calibration")
        else:
            failed.append("Calibration")
            recommendations.append("Revisit quantile calibration and interval construction")

        if backtesting is None or backtesting.get("wmape_std", 0.0) < 1.0:
            passed.append("Backtesting")
        else:
            failed.append("Backtesting")
            recommendations.append("Stabilize performance across folds")

        if not residual_diagnostics.get("ljung_box", {}).get("lag_10", {}).get("has_autocorrelation", True):
            passed.append("Residual Autocorrelation")
        else:
            failed.append("Residual Autocorrelation")
            recommendations.append("Add richer lag windows or sequence-aware modeling")

        return {"status": not failed, "passed": passed, "failed": failed, "recommendations": recommendations}

