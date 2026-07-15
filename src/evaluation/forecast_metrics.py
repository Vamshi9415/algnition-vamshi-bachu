"""Full forecast error metrics: MAE, RMSE, MAPE, WMAPE, SMAPE, MASE, Pinball, PICP, CRPS."""
import numpy as np
import pandas as pd
from loguru import logger


class ForecastMetrics:
    """Computes point and probabilistic forecasting evaluation metrics."""

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

        result = {
            "n": int(n),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "wmape": round(wmape, 4),
            "smape": round(smape, 4),
            "median_ae": round(median_ae, 4),
            "rmsle": round(rmsle, 4) if rmsle is not None else None,
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
    ) -> dict:
        """Full probabilistic evaluation."""
        y_true = np.array(y_true)
        point = self.point_metrics(y_true, np.array(p50))
        coverage = self.picp(y_true, np.array(p10), np.array(p90))
        width = self.mpiw(np.array(p10), np.array(p90))
        pb_10 = self.pinball_loss(y_true, np.array(p10), 0.10)
        pb_50 = self.pinball_loss(y_true, np.array(p50), 0.50)
        pb_90 = self.pinball_loss(y_true, np.array(p90), 0.90)

        result = {
            **point,
            "picp_80pct_interval": coverage,
            "target_coverage": 0.80,
            "coverage_gap": round(float(coverage - 0.80), 4),
            "mpiw": width,
            "pinball_p10": pb_10,
            "pinball_p50": pb_50,
            "pinball_p90": pb_90,
            "calibration_status": (
                "Well-calibrated" if abs(coverage - 0.80) < 0.05
                else "Under-covering" if coverage < 0.75
                else "Over-covering"
            ),
        }
        logger.info(f"Probabilistic metrics: PICP={coverage:.2%}, MPIW={width:.2f}, WMAPE={point['wmape']}%")
        return result
