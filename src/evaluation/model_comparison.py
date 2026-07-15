"""Multi-model comparison table with statistical significance."""
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.forecast_metrics import ForecastMetrics
from src.evaluation.significance_tests import SignificanceTester


@dataclass
class ModelResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    p10: np.ndarray = None
    p50: np.ndarray = None
    p90: np.ndarray = None
    train_time_s: float = 0.0
    inference_time_s: float = 0.0


class ModelComparator:
    """Compares multiple ModelResult objects and outputs a ranked comparison table."""

    def __init__(self):
        self.metrics = ForecastMetrics()
        self.significance = SignificanceTester()

    def compare(self, results: list[ModelResult]) -> dict:
        rows = []
        for r in results:
            m = self.metrics.point_metrics(r.y_true, r.y_pred)
            if r.p10 is not None and r.p90 is not None:
                prob = self.metrics.probabilistic_metrics(r.y_true, r.p10, r.p50, r.p90)
                m.update({"picp": prob["picp_80pct_interval"], "mpiw": prob["mpiw"]})
            m["model"] = r.name
            m["train_time_s"] = round(r.train_time_s, 2)
            m["inference_time_s"] = round(r.inference_time_s, 3)
            rows.append(m)

        df = pd.DataFrame(rows).sort_values("wmape")
        best_model = df.iloc[0]["model"]

        # Pairwise Wilcoxon vs best model
        best_result = next(r for r in results if r.name == best_model)
        sig_tests = []
        for r in results:
            if r.name == best_model:
                continue
            test = self.significance.wilcoxon_test(
                np.abs(best_result.y_true - best_result.y_pred),
                np.abs(r.y_true - r.y_pred),
                name_a=best_model,
                name_b=r.name,
            )
            d = self.significance.cohens_d(
                best_result.y_true - best_result.y_pred,
                r.y_true - r.y_pred,
            )
            sig_tests.append({"comparison": f"{best_model} vs {r.name}", **test, **d})

        logger.info(f"Model comparison complete. Best model (WMAPE): {best_model}")
        return {
            "ranking": df.to_dict(orient="records"),
            "best_model": best_model,
            "significance_tests": sig_tests,
        }
