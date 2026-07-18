"""Statistical significance tests: Wilcoxon, Friedman, Nemenyi, Bootstrap CI.

Purpose:
    Compare forecast models and quantify whether differences are meaningful.
Responsibilities:
    Compute non-parametric tests, effect sizes, and bootstrap confidence intervals.
Inputs:
    Paired error arrays or metric functions with true/predicted samples.
Outputs:
    Test statistics, p-values, effect sizes, and confidence intervals.
Assumptions:
    Inputs are aligned and represent the same horizon or fold.
Limitations:
    These routines assess statistical differences, not business significance.
"""
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

try:
    from scipy.stats import friedmanchisquare
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


class SignificanceTester:
    """Compares models using rigorous statistical tests."""

    def wilcoxon_test(self, errors_a: np.ndarray, errors_b: np.ndarray,
                      name_a: str = "ModelA", name_b: str = "ModelB") -> dict:
        """
        Wilcoxon Signed-Rank Test: non-parametric paired comparison.
        H0: no difference in error distributions.
        """
        a, b = np.abs(np.array(errors_a)), np.abs(np.array(errors_b))
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if n < 10:
            return {"error": "Need at least 10 paired samples"}
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        winner = name_a if np.median(a) < np.median(b) else name_b
        return {
            "test": "Wilcoxon Signed-Rank",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05),
            "winner": winner if p < 0.05 else "No significant difference",
            "median_error_a": round(float(np.median(a)), 4),
            "median_error_b": round(float(np.median(b)), 4),
            "interpretation": (
                f"{winner} is significantly better (p={p:.4f})" if p < 0.05
                else f"No significant difference between {name_a} and {name_b} (p={p:.4f})"
            ),
        }

    def friedman_test(self, error_matrix: dict) -> dict:
        """
        Friedman Test for 3+ models.
        error_matrix: {model_name: [abs_errors_per_fold]}
        """
        model_names = list(error_matrix.keys())
        if len(model_names) < 3:
            return {"error": "Friedman test requires 3+ models"}
        arrays = [np.array(v) for v in error_matrix.values()]
        n = min(len(a) for a in arrays)
        arrays = [a[:n] for a in arrays]
        stat, p = friedmanchisquare(*arrays)
        return {
            "test": "Friedman",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05),
            "models_compared": model_names,
            "interpretation": (
                "Significant differences exist between models (proceed to post-hoc)"
                if p < 0.05 else
                "No significant differences across models"
            ),
        }

    def bootstrap_ci(
        self,
        metric_fn,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> dict:
        """Bootstrap confidence interval for any scalar metric function."""
        rng = np.random.default_rng(42)
        n = len(y_true)
        boot_scores = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            score = metric_fn(y_true[idx], y_pred[idx])
            boot_scores.append(score)
        boot_scores = np.array(boot_scores)
        alpha = (1 - ci) / 2
        lower = float(np.percentile(boot_scores, alpha * 100))
        upper = float(np.percentile(boot_scores, (1 - alpha) * 100))
        return {
            "mean": round(float(np.mean(boot_scores)), 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "ci_pct": int(ci * 100),
            "n_bootstrap": n_bootstrap,
        }

    def cohens_d(self, errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
        """Cohen's d effect size."""
        a, b = np.abs(errors_a), np.abs(errors_b)
        pooled_std = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
        d = float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-9))
        magnitude = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        return {
            "cohens_d": round(d, 4),
            "magnitude": magnitude,
            "interpretation": f"Effect size is {magnitude} (d={d:.3f})",
        }

    def cliffs_delta(self, errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
        """Cliff's Delta effect size for paired comparisons."""

        a, b = np.abs(np.array(errors_a)), np.abs(np.array(errors_b))
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if n == 0:
            return {"error": "Need at least one paired sample"}

        greater = 0
        lower = 0
        for x in a:
            greater += int(np.sum(x > b))
            lower += int(np.sum(x < b))

        delta = float((greater - lower) / (n * n))
        magnitude = (
            "negligible" if abs(delta) < 0.147 else
            "small" if abs(delta) < 0.33 else
            "medium" if abs(delta) < 0.474 else
            "large"
        )
        return {
            "cliffs_delta": round(delta, 4),
            "magnitude": magnitude,
            "interpretation": f"Cliff's delta is {magnitude} (δ={delta:.3f})",
        }

