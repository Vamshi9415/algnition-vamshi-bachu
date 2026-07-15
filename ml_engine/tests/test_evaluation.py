"""Tests for statistical evaluation modules."""
import numpy as np
import pandas as pd
import pytest

from ml_engine.evaluation.descriptive_stats import DescriptiveStats
from ml_engine.evaluation.forecast_metrics import ForecastMetrics
from ml_engine.evaluation.significance_tests import SignificanceTester
from ml_engine.evaluation.residual_diagnostics import ResidualDiagnostics


@pytest.fixture
def sample_series():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(1000, 200, 100))


@pytest.fixture
def y_true_pred():
    rng = np.random.default_rng(42)
    y_true = rng.normal(1000, 200, 100)
    y_pred = y_true + rng.normal(0, 50, 100)  # small noise
    return y_true, y_pred


# --- DescriptiveStats ---

class TestDescriptiveStats:
    def test_compute_returns_all_keys(self, sample_series):
        ds = DescriptiveStats()
        df = pd.DataFrame({"revenue": sample_series})
        result = ds.compute(df, "revenue")
        for key in ["n", "mean", "median", "std", "skewness", "kurtosis", "cv", "iqr"]:
            assert key in result, f"Missing key: {key}"

    def test_cv_is_positive(self, sample_series):
        ds = DescriptiveStats()
        df = pd.DataFrame({"revenue": sample_series.abs()})
        result = ds.compute(df, "revenue")
        assert result["cv"] > 0


# --- ForecastMetrics ---

class TestForecastMetrics:
    def test_perfect_predictions_give_zero_mae(self):
        fm = ForecastMetrics()
        y = np.array([100.0, 200.0, 300.0])
        m = fm.point_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0

    def test_wmape_range(self, y_true_pred):
        y_true, y_pred = y_true_pred
        fm = ForecastMetrics()
        m = fm.point_metrics(y_true, y_pred)
        assert 0 <= m["wmape"] < 100

    def test_picp_perfect_coverage(self):
        fm = ForecastMetrics()
        y = np.array([100.0, 200.0, 300.0])
        coverage = fm.picp(y, y - 10, y + 10)
        assert coverage == 1.0

    def test_picp_zero_coverage(self):
        fm = ForecastMetrics()
        y = np.array([100.0, 200.0, 300.0])
        coverage = fm.picp(y, y + 1000, y + 2000)  # intervals completely miss
        assert coverage == 0.0

    def test_pinball_loss_correct_quantile(self):
        fm = ForecastMetrics()
        y = np.array([100.0, 200.0, 300.0])
        # Perfect prediction at median should give 0
        loss = fm.pinball_loss(y, y, 0.5)
        assert loss == 0.0

    def test_probabilistic_metrics_calibration_status(self, y_true_pred):
        y_true, y_pred = y_true_pred
        fm = ForecastMetrics()
        # Wide intervals → over-covering
        result = fm.probabilistic_metrics(y_true, y_pred - 1e6, y_pred, y_pred + 1e6)
        assert result["calibration_status"] == "Over-covering"


# --- SignificanceTester ---

class TestSignificanceTester:
    def test_wilcoxon_same_model_not_significant(self):
        st = SignificanceTester()
        errors = np.abs(np.random.default_rng(42).normal(0, 100, 50))
        result = st.wilcoxon_test(errors, errors)
        assert not result["significant"]

    def test_wilcoxon_clearly_different_models(self):
        st = SignificanceTester()
        rng = np.random.default_rng(42)
        good = np.abs(rng.normal(10, 5, 100))
        bad  = np.abs(rng.normal(100, 20, 100))
        result = st.wilcoxon_test(good, bad)
        assert result["significant"]

    def test_bootstrap_ci_width_positive(self, y_true_pred):
        y_true, y_pred = y_true_pred
        st = SignificanceTester()
        def mae_fn(yt, yp): return float(np.mean(np.abs(yt - yp)))
        ci = st.bootstrap_ci(mae_fn, y_true, y_pred, n_bootstrap=200)
        assert ci["ci_upper"] > ci["ci_lower"]
        assert ci["ci_lower"] >= 0

    def test_cohens_d_same_distribution_near_zero(self):
        st = SignificanceTester()
        errors = np.random.default_rng(42).normal(0, 100, 100)
        d = st.cohens_d(errors, errors)
        assert abs(d["cohens_d"]) < 0.01


# --- ResidualDiagnostics ---

class TestResidualDiagnostics:
    def test_unbiased_residuals_detected(self):
        rd = ResidualDiagnostics()
        y = pd.Series(np.random.default_rng(42).normal(1000, 100, 100))
        result = rd.diagnose(y, y)  # perfect predictions
        assert result["mean"] == 0.0
        assert result["bias_interpretation"] == "Unbiased"

    def test_under_forecast_bias_detected(self):
        rd = ResidualDiagnostics()
        y_true = pd.Series([100.0] * 50)
        y_pred = pd.Series([80.0] * 50)  # systematic under-forecast
        result = rd.diagnose(y_true, y_pred)
        assert result["bias_interpretation"] == "Under-forecast"

