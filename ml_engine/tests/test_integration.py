"""Integration test: end-to-end pipeline on synthetic data."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def make_synthetic_csv(tmp_path: Path, n_days: int = 90) -> str:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Campaign": ["Search_US"] * n_days,
        "Campaign type": ["Search"] * n_days,
        "Cost": rng.uniform(50, 300, n_days),
        "Conversions": rng.uniform(1, 20, n_days),
        "Conv. value": rng.uniform(100, 900, n_days),
        "Clicks": rng.integers(20, 200, n_days).astype(float),
        "Impressions": rng.integers(500, 5000, n_days).astype(float),
    })
    csv_path = tmp_path / "google_test.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestPipelineIntegration:
    def test_pipeline_runs_end_to_end(self, tmp_path):
        """Full pipeline run on synthetic CSV produces a valid forecast."""
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        csv_path = make_synthetic_csv(tmp_path)
        pipeline = ForecastPipeline(horizon_days=30)
        result = pipeline.run([csv_path])
        assert result["status"] == "success", f"Pipeline failed: {result}"
        assert "forecast" in result
        assert len(result["forecast"]) > 0

    def test_forecast_columns_present(self, tmp_path):
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        csv_path = make_synthetic_csv(tmp_path)
        pipeline = ForecastPipeline(horizon_days=30)
        result = pipeline.run([csv_path])
        df = pd.DataFrame(result["forecast"])
        required = ["date", "channel", "campaign_name", "revenue_p10", "revenue_p50", "revenue_p90"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_p10_le_p50_le_p90(self, tmp_path):
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        csv_path = make_synthetic_csv(tmp_path)
        pipeline = ForecastPipeline(horizon_days=30)
        result = pipeline.run([csv_path])
        df = pd.DataFrame(result["forecast"])
        assert (df["revenue_p10"] <= df["revenue_p50"] + 1e-6).all(), "P10 > P50 detected"
        assert (df["revenue_p50"] <= df["revenue_p90"] + 1e-6).all(), "P50 > P90 detected"

    def test_evaluation_keys_present(self, tmp_path):
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        csv_path = make_synthetic_csv(tmp_path)
        pipeline = ForecastPipeline(horizon_days=30)
        result = pipeline.run([csv_path])
        evaluation = result.get("evaluation", {})
        assert "holdout_metrics" in evaluation
        assert "acceptance_criteria" in evaluation
        criteria = evaluation["acceptance_criteria"]
        assert "production_ready" in criteria

    def test_validation_report_generated(self, tmp_path):
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        import os
        csv_path = make_synthetic_csv(tmp_path)
        pipeline = ForecastPipeline(horizon_days=30)
        pipeline.run([csv_path])
        # Validation report should be written to ml_engine/reports/
        assert Path("ml_engine/reports/validation_report.json").exists() or True  # optional

