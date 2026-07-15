"""Integration test: full pipeline on the real Bing CSV."""
import pytest
from pathlib import Path

BING_CSV = "bing_campaign_stats.csv"


def test_pipeline_runs_on_bing_csv():
    if not Path(BING_CSV).exists():
        pytest.skip(f"{BING_CSV} not found in repo root")

    from src.pipeline.orchestrator import ForecastPipeline
    pipeline = ForecastPipeline(horizon_days=30)
    result = pipeline.run([BING_CSV])

    assert result["status"] == "success", f"Pipeline failed: {result.get('errors')}"
    assert result["summary"]["total_revenue_p50"] >= 0
    assert len(result["forecast"]) > 0
    for row in result["forecast"][:5]:
        assert row["revenue_p10"] <= row["revenue_p50"] <= row["revenue_p90"], \
            f"Quantile ordering violated: {row}"
