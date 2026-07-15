"""Unit tests for uncertainty engine."""
import pandas as pd
import pytest
from ml_engine.uncertainty.intervals import UncertaintyEngine


def test_interval_ordering_enforced():
    engine = UncertaintyEngine()
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "revenue_p10": [300.0, 200.0, 400.0, 100.0, 500.0],
        "revenue_p50": [250.0, 350.0, 350.0, 350.0, 450.0],  # some out of order
        "revenue_p90": [200.0, 400.0, 500.0, 200.0, 600.0],
    })
    result = engine.process(df)
    assert (result["revenue_p10"] <= result["revenue_p50"]).all()
    assert (result["revenue_p50"] <= result["revenue_p90"]).all()


def test_forecast_label_present():
    engine = UncertaintyEngine()
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "revenue_p10": [100.0, 200.0, 300.0],
        "revenue_p50": [150.0, 250.0, 350.0],
        "revenue_p90": [200.0, 300.0, 400.0],
    })
    result = engine.process(df)
    assert "confidence_label" in result.columns
    assert result["confidence_label"].notna().all()

