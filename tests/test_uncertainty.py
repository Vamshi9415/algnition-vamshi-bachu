"""Unit tests for uncertainty engine."""
import pandas as pd
import pytest
from src.uncertainty.intervals import UncertaintyEngine


def test_interval_ordering_enforced():
    engine = UncertaintyEngine()
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "p10": [300.0, 200.0, 400.0, 100.0, 500.0],
        "p50": [250.0, 350.0, 350.0, 350.0, 450.0],  # some out of order
        "p90": [200.0, 400.0, 500.0, 200.0, 600.0],
    })
    result = engine.process(df)
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


def test_forecast_label_present():
    engine = UncertaintyEngine()
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "p10": [100.0, 200.0, 300.0],
        "p50": [150.0, 250.0, 350.0],
        "p90": [200.0, 300.0, 400.0],
    })
    result = engine.process(df)
    assert "forecast_label" in result.columns
    assert result["forecast_label"].notna().all()
