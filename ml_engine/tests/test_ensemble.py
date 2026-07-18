"""Tests for the standard forecasting ensemble."""

import pandas as pd
import pytest

from ml_engine.forecasting.ensemble import EnsembleForecaster


def _make_forecast_frame(value_p10: float, value_p50: float, value_p90: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "channel": ["google"],
            "campaign_name": ["campaign_a"],
            "revenue_p10": [value_p10],
            "revenue_p50": [value_p50],
            "revenue_p90": [value_p90],
        }
    )


def test_two_way_ensemble_blends_available_sources():
    ensemble = EnsembleForecaster(weights={"lgbm": 0.4, "prophet": 0.6})

    lgbm = _make_forecast_frame(10.0, 20.0, 30.0)
    prophet = _make_forecast_frame(20.0, 40.0, 60.0)

    combined = ensemble.combine(lgbm, prophet)

    assert combined.loc[0, "revenue_p10"] == pytest.approx(16.0)
    assert combined.loc[0, "revenue_p50"] == pytest.approx(32.0)
    assert combined.loc[0, "revenue_p90"] == pytest.approx(48.0)
