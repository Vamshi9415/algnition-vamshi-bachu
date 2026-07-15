"""Tests for FeatureStore pipeline."""
import pandas as pd
import pytest
from ml_engine.features.feature_store import FeatureStore


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "channel": "google",
        "campaign_name": "Search_US",
        "campaign_type": "Search",
        "spend": [100 + i for i in range(60)],
        "revenue": [300 + i * 2 for i in range(60)],
        "clicks": [50 + i for i in range(60)],
        "impressions": [1000 + i * 10 for i in range(60)],
        "conversions": [5.0 + i * 0.1 for i in range(60)],
        "roas": [3.0] * 60,
        "currency": "USD",
        "campaign_id": None,
        "daily_budget": 200.0,
    })
    return df


def test_feature_store_adds_calendar_cols(sample_df):
    fs = FeatureStore(output_dir="/tmp/test_features")
    out = fs.build(sample_df, save=False)
    assert "is_weekend" in out.columns
    assert "day_of_week" in out.columns
    assert "sin_day_of_year" in out.columns


def test_feature_store_adds_kpi_cols(sample_df):
    fs = FeatureStore(output_dir="/tmp/test_features")
    out = fs.build(sample_df, save=False)
    assert "ctr" in out.columns
    assert "roas" in out.columns
    assert "cpa" in out.columns


def test_no_future_leakage_in_lag_cols(sample_df):
    fs = FeatureStore(output_dir="/tmp/test_features")
    out = fs.build(sample_df, save=False)
    # First row of lag columns must be NaN (shifted)
    lag_col = "revenue_lag1"
    assert lag_col in out.columns
    # NOTE: groupby(...).first() skips NaNs by default, so it cannot be used to check
    # the true first row of each group. Use nth(0) to get the first row by position.
    first_val = out.sort_values("date").groupby("campaign_name", as_index=False).nth(0)[lag_col]
    assert first_val.isna().all(), "Lag-1 first value must be NaN (no leakage)"

