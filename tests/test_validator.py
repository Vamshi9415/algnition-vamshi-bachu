"""Unit tests for the data validator."""
import pandas as pd
import pytest
from src.validation.validator import DataValidator


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "channel": ["microsoft"] * 10,
        "campaign_name": ["Camp A"] * 10,
        "spend": [100.0] * 10,
        "revenue": [500.0] * 10,
        "clicks": [200] * 10,
        "impressions": [2000] * 10,
        "conversions": [5.0] * 10,
    })


def test_valid_dataframe_passes(valid_df):
    v = DataValidator()
    report = v.validate(valid_df)
    assert report.passed
    assert len(report.errors) == 0


def test_missing_required_column_fails():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "channel": ["google"] * 3})
    v = DataValidator()
    report = v.validate(df)
    assert not report.passed
    assert any("Missing required columns" in e for e in report.errors)


def test_negative_spend_fails(valid_df):
    valid_df.loc[0, "spend"] = -10.0
    v = DataValidator()
    report = v.validate(valid_df)
    assert not report.passed
