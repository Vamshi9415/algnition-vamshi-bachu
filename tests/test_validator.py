"""Tests for DataValidator."""
import pandas as pd
import pytest
from src.validation.validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


def make_df(**overrides):
    base = {
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "channel": ["google", "google"],
        "campaign_name": ["Search_US", "Search_US"],
        "spend": [100.0, 200.0],
        "revenue": [300.0, 600.0],
        "clicks": [50, 80],
        "impressions": [1000, 2000],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_valid_dataframe_passes(validator):
    df = make_df()
    report = validator.validate(df)
    assert report.passed
    assert report.errors == []


def test_negative_spend_fails(validator):
    df = make_df(spend=[-10.0, 100.0])
    report = validator.validate(df)
    assert not report.passed
    assert any("negative spend" in e for e in report.errors)


def test_negative_revenue_fails(validator):
    df = make_df(revenue=[-1.0, 100.0])
    report = validator.validate(df)
    assert not report.passed
    assert any("negative revenue" in e for e in report.errors)


def test_missing_required_column_fails(validator):
    df = make_df()
    df = df.drop(columns=["revenue"])
    report = validator.validate(df)
    assert not report.passed
    assert any("revenue" in e for e in report.errors)


def test_empty_dataframe_fails(validator):
    df = pd.DataFrame(columns=["date", "channel", "campaign_name", "spend", "revenue"])
    report = validator.validate(df)
    assert not report.passed
