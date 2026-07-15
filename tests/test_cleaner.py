"""Tests for DataCleaner."""
import pandas as pd
import pytest
from src.preprocessing.cleaner import DataCleaner


@pytest.fixture
def cleaner():
    return DataCleaner()


def make_dirty_df():
    return pd.DataFrame({
        "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"],
        "channel": ["google", "GOOGLE", "google", "google"],
        "campaign_name": ["  search_us  ", "search_us", "search_us", "search_us"],
        "spend": [100.0, None, 50.0, 50.0],
        "revenue": [300.0, None, 150.0, 150.0],
        "clicks": [50, 20, 30, 30],
        "impressions": [1000, 500, 600, 600],
        "conversions": [5.0, None, 2.0, 2.0],
        "roas": [3.0, None, 3.0, 3.0],
    })


def test_removes_duplicates(cleaner):
    df = make_dirty_df()
    cleaned = cleaner.clean(df)
    assert not cleaned.duplicated(subset=["date", "channel", "campaign_name"]).any()


def test_fills_missing_spend(cleaner):
    df = make_dirty_df()
    cleaned = cleaner.clean(df)
    assert cleaned["spend"].isna().sum() == 0


def test_sorted_by_date(cleaner):
    df = make_dirty_df()
    cleaned = cleaner.clean(df)
    assert cleaned["date"].is_monotonic_increasing


def test_roas_recomputed(cleaner):
    df = make_dirty_df()
    cleaned = cleaner.clean(df)
    mask = cleaned["spend"] > 0
    expected = cleaned.loc[mask, "revenue"] / cleaned.loc[mask, "spend"]
    assert (cleaned.loc[mask, "roas"].round(2) == expected.round(2)).all()
