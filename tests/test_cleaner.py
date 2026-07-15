"""Unit tests for the data cleaning pipeline."""
import pandas as pd
import pytest
from src.preprocessing.cleaner import DataCleaner


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-02"],  # dupe on day 2
        "channel": [" google ", "Google", "google"],
        "campaign_name": ["search_campaign", "Search Campaign", "search_campaign"],
        "spend": [100.0, None, 80.0],
        "revenue": [500.0, 600.0, 600.0],
        "clicks": [200, 220, 220],
        "impressions": [2000, 2200, 2200],
        "conversions": [5.0, 6.0, 6.0],
    })


def test_duplicates_removed(raw_df):
    cleaner = DataCleaner()
    result = cleaner.clean(raw_df)
    assert result.duplicated(subset=["date", "channel", "campaign_name"]).sum() == 0


def test_missing_spend_filled(raw_df):
    cleaner = DataCleaner()
    result = cleaner.clean(raw_df)
    assert result["spend"].isna().sum() == 0


def test_roas_computed(raw_df):
    cleaner = DataCleaner()
    result = cleaner.clean(raw_df)
    assert "roas" in result.columns
    assert (result["roas"] >= 0).all()


def test_sorted_chronologically(raw_df):
    cleaner = DataCleaner()
    result = cleaner.clean(raw_df)
    assert result["date"].is_monotonic_increasing
