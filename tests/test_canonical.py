"""Unit tests for canonical schema builder."""
import pandas as pd
import pytest
from src.canonical.schema import CanonicalSchemaBuilder, CANONICAL_COLUMNS
from src.ingestion.detector import Channel


@pytest.fixture
def bing_df():
    return pd.DataFrame({
        "CampaignId": ["123", "123"],
        "TimePeriod": ["2024-11-01", "2024-11-02"],
        "Revenue": [500.0, 750.0],
        "Spend": [50.0, 60.0],
        "Clicks": [120, 140],
        "Impressions": [1000, 1200],
        "Conversions": [5.0, 7.0],
        "CampaignType": ["Search", "Search"],
        "DailyBudget": [10.0, 10.0],
        "CampaignName": ["Search_TM_01", "Search_TM_01"],
    })


def test_canonical_columns_present(bing_df):
    builder = CanonicalSchemaBuilder()
    result = builder.build(bing_df, Channel.MICROSOFT)
    for col in ["date", "channel", "campaign_name", "spend", "revenue", "roas"]:
        assert col in result.columns


def test_roas_computed(bing_df):
    builder = CanonicalSchemaBuilder()
    result = builder.build(bing_df, Channel.MICROSOFT)
    assert (result["roas"] == result["revenue"] / result["spend"]).all()


def test_channel_tagged(bing_df):
    builder = CanonicalSchemaBuilder()
    result = builder.build(bing_df, Channel.MICROSOFT)
    assert (result["channel"] == "microsoft").all()
