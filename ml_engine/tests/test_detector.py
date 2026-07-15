"""Tests for source detection engine."""
import pandas as pd
import pytest
from ml_engine.ingestion.detector import SourceDetector, Channel


@pytest.fixture
def detector():
    return SourceDetector()


def test_detect_google_from_filename(detector, tmp_path):
    f = tmp_path / "google_ads_data.csv"
    f.write_text("Date,Campaign,Cost\n2024-01-01,Test,10")
    assert detector.detect(str(f)) == Channel.GOOGLE


def test_detect_meta_from_filename(detector, tmp_path):
    f = tmp_path / "meta_report.csv"
    f.write_text("Reporting starts,Campaign name,Amount spent (USD)\n2024-01-01,Test,10")
    assert detector.detect(str(f)) == Channel.META


def test_detect_microsoft_from_filename(detector, tmp_path):
    f = tmp_path / "bing_campaign_stats.csv"
    f.write_text("TimePeriod,CampaignId,Spend\n2024-01-01,123,10")
    assert detector.detect(str(f)) == Channel.MICROSOFT


def test_detect_google_from_columns(detector):
    df = pd.DataFrame(columns=["Cost", "Conv. value", "Campaign type"])
    assert detector.detect_from_df(df) == Channel.GOOGLE


def test_detect_meta_from_columns(detector):
    df = pd.DataFrame(columns=["Amount spent (USD)", "Purchases conversion value", "Reporting starts"])
    assert detector.detect_from_df(df) == Channel.META


def test_detect_unknown(detector):
    df = pd.DataFrame(columns=["foo", "bar"])
    assert detector.detect_from_df(df) == Channel.UNKNOWN

