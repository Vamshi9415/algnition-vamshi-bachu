"""Unit tests for source detection engine."""
import pandas as pd
import pytest
from src.ingestion.detector import SourceDetector, Channel


def test_detect_microsoft_by_columns():
    detector = SourceDetector()
    df = pd.DataFrame(columns=["TimePeriod", "CampaignId", "DailyBudget", "Revenue", "Spend"])
    channel = detector.detect_from_df(df)
    assert channel == Channel.MICROSOFT


def test_detect_meta_by_columns():
    detector = SourceDetector()
    df = pd.DataFrame(columns=["Reporting starts", "Amount spent (USD)", "Purchases conversion value"])
    channel = detector.detect_from_df(df)
    assert channel == Channel.META


def test_detect_google_by_columns():
    detector = SourceDetector()
    df = pd.DataFrame(columns=["Cost", "Conv. value", "Campaign type", "Date"])
    channel = detector.detect_from_df(df)
    assert channel == Channel.GOOGLE


def test_unknown_channel():
    detector = SourceDetector()
    df = pd.DataFrame(columns=["foo", "bar", "baz"])
    channel = detector.detect_from_df(df)
    assert channel == Channel.UNKNOWN
