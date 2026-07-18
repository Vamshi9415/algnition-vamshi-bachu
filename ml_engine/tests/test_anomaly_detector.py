"""Tests for AnomalyDetector: detection correctness, leak-free features, winsorization."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.anomaly.detector import AnomalyDetector


def _series_with_spike(n: int = 60, spike_idx: int = 40, spike_val: float = 900.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rev = np.abs(rng.normal(100, 12, n))
    rev[spike_idx] = spike_val
    return pd.DataFrame({
        "channel": ["g"] * n,
        "campaign_name": ["c"] * n,
        "date": pd.date_range("2025-01-01", periods=n),
        "revenue": rev,
    })


def test_detects_obvious_spike():
    df = _series_with_spike(spike_idx=40)
    det = AnomalyDetector(window=28, threshold=4.0)
    flag = det.flag(df)
    assert bool(flag.iloc[40]), "clear 9x spike should be flagged"
    # a normal day should not be flagged
    assert not bool(flag.iloc[10])


def test_leakfree_features_do_not_depend_on_current_row():
    """Perturbing ONLY the last row's revenue must not change that row's anomaly features
    -- they are all derived from strictly-past (shifted) values."""
    df = _series_with_spike()
    det = AnomalyDetector()
    f1 = det.add_leakfree_features(df)

    df2 = df.copy()
    last = df2.index[-1]
    df2.loc[last, "revenue"] = 99_999.0
    f2 = det.add_leakfree_features(df2)

    anom_cols = [c for c in f1.columns if "_anom" in c]
    assert anom_cols, "expected anomaly feature columns"
    for c in anom_cols:
        a, b = f1.loc[last, c], f2.loc[last, c]
        if pd.isna(a):
            assert pd.isna(b), f"{c}: NaN-ness changed after perturbing today's revenue"
        else:
            assert a == pytest.approx(b), f"{c} leaked: changed when only today's revenue moved"


def test_flag_lag1_fires_day_after_spike():
    df = _series_with_spike(spike_idx=40)
    det = AnomalyDetector(window=28, threshold=4.0)
    out = det.add_leakfree_features(df)
    # the leak-free "yesterday was anomalous" flag should be set the day AFTER the spike
    assert out["revenue_anom_flag_lag1"].iloc[41] == 1
    assert out["revenue_anom_flag_lag1"].iloc[40] == 0
    # days-since resets to 0 the day after the spike, then counts up
    assert out["revenue_anom_days_since"].iloc[41] == 0
    assert out["revenue_anom_days_since"].iloc[42] == 1


def test_winsorize_caps_spike_toward_fence():
    df = _series_with_spike(spike_idx=40, spike_val=900.0)
    det = AnomalyDetector(window=28, threshold=4.0)
    cleaned, n_capped = det.winsorize_training_target(df)
    assert n_capped >= 1
    capped_val = cleaned.loc[cleaned["date"] == df.loc[40, "date"], "revenue"].iloc[0]
    assert capped_val < 900.0, "spike should be pulled down"
    assert capped_val > df["revenue"].drop(index=40).median(), "but not below the normal level"


def test_per_series_isolation():
    """A big campaign's normal level must not make a small campaign's spike invisible,
    and vice versa -- detection is per series, not global."""
    n = 60
    small = _series_with_spike(n=n, spike_idx=40, spike_val=800.0)
    big = pd.DataFrame({
        "channel": ["g"] * n, "campaign_name": ["big"] * n,
        "date": pd.date_range("2025-01-01", periods=n),
        "revenue": np.full(n, 5000.0) + np.random.default_rng(1).normal(0, 50, n),
    })
    df = pd.concat([small, big], ignore_index=True)
    det = AnomalyDetector(window=28, threshold=4.0)
    flag = det.flag(df)
    small_mask = df["campaign_name"] == "c"
    # the small campaign's 800 spike (vs its ~100 baseline) is flagged even though it is
    # far below the big campaign's normal 5000 level
    spike_row = df[small_mask].index[40]
    assert bool(flag.loc[spike_row])


def test_isolation_forest_method_runs():
    df = _series_with_spike()
    det = AnomalyDetector(method="isolation_forest", contamination=0.05)
    flag = det.flag(df)
    assert flag.sum() >= 1  # should flag something in a series with a clear spike
    # winsorization is undefined for isolation_forest -> no-op
    _, n = det.winsorize_training_target(df)
    assert n == 0
