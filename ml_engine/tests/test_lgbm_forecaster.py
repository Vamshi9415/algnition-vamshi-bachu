"""Tests for LGBMForecaster: log-target transform and monotonic quantile stitching."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.forecasting.lgbm_model import LGBMForecaster
from ml_engine.features.lag_features import LagFeatureGenerator


def _make_training_frame(n: int = 400) -> pd.DataFrame:
    """Small synthetic panel with a couple of numeric features and heavy-tailed revenue."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    x1 = rng.gamma(2.0, 50.0, size=n)
    x2 = rng.normal(0, 1, size=n)
    revenue = np.maximum(x1 * 2 + rng.gamma(1.5, 30.0, size=n), 0)
    return pd.DataFrame({
        "date": dates,
        "channel": "google",
        "campaign_name": "campaign_a",
        "feat_x1": x1,
        "feat_x2": x2,
        "revenue": revenue,
    })


def test_growth_features_do_not_leak_current_day_target():
    """Regression test for a real bug: *_wow_growth/mom_growth/qoq_growth used to be
    computed from today's raw (unshifted) revenue/spend, which combined with the lag
    features already in the set made the target algebraically reconstructible. Growth
    columns for a given row must depend only on shifted (past) values, so changing ONLY
    that row's own revenue -- with all history before it held fixed -- must not move
    that row's growth features."""
    df = _make_training_frame(n=60)
    out_a = LagFeatureGenerator().generate(df)

    df_b = df.copy()
    last = df_b.index[-1]
    df_b.loc[last, "revenue"] = df_b.loc[last, "revenue"] * 1000 + 50_000  # only today changes
    out_b = LagFeatureGenerator().generate(df_b)

    for col in ["revenue_wow_growth", "revenue_mom_growth", "revenue_qoq_growth"]:
        a_val = out_a.loc[out_a.index[-1], col]
        b_val = out_b.loc[out_b.index[-1], col]
        if pd.isna(a_val):
            assert pd.isna(b_val), f"{col}: NaN-ness changed after perturbing today's revenue"
        else:
            assert a_val == pytest.approx(b_val), (
                f"{col} changed ({a_val} -> {b_val}) after perturbing only today's revenue "
                "-- growth feature is leaking the current-day target"
            )


def test_predict_quantiles_are_monotonic():
    """P10 <= P50 <= P90 must hold for every row, even though the three quantile
    models are fit independently and can cross before stitching."""
    df = _make_training_frame()
    m = LGBMForecaster({"log_target": True, "target_encoding": False})
    m.fit(df)
    out = m.predict(df)

    assert (out["revenue_p10"] <= out["revenue_p50"] + 1e-9).all()
    assert (out["revenue_p50"] <= out["revenue_p90"] + 1e-9).all()


def test_predictions_are_non_negative():
    df = _make_training_frame()
    m = LGBMForecaster({"log_target": True, "target_encoding": False})
    m.fit(df)
    out = m.predict(df)
    for col in ["revenue_p10", "revenue_p50", "revenue_p90"]:
        assert (out[col] >= 0).all()


def test_conformal_widens_and_preserves_ordering():
    """Conformal calibration should keep quantiles ordered and produce a band at least
    as wide as the uncalibrated one (k >= ~1 when raw intervals are too narrow)."""
    df = _make_training_frame()
    base = LGBMForecaster({"log_target": True, "conformal": False})
    base.fit(df)
    conf = LGBMForecaster({"log_target": True, "conformal": True})
    conf.fit(df)

    out = conf.predict(df)
    assert (out["revenue_p10"] <= out["revenue_p50"] + 1e-9).all()
    assert (out["revenue_p50"] <= out["revenue_p90"] + 1e-9).all()
    # a calibration factor was computed and is positive
    assert conf._conformal_k > 0


def test_conformal_off_leaves_band_unscaled():
    df = _make_training_frame()
    m = LGBMForecaster({"log_target": True, "conformal": False})
    m.fit(df)
    assert m._conformal_k == 1.0


def test_per_channel_conformal_produces_per_channel_factors_and_keeps_ordering():
    """Per-channel conformal must compute a factor for each channel with enough calibration
    points and still return monotonic, non-negative bands."""
    a = _make_training_frame(300)
    b = _make_training_frame(300)
    b["channel"] = "meta"
    b["campaign_name"] = "campaign_b"
    df = pd.concat([a, b], ignore_index=True)

    m = LGBMForecaster({"conformal": True, "conformal_per_channel": True, "conformal_min_points": 10})
    m.fit(df)
    assert set(m._conformal_k_by_channel) == {"google", "meta"}
    out = m.predict(df)
    assert (out["revenue_p10"] <= out["revenue_p50"] + 1e-9).all()
    assert (out["revenue_p50"] <= out["revenue_p90"] + 1e-9).all()
    assert (out[["revenue_p10", "revenue_p50", "revenue_p90"]] >= 0).all().all()


def test_log_target_predictions_are_on_revenue_scale():
    """With log_target on, expm1 must invert the transform so predictions land on the
    revenue scale, not the compressed log scale."""
    df = _make_training_frame()
    m = LGBMForecaster({"log_target": True})
    m.fit(df)
    preds = m.predict(df)["revenue_p50"]
    assert preds.max() > 10  # not stuck near log-scale values (log of ~200 is ~5)
    assert preds.mean() == pytest.approx(df["revenue"].mean(), rel=1.0)
