"""Unit tests for budget simulator."""
import pandas as pd
import pytest
from src.budget.simulator import BudgetSimulator


@pytest.fixture
def canonical_df():
    dates = pd.date_range("2024-10-01", periods=30)
    return pd.DataFrame({
        "date": dates,
        "channel": ["google"] * 30,
        "campaign_name": ["Search"] * 30,
        "spend": [100.0] * 30,
        "revenue": [400.0] * 30,
    })


def test_fit_computes_roas(canonical_df):
    sim = BudgetSimulator()
    sim.fit(canonical_df)
    assert "google" in sim._channel_roas
    assert abs(sim._channel_roas["google"] - 4.0) < 0.01


def test_simulate_positive_delta(canonical_df):
    sim = BudgetSimulator()
    sim.fit(canonical_df)
    result = sim.simulate("google", 20.0, 10000.0)
    assert result["delta_revenue_estimated"] > 0
    assert result["new_revenue_p50"] > 10000.0


def test_simulate_explanation_present(canonical_df):
    sim = BudgetSimulator()
    sim.fit(canonical_df)
    result = sim.simulate("google", 20.0, 10000.0)
    assert "explanation" in result
    assert len(result["explanation"]) > 20
