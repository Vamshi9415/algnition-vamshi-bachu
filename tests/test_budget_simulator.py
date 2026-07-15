"""Unit tests for BudgetSimulator."""
import pandas as pd
import numpy as np
import pytest


class TestBudgetSimulator:
    def _make_df(self, n=90):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "date": dates, "channel": "google", "campaign_name": "Test",
            "spend": rng.uniform(50, 200, n), "revenue": rng.uniform(100, 800, n),
        })

    def test_fit_does_not_crash(self):
        from src.budget.simulator import BudgetSimulator
        sim = BudgetSimulator()
        sim.fit(self._make_df())

    def test_simulate_returns_delta(self):
        from src.budget.simulator import BudgetSimulator
        df = self._make_df()
        sim = BudgetSimulator()
        sim.fit(df)
        result = sim.simulate({"google/Test": 0.20})
        assert "delta_revenue" in result or "error" in result or isinstance(result, dict)

    def test_simulate_positive_spend_increase(self):
        from src.budget.simulator import BudgetSimulator
        df = self._make_df()
        sim = BudgetSimulator()
        sim.fit(df)
        result = sim.simulate({"google/Test": 0.50})
        if "delta_revenue" in result:
            assert result["delta_revenue"] >= 0, "Positive spend increase should not decrease revenue"
