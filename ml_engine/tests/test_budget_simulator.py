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
        from ml_engine.budget.simulator import BudgetSimulator
        sim = BudgetSimulator()
        sim.fit(self._make_df())

    def _make_forecast(self, n=10):
        dates = pd.date_range("2024-04-01", periods=n, freq="D")
        return pd.DataFrame({
            "date": dates, "channel": "google", "campaign_name": "Test",
            "revenue_p10": 100.0, "revenue_p50": 200.0, "revenue_p90": 300.0,
        })

    def test_simulate_returns_delta(self):
        from ml_engine.budget.simulator import BudgetSimulator
        df = self._make_df()
        sim = BudgetSimulator()
        sim.fit(df)
        forecast = self._make_forecast()
        simulated = sim.simulate(forecast, {"Test": 0.20})
        summary = sim.what_if_summary(forecast, simulated)
        assert "delta_revenue" in summary

    def test_simulate_positive_spend_increase(self):
        from ml_engine.budget.simulator import BudgetSimulator
        df = self._make_df()
        sim = BudgetSimulator()
        sim.fit(df)
        forecast = self._make_forecast()
        simulated = sim.simulate(forecast, {"Test": 0.50})
        summary = sim.what_if_summary(forecast, simulated)
        assert summary["delta_revenue"] >= 0, "Positive spend increase should not decrease revenue"

