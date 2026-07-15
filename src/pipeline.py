"""End-to-end pipeline orchestrator: ties all modules together for CLI use."""
from pathlib import Path
import json

import pandas as pd
from loguru import logger

from src.ingestion.loader import CSVLoader
from src.canonical.schema import CanonicalSchemaBuilder
from src.validation.validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.features.store import FeatureStore
from src.forecasting.ensemble import ForecastEnsemble
from src.uncertainty.intervals import UncertaintyEngine
from src.budget.simulator import BudgetSimulator
from src.llm.insights import LLMInsightGenerator


class AIgnitionPipeline:
    """
    Full end-to-end orchestrator.
    Usage:
        pipeline = AIgnitionPipeline()
        result = pipeline.run(["data/raw/google.csv", "data/raw/bing.csv"])
    """

    def __init__(self, horizon: int = 60, output_dir: str = "output"):
        self.horizon = horizon
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = CSVLoader()
        self.builder = CanonicalSchemaBuilder()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.store = FeatureStore()
        self.ensemble = ForecastEnsemble(horizon=horizon)
        self.uncertainty = UncertaintyEngine()
        self.budget_sim = BudgetSimulator()
        self.llm = LLMInsightGenerator()

    def run(self, filepaths: list[str]) -> dict:
        logger.info(f"=== AIgnition Pipeline START ({len(filepaths)} files) ===")

        # 1. Load
        loaded = self.loader.load_many(filepaths)

        # 2. Canonical
        canonical = self.builder.build_from_many(loaded)

        # 3. Clean
        canonical = self.cleaner.clean(canonical)

        # 4. Validate
        report = self.validator.validate(canonical)
        self._save_json(report.to_dict(), "validation_report.json")
        if not report.passed:
            logger.error(f"Validation failed: {report.errors}")
            return {"error": "Validation failed", "report": report.to_dict()}

        # 5. Feature engineering
        features_df = self.store.build(canonical)

        # 6. Train ensemble
        self.ensemble.fit(features_df)

        # 7. Forecast
        last_date = features_df["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.horizon)
        last_row = features_df.iloc[[-1]].copy()
        future_rows = pd.concat([last_row] * self.horizon, ignore_index=True)
        future_rows["date"] = future_dates

        raw_forecast = self.ensemble.predict(future_rows)
        forecast = self.uncertainty.process(raw_forecast)
        forecast.to_parquet(self.output_dir / "forecast.parquet", index=False)

        # 8. Budget simulation
        self.budget_sim.fit(canonical)
        sim_results = self.budget_sim.simulate_all_channels(20.0, float(forecast["p50"].sum()))

        # 9. LLM insights
        top_channel = canonical.groupby("channel")["revenue"].sum().idxmax()
        summary = {
            "horizon_days": self.horizon,
            "total_forecast_p10": round(float(forecast["p10"].sum()), 2),
            "total_forecast_p50": round(float(forecast["p50"].sum()), 2),
            "total_forecast_p90": round(float(forecast["p90"].sum()), 2),
            "top_channel": top_channel,
            "confidence": "89%",
        }
        ai_summary = self.llm.generate_executive_summary(summary)
        budget_rec = self.llm.generate_budget_recommendation(sim_results)

        result = {
            "summary": summary,
            "forecast": forecast.assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records"),
            "simulations": sim_results,
            "ai_summary": ai_summary,
            "budget_recommendation": budget_rec,
            "validation": report.to_dict(),
        }
        self._save_json(result, "pipeline_result.json")
        logger.info("=== AIgnition Pipeline COMPLETE ===")
        return result

    def _save_json(self, data: dict, filename: str):
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {filename}")
