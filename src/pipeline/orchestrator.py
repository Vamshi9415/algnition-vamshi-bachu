"""Pipeline orchestrator: wires every module together end-to-end."""
import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.ingestion.loader import CSVLoader
from src.canonical.schema import CanonicalSchemaBuilder
from src.validation.validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.features.feature_store import FeatureStore
from src.forecasting.lgbm_model import LGBMForecaster
from src.forecasting.prophet_model import ProphetForecaster
from src.forecasting.ensemble import EnsembleForecaster
from src.uncertainty.intervals import UncertaintyEngine
from src.budget.simulator import BudgetSimulator
from src.llm.insights import InsightGenerator


class ForecastPipeline:
    """End-to-end pipeline: raw CSVs → probabilistic forecast JSON."""

    def __init__(self, horizon_days: int = 60, config_dir: str = "config"):
        self.horizon_days = horizon_days
        with open(f"{config_dir}/forecasting.yaml") as f:
            self.fc_cfg = yaml.safe_load(f)

        self.loader = CSVLoader()
        self.canonical_builder = CanonicalSchemaBuilder()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.feature_store = FeatureStore()
        self.lgbm = LGBMForecaster(self.fc_cfg.get("models", {}).get("lightgbm", {}))
        self.prophet = ProphetForecaster(self.fc_cfg.get("models", {}).get("prophet", {}))
        self.ensemble = EnsembleForecaster(self.fc_cfg.get("ensemble", {}).get("weights"))
        self.uncertainty = UncertaintyEngine()
        self.budget_sim = BudgetSimulator()
        self.llm = InsightGenerator()

    def run(self, filepaths: list[str]) -> dict:
        logger.info(f"Pipeline starting: {len(filepaths)} file(s)")

        # 1. Load
        loaded_files = self.loader.load_many(filepaths)

        # 2. Canonicalize
        canonical_df = self.canonical_builder.build_from_many(loaded_files)

        # 3. Validate
        validation_report = self.validator.validate(canonical_df)
        if not validation_report.passed:
            logger.error(f"Validation failed: {validation_report.errors}")
            return {"status": "validation_failed", "errors": validation_report.errors}

        # 4. Clean
        cleaned_df = self.cleaner.clean(canonical_df)

        # 5. Feature engineering
        features_df = self.feature_store.build(cleaned_df, save=False)

        # 6. Train-test split (last 30 days as holdout)
        cutoff = features_df["date"].max() - pd.Timedelta(days=30)
        train_df = features_df[features_df["date"] <= cutoff]
        test_df  = features_df[features_df["date"] >  cutoff]

        # 7. Fit models
        self.lgbm.fit(train_df)
        self.prophet.fit(train_df)
        self.budget_sim.fit(train_df)

        # 8. Generate forecast future dates
        future_rows = self._build_future_frame(features_df)

        # 9. Predict
        lgbm_preds   = self.lgbm.predict(future_rows)
        prophet_preds = self.prophet.predict(self.horizon_days)

        # 10. Ensemble
        forecast = self.ensemble.combine(lgbm_preds, prophet_preds)

        # 11. Uncertainty
        forecast = self.uncertainty.enrich(forecast)

        # 12. LLM insights
        exec_summary = self.llm.executive_summary(forecast)
        risk_text    = self.llm.risk_analysis(cleaned_df, forecast)

        # 13. Feature importance (top 10)
        fi = self.lgbm.feature_importance.head(10).to_dict(orient="records")

        # 14. Compile response
        result = self._compile_result(forecast, validation_report, exec_summary, risk_text, fi)
        logger.info("Pipeline complete")
        return result

    def _build_future_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create a DataFrame of future rows reusing the last known feature state."""
        last_date = features_df["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=self.horizon_days, freq="D")

        # Repeat last row of each campaign as the future template
        templates = features_df.sort_values("date").groupby(["channel", "campaign_name"]).last().reset_index()
        frames = []
        for _, row in templates.iterrows():
            for d in future_dates:
                r = row.copy()
                r["date"] = d
                frames.append(r)

        future_df = pd.DataFrame(frames)
        # Re-generate calendar features for actual future dates
        from src.features.calendar_features import CalendarFeatureGenerator
        cal = CalendarFeatureGenerator()
        future_df = cal.generate(future_df)
        return future_df

    def _compile_result(self, forecast, validation_report, exec_summary, risk_text, feature_importance) -> dict:
        forecast_records = []
        for _, row in forecast.iterrows():
            forecast_records.append({
                "date": str(row["date"])[:10],
                "channel": str(row.get("channel", "")),
                "campaign_name": str(row.get("campaign_name", "")),
                "revenue_p10": round(float(row.get("revenue_p10", 0)), 2),
                "revenue_p50": round(float(row.get("revenue_p50", 0)), 2),
                "revenue_p90": round(float(row.get("revenue_p90", 0)), 2),
                "confidence": str(row.get("confidence_label", "")),
            })

        return {
            "status": "success",
            "horizon_days": self.horizon_days,
            "validation": validation_report.to_dict(),
            "summary": {
                "total_revenue_p10": round(float(forecast["revenue_p10"].sum()), 2),
                "total_revenue_p50": round(float(forecast["revenue_p50"].sum()), 2),
                "total_revenue_p90": round(float(forecast["revenue_p90"].sum()), 2),
                "campaigns_forecasted": int(forecast["campaign_name"].nunique()),
                "channels": forecast["channel"].unique().tolist(),
            },
            "forecast": forecast_records,
            "feature_importance": feature_importance,
            "ai_insights": {
                "executive_summary": exec_summary,
                "risk_analysis": risk_text,
            },
        }
