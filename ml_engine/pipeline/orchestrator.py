"""Pipeline orchestrator: wires every module together end-to-end."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from time import perf_counter

from ml_engine.ingestion.loader import CSVLoader
from ml_engine.canonical.schema import CanonicalSchemaBuilder
from ml_engine.validation.validator import DataValidator
from ml_engine.preprocessing.cleaner import DataCleaner
from ml_engine.features.feature_store import FeatureStore
from ml_engine.forecasting.baseline_models import NaiveForecaster, SeasonalNaiveForecaster
from ml_engine.forecasting.lgbm_model import LGBMForecaster
from ml_engine.forecasting.prophet_model import ProphetForecaster
from ml_engine.forecasting.ensemble import EnsembleForecaster
from ml_engine.evaluation.model_comparison import ModelComparator, ModelResult
from ml_engine.uncertainty.intervals import UncertaintyEngine
from ml_engine.budget.simulator import BudgetSimulator
from ml_engine.llm.insights import InsightGenerator
from ml_engine.evaluation.descriptive_stats import DescriptiveStats
from ml_engine.evaluation.stationarity import StationarityTester
from ml_engine.evaluation.residual_diagnostics import ResidualDiagnostics
from ml_engine.evaluation.forecast_metrics import ForecastMetrics
from ml_engine.evaluation.significance_tests import SignificanceTester
from ml_engine.evaluation.backtester import WalkForwardBacktester
from ml_engine.evaluation.explainability import SHAPExplainer
from ml_engine.evaluation.report_generator import ReportGenerator
from ml_engine.paths import config_path, output_path


class ForecastPipeline:
    """End-to-end pipeline: raw CSVs → probabilistic forecast → statistical validation → JSON."""

    def __init__(self, horizon_days: int = 60, config_dir: str | Path | None = None, skip_train: bool = False):
        self.horizon_days = horizon_days
        self.skip_train = skip_train
        config_file = Path(config_dir) if config_dir else config_path("forecasting.yaml")
        with open(config_file, encoding="utf-8") as f:
            self.fc_cfg = yaml.safe_load(f)

        # Data layer
        self.loader = CSVLoader()
        self.canonical_builder = CanonicalSchemaBuilder()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.feature_store = FeatureStore()

        # Models
        self.lgbm = LGBMForecaster(self.fc_cfg.get("models", {}).get("lightgbm", {}))
        self.prophet = ProphetForecaster(self.fc_cfg.get("models", {}).get("prophet", {}))
        self.ensemble = EnsembleForecaster(self.fc_cfg.get("ensemble", {}).get("weights"))
        self.uncertainty = UncertaintyEngine()
        self.budget_sim = BudgetSimulator()
        self.llm = InsightGenerator()

        # Evaluation
        self.desc_stats = DescriptiveStats()
        self.stationarity = StationarityTester()
        self.residuals = ResidualDiagnostics()
        self.metrics = ForecastMetrics()
        self.significance = SignificanceTester()
        self.backtester = WalkForwardBacktester(n_splits=3, test_size_days=30)
        self.shap_explainer = SHAPExplainer()
        self.reporter = ReportGenerator()
        self.model_comparator = ModelComparator()

    def run(self, filepaths: list[str]) -> dict:
        logger.info(f"Pipeline starting: {len(filepaths)} file(s)")

        # 1. Load + canonicalize + validate + clean
        loaded_files = self.loader.load_many(filepaths)
        canonical_df = self.canonical_builder.build_from_many(loaded_files)
        validation_report = self.validator.validate(canonical_df)
        validation_report.to_json()
        validation_report.to_html()
        if not validation_report.passed:
            return {"status": "validation_failed", "errors": validation_report.errors}
        cleaned_df = self.cleaner.clean(canonical_df)

        # 2. Feature engineering
        features_df = self.feature_store.build(cleaned_df, save=False)

        # 3. Descriptive statistics
        desc = self.desc_stats.compute_all(cleaned_df, ["revenue", "spend", "roas"])
        self.reporter.descriptive_stats_report(desc)

        # 4. Stationarity tests
        stat_results = []
        for campaign, grp in cleaned_df.groupby("campaign_name"):
            s = grp.sort_values("date")["revenue"].reset_index(drop=True)
            r = self.stationarity.test(s, name=campaign)
            stl = self.stationarity.stl_decompose(s)
            r["stl"] = stl
            stat_results.append(r)
        self.reporter.stationarity_report(stat_results)

        # 5. Train/test split (last 30 days holdout)
        cutoff = features_df["date"].max() - pd.Timedelta(days=30)
        train_df = features_df[features_df["date"] <= cutoff]
        test_df  = features_df[features_df["date"] >  cutoff]

        # 6. Fit models (skipped when loading a pre-trained bundle)
        lgbm_train_time = 0.0
        prophet_train_time = 0.0
        budget_train_time = 0.0
        if not self.skip_train:
            start = perf_counter()
            self.lgbm.fit(train_df)
            lgbm_train_time = perf_counter() - start

            start = perf_counter()
            self.prophet.fit(train_df)
            prophet_train_time = perf_counter() - start

            start = perf_counter()
            self.budget_sim.fit(train_df)
            budget_train_time = perf_counter() - start

        # 7. In-sample test predictions for all candidate models
        merge_keys = ["date", "channel", "campaign_name"]
        lgbm_test_preds = self.lgbm.predict(test_df)
        prophet_test_preds = self.prophet.predict_on_frame(test_df)
        ensemble_test_preds = self.ensemble.combine(lgbm_test_preds, prophet_test_preds)

        naive = NaiveForecaster()
        seasonal_naive = SeasonalNaiveForecaster()
        naive.fit(train_df)
        seasonal_naive.fit(train_df)
        naive_test_preds = naive.predict(test_df)
        seasonal_naive_test_preds = seasonal_naive.predict(test_df)

        merged_test = test_df[[*merge_keys, "revenue"]].merge(ensemble_test_preds, on=merge_keys, how="inner")

        # 8. Point + probabilistic metrics on holdout
        y_true = merged_test["revenue"].values
        y_p10  = merged_test["revenue_p10"].values
        y_p50  = merged_test["revenue_p50"].values
        y_p90  = merged_test["revenue_p90"].values
        prob_metrics = self.metrics.probabilistic_metrics(
            y_true, y_p10, y_p50, y_p90, insample=train_df["revenue"].values, seasonality=7
        )
        self.reporter.metrics_report(prob_metrics, model_name="LightGBM + Prophet Ensemble")

        # 9. Residual diagnostics
        residual_diag = self.residuals.diagnose(
            pd.Series(y_true), pd.Series(y_p50)
        )
        self.reporter.residual_report(residual_diag)

        # 10. Bootstrap CI for WMAPE
        def wmape_fn(yt, yp):
            return float(sum(abs(yt - yp)) / (sum(abs(yt)) + 1e-9) * 100)
        wmape_ci = self.significance.bootstrap_ci(wmape_fn, y_true, y_p50)
        comparison_summary = self._build_model_comparison(
            test_df,
            merge_keys,
            y_true,
            lgbm_test_preds,
            prophet_test_preds,
            ensemble_test_preds,
            naive_test_preds,
            seasonal_naive_test_preds,
            lgbm_train_time,
            prophet_train_time,
        )
        self.reporter.model_comparison_report(comparison_summary)

        # 11. SHAP explainability
        shap_result = self.shap_explainer.explain(self.lgbm, test_df)
        self.reporter.feature_importance_report(self.lgbm.feature_importance.head(10).to_dict(orient="records"))

        # 12. Walk-forward backtesting
        backtest = self.backtester.run(features_df, LGBMForecaster,
                                       model_kwargs=self.fc_cfg.get("models", {}).get("lightgbm", {}))

        # 13. Acceptance criteria
        acceptance = self._check_acceptance_criteria(prob_metrics, residual_diag, backtest)
        readiness = self.metrics.production_readiness(prob_metrics, residual_diag, backtest)
        overall_score = self.metrics.overall_score(prob_metrics, residual_diag, backtest)
        sig_tests = self._build_significance_summary(y_true, y_p50, naive_test_preds["revenue_p50"].values, seasonal_naive_test_preds["revenue_p50"].values)

        # 14. Future forecast
        future_rows = self._build_future_frame(features_df)
        lgbm_preds   = self.lgbm.predict(future_rows)
        prophet_preds = self.prophet.predict(self.horizon_days)
        forecast = self.ensemble.combine(lgbm_preds, prophet_preds)
        forecast = self.uncertainty.enrich(forecast)

        # 15. LLM insights
        exec_summary = self.llm.executive_summary(forecast)
        risk_text    = self.llm.risk_analysis(cleaned_df, forecast)

        result = self._compile_result(
            forecast, validation_report, exec_summary, risk_text,
            self.lgbm.feature_importance.head(10).to_dict(orient="records"),
            prob_metrics, residual_diag, wmape_ci, backtest, shap_result,
            stat_results, desc, acceptance, comparison_summary, sig_tests, readiness, overall_score
        )
        self._write_evaluation_summary(result)
        logger.info("Pipeline complete")
        return result

    def _check_acceptance_criteria(self, metrics: dict, residuals: dict, backtest: dict | None = None) -> dict:
        """Evaluate production-readiness against predefined thresholds."""
        criteria = {
            "wmape_below_20pct": metrics.get("wmape", 999) < 20.0,
            "picp_near_target": abs(metrics.get("coverage_gap", 1.0)) < 0.10,
            "no_residual_autocorrelation": not residuals.get("ljung_box", {}).get(
                "lag_10", {}).get("has_autocorrelation", True),
            "calibration_ok": metrics.get("calibration_status") in ["Well-calibrated", "Over-covering"],
            "stable_backtesting": bool(not backtest or backtest.get("wmape_std", 0.0) < 1.0),
        }
        criteria["production_ready"] = all(criteria.values())
        return criteria

    def _build_significance_summary(self, y_true: pd.Series | np.ndarray, ensemble_pred: pd.Series | np.ndarray,
                                    naive_pred: pd.Series | np.ndarray, seasonal_pred: pd.Series | np.ndarray) -> list[dict]:
        """Create significance tests and effect sizes against simple baselines."""

        ensemble_errors = np.abs(np.array(y_true) - np.array(ensemble_pred))
        naive_errors = np.abs(np.array(y_true) - np.array(naive_pred))
        seasonal_errors = np.abs(np.array(y_true) - np.array(seasonal_pred))

        tests = [
            {"comparison": "Ensemble vs Naive", **self.significance.wilcoxon_test(ensemble_errors, naive_errors, "Ensemble", "Naive")},
            {"comparison": "Ensemble vs Seasonal Naive", **self.significance.wilcoxon_test(ensemble_errors, seasonal_errors, "Ensemble", "Seasonal Naive")},
        ]
        for item, other in ((naive_errors, seasonal_errors), (ensemble_errors, naive_errors)):
            if len(item) >= 10 and len(other) >= 10:
                tests.append(self.significance.cohens_d(item, other))
                tests.append(self.significance.cliffs_delta(item, other))
        return tests

    def _build_model_comparison(
        self,
        test_df: pd.DataFrame,
        merge_keys: list[str],
        y_true: np.ndarray,
        lgbm_test_preds: pd.DataFrame,
        prophet_test_preds: pd.DataFrame,
        ensemble_test_preds: pd.DataFrame,
        naive_test_preds: pd.DataFrame,
        seasonal_naive_test_preds: pd.DataFrame,
        lgbm_train_time: float,
        prophet_train_time: float,
    ) -> dict:
        """Build a ranked comparison table across ensemble and baseline forecasters."""

        test_lookup = test_df[[*merge_keys, "revenue"]]
        lookup = lambda preds: test_lookup.merge(preds, on=merge_keys, how="inner")

        comparisons = [
            ("Ensemble", lookup(ensemble_test_preds), lgbm_train_time + prophet_train_time, 0.0),
            ("LightGBM", lookup(lgbm_test_preds), lgbm_train_time, 0.0),
            ("Prophet", lookup(prophet_test_preds), prophet_train_time, 0.0),
            ("Naive", lookup(naive_test_preds), 0.0, 0.0),
            ("Seasonal Naive", lookup(seasonal_naive_test_preds), 0.0, 0.0),
        ]

        results: list[ModelResult] = []
        for name, frame, train_time, inference_time in comparisons:
            if frame.empty:
                continue
            results.append(
                ModelResult(
                    name=name,
                    y_true=frame["revenue"].to_numpy(),
                    y_pred=frame["revenue_p50"].to_numpy(),
                    p10=frame["revenue_p10"].to_numpy() if "revenue_p10" in frame.columns else None,
                    p50=frame["revenue_p50"].to_numpy() if "revenue_p50" in frame.columns else None,
                    p90=frame["revenue_p90"].to_numpy() if "revenue_p90" in frame.columns else None,
                    train_time_s=train_time,
                    inference_time_s=inference_time,
                )
            )

        return self.model_comparator.compare(results)

    def _build_future_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        last_date = features_df["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=self.horizon_days, freq="D")
        templates = features_df.sort_values("date").groupby(["channel", "campaign_name"]).last().reset_index()
        frames = []
        for _, row in templates.iterrows():
            for d in future_dates:
                r = row.copy()
                r["date"] = d
                frames.append(r)
        future_df = pd.DataFrame(frames)
        from ml_engine.features.calendar_features import CalendarFeatureGenerator
        cal = CalendarFeatureGenerator()
        future_df = cal.generate(future_df)
        return future_df

    def _compile_result(self, forecast, validation_report, exec_summary, risk_text,
                        feature_importance, prob_metrics, residual_diag, wmape_ci,
                        backtest, shap_result, stat_results, desc_stats, acceptance,
                        model_comparison, significance_tests, production_readiness, overall_score) -> dict:
        forecast_records = [
            {
                "date": str(row["date"])[:10],
                "channel": str(row.get("channel", "")),
                "campaign_name": str(row.get("campaign_name", "")),
                "revenue_p10": round(float(row.get("revenue_p10", 0)), 2),
                "revenue_p50": round(float(row.get("revenue_p50", 0)), 2),
                "revenue_p90": round(float(row.get("revenue_p90", 0)), 2),
                "confidence": str(row.get("confidence_label", "")),
            }
            for _, row in forecast.iterrows()
        ]
        return {
            "status": "success",
            "horizon_days": self.horizon_days,
            "pipeline_variant": "standard_ensemble",
            "validation": validation_report.to_dict(),
            "summary": {
                "total_revenue_p10": round(float(forecast["revenue_p10"].sum()), 2),
                "total_revenue_p50": round(float(forecast["revenue_p50"].sum()), 2),
                "total_revenue_p90": round(float(forecast["revenue_p90"].sum()), 2),
                "campaigns_forecasted": int(forecast["campaign_name"].nunique()),
                "channels": forecast["channel"].unique().tolist(),
            },
            "forecast": forecast_records,
            "evaluation": {
                "holdout_metrics": prob_metrics,
                "wmape_bootstrap_ci": wmape_ci,
                "residual_diagnostics": residual_diag,
                "backtesting": backtest,
                "acceptance_criteria": acceptance,
                "model_comparison": model_comparison,
                "significance_tests": significance_tests,
                "production_readiness": production_readiness,
                "overall_model_score": overall_score,
            },
            "stationarity": [{
                "series": r.get("series"),
                "verdict": r.get("verdict"),
                "stl": r.get("stl", {}),
            } for r in stat_results],
            "descriptive_stats": desc_stats,
            "feature_importance": feature_importance,
            "feature_importance_top_10": feature_importance,
            "shap": shap_result,
            "ai_insights": {
                "executive_summary": exec_summary,
                "risk_analysis": risk_text,
            },
        }

    def _write_evaluation_summary(self, result: dict) -> None:
        summary_path = output_path("evaluation_summary.json")
        forecast_path = output_path("forecast_details.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        detailed_forecast = result.get("forecast", [])
        forecast_path.write_text(
            json.dumps(detailed_forecast, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        compact_result = {key: value for key, value in result.items() if key != "forecast"}
        compact_result["forecast_count"] = len(detailed_forecast)
        compact_result["forecast_artifact"] = str(forecast_path)
        summary_path.write_text(
            json.dumps(compact_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Forecast details written: {forecast_path}")
        logger.info(f"Evaluation summary written: {summary_path}")

