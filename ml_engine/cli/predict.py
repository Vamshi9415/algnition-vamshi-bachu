"""CLI entry point: train (or load) model, generate forecast, write OUTPUT_PATH CSV."""
import argparse
import sys
from pathlib import Path

from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description="AIgnition Forecast Studio — prediction CLI")
    p.add_argument("--data-dir",    required=True, help="Directory with raw ad-platform CSVs")
    p.add_argument("--model-path",  required=True, help="Path to save/load serialised model bundle")
    p.add_argument("--output-path", required=True, help="Output CSV path for predictions")
    p.add_argument("--horizon-days", type=int, default=60, help="Forecast horizon in days")
    p.add_argument("--skip-train",  type=int, default=0, help="1 = skip training, load MODEL_PATH")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir    = Path(args.data_dir)
    model_path  = Path(args.model_path)
    output_path = Path(args.output_path)
    horizon     = args.horizon_days
    skip_train  = bool(args.skip_train)

    # ------------------------------------------------------------------
    # 1. Discover CSV files
    # ------------------------------------------------------------------
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        sys.exit(1)
    logger.info(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    # ------------------------------------------------------------------
    # 2. Load or train model bundle
    # ------------------------------------------------------------------
    if skip_train and model_path.exists():
        logger.info(f"Loading pre-trained model bundle from {model_path}")
        from ml_engine.model_store.serializer import ModelSerializer
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        bundle = ModelSerializer.load(str(model_path))
        pipeline = ForecastPipeline(horizon_days=horizon, skip_train=True)
        pipeline.lgbm = bundle["lgbm"]
        pipeline.prophet = bundle["prophet"]
        pipeline.budget_sim = bundle["budget_sim"]
    else:
        logger.info("Training model from scratch ...")
        from ml_engine.pipeline.orchestrator import ForecastPipeline
        pipeline = ForecastPipeline(horizon_days=horizon)

    # ------------------------------------------------------------------
    # 3. Run the pipeline
    # ------------------------------------------------------------------
    result = pipeline.run([str(f) for f in csv_files])

    if result.get("status") != "success":
        logger.error(f"Pipeline failed: {result}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Save model bundle
    # ------------------------------------------------------------------
    if not skip_train:
        from ml_engine.model_store.serializer import ModelSerializer
        ModelSerializer.save(
            path=str(model_path),
            lgbm=pipeline.lgbm,
            prophet=pipeline.prophet,
            budget_sim=pipeline.budget_sim,
        )
        logger.info(f"Model bundle saved to {model_path}")

    # ------------------------------------------------------------------
    # 5. Write output CSV
    # ------------------------------------------------------------------
    import pandas as pd
    forecast_records = result["forecast"]
    df_out = pd.DataFrame(forecast_records)

    # Add evaluation summary columns for judge inspection
    metrics = result.get("evaluation", {}).get("holdout_metrics", {})
    df_out["pipeline_wmape"]       = metrics.get("wmape", None)
    df_out["pipeline_picp"]        = metrics.get("picp_80pct_interval", None)
    df_out["production_ready"]     = result.get("evaluation", {}).get(
        "acceptance_criteria", {}).get("production_ready", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(str(output_path), index=False)
    logger.info(f"Output CSV written: {output_path} ({len(df_out)} rows)")

    # ------------------------------------------------------------------
    # 6. Print summary to stdout (judges can grep this)
    # ------------------------------------------------------------------
    summary = result.get("summary", {})
    print("\n===== FORECAST SUMMARY =====")
    print(f"  Campaigns forecasted : {summary.get('campaigns_forecasted')}")
    print(f"  Channels             : {summary.get('channels')}")
    print(f"  Revenue P50 (total)  : ${summary.get('total_revenue_p50', 0):,.0f}")
    print(f"  Revenue P10 (total)  : ${summary.get('total_revenue_p10', 0):,.0f}")
    print(f"  Revenue P90 (total)  : ${summary.get('total_revenue_p90', 0):,.0f}")
    print(f"  WMAPE (holdout)      : {metrics.get('wmape', 'N/A')}%")
    print(f"  PICP (80% interval)  : {metrics.get('picp_80pct_interval', 'N/A')}")
    print(f"  Production ready     : {result.get('evaluation', {}).get('acceptance_criteria', {}).get('production_ready')}")
    print(f"  Output CSV           : {output_path}")
    print("============================\n")


if __name__ == "__main__":
    main()

