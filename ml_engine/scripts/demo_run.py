"""End-to-end demo script using the real sample CSVs from ml_engine/data/raw/."""
import sys
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def main():
    from ml_engine.pipeline.orchestrator import ForecastPipeline

    csv_files = list(Path("ml_engine/data/raw").glob("*.csv"))
    if not csv_files:
        # Fall back to root-level CSVs if present (moved)
        csv_files = [
            f for f in Path(".").glob("*.csv")
            if any(p in f.name for p in ["google", "meta", "bing"])
        ]

    if not csv_files:
        print("[demo_run] No CSV files found. Place CSVs in ml_engine/data/raw/ first.")
        sys.exit(1)

    print(f"[demo_run] Using {len(csv_files)} CSV(s): {[f.name for f in csv_files]}")

    pipeline = ForecastPipeline(horizon_days=60)
    result = pipeline.run([str(f) for f in csv_files])

    if result.get("status") == "success":
        df = pd.DataFrame(result["forecast"])
        print(f"\n[demo_run] Forecast rows: {len(df)}")
        print(df[["date", "channel", "campaign_name", "revenue_p10", "revenue_p50", "revenue_p90"]].head(10).to_string())
        summary = result.get("summary", {})
        print(f"\nRevenue P50 (total 60d): ${summary.get('total_revenue_p50', 0):,.0f}")
        print(f"WMAPE: {result.get('evaluation', {}).get('holdout_metrics', {}).get('wmape', 'N/A')}%")
        print(f"Production Ready: {result.get('evaluation', {}).get('acceptance_criteria', {}).get('production_ready', 'N/A')}")
    else:
        print(f"[demo_run] Pipeline error: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()

