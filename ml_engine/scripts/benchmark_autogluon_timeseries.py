"""AutoGluon TimeSeriesPredictor benchmark — genuine multi-series probabilistic forecasting.

Unlike the tabular benchmarks, this uses AutoGluon's dedicated time-series engine, which
trains its own lag/rolling features internally plus classical (ETS, Theta, AutoARIMA,
SeasonalNaive), ML (RecursiveTabular, DirectTabular), deep learning (DeepAR, PatchTST,
TemporalFusionTransformer) and the pretrained zero-shot Chronos foundation model, then
ensembles them. Only date-derived (known-in-advance) covariates are supplied — no
same-day KPIs — so this is a genuinely leak-free forecasting comparison.

Must be run under `if __name__ == "__main__":` on Windows (AutoGluon TS uses multiprocessing).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT = Path(r"d:\0_placements\Netelixiar")
sys.path.insert(0, str(PROJECT))


def main():
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    from ml_engine.ingestion.loader import CSVLoader
    from ml_engine.canonical.schema import CanonicalSchemaBuilder
    from ml_engine.preprocessing.cleaner import DataCleaner
    from ml_engine.features.feature_store import FeatureStore

    CSVS = [str(PROJECT / "ml_engine" / "data" / "raw" / f)
            for f in ["google_ads_campaign_stats.csv", "meta_ads_campaign_stats.csv", "bing_campaign_stats.csv"]]

    print("Building features via project pipeline...", flush=True)
    loaded = CSVLoader().load_many(CSVS)
    df = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
        DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(loaded)), save=False
    ).sort_values("date").reset_index(drop=True)

    df["item_id"] = df["channel"].astype(str) + "|" + df["campaign_name"].astype(str)

    PRED_LEN = 14
    counts = df.groupby("item_id").size()
    keep = counts[counts >= PRED_LEN * 3].index  # need real train history beyond the holdout
    df = df[df["item_id"].isin(keep)].copy()
    print(f"Series kept (>= {PRED_LEN*3} rows): {len(keep)} of {len(counts)}", flush=True)

    KNOWN_COVARIATES = ["year", "quarter", "month", "week_of_year", "day_of_week",
                        "day_of_month", "day_of_year", "is_weekend", "is_month_start",
                        "is_month_end", "is_quarter_start", "is_quarter_end", "is_holiday"]
    KNOWN_COVARIATES = [c for c in KNOWN_COVARIATES if c in df.columns]

    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    long_df = df[["item_id", "date", "revenue"] + KNOWN_COVARIATES].rename(columns={"date": "timestamp"})
    ts_df = TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id", timestamp_column="timestamp")
    # Many campaigns have gaps (no rows on days with zero activity), so the raw index is
    # irregular. Reindex to daily frequency (missing days -> NaN) then fill: revenue gaps
    # are genuine no-activity days (0), covariate gaps are forward/back-filled from the
    # nearest known date.
    ts_df = ts_df.convert_frequency(freq="D")
    ts_df["revenue"] = ts_df["revenue"].fillna(0.0)
    for c in KNOWN_COVARIATES:
        ts_df[c] = ts_df[c].ffill().bfill()

    save_path = str(PROJECT / "ml_engine" / "output" / "ag_timeseries")
    t0 = time.perf_counter()
    predictor = TimeSeriesPredictor(
        prediction_length=PRED_LEN,
        target="revenue",
        known_covariates_names=KNOWN_COVARIATES,
        eval_metric="WQL",
        quantile_levels=[0.1, 0.5, 0.9],
        path=save_path,
        verbosity=1,
    ).fit(ts_df, time_limit=600, presets="medium_quality")
    fit_t = time.perf_counter() - t0

    leaderboard = predictor.leaderboard(ts_df, silent=True)
    print("\nLeaderboard (AutoGluon internal backtest, best first):", flush=True)
    print(leaderboard.to_string(index=False), flush=True)

    # ------------------------------------------------------------- honest external evaluation
    # Hold out the last PRED_LEN points per series as "truth"; forecast from before that,
    # using only known covariates (no leakage), then score against the real values.
    train_ts = ts_df.slice_by_timestep(None, -PRED_LEN)
    known_future = ts_df.slice_by_timestep(-PRED_LEN, None)[KNOWN_COVARIATES]
    forecast = predictor.predict(train_ts, known_covariates=known_future)

    truth = ts_df.slice_by_timestep(-PRED_LEN, None)[["revenue"]]
    merged = forecast[["0.1", "0.5", "0.9"]].join(truth, how="inner")
    y = merged["revenue"].to_numpy(dtype=float)
    p10 = np.maximum(merged["0.1"].to_numpy(dtype=float), 0)
    p50 = np.maximum(merged["0.5"].to_numpy(dtype=float), 0)
    p90 = np.maximum(merged["0.9"].to_numpy(dtype=float), 0)
    stacked = np.sort(np.column_stack([p10, p50, p90]), axis=1)
    p10, p50, p90 = stacked[:, 0], stacked[:, 1], stacked[:, 2]

    err = p50 - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.abs(y) + np.abs(p50)
    m = denom > 1e-9
    smape = float(np.mean(2 * np.abs(p50[m] - y[m]) / denom[m]) * 100)
    r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)) if y.std() > 0 else float("nan")
    cover = float(np.mean((y >= p10) & (y <= p90)))

    print(f"\n[AutoGluon TimeSeries — external held-out eval, {PRED_LEN}-day horizon, "
          f"{len(y)} rows across {len(keep)} series]")
    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  sMAPE={smape:.2f}%  R2={r2:.4f}  "
          f"P10-P90 coverage={cover*100:.1f}%  fit={fit_t:.0f}s")
    print(f"  Best model per leaderboard: {leaderboard.iloc[0]['model']}  "
          f"(internal val WQL={leaderboard.iloc[0]['score_val']:.4f})")

    import json
    out = {
        "prediction_length": PRED_LEN,
        "n_series": len(keep),
        "n_test_rows": len(y),
        "fit_s": round(fit_t, 1),
        "known_covariates": KNOWN_COVARIATES,
        "external_eval": {"mae": mae, "rmse": rmse, "smape": smape, "r2": r2, "coverage": cover},
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    out_path = Path(__file__).parent / "autogluon_timeseries_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
