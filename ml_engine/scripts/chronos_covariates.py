"""Does 'good feature engineering' help Chronos zero-shot? Empirical head-to-head.

Chronos is univariate; the only way to inject features is a covariate_regressor over
KNOWN covariates (features known for the future horizon). For multi-step forecasting the
useful lag/rolling/anomaly features are NOT known ahead (they need future target values),
so only calendar features qualify. This tests whether that helps at all:

  1. Chronos2 zero-shot, no covariates
  2. Chronos2 + LightGBM covariate_regressor over calendar known-covariates
  3. (reference) DirectTabular -- a pure feature-using model, same covariates
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
    logger.remove(); logger.add(sys.stderr, level="ERROR")

    from ml_engine.ingestion.loader import CSVLoader
    from ml_engine.canonical.schema import CanonicalSchemaBuilder
    from ml_engine.preprocessing.cleaner import DataCleaner
    from ml_engine.features.feature_store import FeatureStore
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    CSVS = [str(PROJECT / "ml_engine" / "data" / "raw" / f)
            for f in ["google_ads_campaign_stats.csv", "meta_ads_campaign_stats.csv", "bing_campaign_stats.csv"]]
    df = FeatureStore(anomaly_features=False).build(
        DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(CSVLoader().load_many(CSVS))), save=False
    ).sort_values("date").reset_index(drop=True)
    df["item_id"] = df["channel"].astype(str) + "|" + df["campaign_name"].astype(str)

    PRED_LEN = 14
    counts = df.groupby("item_id").size()
    keep = counts[counts >= PRED_LEN * 3].index
    df = df[df["item_id"].isin(keep)].copy()

    # KNOWN covariates: calendar only (the only features knowable for the future horizon)
    KNOWN = [c for c in ["year","quarter","month","week_of_year","day_of_week","day_of_month",
                         "day_of_year","is_weekend","is_month_start","is_month_end",
                         "is_quarter_start","is_quarter_end","is_holiday"] if c in df.columns]

    long_df = df[["item_id","date","revenue"]+KNOWN].rename(columns={"date":"timestamp"})
    ts = TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id", timestamp_column="timestamp")
    ts = ts.convert_frequency(freq="D")
    ts["revenue"] = ts["revenue"].fillna(0.0)
    for c in KNOWN:
        ts[c] = ts[c].ffill().bfill()

    hyperparameters = {
        "Chronos2": [
            {"ag_args": {"name_suffix": "ZeroShot"}},
            {"covariate_regressor": "GBM", "ag_args": {"name_suffix": "PlusFeatures"}},
        ],
        "DirectTabular": [{}],  # reference: pure feature-using model
    }

    t0 = time.perf_counter()
    predictor = TimeSeriesPredictor(
        prediction_length=PRED_LEN, target="revenue",
        known_covariates_names=KNOWN, eval_metric="WQL",
        quantile_levels=[0.1, 0.5, 0.9], path=str(PROJECT/"ml_engine"/"output"/"ag_chronos_cov"),
        verbosity=1,
    ).fit(ts, hyperparameters=hyperparameters, time_limit=900)
    fit_t = time.perf_counter() - t0

    lb = predictor.leaderboard(ts, silent=True)
    print("\nLeaderboard (internal backtest, lower WQL better):", flush=True)
    print(lb[["model","score_test","score_val","fit_time_marginal"]].to_string(index=False), flush=True)

    # external held-out eval per model
    train_ts = ts.slice_by_timestep(None, -PRED_LEN)
    known_future = ts.slice_by_timestep(-PRED_LEN, None)[KNOWN]
    truth = ts.slice_by_timestep(-PRED_LEN, None)[["revenue"]]

    def smape(y, p):
        d = np.abs(y)+np.abs(p); m = d > 1e-9
        return float(np.mean(2*np.abs(p[m]-y[m])/d[m])*100)

    print("\nExternal held-out eval (14-day horizon, per model):", flush=True)
    print(f"{'model':<28} {'MAE':>9} {'RMSE':>9} {'sMAPE':>8} {'R2':>8} {'cover':>7}", flush=True)
    results = {}
    for model in lb["model"].tolist():
        try:
            fc = predictor.predict(train_ts, known_covariates=known_future, model=model)
            merged = fc[["0.1","0.5","0.9"]].join(truth, how="inner")
            y = merged["revenue"].to_numpy(float)
            p10,p50,p90 = (np.maximum(merged[q].to_numpy(float),0) for q in ["0.1","0.5","0.9"])
            st = np.sort(np.column_stack([p10,p50,p90]),axis=1); p10,p50,p90 = st[:,0],st[:,1],st[:,2]
            err = p50-y
            mae=float(np.mean(np.abs(err))); rmse=float(np.sqrt(np.mean(err**2)))
            r2=float(1-np.sum(err**2)/np.sum((y-y.mean())**2)); cover=float(np.mean((y>=p10)&(y<=p90)))
            results[model]=dict(mae=mae,rmse=rmse,smape=smape(y,p50),r2=r2,cover=cover)
            print(f"{model:<28} {mae:9.2f} {rmse:9.2f} {smape(y,p50):7.1f}% {r2:8.4f} {cover*100:6.1f}%", flush=True)
        except Exception as e:
            print(f"{model:<28} FAILED: {str(e)[:60]}", flush=True)

    import json
    (Path(__file__).parent/"chronos_covariates_results.json").write_text(
        json.dumps({"fit_s":round(fit_t,1),"known_covariates":KNOWN,
                    "leaderboard":lb[["model","score_test","score_val"]].to_dict("records"),
                    "external":results}, indent=2, default=str))
    print(f"\nfit_time={fit_t:.0f}s  Saved chronos_covariates_results.json", flush=True)


if __name__ == "__main__":
    main()
