"""AutoGluon TabularPredictor benchmark — point regression + native quantile regression.

Uses the exact same project pipeline and time split as the other benchmarks so results
are directly comparable to reports/model_benchmark.md and reports/recipe_verification.md.
Tests AutoGluon on both the leak-free (honest forecasting) feature set and the full
same-day-KPI (nowcast) set, mirroring the earlier ablation.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT = Path(r"d:\0_placements\Netelixiar")
sys.path.insert(0, str(PROJECT))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from ml_engine.ingestion.loader import CSVLoader
from ml_engine.canonical.schema import CanonicalSchemaBuilder
from ml_engine.preprocessing.cleaner import DataCleaner
from ml_engine.features.feature_store import FeatureStore
from ml_engine.forecasting.lgbm_model import FEATURE_COLS_EXCLUDE

CSVS = [str(PROJECT / "ml_engine" / "data" / "raw" / f)
        for f in ["google_ads_campaign_stats.csv", "meta_ads_campaign_stats.csv", "bing_campaign_stats.csv"]]

print("Building features via project pipeline...", flush=True)
loaded = CSVLoader().load_many(CSVS)
df = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(loaded)), save=False
).sort_values("date").reset_index(drop=True)

exclude = set(FEATURE_COLS_EXCLUDE)
all_numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
LEAKY = {"clicks", "impressions", "conversions", "spend", "video_views",
         "ctr", "cpc", "cpm", "cpa", "cvr", "rpc", "roas", "aov",
         "spend_x_month", "spend_x_holiday", "conversions_value"}
LEAKY |= {c for c in all_numeric if c.endswith(("_wow_growth", "_mom_growth", "_qoq_growth"))}
leakfree_cols = [c for c in all_numeric if c not in LEAKY]

dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_df = df[df["date"] < cutoff].copy()
test_df = df[df["date"] >= cutoff].copy()
y_te = test_df["revenue"].fillna(0).values

print(f"train={len(train_df)} test={len(test_df)} leak-free-features={len(leakfree_cols)} "
      f"full-features={len(all_numeric)}", flush=True)

def smape(y, p):
    d = np.abs(y) + np.abs(p)
    m = d > 1e-9
    return float(np.mean(2 * np.abs(p[m] - y[m]) / d[m]) * 100)

def score(p):
    p = np.maximum(np.asarray(p, dtype=float), 0)
    err = p - y_te
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "smape": smape(y_te, p),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y_te - y_te.mean()) ** 2))}

from autogluon.tabular import TabularPredictor, TabularDataset

results = []

def run_point(label, cols, time_limit):
    tr = TabularDataset(train_df[cols + ["revenue"]].fillna(0))
    te = TabularDataset(test_df[cols].fillna(0))
    save_path = str(PROJECT / "ml_engine" / "output" / f"ag_{label.replace(' ', '_').lower()}")
    t0 = time.perf_counter()
    predictor = TabularPredictor(
        label="revenue", problem_type="regression", eval_metric="mean_absolute_error",
        path=save_path, verbosity=0,
    ).fit(tr, time_limit=time_limit, presets="medium_quality")
    fit_t = time.perf_counter() - t0
    p = predictor.predict(te).to_numpy()
    s = score(p)
    s.update({"label": label, "n_features": len(cols), "fit_s": round(fit_t, 1),
              "leaderboard_top": predictor.leaderboard(silent=True).iloc[0]["model"]})
    print(f"  {label:<38} MAE={s['mae']:>8.2f} RMSE={s['rmse']:>8.2f} "
          f"sMAPE={s['smape']:>6.2f}% R2={s['r2']:>7.4f} fit={fit_t:.0f}s "
          f"best_model={s['leaderboard_top']}", flush=True)
    results.append(s)
    return predictor

print("\n[AutoGluon Tabular — point regression]", flush=True)
run_point("Leak-free (honest forecasting)", leakfree_cols, time_limit=240)
run_point("Full production set (nowcast)", all_numeric, time_limit=240)

# ---------------------------------------------------------------- native quantile regression
print("\n[AutoGluon Tabular — native quantile regression P10/P50/P90, leak-free]", flush=True)
tr = TabularDataset(train_df[leakfree_cols + ["revenue"]].fillna(0))
te = TabularDataset(test_df[leakfree_cols].fillna(0))
save_path = str(PROJECT / "ml_engine" / "output" / "ag_quantile_leakfree")
t0 = time.perf_counter()
qpredictor = TabularPredictor(
    label="revenue", problem_type="quantile", quantile_levels=[0.1, 0.5, 0.9],
    path=save_path, verbosity=0,
).fit(tr, time_limit=240, presets="medium_quality")
fit_t = time.perf_counter() - t0
qpred = qpredictor.predict(te)
p10 = np.maximum(qpred[0.1].to_numpy(), 0)
p50 = np.maximum(qpred[0.5].to_numpy(), 0)
p90 = np.maximum(qpred[0.9].to_numpy(), 0)
stacked = np.sort(np.column_stack([p10, p50, p90]), axis=1)
p10, p50, p90 = stacked[:, 0], stacked[:, 1], stacked[:, 2]
s = score(p50)
cover = float(np.mean((y_te >= p10) & (y_te <= p90)))
s.update({"label": "Quantile AutoGluon (leak-free)", "n_features": len(leakfree_cols),
          "fit_s": round(fit_t, 1), "coverage": cover,
          "leaderboard_top": qpredictor.leaderboard(silent=True).iloc[0]["model"]})
print(f"  Quantile AutoGluon (leak-free)      MAE={s['mae']:>8.2f} RMSE={s['rmse']:>8.2f} "
      f"sMAPE={s['smape']:>6.2f}% R2={s['r2']:>7.4f} coverage={cover*100:.1f}% "
      f"fit={fit_t:.0f}s best_model={s['leaderboard_top']}", flush=True)
results.append(s)

out_path = Path(__file__).parent / "autogluon_tabular_results.json"
out_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved {out_path}", flush=True)
