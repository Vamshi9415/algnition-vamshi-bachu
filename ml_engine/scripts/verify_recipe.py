"""End-to-end verification of the recipe applied to the production LGBMForecaster.

Uses the REAL production class (ml_engine.forecasting.lgbm_model.LGBMForecaster),
built through the REAL feature store, on the same time split as the benchmarks.
Compares OLD behaviour (log_target=off, target_encoding=off) vs NEW defaults.

To keep the comparison honest (forecasting, not nowcasting), the same-day KPI
columns that leak the target are excluded from BOTH runs, so any delta is due to
the recipe, not to leakage.
"""
import sys
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
from ml_engine.forecasting import lgbm_model
from ml_engine.forecasting.lgbm_model import LGBMForecaster

CSVS = [str(PROJECT / "ml_engine" / "data" / "raw" / f)
        for f in ["google_ads_campaign_stats.csv", "meta_ads_campaign_stats.csv", "bing_campaign_stats.csv"]]

print("Building features through production pipeline...", flush=True)
loaded = CSVLoader().load_many(CSVS)
df = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(loaded)), save=False
).sort_values("date").reset_index(drop=True)

# Confirm the two new engineered features actually appeared from the feature store
for f in ["days_since_series_start", "rev_lag1_over_lag7"]:
    print(f"  feature-store produced '{f}': {f in df.columns}", flush=True)

dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_df = df[df["date"] < cutoff].copy()
test_df = df[df["date"] >= cutoff].copy()
y_te = test_df["revenue"].fillna(0).values

# Exclude same-day KPI leakers so this is genuine forecasting for BOTH runs.
LEAKY = {"clicks", "impressions", "conversions", "spend", "video_views",
         "ctr", "cpc", "cpm", "cpa", "cvr", "rpc", "roas", "aov",
         "spend_x_month", "spend_x_holiday", "conversions_value"}
orig_exclude = list(lgbm_model.FEATURE_COLS_EXCLUDE)
lgbm_model.FEATURE_COLS_EXCLUDE = orig_exclude + [c for c in LEAKY if c in df.columns]

def smape(y, p):
    d = np.abs(y) + np.abs(p); m = d > 1e-9
    return float(np.mean(2 * np.abs(p[m] - y[m]) / d[m]) * 100)

def run(label, cfg):
    m = LGBMForecaster(cfg)
    m.fit(train_df)
    out = m.predict(test_df)
    p = np.maximum(out["revenue_p50"].values, 0)
    err = p - y_te
    # quantile calibration + ordering sanity
    p10, p90 = out["revenue_p10"].values, out["revenue_p90"].values
    cover = float(np.mean((y_te >= p10) & (y_te <= p90)))
    ordered = bool(np.all(p10 <= p + 1e-6) and np.all(p <= p90 + 1e-6))
    res = dict(mae=float(np.mean(np.abs(err))),
               rmse=float(np.sqrt(np.mean(err ** 2))),
               smape=smape(y_te, p),
               r2=float(1 - np.sum(err ** 2) / np.sum((y_te - y_te.mean()) ** 2)),
               cover80=cover, ordered=ordered, nfeat=len(m.feature_cols))
    print(f"\n[{label}]  ({res['nfeat']} features)")
    print(f"  MAE={res['mae']:8.2f}  RMSE={res['rmse']:8.2f}  sMAPE={res['smape']:6.2f}%  "
          f"R2={res['r2']:.4f}  P10-P90 coverage={cover*100:.1f}%  ordered={ordered}")
    return res

old = run("OLD baseline (log=off, TE=off)",
          {"log_target": False, "target_encoding": False})
te_only = run("TE only     (log=off, TE=on)",
              {"log_target": False, "target_encoding": True})
log_only = run("log only    (log=on,  TE=off)",
               {"log_target": True, "target_encoding": False})
new = run("NEW both     (log=on,  TE=on)",
          {"log_target": True, "target_encoding": True})

def cmp(label, a, b):
    print(f"\n{label}")
    for k in ["mae", "rmse", "smape"]:
        d = (b[k] - a[k]) / a[k] * 100
        print(f"  {k.upper():6} {a[k]:9.2f} -> {b[k]:9.2f}  ({d:+.1f}%)")
    print(f"  R2     {a['r2']:9.4f} -> {b['r2']:9.4f}  ({b['r2']-a['r2']:+.4f})")
    print(f"  P10-P90 coverage {a['cover80']*100:.1f}% -> {b['cover80']*100:.1f}%  (nominal 80%)")

print("\n" + "=" * 66)
cmp("Target encoding alone (vs OLD):", old, te_only)
cmp("log1p alone (vs OLD):", old, log_only)
cmp("Both / NEW default (vs OLD):", old, new)
print("=" * 66)
