"""Final end-to-end verification: cumulative effect of every shipped change.

Real production LGBMForecaster + real feature store, honest forecasting split
(same-day leakers excluded from every run). Reports point accuracy AND interval
quality (crossing rate, P10-P90 coverage) for each cumulative configuration.
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
logger.remove(); logger.add(sys.stderr, level="ERROR")

from ml_engine.ingestion.loader import CSVLoader
from ml_engine.canonical.schema import CanonicalSchemaBuilder
from ml_engine.preprocessing.cleaner import DataCleaner
from ml_engine.features.feature_store import FeatureStore
from ml_engine.forecasting import lgbm_model
from ml_engine.forecasting.lgbm_model import LGBMForecaster

CSVS = [str(PROJECT / "ml_engine" / "data" / "raw" / f)
        for f in ["google_ads_campaign_stats.csv", "meta_ads_campaign_stats.csv", "bing_campaign_stats.csv"]]
print("Building features...", flush=True)
df = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(CSVLoader().load_many(CSVS))), save=False
).sort_values("date").reset_index(drop=True)

dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_df = df[df["date"] < cutoff].copy()
test_df = df[df["date"] >= cutoff].copy()
y_te = test_df["revenue"].fillna(0).values

LEAKY = {"clicks","impressions","conversions","spend","video_views","ctr","cpc","cpm",
         "cpa","cvr","rpc","roas","aov","spend_x_month","spend_x_holiday","conversions_value"}
lgbm_model.FEATURE_COLS_EXCLUDE = list(lgbm_model.FEATURE_COLS_EXCLUDE) + [c for c in LEAKY if c in df.columns]

def smape(y, p):
    d = np.abs(y) + np.abs(p); m = d > 1e-9
    return float(np.mean(2 * np.abs(p[m] - y[m]) / d[m]) * 100)

def evaluate(label, cfg):
    m = LGBMForecaster(cfg)
    m.fit(train_df)
    out = m.predict(test_df)
    p10, p50, p90 = out["revenue_p10"].values, out["revenue_p50"].values, out["revenue_p90"].values
    err = p50 - y_te
    cover = float(np.mean((y_te >= p10) & (y_te <= p90)))
    ordered = bool(np.all(p10 <= p50 + 1e-9) and np.all(p50 <= p90 + 1e-9))
    r = dict(label=label,
             mae=float(np.mean(np.abs(err))),
             rmse=float(np.sqrt(np.mean(err**2))),
             smape=smape(y_te, p50),
             r2=float(1 - np.sum(err**2)/np.sum((y_te - y_te.mean())**2)),
             cover=cover, ordered=ordered, k=getattr(m, "_conformal_k", 1.0))
    print(f"{label:<46} MAE={r['mae']:7.2f} RMSE={r['rmse']:7.2f} R2={r['r2']:.4f} "
          f"cover={cover*100:5.1f}% ordered={str(ordered):5} k={r['k']:.2f}", flush=True)
    return r

print("\nCumulative configuration ladder (honest forecasting):\n" + "-"*110)
rows = []
rows.append(evaluate("1. OLD baseline (raw target, no fixes)",
                     {"log_target": False, "target_encoding": False, "conformal": False}))
rows.append(evaluate("2. + log1p target",
                     {"log_target": True, "target_encoding": False, "conformal": False}))
rows.append(evaluate("3. + monotonic stitching (default predict)",
                     {"log_target": True, "target_encoding": False, "conformal": False}))
# stitching is always on in predict(); rows 2 and 3 are identical by construction -> drop dup
rows = rows[:2]
rows.append(evaluate("3. + conformal calibration = FULL SHIPPED",
                     {"log_target": True, "target_encoding": False, "conformal": True}))

print("-"*110)
old, ship = rows[0], rows[-1]
print("\nFULL SHIPPED vs OLD baseline:")
for k in ["mae", "rmse", "smape"]:
    print(f"  {k.upper():6} {old[k]:8.2f} -> {ship[k]:8.2f}  ({(ship[k]-old[k])/old[k]*100:+.1f}%)")
print(f"  R2     {old['r2']:8.4f} -> {ship['r2']:8.4f}  ({ship['r2']-old['r2']:+.4f})")
print(f"  P10-P90 coverage {old['cover']*100:.1f}% -> {ship['cover']*100:.1f}%  (nominal 80%)")
print(f"  Quantile ordering guaranteed: {ship['ordered']}")
