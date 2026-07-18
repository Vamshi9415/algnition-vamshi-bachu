"""Interval-quality verification: crossing rate, coverage, and a conformal widening test.

Shipped config = log_target on, target_encoding off. Honest forecasting (leakers excluded).
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
df = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(CSVLoader().load_many(CSVS))), save=False
).sort_values("date").reset_index(drop=True)

dates = np.sort(df["date"].unique())
# three-way split: train (60%) | calib (20%) | test (20%)
c1, c2 = dates[int(len(dates)*0.6)], dates[int(len(dates)*0.8)]
train_df = df[df["date"] < c1].copy()
calib_df = df[(df["date"] >= c1) & (df["date"] < c2)].copy()
test_df = df[df["date"] >= c2].copy()

LEAKY = {"clicks","impressions","conversions","spend","video_views","ctr","cpc","cpm",
         "cpa","cvr","rpc","roas","aov","spend_x_month","spend_x_holiday","conversions_value"}
lgbm_model.FEATURE_COLS_EXCLUDE = list(lgbm_model.FEATURE_COLS_EXCLUDE) + [c for c in LEAKY if c in df.columns]

def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))

m = LGBMForecaster({"log_target": True, "target_encoding": False})
m.fit(train_df)

# raw (pre-stitch) vs stitched — reach into per-quantile models to see crossing
def raw_preds(frame):
    X = frame[m.feature_cols].fillna(0)
    cols = {}
    for q in [0.1, 0.5, 0.9]:
        p = np.expm1(m.models[q].predict(X))
        cols[q] = np.maximum(p, 0)
    return cols

test_raw = raw_preds(test_df)
cross = float(np.mean((test_raw[0.1] > test_raw[0.5]) | (test_raw[0.5] > test_raw[0.9])))
y_te = test_df["revenue"].fillna(0).values

# production predict() (stitched)
out = m.predict(test_df)
p10, p50, p90 = out["revenue_p10"].values, out["revenue_p50"].values, out["revenue_p90"].values
ordered = bool(np.all(p10 <= p50 + 1e-9) and np.all(p50 <= p90 + 1e-9))

print(f"Raw per-quantile crossing rate : {cross*100:.2f}% of rows")
print(f"After stitching, ordered       : {ordered}")
print(f"Stitched P10-P90 coverage      : {coverage(y_te, p10, p90)*100:.1f}%  (nominal 80%)")

# ---- conformal widening: scale the interval half-width on the calibration set ----
cal = m.predict(calib_df)
y_ca = calib_df["revenue"].fillna(0).values
cl, cm, ch = cal["revenue_p10"].values, cal["revenue_p50"].values, cal["revenue_p90"].values
# nonconformity: how many interval half-widths away is the truth?
lo_gap = (cm - y_ca) / np.maximum(cm - cl, 1e-6)
hi_gap = (y_ca - cm) / np.maximum(ch - cm, 1e-6)
scores = np.maximum(lo_gap, hi_gap)
k = float(np.quantile(scores, 0.80))  # scale so ~80% fall inside
p10c = np.maximum(p50 - k * (p50 - p10), 0)
p90c = p50 + k * (p90 - p50)
print(f"\nConformal scale factor k       : {k:.2f}")
print(f"Conformal P10-P90 coverage     : {coverage(y_te, p10c, p90c)*100:.1f}%  (nominal 80%)")
print(f"Median interval width  raw     : {np.median(p90 - p10):.1f}")
print(f"Median interval width  conformal: {np.median(p90c - p10c):.1f}")
