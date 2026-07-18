"""Does per-channel conformal calibration fix the coverage split? Global vs per-channel.

Point accuracy (MAE/RMSE/R2) is identical by construction -- conformal only rescales the
P10/P90 band, never P50 -- so the only thing that moves is interval coverage. The goal:
each channel's coverage should sit near the nominal 80%, instead of 73% (google/meta) and
95% (bing) under a single global factor.
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
df = FeatureStore(anomaly_features=False).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(CSVLoader().load_many(CSVS))), save=False
).sort_values("date").reset_index(drop=True)

LEAKY = {"clicks","impressions","conversions","spend","video_views","ctr","cpc","cpm","cpa",
         "cvr","rpc","roas","aov","spend_x_month","spend_x_holiday","conversions_value","revenue_outlier_flag"}
lgbm_model.FEATURE_COLS_EXCLUDE = list(set(lgbm_model.FEATURE_COLS_EXCLUDE) | {c for c in LEAKY if c in df.columns})

dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_df = df[df["date"] < cutoff].copy()
test_df = df[df["date"] >= cutoff].copy()

def coverage_by_channel(out, truth):
    e = truth[["channel", "revenue"]].reset_index(drop=True).copy()
    e["p10"] = out["revenue_p10"].values; e["p90"] = out["revenue_p90"].values
    e["inside"] = (e["revenue"] >= e["p10"]) & (e["revenue"] <= e["p90"])
    e["width"] = e["p90"] - e["p10"]
    g = e.groupby("channel").agg(coverage=("inside", "mean"), med_width=("width", "median"), rows=("inside", "size"))
    g["coverage"] = (g["coverage"] * 100).round(1)
    overall = e["inside"].mean() * 100
    return g, overall

for label, per_ch in [("GLOBAL conformal (single k)", False), ("PER-CHANNEL conformal", True)]:
    m = LGBMForecaster({"conformal": True, "conformal_per_channel": per_ch})
    m.fit(train_df)
    out = m.predict(test_df)
    g, overall = coverage_by_channel(out, test_df)
    print(f"\n=== {label} ===")
    if m._conformal_k_by_channel:
        print("  factors:", {c: round(k, 3) for c, k in m._conformal_k_by_channel.items()},
              f"(global={m._conformal_k:.3f})")
    else:
        print(f"  factor: global k={m._conformal_k:.3f}")
    print(g.to_string())
    print(f"  overall coverage: {overall:.1f}%  (nominal 80%)")
    # confirm point accuracy unchanged
    y = test_df["revenue"].fillna(0).values
    mae = float(np.mean(np.abs(np.maximum(out["revenue_p50"].values, 0) - y)))
    print(f"  MAE (P50, unchanged by conformal): {mae:.2f}")
