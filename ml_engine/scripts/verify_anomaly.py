"""Does anomaly detection improve the forecast? Honest leak-free A/B.

Four configs on the identical time split, real production classes:
  A. baseline           : no anomaly features, no winsorization
  B. + anomaly features : leak-free trailing anomaly columns in the feature set
  C. + winsorize target : cap anomalous training spikes (no anomaly features)
  D. + both             : anomaly features AND winsorized training target
Same-day leakers excluded from every run so deltas are purely the anomaly work.
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
loaded = CSVLoader().load_many(CSVS)
cleaned = DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(loaded))

# Build features WITH anomaly columns (they're leak-free; configs that don't want them
# just drop them from the model's feature list).
df = FeatureStore(anomaly_features=True).build(cleaned, save=False).sort_values("date").reset_index(drop=True)

anom_cols = [c for c in df.columns if "_anom" in c]
LEAKY = {"clicks","impressions","conversions","spend","video_views","ctr","cpc","cpm","cpa",
         "cvr","rpc","roas","aov","spend_x_month","spend_x_holiday","conversions_value","revenue_outlier_flag"}
lgbm_model.FEATURE_COLS_EXCLUDE = list(set(lgbm_model.FEATURE_COLS_EXCLUDE) | {c for c in LEAKY if c in df.columns})

dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_df = df[df["date"] < cutoff].copy()
test_df = df[df["date"] >= cutoff].copy()
y_te = test_df["revenue"].fillna(0).values

def smape(y, p):
    d = np.abs(y) + np.abs(p); m = d > 1e-9
    return float(np.mean(2 * np.abs(p[m] - y[m]) / d[m]) * 100)

def run(label, drop_anom, winsorize):
    exclude_extra = set(anom_cols) if drop_anom else set()
    saved = list(lgbm_model.FEATURE_COLS_EXCLUDE)
    lgbm_model.FEATURE_COLS_EXCLUDE = saved + list(exclude_extra)
    try:
        m = LGBMForecaster({"winsorize_target": winsorize})
        m.fit(train_df)
        out = m.predict(test_df)
    finally:
        lgbm_model.FEATURE_COLS_EXCLUDE = saved
    p10, p50, p90 = out["revenue_p10"].values, out["revenue_p50"].values, out["revenue_p90"].values
    err = p50 - y_te
    cover = float(np.mean((y_te >= p10) & (y_te <= p90)))
    mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err**2)))
    r2 = float(1 - np.sum(err**2)/np.sum((y_te - y_te.mean())**2))
    nfeat = len(m.feature_cols)
    print(f"{label:<34} MAE={mae:7.2f} RMSE={rmse:7.2f} sMAPE={smape(y_te,p50):6.2f}% "
          f"R2={r2:.4f} cover={cover*100:4.1f}% nfeat={nfeat}", flush=True)
    return dict(mae=mae, rmse=rmse, r2=r2, cover=cover)

print(f"anomaly cols: {anom_cols}")
print(f"train={len(train_df)} test={len(test_df)}\n")
A = run("A. baseline (no anom, no winsor)", drop_anom=True,  winsorize=False)
B = run("B. + anomaly features",           drop_anom=False, winsorize=False)
C = run("C. + winsorize target only",      drop_anom=True,  winsorize=True)
D = run("D. + both",                       drop_anom=False, winsorize=True)

print("\nDeltas vs baseline A (MAE):")
for name, r in [("B anomaly feats", B), ("C winsorize", C), ("D both", D)]:
    print(f"  {name:<16} {A['mae']:.2f} -> {r['mae']:.2f}  ({(r['mae']-A['mae'])/A['mae']*100:+.1f}%)  "
          f"R2 {A['r2']:.4f}->{r['r2']:.4f}")
