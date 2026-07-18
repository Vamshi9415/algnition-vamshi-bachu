"""Per-channel error breakdown of the shipped production LightGBM on the honest leak-free split.

Same config and split as verify_final.py (raw target, conformal on, all three leaks excluded),
but errors are reported per channel (google / meta / microsoft-bing) instead of pooled, plus a
naive lag-1 baseline per channel for context.
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

# shipped config: raw target, conformal on
m = LGBMForecaster({"log_target": False, "target_encoding": False, "conformal": True})
m.fit(train_df)
pred = m.predict(test_df)

ev = test_df[["channel", "campaign_name", "date", "revenue"]].reset_index(drop=True).copy()
ev["p10"] = pred["revenue_p10"].values
ev["p50"] = pred["revenue_p50"].values
ev["p90"] = pred["revenue_p90"].values
ev["naive"] = test_df["revenue_lag1"].fillna(0).values  # per-channel baseline

def smape(y, p):
    d = np.abs(y) + np.abs(p); mask = d > 1e-9
    return float(np.mean(2 * np.abs(p[mask] - y[mask]) / d[mask]) * 100)

def block(g):
    y = g["revenue"].to_numpy(float)
    p = np.maximum(g["p50"].to_numpy(float), 0)
    nv = np.maximum(g["naive"].to_numpy(float), 0)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    wmape = float(np.sum(np.abs(err)) / (np.sum(np.abs(y)) + 1e-9) * 100)
    r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)) if y.std() > 0 else float("nan")
    cover = float(np.mean((y >= g["p10"]) & (y <= g["p90"])))
    naive_mae = float(np.mean(np.abs(nv - y)))
    return pd.Series({
        "rows": len(g), "mean_rev": float(y.mean()),
        "MAE": mae, "naive_MAE": naive_mae, "skill_vs_naive_%": (1 - mae / naive_mae) * 100 if naive_mae else np.nan,
        "RMSE": rmse, "WMAPE_%": wmape, "sMAPE_%": smape(y, p), "R2": r2, "cover_%": cover * 100,
    })

print("=== Per-channel error breakdown (shipped config, honest leak-free split) ===\n")
per_ch = ev.groupby("channel").apply(block)
pooled = block(ev); pooled.name = "POOLED"
out = pd.concat([per_ch, pooled.to_frame().T])
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.width", 160)
print(out.to_string())

print("\n=== Per-channel share of total absolute error (who drives the pooled MAE) ===")
ev["abs_err"] = np.abs(np.maximum(ev["p50"], 0) - ev["revenue"])
share = ev.groupby("channel")["abs_err"].sum()
share_pct = (share / share.sum() * 100).round(1)
rows_pct = (ev.groupby("channel").size() / len(ev) * 100).round(1)
summary = pd.DataFrame({"rows_%": rows_pct, "abs_err_share_%": share_pct})
print(summary.to_string())
