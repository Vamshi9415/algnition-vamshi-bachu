"""Feature-engineering ablation: which feature groups drive accuracy?

Fixed models (LightGBM prod-cfg, XGBoost, Random Forest) evaluated over multiple
feature sets on the same time split as benchmark_models.py. Also tests new
engineered features (target encoding, ratios, trend) and a log1p target transform.
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

print("Building features...", flush=True)
loaded = CSVLoader().load_many(CSVS)
df = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(loaded)), save=False
).sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------- extra engineered features
dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_mask = df["date"] < cutoff

# 1) target encodings computed on TRAIN ONLY (no leakage)
tr = df[train_mask]
camp_mean = tr.groupby(["channel", "campaign_name"])["revenue"].mean().rename("te_campaign_mean_rev")
camp_std = tr.groupby(["channel", "campaign_name"])["revenue"].std().rename("te_campaign_std_rev")
chan_mean = tr.groupby("channel")["revenue"].mean().rename("te_channel_mean_rev")
dow_mean = tr.groupby(["channel", "campaign_name", tr["date"].dt.dayofweek])["revenue"].mean()
dow_mean.index.names = ["channel", "campaign_name", "dow"]
dow_mean = dow_mean.rename("te_campaign_dow_mean_rev")

df = df.merge(camp_mean.reset_index(), on=["channel", "campaign_name"], how="left")
df = df.merge(camp_std.reset_index(), on=["channel", "campaign_name"], how="left")
df = df.merge(chan_mean.reset_index(), on="channel", how="left")
df["dow"] = df["date"].dt.dayofweek
df = df.merge(dow_mean.reset_index(), on=["channel", "campaign_name", "dow"], how="left")
df = df.drop(columns=["dow"])

# 2) lag ratios & momentum (all from shifted values -> leak-free)
eps = 1e-9
df["rev_lag1_over_lag7"] = df["revenue_lag1"] / (df["revenue_lag7"] + eps)
df["rev_lag7_over_lag30"] = df["revenue_lag7"] / (df["revenue_lag30"] + eps)
df["rev_roll7_over_roll30"] = df["revenue_roll_mean_7d"] / (df["revenue_roll_mean_30d"] + eps)
df["rev_roll_cv_7d"] = df["revenue_roll_std_7d"] / (df["revenue_roll_mean_7d"] + eps)
df["spend_roll7_over_roll30"] = df["spend_roll_mean_7d"] / (df["spend_roll_mean_30d"] + eps)
for c in ["rev_lag1_over_lag7", "rev_lag7_over_lag30", "rev_roll7_over_roll30",
          "rev_roll_cv_7d", "spend_roll7_over_roll30"]:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-100, 100)

# 3) trend / lifecycle
df["days_since_series_start"] = (
    df.groupby(["channel", "campaign_name"])["date"].transform(lambda s: (s - s.min()).dt.days))
df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * (df["date"].dt.month - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["date"].dt.month - 1) / 12)

NEW_FEATS = ["te_campaign_mean_rev", "te_campaign_std_rev", "te_channel_mean_rev",
             "te_campaign_dow_mean_rev", "rev_lag1_over_lag7", "rev_lag7_over_lag30",
             "rev_roll7_over_roll30", "rev_roll_cv_7d", "spend_roll7_over_roll30",
             "days_since_series_start", "dow_sin", "dow_cos", "month_sin", "month_cos"]

# ---------------------------------------------------------------- feature groups
exclude = set(FEATURE_COLS_EXCLUDE)
all_numeric = [c for c in df.columns
               if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
prod_cols = [c for c in all_numeric if c not in NEW_FEATS]  # production set

CAL = [c for c in prod_cols if c in {
    "year", "quarter", "month", "week_of_year", "day_of_week", "day_of_month",
    "day_of_year", "is_weekend", "is_month_start", "is_month_end",
    "is_quarter_start", "is_quarter_end", "is_holiday", "days_to_next_holiday",
    "days_since_prev_holiday", "channel_weekend"} or c.startswith("is_")]
LAG = [c for c in prod_cols if "_lag" in c]
ROLL_EWM = [c for c in prod_cols if "_roll_" in c or "_ewm" in c]
SAMEDAY = [c for c in prod_cols if c in {
    "clicks", "impressions", "conversions", "spend", "video_views",
    "ctr", "cpc", "cpm", "cpa", "cvr", "rpc", "roas", "aov",
    "spend_x_month", "spend_x_holiday"}
    or c.endswith(("_wow_growth", "_mom_growth", "_qoq_growth"))]
LEAKFREE = sorted(set(CAL + LAG + ROLL_EWM))

SETS = {
    "Calendar only": CAL,
    "Lags only": LAG,
    "Rolling+EWM only": ROLL_EWM,
    "Leak-free (calendar+lags+rolling+EWM)": LEAKFREE,
    "Leak-free + new engineered feats": sorted(set(LEAKFREE + NEW_FEATS)),
    "Same-day KPIs only (nowcast)": SAMEDAY,
    "Full production set (leaky)": prod_cols,
    "Production + new engineered feats": sorted(set(prod_cols + NEW_FEATS)),
}

train_df = df[df["date"] < cutoff]
test_df = df[df["date"] >= cutoff]
y_tr = train_df["revenue"].fillna(0).values
y_te = test_df["revenue"].fillna(0).values

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

def models():
    return {
        "LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                      subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                      verbose=-1, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                    random_state=42, verbosity=0),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                              n_jobs=-1, random_state=42),
    }

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

results = []
for set_name, cols in SETS.items():
    print(f"\n[{set_name}] ({len(cols)} features)", flush=True)
    X_tr = train_df[cols].fillna(0).values
    X_te = test_df[cols].fillna(0).values
    for mname, m in models().items():
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        s = score(m.predict(X_te))
        s.update({"set": set_name, "n_features": len(cols), "model": mname,
                  "fit_s": round(time.perf_counter() - t0, 1)})
        results.append(s)
        print(f"  {mname:<14} MAE={s['mae']:>8.2f} RMSE={s['rmse']:>8.2f} "
              f"sMAPE={s['smape']:>6.2f}% R2={s['r2']:>7.4f} fit={s['fit_s']}s", flush=True)

# ---------------------------------------------------------------- log1p target transform
print("\n[Target transform: log1p, leak-free + new feats, LightGBM]", flush=True)
cols = SETS["Leak-free + new engineered feats"]
X_tr = train_df[cols].fillna(0).values
X_te = test_df[cols].fillna(0).values
m = models()["LightGBM"]
t0 = time.perf_counter()
m.fit(X_tr, np.log1p(y_tr))
s = score(np.expm1(m.predict(X_te)))
s.update({"set": "Leak-free + new feats, log1p target", "n_features": len(cols),
          "model": "LightGBM", "fit_s": round(time.perf_counter() - t0, 1)})
results.append(s)
print(f"  LightGBM(log1p) MAE={s['mae']:.2f} RMSE={s['rmse']:.2f} "
      f"sMAPE={s['smape']:.2f}% R2={s['r2']:.4f}", flush=True)

# also raw comparison for XGB with log
m = models()["XGBoost"]
t0 = time.perf_counter()
m.fit(X_tr, np.log1p(y_tr))
s = score(np.expm1(m.predict(X_te)))
s.update({"set": "Leak-free + new feats, log1p target", "n_features": len(cols),
          "model": "XGBoost", "fit_s": round(time.perf_counter() - t0, 1)})
results.append(s)
print(f"  XGBoost(log1p)  MAE={s['mae']:.2f} RMSE={s['rmse']:.2f} "
      f"sMAPE={s['smape']:.2f}% R2={s['r2']:.4f}", flush=True)

# ---------------------------------------------------------------- importance of new feats
m = models()["LightGBM"]
m.fit(train_df[cols].fillna(0).values, y_tr)
imp = sorted(zip(cols, m.feature_importances_.tolist()), key=lambda t: -t[1])
new_ranked = [(f, i, rank + 1) for rank, (f, i) in enumerate(imp) if f in NEW_FEATS]

out = {"cutoff": str(pd.Timestamp(cutoff).date()),
       "set_sizes": {k: len(v) for k, v in SETS.items()},
       "results": results,
       "top20_leakfree_new": imp[:20],
       "new_feature_ranks": new_ranked}
out_path = Path(__file__).parent / "feature_results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved {out_path}", flush=True)
