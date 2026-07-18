"""Manual time-ordered stacking (fixes StackingRegressor + TimeSeriesSplit incompatibility).

Meta-features are built out-of-sample: base models trained on the first 75% of the
training window predict the last 25%; a meta-learner is fit on those predictions.
Base models are then refit on the full training window to predict the test set.
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

loaded = CSVLoader().load_many(CSVS)
features = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(
    DataCleaner().clean(CanonicalSchemaBuilder().build_from_many(loaded)), save=False
).sort_values("date").reset_index(drop=True)

exclude = set(FEATURE_COLS_EXCLUDE)
feat_cols = [c for c in features.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]
dates = np.sort(features["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
inner_cutoff = dates[int(len(dates) * 0.6)]  # 75% of the training window

train_df = features[features["date"] < cutoff]
test_df = features[features["date"] >= cutoff]
core_df = train_df[train_df["date"] < inner_cutoff]
holdout_df = train_df[train_df["date"] >= inner_cutoff]

def xy(df):
    return df[feat_cols].fillna(0).values, df["revenue"].fillna(0).values

X_core, y_core = xy(core_df)
X_hold, y_hold = xy(holdout_df)
X_tr, y_tr = xy(train_df)
X_te, y_te = xy(test_df)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

def bases():
    return {
        "lgbm": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                  verbose=-1, random_state=42),
        "xgb": xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", random_state=42, verbosity=0),
        "cat": CatBoostRegressor(iterations=400, learning_rate=0.05, depth=7, subsample=0.8,
                                 random_seed=42, verbose=False),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42),
    }

def smape(y, p):
    d = np.abs(y) + np.abs(p)
    m = d > 1e-9
    return float(np.mean(2 * np.abs(p[m] - y[m]) / d[m]) * 100)

def stack(name, keys, meta_maker):
    t0 = time.perf_counter()
    # 1) out-of-sample meta features from the inner holdout
    Z_hold, Z_test = [], []
    for k in keys:
        m = bases()[k]
        m.fit(X_core, y_core)
        Z_hold.append(m.predict(X_hold))
        m2 = bases()[k]          # refit on full train for test-time base preds
        m2.fit(X_tr, y_tr)
        Z_test.append(m2.predict(X_te))
    Z_hold = np.column_stack(Z_hold)
    Z_test = np.column_stack(Z_test)
    # 2) meta learner
    meta = meta_maker()
    meta.fit(Z_hold, y_hold)
    p = np.maximum(meta.predict(Z_test), 0)
    fit_t = time.perf_counter() - t0
    err = p - y_te
    res = {"model": name, "family": "ensemble",
           "mae": float(np.mean(np.abs(err))),
           "rmse": float(np.sqrt(np.mean(err ** 2))),
           "smape": smape(y_te, p),
           "r2": float(1 - np.sum(err ** 2) / np.sum((y_te - y_te.mean()) ** 2)),
           "fit_s": round(fit_t, 2), "pred_s": 0.0}
    w = getattr(meta, "coef_", None)
    print(f"  {name:<44} MAE={res['mae']:>8.2f} RMSE={res['rmse']:>8.2f} "
          f"sMAPE={res['smape']:>6.2f}% R2={res['r2']:.4f} fit={fit_t:.1f}s"
          + (f" weights={np.round(w, 3).tolist()}" if w is not None else ""), flush=True)
    return res

results = []
results.append(stack("Stacking (LGBM+XGB+Cat -> Ridge)", ["lgbm", "xgb", "cat"], lambda: Ridge(alpha=1.0)))
results.append(stack("Stacking (LGBM+XGB+Cat+RF -> Ridge)", ["lgbm", "xgb", "cat", "rf"], lambda: Ridge(alpha=1.0)))
results.append(stack("Stacking (LGBM+XGB+Cat+RF -> LGBM meta)", ["lgbm", "xgb", "cat", "rf"],
                     lambda: lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)))

out_path = Path(__file__).parent / "stacking_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"Saved {out_path}", flush=True)
