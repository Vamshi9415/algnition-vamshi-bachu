"""Benchmark every viable model family on the Netelixiar revenue-forecasting task.

Reuses the project's own pipeline (loader -> canonical -> cleaner -> feature store)
so results are directly comparable to the production LightGBM/Prophet ensemble.

Evaluation: time-based holdout — last 20% of dates held out as test.
Metrics: MAE, RMSE, sMAPE, R2, fit time, predict time.
Output: JSON results consumed by the report generator.
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
logger.add(sys.stderr, level="WARNING")

from ml_engine.ingestion.loader import CSVLoader
from ml_engine.canonical.schema import CanonicalSchemaBuilder
from ml_engine.preprocessing.cleaner import DataCleaner
from ml_engine.features.feature_store import FeatureStore
from ml_engine.forecasting.lgbm_model import FEATURE_COLS_EXCLUDE

# ---------------------------------------------------------------- data prep
CSVS = [
    str(PROJECT / "ml_engine" / "data" / "raw" / "google_ads_campaign_stats.csv"),
    str(PROJECT / "ml_engine" / "data" / "raw" / "meta_ads_campaign_stats.csv"),
    str(PROJECT / "ml_engine" / "data" / "raw" / "bing_campaign_stats.csv"),
]

print("Loading + building features via project pipeline...", flush=True)
loader = CSVLoader()
loaded = loader.load_many(CSVS)
canonical = CanonicalSchemaBuilder().build_from_many(loaded)
cleaned = DataCleaner().clean(canonical)
features = FeatureStore(output_dir=str(PROJECT / "data" / "features")).build(cleaned, save=False)
features = features.sort_values("date").reset_index(drop=True)

# numeric feature columns, same exclusion list as production LGBMForecaster
exclude = set(FEATURE_COLS_EXCLUDE)
feat_cols = [c for c in features.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]

dates = np.sort(features["date"].unique())
cutoff = dates[int(len(dates) * 0.8)]
train_df = features[features["date"] < cutoff]
test_df = features[features["date"] >= cutoff]

X_tr = train_df[feat_cols].fillna(0).values
y_tr = train_df["revenue"].fillna(0).values
X_te = test_df[feat_cols].fillna(0).values
y_te = test_df["revenue"].fillna(0).values

meta = {
    "n_rows": int(len(features)),
    "n_train": int(len(train_df)),
    "n_test": int(len(test_df)),
    "n_features": len(feat_cols),
    "n_series": int(features.groupby(["channel", "campaign_name"]).ngroups),
    "date_min": str(pd.Timestamp(dates[0]).date()),
    "date_max": str(pd.Timestamp(dates[-1]).date()),
    "cutoff": str(pd.Timestamp(cutoff).date()),
    "target_mean_test": float(np.mean(y_te)),
    "target_std_test": float(np.std(y_te)),
}
print(f"rows={meta['n_rows']} train={meta['n_train']} test={meta['n_test']} "
      f"features={meta['n_features']} series={meta['n_series']} cutoff={meta['cutoff']}", flush=True)

# ---------------------------------------------------------------- metrics
def smape(y, p):
    denom = (np.abs(y) + np.abs(p))
    mask = denom > 1e-9
    return float(np.mean(2.0 * np.abs(p[mask] - y[mask]) / denom[mask]) * 100)

def evaluate(name, family, fit_fn, pred_fn):
    try:
        t0 = time.perf_counter()
        fit_fn()
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = np.asarray(pred_fn(), dtype=float)
        pred_t = time.perf_counter() - t0
        p = np.maximum(p, 0)  # revenue can't be negative
        err = p - y_te
        res = {
            "model": name,
            "family": family,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "smape": smape(y_te, p),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y_te - y_te.mean()) ** 2)),
            "fit_s": round(fit_t, 2),
            "pred_s": round(pred_t, 3),
        }
        print(f"  {name:<38} MAE={res['mae']:>9.2f} RMSE={res['rmse']:>9.2f} "
              f"sMAPE={res['smape']:>6.2f}% R2={res['r2']:>7.4f} fit={fit_t:.1f}s", flush=True)
        return res, p
    except Exception as e:
        print(f"  {name:<38} FAILED: {e}", flush=True)
        return {"model": name, "family": family, "error": str(e)}, None

results = []
preds_store = {}  # name -> test predictions (for blends)

def run(name, family, model, scale=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    est = make_pipeline(StandardScaler(), model) if scale else model
    res, p = evaluate(name, family, lambda: est.fit(X_tr, y_tr), lambda: est.predict(X_te))
    results.append(res)
    if p is not None:
        preds_store[name] = p
    return est

# ---------------------------------------------------------------- baselines
print("\n[Baselines]", flush=True)
lag1 = test_df["revenue_lag1"].fillna(0).values
lag7 = test_df["revenue_lag7"].fillna(0).values if "revenue_lag7" in test_df else lag1
roll7 = test_df["revenue_roll_mean_7d"].fillna(0).values if "revenue_roll_mean_7d" in test_df else lag1

for nm, arr in [("Naive (lag-1)", lag1), ("Seasonal Naive (lag-7)", lag7),
                ("Moving Average (7d)", roll7)]:
    res, p = evaluate(nm, "baseline", lambda: None, lambda a=arr: a)
    results.append(res)
    if p is not None:
        preds_store[nm] = p

# ---------------------------------------------------------------- linear
print("\n[Linear models]", flush=True)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
run("Linear Regression", "linear", LinearRegression(), scale=True)
run("Ridge", "linear", Ridge(alpha=1.0), scale=True)
run("Lasso", "linear", Lasso(alpha=0.1, max_iter=5000), scale=True)
run("ElasticNet", "linear", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), scale=True)
run("Huber (robust)", "linear", HuberRegressor(max_iter=500), scale=True)

# ---------------------------------------------------------------- other classic
print("\n[Classic non-linear]", flush=True)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
run("KNN (k=10)", "instance", KNeighborsRegressor(n_neighbors=10), scale=True)
run("Decision Tree", "tree", DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42))
run("Linear SVR", "svm", LinearSVR(C=1.0, max_iter=5000, random_state=42), scale=True)
from sklearn.neural_network import MLPRegressor
run("MLP Neural Net (2x64)", "neural", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300,
                                                    early_stopping=True, random_state=42), scale=True)

# ---------------------------------------------------------------- bagging
print("\n[Bagging]", flush=True)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
run("Random Forest (400)", "bagging", RandomForestRegressor(
    n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=42))
run("Extra Trees (400)", "bagging", ExtraTreesRegressor(
    n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=42))
run("Bagging (100 trees)", "bagging", BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=12, random_state=42),
    n_estimators=100, n_jobs=-1, random_state=42))

# ---------------------------------------------------------------- boosting
print("\n[Boosting]", flush=True)
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

run("AdaBoost (200)", "boosting", AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=6, random_state=42),
    n_estimators=200, learning_rate=0.05, random_state=42))
run("Gradient Boosting (sklearn)", "boosting", GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42))
run("HistGradientBoosting", "boosting", HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, random_state=42))
run("LightGBM (production cfg)", "boosting", lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=20, verbose=-1, random_state=42))
run("XGBoost", "boosting", xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
    colsample_bytree=0.8, tree_method="hist", random_state=42, verbosity=0))
run("CatBoost", "boosting", CatBoostRegressor(
    iterations=400, learning_rate=0.05, depth=7, subsample=0.8,
    random_seed=42, verbose=False))

# ---------------------------------------------------------------- voting / stacking
print("\n[Voting & Stacking ensembles]", flush=True)
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit

def fresh_lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                             verbose=-1, random_state=42)
def fresh_xgb():
    return xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", random_state=42, verbosity=0)
def fresh_cat():
    return CatBoostRegressor(iterations=400, learning_rate=0.05, depth=7, subsample=0.8,
                             random_seed=42, verbose=False)
def fresh_rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)

run("Voting (LGBM+XGB+Cat)", "ensemble", VotingRegressor(
    [("lgbm", fresh_lgbm()), ("xgb", fresh_xgb()), ("cat", fresh_cat())]))
run("Voting (LGBM+XGB+Cat+RF)", "ensemble", VotingRegressor(
    [("lgbm", fresh_lgbm()), ("xgb", fresh_xgb()), ("cat", fresh_cat()), ("rf", fresh_rf())]))

run("Stacking (LGBM+XGB+Cat -> Ridge)", "ensemble", StackingRegressor(
    estimators=[("lgbm", fresh_lgbm()), ("xgb", fresh_xgb()), ("cat", fresh_cat())],
    final_estimator=Ridge(alpha=1.0), cv=TimeSeriesSplit(n_splits=3), n_jobs=-1))
run("Stacking (LGBM+XGB+Cat+RF -> LGBM)", "ensemble", StackingRegressor(
    estimators=[("lgbm", fresh_lgbm()), ("xgb", fresh_xgb()), ("cat", fresh_cat()), ("rf", fresh_rf())],
    final_estimator=lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42),
    cv=TimeSeriesSplit(n_splits=3), n_jobs=-1))

# ---------------------------------------------------------------- blends of stored preds
print("\n[Simple blends]", flush=True)
def blend(name, names, weights=None):
    parts = [preds_store[n] for n in names if n in preds_store]
    if len(parts) != len(names):
        print(f"  {name}: skipped (missing base preds)")
        return
    w = weights or [1 / len(parts)] * len(parts)
    p = np.sum([wi * pi for wi, pi in zip(w, parts)], axis=0)
    res, _ = evaluate(name, "ensemble", lambda: None, lambda: p)
    results.append(res)

blend("Mean blend (LGBM+XGB+Cat)", ["LightGBM (production cfg)", "XGBoost", "CatBoost"])
blend("Weighted blend 0.5/0.3/0.2 (LGBM/XGB/Cat)",
      ["LightGBM (production cfg)", "XGBoost", "CatBoost"], [0.5, 0.3, 0.2])
blend("Mean blend (all boosters + RF + ET)",
      ["LightGBM (production cfg)", "XGBoost", "CatBoost", "HistGradientBoosting",
       "Random Forest (400)", "Extra Trees (400)"])

# ---------------------------------------------------------------- prophet (per-series, project impl)
print("\n[Prophet — project implementation, per-campaign]", flush=True)
try:
    from ml_engine.forecasting.prophet_model import ProphetForecaster
    horizon = int((pd.Timestamp(dates[-1]) - pd.Timestamp(cutoff)).days) + 1
    pf = ProphetForecaster()
    t0 = time.perf_counter()
    pf.fit(train_df)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    fc = pf.predict(horizon)
    pred_t = time.perf_counter() - t0
    if fc is not None and not fc.empty:
        fc["date"] = pd.to_datetime(fc["date"])
        m = test_df[["date", "channel", "campaign_name", "revenue"]].merge(
            fc[["date", "channel", "campaign_name", "revenue_p50"]],
            on=["date", "channel", "campaign_name"], how="inner")
        if len(m):
            y = m["revenue"].fillna(0).values
            p = np.maximum(m["revenue_p50"].fillna(0).values, 0)
            err = p - y
            res = {"model": f"Prophet (per-campaign, {len(m)}/{len(test_df)} rows matched)",
                   "family": "statistical",
                   "mae": float(np.mean(np.abs(err))),
                   "rmse": float(np.sqrt(np.mean(err ** 2))),
                   "smape": smape(y, p),
                   "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                   "fit_s": round(fit_t, 2), "pred_s": round(pred_t, 2)}
            results.append(res)
            print(f"  Prophet MAE={res['mae']:.2f} RMSE={res['rmse']:.2f} "
                  f"sMAPE={res['smape']:.2f}% R2={res['r2']:.4f} fit={fit_t:.1f}s", flush=True)
    else:
        print("  Prophet produced no forecasts", flush=True)
except Exception as e:
    print(f"  Prophet FAILED: {e}", flush=True)
    results.append({"model": "Prophet (per-campaign)", "family": "statistical", "error": str(e)})

# ---------------------------------------------------------------- feature importance (LGBM)
imp = None
try:
    m = fresh_lgbm().fit(X_tr, y_tr)
    imp = sorted(zip(feat_cols, m.feature_importances_.tolist()),
                 key=lambda t: -t[1])[:15]
except Exception:
    pass

out = {"meta": meta, "results": results, "top_features": imp}
out_path = Path(__file__).parent / "benchmark_results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved {out_path}", flush=True)
