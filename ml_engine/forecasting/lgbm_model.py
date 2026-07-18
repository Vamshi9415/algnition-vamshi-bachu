"""LightGBM quantile regression forecaster."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger


FEATURE_COLS_EXCLUDE = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "currency", "daily_budget", "holiday_name", "revenue",
    # revenue_outlier_flag is computed from TODAY's revenue (DataCleaner step 8), so using
    # it as a feature leaks the target -- flagged rows average ~20x the revenue of normal
    # rows. Excluded here; the leak-free per-series signal lives in the *_anom_* columns
    # produced by ml_engine/anomaly (added in the feature store). See reports/anomaly_detection.md.
    "revenue_outlier_flag",
]

QUANTILES = [0.1, 0.5, 0.9]


class LGBMForecaster:
    """Trains one LightGBM model per quantile and returns P10/P50/P90 forecasts."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.models: dict[float, lgb.Booster] = {}
        self.feature_cols: list[str] = []
        # Revenue is heavy-tailed, so log1p(revenue)/expm1 training looked like a big win
        # in early benchmarking (~10-27% lower MAE) -- but that benchmark ran on data
        # where ml_engine/features/lag_features.py's *_wow/mom/qoq_growth columns still
        # leaked the current-day target (fixed 2026-07-17; see the errata in
        # reports/recipe_verification.md). Re-verified on genuinely leak-free data, log1p
        # is flat-to-slightly-worse (MAE +0.5%, RMSE +2.3%, R2 -0.011) vs the raw target.
        # OFF by default. Kept config-gated in case a different dataset benefits.
        self.log_target = self.config.get("log_target", False)
        # Train-only target encoding: mean revenue per (channel, campaign, day-of-week).
        # Ranked #1 by LightGBM gain, but end-to-end verification showed it OVERFITS the
        # holdout window (MAE +10% and P10-P90 coverage 76%->63% when enabled), so it is
        # OFF by default. Kept config-gated for experimentation. See reports/recipe_verification.md.
        self.target_encoding = self.config.get("target_encoding", False)
        self._te_map: pd.DataFrame | None = None
        self._te_global: float = 0.0
        # Split-conformal interval calibration: hold out the most recent slice of the
        # training data, measure how far the truth falls outside the raw P10-P90 band,
        # and scale the band by a factor so empirical coverage matches nominal.
        # Verified to lift P10-P90 coverage ~73% -> ~80% at negligible width cost
        # (see reports/recipe_verification.md). ON by default.
        self.conformal = self.config.get("conformal", True)
        self.calibration_fraction = self.config.get("calibration_fraction", 0.2)
        # Per-channel calibration: computes one factor per channel. Verified NEUTRAL on this
        # dataset -- the coverage split (73% google/meta vs 95% bing) is driven by TEMPORAL
        # revenue drift between the calibration window and the future (google +22%, meta +45%,
        # bing -94%), not by static between-channel differences, so per-channel factors all
        # come out ~1.0 and coverage is unchanged (see reports/per_channel_analysis.md). OFF
        # by default; kept because it's correct and would help on a more stationary dataset.
        self.conformal_per_channel = self.config.get("conformal_per_channel", False)
        self._min_calib_points = self.config.get("conformal_min_points", 100)
        self._conformal_k: float = 1.0
        self._conformal_k_by_channel: dict[str, float] = {}
        # Winsorize anomalous spikes in the TRAINING target toward a robust per-series
        # fence before fitting, so a handful of huge days don't dominate the loss. Only
        # touches training data (never inference), so it's leak-free. OFF by default until
        # verified to help on this dataset -- see reports/anomaly_detection.md.
        self.winsorize_target = self.config.get("winsorize_target", False)
        self._winsor_threshold = self.config.get("winsorize_threshold", 4.0)

    TE_COL = "te_campaign_dow_mean_rev"

    def _fit_target_encoding(self, df: pd.DataFrame):
        tmp = df[["channel", "campaign_name", "date", "revenue"]].copy()
        tmp["dow"] = tmp["date"].dt.dayofweek
        self._te_global = float(tmp["revenue"].fillna(0).mean())
        self._te_map = (
            tmp.groupby(["channel", "campaign_name", "dow"])["revenue"]
            .mean().rename(self.TE_COL).reset_index()
        )

    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["dow"] = df["date"].dt.dayofweek
        df = df.merge(self._te_map, on=["channel", "campaign_name", "dow"], how="left")
        df[self.TE_COL] = df[self.TE_COL].fillna(self._te_global)
        return df.drop(columns=["dow"])

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        exclude = set(FEATURE_COLS_EXCLUDE)
        return [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    def _base_params(self) -> dict:
        return {
            "n_estimators": self.config.get("n_estimators", 400),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "num_leaves": self.config.get("num_leaves", 63),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "verbose": -1,
        }

    def _train_quantile_models(self, X, y) -> dict:
        models = {}
        for q in QUANTILES:
            params = {**self._base_params(), "objective": "quantile", "alpha": q}
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            models[q] = model
        return models

    def _predict_band(self, models: dict, X) -> tuple:
        """Return stitched (p10, p50, p90) on the revenue scale for a feature matrix."""
        cols = []
        for q in QUANTILES:
            preds = models[q].predict(X)
            if self.log_target:
                preds = np.expm1(preds)
            cols.append(np.maximum(preds, 0))
        stacked = np.sort(np.column_stack(cols), axis=1)  # monotonic stitching
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]

    def _calibrate_conformal(self, df: pd.DataFrame) -> float:
        """Split-conformal: fit on the earlier part, score the recent calibration tail,
        and return the band scale factor that hits the nominal P10-P90 coverage."""
        target_cov = QUANTILES[-1] - QUANTILES[0]  # 0.9 - 0.1 = 0.8
        cal_dates = np.sort(df["date"].unique())
        if len(cal_dates) < 10:
            return 1.0  # not enough history to calibrate
        split = cal_dates[int(len(cal_dates) * (1 - self.calibration_fraction))]
        proper = df[df["date"] < split]
        calib = df[df["date"] >= split]
        if len(proper) == 0 or len(calib) == 0:
            return 1.0

        Xp = proper[self.feature_cols].fillna(0)
        yp = proper["revenue"].fillna(0)
        if self.log_target:
            yp = np.log1p(yp.clip(lower=0))
        cal_models = self._train_quantile_models(Xp, yp)

        p10, p50, p90 = self._predict_band(cal_models, calib[self.feature_cols].fillna(0))
        y = calib["revenue"].fillna(0).to_numpy()
        # nonconformity: factor needed to just cover each point, relative to the
        # p50->edge half-width. The 0.8-quantile of these scores is the band scale that
        # makes ~80% of calibration points fall inside.
        lo = (p50 - y) / np.maximum(p50 - p10, 1e-6)
        hi = (y - p50) / np.maximum(p90 - p50, 1e-6)
        scores = np.maximum(lo, hi)
        global_k = max(float(np.quantile(scores, target_cov)), 1e-6)

        # Per-channel factors, falling back to global_k where a channel has too few points.
        if self.conformal_per_channel and "channel" in calib.columns:
            ch = calib["channel"].to_numpy()
            for channel in pd.unique(ch):
                mask = ch == channel
                if mask.sum() >= self._min_calib_points:
                    self._conformal_k_by_channel[channel] = max(
                        float(np.quantile(scores[mask], target_cov)), 1e-6
                    )
                else:
                    self._conformal_k_by_channel[channel] = global_k
        return global_k

    def fit(self, df: pd.DataFrame):
        if self.winsorize_target and {"channel", "campaign_name", "date", "revenue"} <= set(df.columns):
            from ml_engine.anomaly.detector import AnomalyDetector
            detector = AnomalyDetector(column="revenue", threshold=self._winsor_threshold)
            df, n_capped = detector.winsorize_training_target(df)
            logger.info(f"Winsorized {n_capped} anomalous training targets before fit")
        if self.target_encoding:
            self._fit_target_encoding(df)
            df = self._apply_target_encoding(df)
        self.feature_cols = self._get_feature_cols(df)

        if self.conformal:
            self._conformal_k = self._calibrate_conformal(df)
            if self._conformal_k_by_channel:
                per_ch = ", ".join(f"{c}={k:.2f}" for c, k in self._conformal_k_by_channel.items())
                logger.info(f"Conformal factors: global={self._conformal_k:.3f} | per-channel: {per_ch}")
            else:
                logger.info(f"Conformal calibration factor k={self._conformal_k:.3f}")

        X = df[self.feature_cols].fillna(0)
        y = df["revenue"].fillna(0)
        if self.log_target:
            y = np.log1p(y.clip(lower=0))

        # Final deployment models are trained on ALL data (calibration slice included).
        self.models = self._train_quantile_models(X, y)
        logger.info(f"LightGBM trained for quantiles={QUANTILES}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_encoding and self._te_map is not None:
            df = self._apply_target_encoding(df)
        X = df[self.feature_cols].fillna(0)
        out = df[["date", "channel", "campaign_name"]].copy()

        # Stitched (monotonic) P10 <= P50 <= P90 on the revenue scale.
        p10, p50, p90 = self._predict_band(self.models, X)

        # Conformal widening: scale the band around the median so empirical coverage
        # matches nominal. Per-channel factor when available (channels differ a lot),
        # else the single global factor. k==1.0 leaves the band unchanged.
        if self.conformal:
            if self._conformal_k_by_channel and "channel" in df.columns:
                k = df["channel"].map(self._conformal_k_by_channel).fillna(self._conformal_k).to_numpy()
            else:
                k = self._conformal_k
            p10 = np.maximum(p50 - k * (p50 - p10), 0)
            p90 = p50 + k * (p90 - p50)

        out["revenue_p10"] = p10
        out["revenue_p50"] = p50
        out["revenue_p90"] = p90
        return out

    @property
    def feature_importance(self) -> pd.DataFrame:
        model = self.models.get(0.5)
        if model is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp

