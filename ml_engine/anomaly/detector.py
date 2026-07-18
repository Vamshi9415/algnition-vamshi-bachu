"""Per-series, leak-free anomaly detection for campaign revenue.

Design notes
------------
The existing DataCleaner flagged outliers with a single GLOBAL IQR over pooled revenue and
compared each row against a threshold derived from *today's* value -- both wrong for a
forecaster: a small campaign's normal day looks anomalous next to a large campaign, and a
flag computed from today's revenue leaks the target if used as a feature.

This detector instead works PER SERIES (channel x campaign) using a robust rolling
statistic (median + MAD), and exposes anomaly signal only through columns that depend on
shifted (strictly past) values -- so it is safe to use as a model feature. The same core
detector can also clean the TRAINING target (winsorize spikes), which is allowed to look at
the current value because it only ever touches training data, never inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

GROUP_KEYS = ["channel", "campaign_name"]
# 0.6745 = inverse of the 75th-percentile of the standard normal; scales MAD to a
# std-equivalent so the z-score threshold is interpretable on the usual normal scale.
_MAD_TO_SIGMA = 0.6745


class AnomalyDetector:
    """Robust rolling-MAD anomaly detector (optionally IsolationForest) for revenue series."""

    def __init__(
        self,
        column: str = "revenue",
        method: str = "robust_zscore",
        window: int = 28,
        threshold: float = 4.0,
        min_periods: int = 7,
        contamination: float = 0.03,
    ):
        if method not in {"robust_zscore", "isolation_forest"}:
            raise ValueError(f"Unknown method: {method!r}")
        self.column = column
        self.method = method
        self.window = window
        self.threshold = threshold
        self.min_periods = min_periods
        self.contamination = contamination

    # ------------------------------------------------------------------ core scoring
    def _rolling_robust_z(self, s: pd.Series) -> pd.Series:
        """Robust z-score of each point vs its own trailing window (inclusive of the point).

        Uses the current value, so this is the IN-SAMPLE score -- correct for detecting
        anomalies in known (training) data, but must be shifted before use as a feature.
        """
        med = s.rolling(self.window, min_periods=self.min_periods).median()
        mad = (s - med).abs().rolling(self.window, min_periods=self.min_periods).median()
        # Fall back to rolling std when MAD collapses to 0 (long flat/zero stretches)
        std = s.rolling(self.window, min_periods=self.min_periods).std()
        scale = (mad / _MAD_TO_SIGMA).where(mad > 1e-9, std)
        z = (s - med) / (scale + 1e-9)
        return z

    def score(self, df: pd.DataFrame) -> pd.Series:
        """In-sample anomaly score per row (higher magnitude = more anomalous).

        For robust_zscore this is the signed robust z-score; for isolation_forest it is the
        negative outlier score (higher = more anomalous). Uses the current value, so it is
        NOT leak-free -- use add_leakfree_features() for model inputs.
        """
        df = df.sort_values(GROUP_KEYS + ["date"])
        if self.method == "robust_zscore":
            return df.groupby(GROUP_KEYS)[self.column].transform(self._rolling_robust_z)
        return self._isolation_forest_score(df)

    def _isolation_forest_score(self, df: pd.DataFrame) -> pd.Series:
        from sklearn.ensemble import IsolationForest

        # Score globally but with per-series context features so the model isn't fooled by
        # scale differences between campaigns.
        feats = pd.DataFrame(index=df.index)
        grp = df.groupby(GROUP_KEYS)[self.column]
        med = grp.transform(lambda x: x.rolling(self.window, min_periods=self.min_periods).median())
        std = grp.transform(lambda x: x.rolling(self.window, min_periods=self.min_periods).std())
        feats["dev"] = (df[self.column] - med) / (std + 1e-9)
        feats["level"] = np.log1p(df[self.column].clip(lower=0))
        feats = feats.fillna(0.0)
        iso = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=200)
        raw = iso.fit_predict(feats.to_numpy())  # +1 inlier, -1 outlier
        score = -iso.score_samples(feats.to_numpy())  # higher = more anomalous
        return pd.Series(np.where(raw == -1, score, 0.0), index=df.index)

    def flag(self, df: pd.DataFrame) -> pd.Series:
        """Boolean in-sample anomaly flag per row (not leak-free; for training use)."""
        score = self.score(df)
        if self.method == "robust_zscore":
            return score.abs() > self.threshold
        return score > 0.0  # isolation_forest already returns 0 for inliers

    # ------------------------------------------------------------------ leak-free features
    def add_leakfree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly features that depend only on strictly-past values.

        Every column is shifted by 1 within each series, so the row for day t sees only
        data up to day t-1 -- safe to feed the forecaster. Columns added:
          {col}_anom_score_lag1   signed robust z-score of yesterday's value
          {col}_anom_flag_lag1     was yesterday an anomaly (0/1)
          {col}_anom_count_7d      # anomalies in the trailing 7 days (t-7..t-1)
          {col}_anom_count_28d     # anomalies in the trailing 28 days
          {col}_anom_days_since    days since the most recent past anomaly (capped at window)
        """
        df = df.sort_values(GROUP_KEYS + ["date"]).copy()
        c = self.column

        in_sample_score = self.score(df)
        in_sample_flag = (
            (in_sample_score.abs() > self.threshold) if self.method == "robust_zscore"
            else (in_sample_score > 0.0)
        ).astype(int)

        tmp = df[GROUP_KEYS].copy()
        tmp["_score"] = in_sample_score.to_numpy()
        tmp["_flag"] = in_sample_flag.to_numpy()

        g = tmp.groupby(GROUP_KEYS, group_keys=False)
        df[f"{c}_anom_score_lag1"] = g["_score"].apply(lambda s: s.shift(1)).to_numpy()
        df[f"{c}_anom_flag_lag1"] = g["_flag"].apply(lambda s: s.shift(1)).to_numpy()
        df[f"{c}_anom_count_7d"] = g["_flag"].apply(
            lambda s: s.shift(1).rolling(7, min_periods=1).sum()
        ).to_numpy()
        df[f"{c}_anom_count_28d"] = g["_flag"].apply(
            lambda s: s.shift(1).rolling(28, min_periods=1).sum()
        ).to_numpy()

        def _days_since(flag: pd.Series) -> pd.Series:
            past = flag.shift(1).fillna(0).to_numpy()
            out = np.empty(len(past))
            since = self.window
            for i, f in enumerate(past):
                since = 0 if f else min(since + 1, self.window)
                out[i] = since
            return pd.Series(out, index=flag.index)

        df[f"{c}_anom_days_since"] = g["_flag"].apply(_days_since).to_numpy()

        n_feats = 5
        logger.info(f"Anomaly features added ({n_feats} leak-free cols, method={self.method})")
        return df

    # ------------------------------------------------------------------ training cleaning
    def winsorize_training_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Cap anomalous spikes in the target column toward the robust upper/lower fence.

        Applied to TRAINING data only. Returns (cleaned_df, n_capped). Uses the in-sample
        score (current value allowed here -- we are cleaning known training targets, not
        creating an inference-time feature).
        """
        df = df.sort_values(GROUP_KEYS + ["date"]).copy()
        c = self.column
        z = self.score(df) if self.method == "robust_zscore" else None
        if z is None:
            # For isolation_forest, winsorization is undefined (no signed fence); skip.
            return df, 0

        med = df.groupby(GROUP_KEYS)[c].transform(
            lambda x: x.rolling(self.window, min_periods=self.min_periods).median()
        )
        mad = df.groupby(GROUP_KEYS)[c].transform(
            lambda x: (x - x.rolling(self.window, min_periods=self.min_periods).median())
            .abs().rolling(self.window, min_periods=self.min_periods).median()
        )
        std = df.groupby(GROUP_KEYS)[c].transform(
            lambda x: x.rolling(self.window, min_periods=self.min_periods).std()
        )
        scale = (mad / _MAD_TO_SIGMA).where(mad > 1e-9, std).fillna(0.0)
        upper = med + self.threshold * scale
        lower = (med - self.threshold * scale).clip(lower=0)

        is_anom = z.abs() > self.threshold
        capped = df[c].copy()
        capped = capped.where(~(is_anom & (df[c] > upper)), upper)
        capped = capped.where(~(is_anom & (df[c] < lower)), lower)
        n_capped = int((capped != df[c]).sum())
        df[c] = capped
        if n_capped:
            logger.info(f"Winsorized {n_capped} anomalous '{c}' training rows toward robust fence")
        return df, n_capped
