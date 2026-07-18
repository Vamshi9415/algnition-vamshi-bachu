"""Feature store: orchestrates all feature generators and saves parquet artifacts."""
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_engine.anomaly.detector import AnomalyDetector
from ml_engine.features.calendar_features import CalendarFeatureGenerator
from ml_engine.features.kpi_features import KPIFeatureGenerator
from ml_engine.features.lag_features import LagFeatureGenerator, LAG_COLS


class FeatureStore:
    """Runs the full feature pipeline and persists output to disk."""

    def __init__(self, output_dir: str = "data/features", anomaly_features: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calendar = CalendarFeatureGenerator()
        self.kpi = KPIFeatureGenerator()
        self.lag = LagFeatureGenerator()
        # Leak-free trailing anomaly features. Verified NEUTRAL on this dataset (MAE flat,
        # RMSE/R2/coverage move within noise -- see reports/anomaly_detection.md), so OFF
        # by default; kept one flag away. The concrete win from the anomaly work was
        # catching and excluding the leaky revenue_outlier_flag, not these features.
        self.anomaly_features = anomaly_features
        self.anomaly = AnomalyDetector(column="revenue")

    def build(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """Run all feature generators in order and return enriched DataFrame."""
        logger.info("Building feature store")
        df = self.kpi.generate(df)
        df = self.calendar.generate(df)
        pre_lag_df = df  # kept for the perturbation leak check below
        df = self.lag.generate(df)
        if self.anomaly_features:
            df = self.anomaly.add_leakfree_features(df)

        # Leakage checks: ensure no future/same-day values reach lag/anomaly-derived columns.
        self._assert_no_leakage(df)
        self._assert_no_perturbation_leak(pre_lag_df, df)

        if save:
            out_path = self.output_dir / "daily_features.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Feature store saved: {out_path} ({len(df)} rows, {df.shape[1]} cols)")

        return df

    def _assert_no_leakage(self, df: pd.DataFrame):
        """Basic leakage guard: lag/rolling cols must have NaN at start of each series."""
        # _roll_std_ is deliberately fillna(0)'d (std of a single point is undefined,
        # not a leak) so it's excluded from the strict first-row-must-be-NaN check.
        # NOTE: this check is necessary but NOT sufficient. A column whose formula only
        # partially shifts its inputs (e.g. the old buggy *_wow_growth, which divided by
        # a shifted denominator but used the CURRENT unshifted numerator) still produces
        # NaN on early rows purely because the denominator lacks history yet -- so this
        # check alone would not have caught that bug. See _assert_no_perturbation_leak
        # for the check that actually catches partial-shift leaks.
        lag_cols = [
            c for c in df.columns
            if ("_lag" in c or "_roll" in c or "_growth" in c) and "_roll_std_" not in c
        ]
        sorted_df = df.sort_values(["channel", "campaign_name", "date"])
        first_rows = sorted_df.groupby(["channel", "campaign_name"], as_index=False).nth(0)
        for col in lag_cols:
            leaked = first_rows[col].notna()
            if leaked.any():
                bad_campaigns = first_rows.loc[leaked, "campaign_name"].unique().tolist()
                raise ValueError(
                    f"Leakage detected in '{col}': first-row value is not NaN for campaigns {bad_campaigns}"
                )
        logger.info(f"Leakage check passed ({len(lag_cols)} lag/roll cols confirmed shifted)")

    def _assert_no_perturbation_leak(self, pre_lag_df: pd.DataFrame, post_lag_df: pd.DataFrame):
        """Catch partial-shift leaks the first-row-NaN check above cannot see: mutate the
        LAST row of each series' raw target-adjacent columns (revenue, spend, roas,
        conversions, clicks, ctr), regenerate lag/roll/growth (and anomaly) features on
        that mutated frame, and confirm the last row's derived columns are unchanged. If a
        derived column depends -- even partially -- on that row's own current-day raw
        value, the perturbation will move it and this raises. This is how the real
        *_wow_growth target leak (fixed in ml_engine/features/lag_features.py) would have
        been caught, and it also guards the anomaly features.
        """
        raw_cols = [c for c in LAG_COLS if c in pre_lag_df.columns]
        if not raw_cols:
            return

        perturbed = pre_lag_df.copy()
        last_idx = (
            perturbed.sort_values(["channel", "campaign_name", "date"])
            .groupby(["channel", "campaign_name"], as_index=False)
            .tail(1)
            .index
        )
        rng = np.random.default_rng(0)
        for col in raw_cols:
            perturbed[col] = perturbed[col].astype(float)
            perturbed.loc[last_idx, col] = (
                perturbed.loc[last_idx, col] * 137.0 + rng.uniform(1000, 5000, size=len(last_idx))
            )

        perturbed_out = LagFeatureGenerator().generate(perturbed)
        if self.anomaly_features:
            perturbed_out = self.anomaly.add_leakfree_features(perturbed_out)

        derived_cols = [
            c for c in post_lag_df.columns
            if ("_lag" in c or "_roll" in c or "_growth" in c or "_anom_" in c)
            and "_roll_std_" not in c
        ]
        a = post_lag_df.loc[last_idx, derived_cols]
        b = perturbed_out.loc[last_idx, derived_cols]
        diff_mask = ~np.isclose(a.fillna(-999999).to_numpy(), b.fillna(-999999).to_numpy(), equal_nan=True)
        if diff_mask.any():
            bad_cols = a.columns[diff_mask.any(axis=0)].tolist()
            raise ValueError(
                f"Target-leak detected: {bad_cols} changed when only the same row's raw "
                f"value was perturbed. These columns depend on same-day data they shouldn't."
            )
        logger.info(f"Perturbation leak check passed ({len(derived_cols)} derived cols "
                    f"confirmed independent of same-day raw values)")

