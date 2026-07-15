"""Data leakage checker for feature sets."""
import pandas as pd
from loguru import logger

# Known leaky columns (computed from the same period's target)
LEAKY_COLS = [
    "revenue",      # target itself
    "roas",         # revenue / spend at same timestep
    "conversions",  # post-event, not available before auction
    "cpa",          # spend / conversions, post-event
    "revenue_per_click", "revenue_per_impression",
]


class LeakageChecker:
    """
    Checks a feature DataFrame for data leakage.
    Raises warnings for known leaky columns that should not appear in future-row prediction.
    """

    def check(self, df: pd.DataFrame, is_future: bool = False) -> dict:
        present_leaky = [c for c in LEAKY_COLS if c in df.columns]
        report = {
            "leaky_columns_found": present_leaky,
            "leaky_in_future_rows": present_leaky if is_future else [],
            "status": "clean",
        }

        if is_future and present_leaky:
            report["status"] = "WARNING"
            for col in present_leaky:
                logger.warning(f"[LeakageChecker] Column '{col}' present in future rows — potential data leak")
        else:
            logger.info(f"[LeakageChecker] No leakage detected in {'future' if is_future else 'training'} set")

        return report

