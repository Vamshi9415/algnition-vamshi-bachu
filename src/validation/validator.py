"""Validation engine: checks canonical DataFrames against data contracts."""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml
from loguru import logger


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False
        logger.error(f"[VALIDATION ERROR] {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"[VALIDATION WARN] {msg}")

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class DataValidator:
    """Validates a canonical DataFrame against schema rules."""

    REQUIRED_COLUMNS = ["date", "channel", "campaign_name", "spend", "revenue"]

    def __init__(self, config_path: str = "config/schema.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()

        # 1. Required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            report.add_error(f"Missing required columns: {missing}")
            return report  # Cannot continue without required cols

        # 2. Row count
        report.stats["row_count"] = len(df)
        if len(df) == 0:
            report.add_error("Dataset is empty.")
            return report

        # 3. Null percentages
        for col in self.REQUIRED_COLUMNS:
            null_pct = df[col].isna().mean() * 100
            report.stats[f"{col}_null_pct"] = round(null_pct, 2)
            if null_pct > 20:
                report.add_warning(f"Column '{col}' has {null_pct:.1f}% null values.")

        # 4. Negative spend
        neg_spend = (df["spend"] < 0).sum()
        if neg_spend > 0:
            report.add_error(f"Found {neg_spend} rows with negative spend.")

        # 5. Negative revenue
        neg_rev = (df["revenue"] < 0).sum()
        if neg_rev > 0:
            report.add_error(f"Found {neg_rev} rows with negative revenue.")

        # 6. Impressions >= Clicks
        if "impressions" in df.columns and "clicks" in df.columns:
            bad = (df["impressions"] < df["clicks"]).sum()
            if bad > 0:
                report.add_warning(f"{bad} rows where impressions < clicks.")

        # 7. Date validity
        invalid_dates = df["date"].isna().sum()
        if invalid_dates > 0:
            report.add_error(f"{invalid_dates} rows have unparseable dates.")

        # 8. Duplicates
        dup_cols = ["date", "channel", "campaign_name"]
        n_dups = df.duplicated(subset=dup_cols).sum()
        if n_dups > 0:
            report.add_warning(f"{n_dups} duplicate rows on (date, channel, campaign_name).")

        # 9. Stats summary
        report.stats["date_range"] = {
            "min": str(df["date"].min().date()),
            "max": str(df["date"].max().date()),
        }
        report.stats["channels"] = df["channel"].unique().tolist()
        report.stats["campaigns"] = int(df["campaign_name"].nunique())
        report.stats["total_revenue"] = round(float(df["revenue"].sum()), 2)
        report.stats["total_spend"] = round(float(df["spend"].sum()), 2)

        logger.info(f"Validation complete. Passed: {report.passed}")
        return report
