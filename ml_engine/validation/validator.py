"""Data validation engine: produces ValidationReport with JSON and HTML output."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_engine.paths import reports_path

REQUIRED_COLS = ["date", "channel", "campaign_name", "spend", "revenue"]


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }

    def to_json(self, path: str | Path | None = None):
        target = Path(path) if path else reports_path("validation_report.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info(f"Validation JSON written: {target}")

    def to_html(self, path: str | Path | None = None):
        target = Path(path) if path else reports_path("validation_report.html")
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        color = "#22c55e" if self.passed else "#ef4444"
        errors_html = "".join(f"<li style='color:#ef4444'>{e}</li>" for e in self.errors) or "<li>None</li>"
        warnings_html = "".join(f"<li style='color:#f59e0b'>{w}</li>" for w in self.warnings) or "<li>None</li>"
        stats_rows = "".join(
            f"<tr><td style='padding:4px 12px'>{k}</td><td style='padding:4px 12px'>{v}</td></tr>"
            for k, v in self.stats.items()
        )
        html = f"""<!DOCTYPE html>
<html><head><title>Validation Report</title>
<style>body{{font-family:sans-serif;background:#0f172a;color:#f1f5f9;padding:32px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #334155;padding:8px}}
</style></head><body>
<h1>⚡ AIgnition — Data Validation Report</h1>
<h2 style='color:{color}'>{status}</h2>
<h3>Dataset Statistics</h3>
<table><tr><th>Metric</th><th>Value</th></tr>{stats_rows}</table>
<h3>Errors ({len(self.errors)})</h3><ul>{errors_html}</ul>
<h3>Warnings ({len(self.warnings)})</h3><ul>{warnings_html}</ul>
</body></html>"""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        logger.info(f"Validation HTML written: {target}")


class DataValidator:
    """Runs 12 validation checks and returns a ValidationReport."""

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()

        # 1. Empty DataFrame
        if df.empty:
            report.add_error("DataFrame is empty")
            return report

        # 2. Required columns
        for col in REQUIRED_COLS:
            if col not in df.columns:
                report.add_error(f"Missing required column: '{col}'")

        if not report.passed:
            return report

        # 3. Negative spend
        neg_spend = (df["spend"] < 0).sum()
        if neg_spend > 0:
            report.add_error(f"Found {neg_spend} rows with negative spend")

        # 4. Negative revenue
        neg_rev = (df["revenue"] < 0).sum()
        if neg_rev > 0:
            report.add_error(f"Found {neg_rev} rows with negative revenue")

        # 5. Invalid ROAS (if column present)
        if "roas" in df.columns:
            invalid_roas = (df["roas"] < 0).sum()
            if invalid_roas > 0:
                report.add_error(f"Found {invalid_roas} rows with negative ROAS")
            very_high = (df["roas"] > 500).sum()
            if very_high > 0:
                report.add_warning(f"{very_high} rows with ROAS > 500 (likely data error)")

        # 6. Null percentages
        for col in ["spend", "revenue", "clicks", "impressions"]:
            if col in df.columns:
                null_pct = df[col].isna().mean() * 100
                if null_pct > 50:
                    report.add_error(f"Column '{col}' has {null_pct:.1f}% missing values")
                elif null_pct > 10:
                    report.add_warning(f"Column '{col}' has {null_pct:.1f}% missing values")

        # 7. Duplicate rows
        dup_count = df.duplicated(subset=["date", "channel", "campaign_name"]).sum()
        if dup_count > 0:
            report.add_warning(f"{dup_count} duplicate (date, channel, campaign) rows")

        # 8. Missing dates
        if "date" in df.columns:
            null_dates = df["date"].isna().sum()
            if null_dates > 0:
                report.add_error(f"{null_dates} rows with missing/invalid date")

        # 9. Data type checks
        for num_col in ["spend", "revenue"]:
            if num_col in df.columns and not pd.api.types.is_numeric_dtype(df[num_col]):
                report.add_error(f"Column '{num_col}' is not numeric (dtype={df[num_col].dtype})")

        # Stats
        report.stats = {
            "rows": len(df),
            "campaigns": df["campaign_name"].nunique() if "campaign_name" in df.columns else "N/A",
            "channels": df["channel"].nunique() if "channel" in df.columns else "N/A",
            "date_range": f"{df['date'].min()} – {df['date'].max()}" if "date" in df.columns else "N/A",
            "total_spend": f"${df['spend'].sum():,.2f}" if "spend" in df.columns else "N/A",
            "total_revenue": f"${df['revenue'].sum():,.2f}" if "revenue" in df.columns else "N/A",
            "null_pct_spend": f"{df['spend'].isna().mean()*100:.1f}%" if "spend" in df.columns else "N/A",
            "null_pct_revenue": f"{df['revenue'].isna().mean()*100:.1f}%" if "revenue" in df.columns else "N/A",
            "duplicate_rows": int(dup_count) if "date" in df.columns else "N/A",
        }

        logger.info(f"Validation: {'PASSED' if report.passed else 'FAILED'} "
                    f"({len(report.errors)} errors, {len(report.warnings)} warnings)")
        return report

