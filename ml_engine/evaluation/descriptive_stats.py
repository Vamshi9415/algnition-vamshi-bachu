"""Descriptive statistics engine for campaign-level EDA."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class DescriptiveStats:
    """Computes full descriptive statistics for a numeric series or DataFrame."""

    def compute(self, df: pd.DataFrame, col: str = "revenue") -> dict:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        result = {
            "column": col,
            "n": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "variance": round(float(s.var()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(q3 - q1), 4),
            "range": round(float(s.max() - s.min()), 4),
            "skewness": round(float(stats.skew(s)), 4),
            "kurtosis": round(float(stats.kurtosis(s)), 4),
            "cv": round(float(s.std() / (s.mean() + 1e-9)), 4),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
        }
        logger.info(f"Descriptive stats computed for '{col}': n={result['n']}")
        return result

    def compute_all(self, df: pd.DataFrame, cols: list[str] = None) -> dict:
        numeric_cols = cols or df.select_dtypes(include="number").columns.tolist()
        return {col: self.compute(df, col) for col in numeric_cols if col in df.columns}

    def to_markdown(self, stats_dict: dict) -> str:
        lines = ["# Descriptive Statistics\n"]
        for col, s in stats_dict.items():
            lines.append(f"## {col}\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in s.items():
                if k != "column":
                    lines.append(f"| {k} | {v} |")
            lines.append("")
        return "\n".join(lines)

