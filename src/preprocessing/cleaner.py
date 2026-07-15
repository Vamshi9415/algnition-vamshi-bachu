"""Data cleaning pipeline: deterministic, ordered, logged."""
import re

import pandas as pd
from loguru import logger


class DataCleaner:
    """Applies cleaning steps in strict order to a canonical DataFrame."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning pipeline")
        df = df.copy()
        df = self._convert_dates(df)
        df = self._standardize_strings(df)
        df = self._trim_whitespace(df)
        df = self._handle_missing_values(df)
        df = self._remove_duplicates(df)
        df = self._sort_chronologically(df)
        df = self._compute_roas(df)
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df

    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad = df["date"].isna().sum()
        if bad:
            logger.warning(f"Dropped {bad} rows with unparseable dates")
        df = df.dropna(subset=["date"])
        return df

    def _standardize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["campaign_name", "campaign_type", "channel"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.strip()
                    .str.title()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace("_", " ")
                )
        return df

    def _trim_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["spend", "revenue", "clicks", "impressions", "conversions"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["date", "channel", "campaign_name"], keep="first")
        dropped = before - len(df)
        if dropped:
            logger.info(f"Removed {dropped} duplicate rows")
        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["channel", "campaign_name", "date"]).reset_index(drop=True)

    def _compute_roas(self, df: pd.DataFrame) -> pd.DataFrame:
        df["roas"] = (
            df["revenue"] / df["spend"].replace(0, float("nan"))
        ).fillna(0.0).round(4)
        return df
