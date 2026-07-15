"""Calendar and holiday feature generator."""
import numpy as np
import pandas as pd
import holidays
from loguru import logger


class CalendarFeatureGenerator:
    """Generates date-based and holiday features from a canonical DataFrame."""

    def __init__(self, country: str = "US"):
        self.country = country

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        d = df["date"]

        df["year"] = d.dt.year
        df["quarter"] = d.dt.quarter
        df["month"] = d.dt.month
        df["week_of_year"] = d.dt.isocalendar().week.astype(int)
        df["day_of_week"] = d.dt.dayofweek          # 0=Monday
        df["day_of_month"] = d.dt.day
        df["day_of_year"] = d.dt.dayofyear
        df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = d.dt.is_month_start.astype(int)
        df["is_month_end"] = d.dt.is_month_end.astype(int)
        df["is_quarter_start"] = d.dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = d.dt.is_quarter_end.astype(int)

        # Days until end of month
        month_end = d + pd.offsets.MonthEnd(0)
        df["days_until_month_end"] = (month_end - d).dt.days

        # Fourier seasonality features
        df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Holiday flags
        df = self._add_holiday_features(df)

        logger.info(f"Calendar features added: {df.shape[1]} total columns")
        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        years = df["date"].dt.year.unique().tolist()
        us_holidays = {}
        for y in years:
            us_holidays.update(holidays.US(years=y))

        df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in us_holidays)).fillna(0).astype(int)
        df["holiday_name"] = df["date"].dt.date.map(lambda d: us_holidays.get(d, ""))

        # Black Friday / Cyber Monday detection
        df["is_black_friday"] = df["holiday_name"].str.contains("Thanksgiving", na=False).astype(int)
        # Proximity to any holiday
        holiday_dates = pd.to_datetime(list(us_holidays.keys()))
        df["days_to_nearest_holiday"] = df["date"].apply(
            lambda d: int(min(abs((holiday_dates - d).days))) if len(holiday_dates) else 99
        )
        return df

