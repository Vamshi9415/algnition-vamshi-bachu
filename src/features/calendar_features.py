"""Calendar and holiday feature generator."""
import numpy as np
import pandas as pd
import holidays


class CalendarFeatureGenerator:
    def __init__(self, country: str = "US"):
        self.country = country

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        d = df["date"]

        df["year"] = d.dt.year
        df["quarter"] = d.dt.quarter
        df["month"] = d.dt.month
        df["week_of_year"] = d.dt.isocalendar().week.astype(int)
        df["day_of_week"] = d.dt.dayofweek        # 0=Mon
        df["day_of_month"] = d.dt.day
        df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = d.dt.is_month_start.astype(int)
        df["is_month_end"] = d.dt.is_month_end.astype(int)
        df["is_quarter_start"] = d.dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = d.dt.is_quarter_end.astype(int)

        # Fourier seasonality
        doy = d.dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Holiday flags
        years = df["year"].unique().tolist()
        country_holidays = holidays.country_holidays(self.country, years=years)
        df["is_holiday"] = d.apply(lambda x: int(x in country_holidays))
        df["holiday_name"] = d.apply(lambda x: country_holidays.get(x, ""))

        # Days until/since key marketing events
        df["days_until_month_end"] = (
            d.apply(lambda x: (x + pd.offsets.MonthEnd(0)) - x).dt.days
        )
        df["days_until_year_end"] = (
            d.apply(lambda x: pd.Timestamp(x.year, 12, 31) - x).dt.days
        )

        # Black Friday proximity (approx 4th Thursday Nov + 1)
        def days_to_black_friday(dt):
            import calendar
            year = dt.year
            nov = pd.Timestamp(year, 11, 1)
            # 4th Thursday
            thursdays = [nov + pd.Timedelta(days=i) for i in range(30) if (nov + pd.Timedelta(days=i)).weekday() == 3]
            bf = thursdays[3] + pd.Timedelta(days=1)
            return (bf - dt).days

        df["days_to_black_friday"] = d.apply(days_to_black_friday)
        df["near_black_friday"] = (df["days_to_black_friday"].abs() <= 7).astype(int)
        df["near_christmas"] = (
            d.apply(lambda x: abs((pd.Timestamp(x.year, 12, 25) - x).days) <= 10).astype(int)
        )

        return df
