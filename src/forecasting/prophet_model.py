"""Prophet forecasting model (aggregated daily revenue)."""
import pandas as pd
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Prophet model disabled.")


class ProphetForecaster:
    """Wraps Facebook Prophet for daily revenue forecasting with uncertainty."""

    def __init__(self, horizon: int = 60):
        self.horizon = horizon
        self.model: object = None
        self.last_date: object = None

    def fit(self, df: pd.DataFrame) -> None:
        if not PROPHET_AVAILABLE:
            raise RuntimeError("Prophet not installed.")
        daily = (
            df.groupby("date")["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"date": "ds", "revenue": "y"})
        )
        self.last_date = daily["ds"].max()
        self.model = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        self.model.add_country_holidays(country_name="US")
        self.model.fit(daily)
        logger.info(f"Prophet fitted on {len(daily)} daily observations")

    def predict(self) -> pd.DataFrame:
        if not PROPHET_AVAILABLE or self.model is None:
            raise RuntimeError("Prophet model not fitted.")
        future = self.model.make_future_dataframe(periods=self.horizon)
        forecast = self.model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(self.horizon)
        result = result.rename(columns={
            "ds": "date",
            "yhat": "p50",
            "yhat_lower": "p10",
            "yhat_upper": "p90",
        })
        result[["p10", "p50", "p90"]] = result[["p10", "p50", "p90"]].clip(lower=0)
        return result.reset_index(drop=True)
