"""Prophet forecaster for aggregate daily revenue per channel."""
import pandas as pd
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. ProphetForecaster will be skipped.")


class ProphetForecaster:
    """Wraps Facebook Prophet for daily revenue forecasting."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.models: dict[str, "Prophet"] = {}  # keyed by channel+campaign

    def fit(self, df: pd.DataFrame):
        if not PROPHET_AVAILABLE:
            return
        groups = df.groupby(["channel", "campaign_name"])
        n_errors = 0
        for (channel, campaign), grp in groups:
            key = f"{channel}|{campaign}"
            ts = grp[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
            ts = ts.sort_values("ds").dropna()
            if len(ts) < 14:
                continue
            try:
                m = Prophet(
                    seasonality_mode=self.config.get("seasonality_mode", "multiplicative"),
                    yearly_seasonality=self.config.get("yearly_seasonality", True),
                    weekly_seasonality=self.config.get("weekly_seasonality", True),
                    daily_seasonality=False,
                    changepoint_prior_scale=self.config.get("changepoint_prior_scale", 0.05),
                )
                m.fit(ts)
                self.models[key] = m
            except Exception as e:
                n_errors += 1
                if n_errors == 1:
                    logger.warning(f"Prophet fit failed for '{key}' (and possibly others): {e}. "
                                    f"Falling back to LightGBM-only for affected campaigns.")
        logger.info(f"Prophet fitted for {len(self.models)} campaign series ({n_errors} failed)")

    def predict(self, horizon_days: int = 60) -> pd.DataFrame:
        if not PROPHET_AVAILABLE or not self.models:
            return pd.DataFrame()
        frames = []
        for key, model in self.models.items():
            channel, campaign = key.split("|", 1)
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future).tail(horizon_days)
            forecast["channel"] = channel
            forecast["campaign_name"] = campaign
            forecast = forecast.rename(columns={"ds": "date", "yhat": "revenue_p50",
                                                 "yhat_lower": "revenue_p10",
                                                 "yhat_upper": "revenue_p90"})
            frames.append(forecast[["date", "channel", "campaign_name",
                                     "revenue_p10", "revenue_p50", "revenue_p90"]])
        out = pd.concat(frames, ignore_index=True)
        out[["revenue_p10", "revenue_p50", "revenue_p90"]] = out[
            ["revenue_p10", "revenue_p50", "revenue_p90"]
        ].clip(lower=0)
        return out

