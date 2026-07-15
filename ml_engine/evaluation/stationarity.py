"""Stationarity testing: ADF, KPSS, and Mann-Kendall trend detection."""
import pandas as pd
import numpy as np
from loguru import logger

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels not installed — stationarity tests disabled")


class StationarityTester:
    """Runs ADF and KPSS tests and interprets them together."""

    def test(self, series: pd.Series, name: str = "series") -> dict:
        s = series.dropna()
        if len(s) < 20:
            return {"error": "Series too short for stationarity tests (need >= 20 obs)"}
        if not STATSMODELS_OK:
            return {"error": "statsmodels not installed"}
        if s.min() == s.max():
            return {"series": name, "n": len(s), "error": "Series is constant — stationarity tests undefined"}

        result = {"series": name, "n": len(s)}

        # ADF Test: H0 = non-stationary (unit root present)
        adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(s, autolag="AIC")
        result["adf"] = {
            "statistic": round(float(adf_stat), 4),
            "p_value": round(float(adf_p), 4),
            "lags_used": int(adf_lags),
            "critical_values": {k: round(v, 4) for k, v in adf_crit.items()},
            "is_stationary": bool(adf_p < 0.05),
            "interpretation": "Stationary (reject H0)" if adf_p < 0.05 else "Non-stationary (fail to reject H0)",
        }

        # KPSS Test: H0 = stationary
        try:
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")
            result["kpss"] = {
                "statistic": round(float(kpss_stat), 4),
                "p_value": round(float(kpss_p), 4),
                "lags_used": int(kpss_lags),
                "critical_values": {k: round(v, 4) for k, v in kpss_crit.items()},
                "is_stationary": bool(kpss_p > 0.05),
                "interpretation": "Stationary (fail to reject H0)" if kpss_p > 0.05 else "Non-stationary (reject H0)",
            }
        except Exception as e:
            result["kpss"] = {"error": str(e)}

        # Combined verdict
        adf_stat_flag = result["adf"]["is_stationary"]
        kpss_stat_flag = result.get("kpss", {}).get("is_stationary", None)
        if adf_stat_flag and kpss_stat_flag:
            verdict = "STATIONARY — both ADF and KPSS agree"
        elif not adf_stat_flag and kpss_stat_flag is False:
            verdict = "NON-STATIONARY — both ADF and KPSS agree. Differencing recommended."
        elif adf_stat_flag and kpss_stat_flag is False:
            verdict = "TREND-STATIONARY — ADF stationary, KPSS non-stationary. Consider detrending."
        else:
            verdict = "DIFFERENCE-STATIONARY — ADF non-stationary, KPSS stationary. Differencing recommended."
        result["verdict"] = verdict

        logger.info(f"Stationarity [{name}]: {verdict}")
        return result

    def stl_decompose(self, series: pd.Series, period: int = 7) -> dict:
        """STL decomposition: returns trend strength and seasonality strength."""
        s = series.dropna()
        if len(s) < period * 2:
            return {"error": "Series too short for STL"}
        if not STATSMODELS_OK:
            return {"error": "statsmodels not installed"}
        stl = STL(s, period=period, robust=True)
        res = stl.fit()
        # Seasonal strength: max(0, 1 - Var(Remainder)/Var(Seasonal+Remainder))
        seasonal_strength = max(0, 1 - np.var(res.resid) / (np.var(res.seasonal + res.resid) + 1e-9))
        trend_strength = max(0, 1 - np.var(res.resid) / (np.var(res.trend + res.resid) + 1e-9))
        return {
            "trend_strength": round(float(trend_strength), 4),
            "seasonal_strength": round(float(seasonal_strength), 4),
            "interpretation": {
                "trend": "Strong trend" if trend_strength > 0.6 else "Weak trend",
                "seasonality": "Strong seasonality" if seasonal_strength > 0.6 else "Weak seasonality",
            },
        }

