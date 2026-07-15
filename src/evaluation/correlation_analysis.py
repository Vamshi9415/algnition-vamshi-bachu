"""Correlation analysis: Pearson, Spearman, Kendall, mutual information."""
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class CorrelationAnalyzer:
    """Computes pairwise correlations and mutual information between features and target."""

    def pairwise(self, df: pd.DataFrame, cols: list[str] = None) -> dict:
        num_df = df[cols].dropna() if cols else df.select_dtypes(include="number").dropna()
        return {
            "pearson":  num_df.corr(method="pearson").round(4).to_dict(),
            "spearman": num_df.corr(method="spearman").round(4).to_dict(),
            "kendall":  num_df.corr(method="kendall").round(4).to_dict(),
        }

    def feature_target_correlation(self, df: pd.DataFrame, target: str = "revenue",
                                    top_n: int = 20) -> dict:
        """Correlations between all numeric features and the target column."""
        num_df = df.select_dtypes(include="number").dropna()
        if target not in num_df.columns:
            return {"error": f"Target '{target}' not found"}

        pearson  = num_df.corr(method="pearson")[target].drop(target).abs().sort_values(ascending=False).head(top_n)
        spearman = num_df.corr(method="spearman")[target].drop(target).abs().sort_values(ascending=False).head(top_n)

        result = {
            "pearson_top": pearson.round(4).to_dict(),
            "spearman_top": spearman.round(4).to_dict(),
        }

        if SKLEARN_OK:
            X = num_df.drop(columns=[target]).fillna(0)
            y = num_df[target].fillna(0)
            mi = mutual_info_regression(X, y, random_state=42)
            mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(top_n)
            result["mutual_information_top"] = mi_series.round(4).to_dict()

        logger.info(f"Correlation analysis complete. Top Pearson: {list(pearson.index[:3])}")
        return result
