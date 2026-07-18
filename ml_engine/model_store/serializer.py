"""Model serializer: saves and loads the full model bundle as a pickle."""
import pickle
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from ml_engine.model_store.registry import build_metadata, load_registry, save_registry


class ModelSerializer:
    """
    Saves and loads the model bundle (LightGBM + Prophet + BudgetSimulator)
    as a single pickle file under pickle/model.pkl.
    """

    @staticmethod
    def save(
        path: str,
        lgbm,
        prophet,
        budget_sim,
        metadata: dict = None,
    ) -> None:
        feature_count = None
        if getattr(lgbm, "feature_cols", None):
            feature_count = len(lgbm.feature_cols)

        registry_metadata = build_metadata(
            wmape=(metadata or {}).get("wmape"),
            trained=(metadata or {}).get("trained"),
            features=feature_count,
            extra=metadata,
        )
        bundle = {
            "lgbm": lgbm,
            "prophet": prophet,
            "budget_sim": budget_sim,
            "metadata": registry_metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_registry(path, registry_metadata)
        size_mb = Path(path).stat().st_size / 1_048_576
        logger.info(f"Model bundle saved: {path} ({size_mb:.2f} MB)")

    @staticmethod
    def load(path: str) -> dict:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        registry_metadata = load_registry(path)
        if registry_metadata:
            bundle["metadata"] = {**registry_metadata, **bundle.get("metadata", {})}
        logger.info(f"Model bundle loaded: {path}")
        return bundle

