"""Model serializer: saves and loads the full model bundle as a pickle."""
import pickle
from pathlib import Path
from loguru import logger


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
        bundle = {
            "lgbm": lgbm,
            "prophet": prophet,
            "budget_sim": budget_sim,
            "metadata": metadata or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = Path(path).stat().st_size / 1_048_576
        logger.info(f"Model bundle saved: {path} ({size_mb:.2f} MB)")

    @staticmethod
    def load(path: str) -> dict:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        logger.info(f"Model bundle loaded: {path}")
        return bundle
