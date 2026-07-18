"""Loads the pre-trained model bundle and its paired historical dataset once, and
serves them to every route — no per-session upload or retraining."""
from pathlib import Path

from loguru import logger

from ml_engine.canonical.schema import CanonicalSchemaBuilder
from ml_engine.ingestion.loader import CSVLoader
from ml_engine.preprocessing.cleaner import DataCleaner
from ml_engine.features.feature_store import FeatureStore
from ml_engine.forecasting.ensemble import ForecastEnsemble
from ml_engine.model_store.serializer import ModelSerializer

MODEL_PATH = Path("ml_engine/pickle/model.pkl")
DATA_FILES = [
    "google_ads_campaign_stats.csv",
    "meta_ads_campaign_stats.csv",
    "bing_campaign_stats.csv",
]

_state: dict = {}


def get_state() -> dict:
    """Returns the loaded {df, features_df, ensemble, metadata} state, loading it on
    first call and caching it in memory for the lifetime of the process."""
    if _state:
        return _state

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Train one first: "
            f"DATA_DIR=ml_engine/data/raw MODEL_PATH={MODEL_PATH} bash run.sh"
        )

    files = [f for f in DATA_FILES if Path(f).exists()]
    if not files:
        raise RuntimeError(
            f"None of the expected historical data files were found: {DATA_FILES}"
        )

    logger.info(f"Loading historical dataset from {files} ...")
    loader = CSVLoader()
    loaded = loader.load_many(files)
    canonical = CanonicalSchemaBuilder().build_from_many(loaded)
    canonical = DataCleaner().clean(canonical)
    # Must match the feature store the training pipeline used (ml_engine/pipeline/orchestrator.py),
    # or the served features won't line up with what the pre-trained model's feature_cols expect.
    features_df = FeatureStore().build(canonical, save=False)

    logger.info(f"Loading pre-trained model bundle from {MODEL_PATH} ...")
    bundle = ModelSerializer.load(str(MODEL_PATH))
    ensemble = ForecastEnsemble()
    ensemble.lgbm = bundle["lgbm"]
    ensemble.prophet = bundle["prophet"]

    _state.update(
        df=canonical,
        features_df=features_df,
        ensemble=ensemble,
        metadata=bundle.get("metadata", {}),
    )
    logger.info(
        f"Model ready: {canonical['campaign_name'].nunique()} campaigns, "
        f"data through {features_df['date'].max().date()}"
    )
    return _state
