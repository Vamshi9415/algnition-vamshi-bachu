"""Pipeline entry points for the ML engine."""

from ml_engine.legacy_pipeline import AIgnitionPipeline
from .orchestrator import ForecastPipeline

__all__ = ["AIgnitionPipeline", "ForecastPipeline"]
