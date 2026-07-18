"""Anomaly detection for revenue time series.

Two leak-free uses in the forecasting pipeline:
  1. add_leakfree_features() -- trailing anomaly signals (was the recent past anomalous?),
     safe to use as model features because every column depends only on shifted (past) values.
  2. winsorize_training_target() -- caps extreme spikes in the TRAINING target so a handful
     of huge days don't dominate the loss. Applied to training data only; never at inference.
"""
from ml_engine.anomaly.detector import AnomalyDetector

__all__ = ["AnomalyDetector"]
