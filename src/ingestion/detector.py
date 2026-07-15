"""Source detection engine: identifies which ad platform a CSV belongs to."""
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


class Channel(str, Enum):
    GOOGLE = "google"
    META = "meta"
    MICROSOFT = "microsoft"
    UNKNOWN = "unknown"


class SourceDetector:
    """Detects the ad channel of a CSV file by inspecting filename and columns."""

    def __init__(self, config_path: str = "config/channels.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.channels = cfg["channels"]

    def detect(self, filepath: str) -> Channel:
        """Return the Channel enum for the given CSV filepath."""
        path = Path(filepath)
        name_lower = path.stem.lower()

        # Filename-based heuristic
        if "google" in name_lower:
            return Channel.GOOGLE
        if "meta" in name_lower or "facebook" in name_lower:
            return Channel.META
        if "bing" in name_lower or "microsoft" in name_lower:
            return Channel.MICROSOFT

        # Column-based detection
        try:
            df = pd.read_csv(filepath, nrows=1)
            cols = set(df.columns.tolist())
            return self._detect_from_columns(cols)
        except Exception:
            return Channel.UNKNOWN

    def detect_from_df(self, df: pd.DataFrame) -> Channel:
        """Detect channel from a DataFrame's column names."""
        cols = set(df.columns.tolist())
        return self._detect_from_columns(cols)

    def _detect_from_columns(self, cols: set) -> Channel:
        for channel_name, cfg in self.channels.items():
            detection_cols = set(cfg.get("detection_columns", []))
            if detection_cols and detection_cols.issubset(cols):
                return Channel(channel_name)
        return Channel.UNKNOWN
