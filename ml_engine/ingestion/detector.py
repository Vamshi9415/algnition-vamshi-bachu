"""Source detection engine: identifies which ad platform a CSV belongs to."""
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml

from ml_engine.paths import config_path as default_config_path


class Channel(str, Enum):
    GOOGLE = "google"
    META = "meta"
    MICROSOFT = "microsoft"
    UNKNOWN = "unknown"


class SourceDetector:
    """Detects the ad channel of a CSV file by inspecting filename and columns."""

    def __init__(self, config_path: str | Path | None = None):
        resolved_path = Path(config_path) if config_path else default_config_path("channels.yaml")
        with open(resolved_path, encoding="utf-8") as f:
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
            candidate_sets = cfg.get("detection_columns", [])
            # Support either a flat list of columns (single required set) or a
            # list of alternative sets (a file matches if ANY set is a subset).
            if candidate_sets and not isinstance(candidate_sets[0], list):
                candidate_sets = [candidate_sets]
            for detection_cols in candidate_sets:
                if set(detection_cols).issubset(cols):
                    return Channel(channel_name)
        return Channel.UNKNOWN

