"""CSV loader: reads raw CSVs and returns DataFrames with metadata."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from src.ingestion.detector import Channel, SourceDetector


@dataclass
class LoadedFile:
    filepath: str
    channel: Channel
    df: pd.DataFrame
    row_count: int
    columns: list[str]


class CSVLoader:
    """Loads one or multiple raw CSV files and attaches channel metadata."""

    def __init__(self):
        self.detector = SourceDetector()

    def load(self, filepath: str) -> LoadedFile:
        logger.info(f"Loading: {filepath}")
        df = pd.read_csv(filepath)
        channel = self.detector.detect(filepath)
        logger.info(f"Detected channel: {channel} ({len(df)} rows)")
        return LoadedFile(
            filepath=filepath,
            channel=channel,
            df=df,
            row_count=len(df),
            columns=df.columns.tolist(),
        )

    def load_many(self, filepaths: list[str]) -> list[LoadedFile]:
        return [self.load(fp) for fp in filepaths]
