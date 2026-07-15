"""Canonical schema builder: maps any platform's DataFrame to unified format."""
import pandas as pd
import yaml
from loguru import logger

from src.ingestion.detector import Channel


CANONICAL_COLUMNS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "clicks", "impressions", "conversions", "revenue",
    "roas", "currency", "daily_budget",
]


class CanonicalSchemaBuilder:
    """Maps raw DataFrames to the canonical schema based on channel config."""

    def __init__(self, config_path: str = "config/channels.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.channels_cfg = cfg["channels"]

    def build(self, df: pd.DataFrame, channel: Channel) -> pd.DataFrame:
        """Return a DataFrame conforming to the canonical schema."""
        cfg = self.channels_cfg.get(channel.value)
        if cfg is None:
            raise ValueError(f"No config for channel: {channel}")

        out = pd.DataFrame()

        # Date
        out["date"] = pd.to_datetime(df[cfg["date_column"]], errors="coerce")

        # Channel tag
        out["channel"] = channel.value

        # Campaign ID
        cid = cfg.get("campaign_id_column")
        out["campaign_id"] = df[cid].astype(str) if cid and cid in df.columns else None

        # Campaign name
        out["campaign_name"] = df[cfg["campaign_column"]].astype(str).str.strip()

        # Campaign type
        ct = cfg.get("campaign_type_column")
        out["campaign_type"] = df[ct].astype(str).str.strip() if ct and ct in df.columns else "Unknown"

        # Numeric fields
        out["spend"] = pd.to_numeric(df[cfg["spend_column"]], errors="coerce").fillna(0.0)
        out["clicks"] = pd.to_numeric(df[cfg["clicks_column"]], errors="coerce").fillna(0).astype(int) if cfg.get("clicks_column") and cfg["clicks_column"] in df.columns else 0
        out["impressions"] = pd.to_numeric(df[cfg["impressions_column"]], errors="coerce").fillna(0).astype(int) if cfg.get("impressions_column") and cfg["impressions_column"] in df.columns else 0
        out["conversions"] = pd.to_numeric(df[cfg["conversions_column"]], errors="coerce").fillna(0.0)
        out["revenue"] = pd.to_numeric(df[cfg["revenue_column"]], errors="coerce").fillna(0.0)

        # Derived
        out["roas"] = (out["revenue"] / out["spend"]).replace([float("inf"), -float("inf")], 0.0).fillna(0.0)
        out["currency"] = cfg.get("currency", "USD")

        # Daily budget
        out["daily_budget"] = pd.to_numeric(df["DailyBudget"], errors="coerce") if "DailyBudget" in df.columns else None

        out = out.sort_values("date").reset_index(drop=True)
        logger.info(f"Canonical schema built: {len(out)} rows, channel={channel.value}")
        return out

    def build_from_many(self, loaded_files: list) -> pd.DataFrame:
        """Merge multiple channel DataFrames into one canonical dataset."""
        frames = []
        for lf in loaded_files:
            try:
                canonical = self.build(lf.df, lf.channel)
                frames.append(canonical)
            except Exception as e:
                logger.error(f"Failed to canonicalize {lf.filepath}: {e}")
        if not frames:
            raise RuntimeError("No files could be canonicalized.")
        combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        logger.info(f"Combined canonical dataset: {len(combined)} rows from {len(frames)} sources")
        return combined
