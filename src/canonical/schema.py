"""Canonical schema builder: maps any platform's raw columns to the unified 15-column schema."""
import pandas as pd
from loguru import logger

# Full 15-column canonical schema
CANONICAL_COLUMNS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "clicks", "impressions", "conversions", "revenue", "roas",
    "currency", "device", "country", "status",
]

# Column mapping tables per platform
GOOGLE_MAP = {
    "Date": "date",
    "Campaign": "campaign_name",
    "Campaign ID": "campaign_id",
    "Campaign type": "campaign_type",
    "Campaign Type": "campaign_type",
    "Cost": "spend",
    "Clicks": "clicks",
    "Impressions": "impressions",
    "Conversions": "conversions",
    "Conv. value": "revenue",
    "Conv. value / cost": "roas",
    "Currency code": "currency",
    "Device": "device",
    "Country": "country",
    "Campaign status": "status",
    "Status": "status",
}

META_MAP = {
    "Reporting starts": "date",
    "Campaign name": "campaign_name",
    "Campaign ID": "campaign_id",
    "Objective": "campaign_type",
    "Amount spent (USD)": "spend",
    "Link clicks": "clicks",
    "Impressions": "impressions",
    "Purchases": "conversions",
    "Purchases conversion value": "revenue",
    "Purchase ROAS (return on ad spend)": "roas",
    "Currency": "currency",
    "Device platforms": "device",
    "Country": "country",
    "Delivery": "status",
}

MICROSOFT_MAP = {
    "TimePeriod": "date",
    "CampaignName": "campaign_name",
    "CampaignId": "campaign_id",
    "CampaignType": "campaign_type",
    "Spend": "spend",
    "Clicks": "clicks",
    "Impressions": "impressions",
    "Conversions": "conversions",
    "Revenue": "revenue",
    "ReturnOnAdSpend": "roas",
    "CurrencyCode": "currency",
    "DeviceType": "device",
    "Country": "country",
    "CampaignStatus": "status",
}

PLATFORM_MAPS = {
    "google": GOOGLE_MAP,
    "meta": META_MAP,
    "microsoft": MICROSOFT_MAP,
}


class CanonicalSchemaBuilder:
    """Maps any platform DataFrame to the 15-column canonical schema."""

    def build(self, df: pd.DataFrame, channel: str) -> pd.DataFrame:
        col_map = PLATFORM_MAPS.get(channel, {})
        renamed = df.rename(columns=col_map)

        # Ensure all canonical columns exist
        for col in CANONICAL_COLUMNS:
            if col not in renamed.columns:
                renamed[col] = None

        canonical = renamed[CANONICAL_COLUMNS].copy()
        canonical["channel"] = channel
        canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")

        # Numeric coercion
        for num_col in ["spend", "clicks", "impressions", "conversions", "revenue", "roas"]:
            canonical[num_col] = pd.to_numeric(canonical[num_col], errors="coerce")

        # String normalization
        for str_col in ["campaign_name", "campaign_type", "currency", "device", "country", "status"]:
            canonical[str_col] = canonical[str_col].astype(str).str.strip().str.lower()

        logger.info(f"Canonical schema built for '{channel}': {len(canonical)} rows")
        return canonical

    def build_from_many(self, loaded_files: list) -> pd.DataFrame:
        frames = []
        for item in loaded_files:
            df = self.build(item["df"], item["channel"])
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Combined canonical DataFrame: {len(combined)} rows, {len(frames)} platforms")
        return combined
