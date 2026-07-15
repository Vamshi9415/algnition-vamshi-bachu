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
# Each platform is supported in two possible export shapes:
#  - the human-readable Google/Meta Ads UI export column names, and
#  - the API-style column names (e.g. Google Ads API "metrics_*"/"segments_*"),
#    which is the shape of the CSVs actually shipped in data/raw/.
GOOGLE_MAP = {
    "Date": "date",
    "segments_date": "date",
    "Campaign": "campaign_name",
    "Campaign ID": "campaign_id",
    "Campaign type": "campaign_type",
    "Campaign Type": "campaign_type",
    "campaign_advertising_channel_type": "campaign_type",
    "Cost": "spend",
    "metrics_cost_micros": "spend",  # raw value is in micros; converted to currency units in build()
    "Clicks": "clicks",
    "metrics_clicks": "clicks",
    "Impressions": "impressions",
    "metrics_impressions": "impressions",
    "Conversions": "conversions",
    "metrics_conversions": "conversions",
    "Conv. value": "revenue",
    "metrics_conversions_value": "revenue",
    "Conv. value / cost": "roas",
    "Currency code": "currency",
    "Device": "device",
    "Country": "country",
    "Campaign status": "status",
    "Status": "status",
}

META_MAP = {
    "Reporting starts": "date",
    "date_start": "date",
    "Campaign name": "campaign_name",
    "Campaign ID": "campaign_id",
    "Objective": "campaign_type",
    "Amount spent (USD)": "spend",
    "Link clicks": "clicks",
    "Impressions": "impressions",
    "Purchases": "conversions",
    "Purchases conversion value": "revenue",
    # This sample export only has a single "conversion" value column (no separate
    # purchase-count column). Its magnitude/decimals match a dollar value, not a count.
    "conversion": "revenue",
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
        is_micros_cost = "metrics_cost_micros" in df.columns
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

        # Google Ads API "cost_micros" is denominated in micros (1e-6 currency units)
        if is_micros_cost:
            canonical["spend"] = canonical["spend"] / 1_000_000

        # String normalization
        for str_col in ["campaign_name", "campaign_type", "currency", "device", "country", "status"]:
            canonical[str_col] = canonical[str_col].astype(str).str.strip().str.lower()

        logger.info(f"Canonical schema built for '{channel}': {len(canonical)} rows")
        return canonical

    def build_from_many(self, loaded_files: list) -> pd.DataFrame:
        frames = []
        for item in loaded_files:
            channel = item.channel.value if hasattr(item.channel, "value") else item.channel
            df = self.build(item.df, channel)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Combined canonical DataFrame: {len(combined)} rows, {len(frames)} platforms")
        return combined

