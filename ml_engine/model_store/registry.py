"""Model registry helpers for bundle metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def registry_path(bundle_path: str | Path) -> Path:
    return Path(bundle_path).with_name("metadata.json")


def build_metadata(
    *,
    version: str = "1.0.0",
    algorithm: str = "LightGBM + Prophet",
    wmape: float | None = None,
    trained: str | None = None,
    features: int | None = None,
    extra: dict | None = None,
) -> dict:
    metadata = {
        "version": version,
        "algorithm": algorithm,
        "wmape": wmape,
        "trained": trained or datetime.now(timezone.utc).date().isoformat(),
        "features": features,
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_registry(bundle_path: str | Path, metadata: dict) -> Path:
    path = registry_path(bundle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_registry(bundle_path: str | Path) -> dict:
    path = registry_path(bundle_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))