"""Backend API entry point.

This module exposes the FastAPI app so the backend can be run as:
    uvicorn backend.app:app --reload
"""

from backend.api.main import app

__all__ = ["app"]

