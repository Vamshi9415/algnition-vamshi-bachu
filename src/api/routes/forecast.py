"""Forecast endpoint: accepts CSV uploads, returns probabilistic forecast JSON."""
import tempfile
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from src.pipeline.orchestrator import ForecastPipeline

router = APIRouter()


@router.post("/forecast")
async def run_forecast(
    files: Annotated[list[UploadFile], File(description="One or more ad platform CSVs")],
    horizon_days: int = 60,
):
    """Upload Google/Meta/Microsoft CSVs and receive a full probabilistic forecast."""
    if not files:
        raise HTTPException(400, "No files provided")

    saved_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            dest = Path(tmpdir) / f.filename
            dest.write_bytes(await f.read())
            saved_paths.append(str(dest))
            logger.info(f"Received file: {f.filename}")

        try:
            pipeline = ForecastPipeline(horizon_days=horizon_days)
            result = pipeline.run(saved_paths)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise HTTPException(500, f"Forecasting failed: {e}")

    return JSONResponse(content=result)
